"""hZZ signal-strength (mu) regression with the toy model, verbatim.

The MODEL is the toy's, unchanged: TemplateDiscriminator([128,256,128]), 100
epochs, lr 3e-3, wd 1e-4, batch 2048, 80/20 split, 100k events per class,
trained by `train_discriminator` and profiled by `profile_mass_estimate`
(mean log-probs -> -2dlogL -> quadratic vertex -> monotonic-anchor calibration).

Only TEMPLATE CONSTRUCTION differs, and it is forced: the toy simulates a
separate sample per hypothesis; hZZ gives ONE pool that must be reweighted by
    w(mu) = mu*msq_sig + sqrt(mu)*msq_int + msq_bkg,  divided by msq_sbi,
then hard-label resampled. Grids are per-region: 0.4 and 2.5 are too far apart
for one 3-node grid to bracket both, and a node at mu=0 sits on the boundary
where d(sqrt(mu))/dmu diverges, which dominates the bias (+0.367 -> -0.059 on an
otherwise identical grid). Pool is cut 60/15/25 into train / calib / eval.

Subcommands: floor (analytic ceiling, no NN) | train (members) | combine (scan)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sbi_reasoning.models import (  # noqa: E402
    Standardizer, TemplateDiscriminator, predict_log_probabilities, train_discriminator)
from sbi_reasoning.pipeline import profile_mass_estimate  # noqa: E402

RAW = [f"l{i}_{v}" for i in (1, 2, 3, 4) for v in ("pt", "eta", "phi", "energy")]
FEATURES = RAW + ["4l_mass"]
MSQ = ["msq_sig_sm", "msq_int_sm", "msq_sbi_sm", "msq_bkg_sm"]
GRIDS = {
    "L": {"templates": [0.3, 1.3, 2.3], "sweep": [0.4, 0.8, 1.3, 1.8],
          "observed": "obs_mu0p4", "truth": 0.4},
    "H": {"templates": [1.0, 2.0, 3.0], "sweep": [1.5, 2.0, 2.5, 2.8],
          "observed": "obs_mu2p5", "truth": 2.5},
}
TOY = {"profiler_learning_rate": 0.003, "weight_decay": 0.0001,
       "profiler_epochs": 100, "batch_size": 2048, "mixed_precision": "none"}
HIDDEN, N_PER_CLASS, N_MEMBERS = [128, 256, 128], 100_000, 16
N_EVAL, N_CALIB, OBS_SHARDS, EVAL_BATCH = 200_000, 120_000, 8, 1024
K_VALUES, N_DRAWS, SEEDS = [1, 2, 4, 8], 8, [0, 1, 2]
CACHE = Path("outputs/toy_verbatim_mu")
DEVICE = torch.device("cpu")


def cached(name, build):
    path = CACHE / name
    if path.exists():
        return np.load(path)
    data = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, data)
    return data


def load_pool():
    def build():
        with open("sbi_reduced.csv") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            index = [header.index(c) for c in FEATURES + MSQ]
            return np.asarray([[row[i] for i in index] for row in reader], dtype=np.float64)
    pool = cached("pool.npy", build)
    return pool[:, :len(FEATURES)], pool[:, len(FEATURES):]


def load_observed(tag):
    """Observed shards ship no 4l_mass column, so it is rebuilt from the four-vectors."""
    def build():
        rows = []
        for shard in sorted(Path(tag).glob("observed_*.csv"))[:OBS_SHARDS]:
            with open(shard) as stream:
                reader = csv.reader(stream)
                header = next(reader)
                index = [header.index(c) for c in RAW] + [header.index("n")]
                rows += [[row[i] for i in index] for row in reader if row]
        array = np.asarray(rows, dtype=np.float64)
        energy = px = py = pz = 0.0
        for i in range(4):
            pt, eta, phi, e = (array[:, 4 * i + k] for k in range(4))
            px = px + pt * np.cos(phi)
            py = py + pt * np.sin(phi)
            pz = pz + pt * np.sinh(eta)
            energy = energy + e
        m4l = np.sqrt(np.clip(energy**2 - px**2 - py**2 - pz**2, 0.0, None))
        return np.column_stack([array[:, :16], m4l, array[:, 16]])
    data = cached(f"obs_{tag}.npy", build)
    return data[:, :-1], data[:, -1]


def density_ratio(msq, mu):
    sig, interference, sbi, bkg = (msq[:, k] for k in range(4))
    return np.clip(mu * sig + np.sqrt(mu) * interference + bkg, 0.0, None) / sbi


def resample(index, msq, mu, count, rng):
    weight = density_ratio(msq[index], mu)
    return rng.choice(index, size=count, replace=True, p=weight / weight.sum())


def build_templates(index, msq, grid, count, rng):
    draws = [resample(index, msq, mu, count, rng) for mu in grid]
    labels = [np.full(count, k) for k in range(len(grid))]
    return np.concatenate(draws), np.concatenate(labels)


def split_pool(n, seed=240513):
    """60/15/25 train/calib/eval, disjoint: no member is profiled or calibrated
    on events it trained on."""
    order = np.random.default_rng(seed).permutation(n)
    a = int(0.60 * n)
    b = a + int(0.15 * n)
    return order[:a], order[a:b], order[b:]


def profile(log_probabilities, grid, calibration=None, weight=None):
    """profile_mass_estimate, with an optional n-weighted mean for observed shards."""
    if weight is None:
        return profile_mass_estimate(log_probabilities, grid, calibration)[0]
    nodes = np.asarray(grid, dtype=np.float64)
    mean = (log_probabilities * weight[:, None]).sum(axis=0) / weight.sum()
    n2dll = -2.0 * (mean - mean.max())
    quadratic, linear, _ = np.polyfit(nodes, n2dll, 2)
    if quadratic <= 0.0:
        estimate = float(nodes[np.argmin(n2dll)])
    else:
        estimate = float(np.clip(-linear / (2.0 * quadratic), nodes.min(), nodes.max()))
    if calibration is None:
        return estimate
    return float(np.interp(estimate, calibration["raw_mass_gev"],
                           calibration["calibrated_mass_gev"]))


def calibration_from(anchors, grid):
    """The toy refuses to calibrate on non-monotonic anchors; that refusal is a result."""
    if not bool(np.all(np.diff(anchors) > 0.0)):
        return None
    return {"raw_mass_gev": anchors, "calibrated_mass_gev": grid}


def cmd_floor(args):
    """Exact analytic posterior in place of the network: isolates ESTIMATOR bias."""
    _, msq = load_pool()
    _, calib_pool, eval_pool = split_pool(len(msq))

    def log_posterior(grid, index):
        norm = [float(np.mean(density_ratio(msq, mu))) for mu in grid]
        stacked = np.stack([np.log(np.clip(density_ratio(msq[index], mu) / z, 1e-300, None))
                            for mu, z in zip(grid, norm)], axis=1)
        return stacked - np.log(np.exp(stacked).sum(axis=1, keepdims=True))

    print(f"{'grid':<24}{'truth':>7}{'recovered':>11}{'bias':>9}")
    for name, spec in GRIDS.items():
        grid = spec["templates"]
        rng = np.random.default_rng(7)
        anchors = [profile(log_posterior(grid, resample(calib_pool, msq, mu, N_CALIB, rng)), grid)
                   for mu in grid]
        calibration = calibration_from(anchors, grid)
        for mu in spec["sweep"] + [spec["truth"]]:
            index = resample(eval_pool, msq, mu, N_EVAL, np.random.default_rng(100))
            estimate = profile(log_posterior(grid, index), grid, calibration)
            print(f"{name} {str(grid):<20}{mu:>7}{estimate:>11.3f}{estimate - mu:>+9.3f}")


def cmd_train(args):
    spec = GRIDS[args.grid]
    grid = spec["templates"]
    features, msq = load_pool()
    train_pool, calib_pool, eval_pool = split_pool(len(msq))
    out = CACHE / f"grid{args.grid}"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    calib_index, calib_labels = build_templates(calib_pool, msq, grid, N_CALIB, rng)
    evaluation = {"calib": features[calib_index]}
    for mu in spec["sweep"]:
        for seed in SEEDS:
            evaluation[f"sweep_{mu}_s{seed}"] = features[
                resample(eval_pool, msq, mu, N_EVAL, np.random.default_rng(100 + seed))]
    observed_features, observed_weight = load_observed(spec["observed"])
    evaluation[spec["observed"]] = observed_features
    meta = out / "eval_meta.npz"
    if not meta.exists():
        np.savez_compressed(meta, calib_labels=calib_labels,
                            **{f"{spec['observed']}_weight": observed_weight})

    for member in range(args.members_from, args.members_to):
        target = out / f"member_{member:02d}.npz"
        if target.exists():
            continue
        rng = np.random.default_rng(1000 + member)
        torch.manual_seed(2000 + member)
        draw, labels = build_templates(train_pool, msq, grid, N_PER_CLASS, rng)
        # Split on POOL EVENT ID: resampling one shared pool would otherwise place
        # the same event on both sides of the train/validation split.
        unique = rng.permutation(np.unique(draw))
        is_train_id = np.zeros(len(features), dtype=bool)
        is_train_id[unique[:int(0.8 * len(unique))]] = True
        mask = is_train_id[draw]

        xt, yt = features[draw[mask]].astype(np.float32), labels[mask].astype(np.int64)
        xv, yv = features[draw[~mask]].astype(np.float32), labels[~mask].astype(np.int64)
        scaler = Standardizer.fit(xt)
        model = TemplateDiscriminator(len(FEATURES), len(grid), HIDDEN)
        history = train_discriminator(model, scaler.transform(xt), yt,
                                      scaler.transform(xv), yv, TOY, DEVICE, rng,
                                      f"{args.grid}-m{member:02d}")
        result = {"loss": np.asarray(history["loss"]),
                  "accuracy": np.asarray(history["validation_accuracy"]),
                  "validation_labels": yv}
        for name, array in evaluation.items():
            result[name] = predict_log_probabilities(
                model, scaler.transform(array.astype(np.float32)), DEVICE, EVAL_BATCH
            ).astype(np.float32)
        np.savez_compressed(target, **result)
        print(f"[member {member}] acc={result['accuracy'][-1]:.4f} "
              f"loss={result['loss'][-1]:.4f}", flush=True)


def subsets_of_size(count, k, rng):
    if k == 1:
        return [(i,) for i in range(min(count, N_DRAWS))]
    combos = list(itertools.combinations(range(count), k))
    picked = rng.choice(len(combos), size=min(N_DRAWS, len(combos)), replace=False)
    return [combos[i] for i in picked]


def cmd_combine(args):
    spec = GRIDS[args.grid]
    grid, sweep = spec["templates"], spec["sweep"]
    tag, truth = spec["observed"], spec["truth"]
    directory = CACHE / f"grid{args.grid}"
    members = [np.load(p) for p in sorted(directory.glob("member_*.npz"))]
    meta = np.load(directory / "eval_meta.npz")
    calib_labels = meta["calib_labels"]

    loss = [float(m["loss"][-1]) for m in members]
    accuracy = [float(m["accuracy"][-1]) for m in members]
    print(f"[grid {args.grid}] {len(members)} members | loss {np.mean(loss):.4f} "
          f"(ln3={np.log(3):.4f}) | acc {np.mean(accuracy):.4f} (chance={1/3:.4f})")

    rows = []
    rng = np.random.default_rng(0)
    for k in K_VALUES:
        monotonic_flags, per_truth, observed = [], {mu: [] for mu in sweep}, []
        for subset in subsets_of_size(len(members), k, rng):
            calib = np.mean([members[i]["calib"] for i in subset], axis=0)
            anchors = [profile(calib[calib_labels == c], grid) for c in range(len(grid))]
            calibration = calibration_from(anchors, grid)
            monotonic_flags.append(calibration is not None)
            for mu in sweep:
                for seed in SEEDS:
                    stacked = np.mean([members[i][f"sweep_{mu}_s{seed}"] for i in subset], axis=0)
                    per_truth[mu].append(profile(stacked, grid, calibration))
            stacked = np.mean([members[i][tag] for i in subset], axis=0)
            observed.append(profile(stacked, grid, calibration, weight=meta[f"{tag}_weight"]))

        rate = round(float(np.mean(monotonic_flags)), 3)
        entries = [("simulated", mu, per_truth[mu]) for mu in sweep] + [("observed", truth, observed)]
        for dataset, mu, values in entries:
            values = np.asarray(values)
            rows.append({"grid": args.grid, "dataset": dataset, "truth_mu": mu, "K": k,
                         "recovered_mu": round(float(values.mean()), 4),
                         "bias": round(float(values.mean() - mu), 4),
                         "spread": round(float(values.std()), 4),
                         "monotonic_anchor_rate": rate})
        print(f"  K={k}: monotonic anchors in {rate:.0%} of draws", flush=True)

    out = CACHE / f"ensemble_scan_grid{args.grid}.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["floor", "train", "combine"])
    parser.add_argument("--grid", choices=sorted(GRIDS), default="L")
    parser.add_argument("--members-from", type=int, default=0)
    parser.add_argument("--members-to", type=int, default=N_MEMBERS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    DEVICE = torch.device(args.device)
    {"floor": cmd_floor, "train": cmd_train, "combine": cmd_combine}[args.command](args)
