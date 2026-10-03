"""hZZ (gg->ZZ->4l) signal-strength mu regression with the toy's profiler.

The toy MODEL is used verbatim: TemplateDiscriminator([128,256,128]), 100 epochs,
lr 3e-3, wd 1e-4, batch 2048, trained by `train_discriminator` and profiled by
`profile_mass_estimate` (mean log-probs -> -2dlogL -> parabola through the three
template nodes -> clipped vertex -> monotonic-anchor calibration).

Only TEMPLATE CONSTRUCTION differs, and it is forced: the toy simulates one sample
per hypothesis, hZZ ships ONE pool (ggZZ_sbi) carrying per-event matrix elements, so
templates are hard-label importance resamples of it with
    w(mu) = mu*msq_sig + sqrt(mu)*msq_int + msq_bkg,   divided by msq_sbi (generation).
Grids are per region: 0.4 and 2.5 are too far apart for one 3-node grid, and a node
at mu=0 sits where d(sqrt(mu))/dmu diverges.  Pool is cut 60/15/25 train/calib/eval.

Every estimate also records the RAW vertex -b/(2a) (before clip and calibration) and
whether the parabola is concave (quadratic<=0): then no minimum exists and the toy
falls back to argmin over the nodes, which is a category label, not an estimate.

Commands
  floor  exact analytic posterior instead of a network -> isolates ESTIMATOR bias
  mlp    toy MLP members, ensembles of K = 1/2/4/8 (trains missing members first)
  heads  trainable head on the FROZEN pretrained EveNet-Lite encoder (no ensembling)
"""
from __future__ import annotations

import argparse
import copy
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
    "L": {"templates": [0.3, 1.3, 2.3], "sweep": [0.4, 0.8, 1.3, 1.8], "observed": "obs_mu0p4", "truth": 0.4},
    "H": {"templates": [1.0, 2.0, 3.0], "sweep": [1.5, 2.0, 2.5, 2.8], "observed": "obs_mu2p5", "truth": 2.5},
}
# perfect-classifier grid-geometry comparison (the mu = 0 boundary result)
GEOMETRY = [[0.0, 1.0, 3.0], [0.0, 1.0, 2.0], [0.3, 1.3, 2.3], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]
TOY = {"profiler_learning_rate": 0.003, "weight_decay": 0.0001,
       "profiler_epochs": 100, "batch_size": 2048, "mixed_precision": "none"}
HIDDEN, N_PER_CLASS, N_EVAL, N_CALIB = [128, 256, 128], 100_000, 200_000, 120_000
OBS_SHARDS, K_VALUES, N_DRAWS, SEEDS = 8, [1, 2, 4, 8], 8, [0, 1, 2]
HEAD_SPECS = {"linear": None, "tiny": 64, "small": 256, "full": "evenet"}
CACHE, RESULTS = Path("outputs/cache"), Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """First OBS_SHARDS shards (~500k events, the agreed budget).  The shards ship no
    4l_mass column, so it is rebuilt from the four-vectors.  Returns features, weight n."""
    def build():
        rows = []
        for shard in sorted(Path(tag).glob("observed_*.csv"))[:OBS_SHARDS]:
            with open(shard) as stream:
                reader = csv.reader(stream)
                header = next(reader)
                index = [header.index(c) for c in RAW] + [header.index("n")]
                rows += [[row[i] for i in index] for row in reader if row]
        a = np.asarray(rows, dtype=np.float64)
        e = px = py = pz = 0.0
        for i in range(4):
            pt, eta, phi, en = (a[:, 4 * i + k] for k in range(4))
            px, py, pz, e = px + pt * np.cos(phi), py + pt * np.sin(phi), pz + pt * np.sinh(eta), e + en
        m4l = np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0.0, None))
        return np.column_stack([a[:, :16], m4l, a[:, 16]])
    data = cached(f"{tag}.npy", build)
    return data[:, :-1], data[:, -1]


def density_ratio(msq, mu):
    sig, interference, sbi, bkg = (msq[:, k] for k in range(4))
    return np.clip(mu * sig + np.sqrt(mu) * interference + bkg, 0.0, None) / sbi


def resample(index, msq, mu, count, rng):
    weight = density_ratio(msq[index], mu)
    return rng.choice(index, size=count, replace=True, p=weight / weight.sum())


def build_templates(index, msq, grid, count, rng):
    draws = [resample(index, msq, mu, count, rng) for mu in grid]
    return np.concatenate(draws), np.concatenate([np.full(count, k) for k in range(len(grid))])


def split_pool(n, seed=240513):
    order = np.random.default_rng(seed).permutation(n)
    a, b = int(0.60 * n), int(0.60 * n) + int(0.15 * n)
    return order[:a], order[a:b], order[b:]


def train_mask(draw, rng, n):
    """80/20 split on POOL EVENT ID: resampling one shared pool would otherwise put
    the same event on both sides (the toy's separate simulations cannot)."""
    unique = rng.permutation(np.unique(draw))
    is_train = np.zeros(n, dtype=bool)
    is_train[unique[:int(0.8 * len(unique))]] = True
    return is_train[draw]


def profile(logp, grid, calibration=None, weight=None):
    """The toy's estimate; observed shards use an n-weighted mean of log-probs."""
    if weight is None:
        return profile_mass_estimate(logp, grid, calibration)[0]
    nodes = np.asarray(grid, dtype=np.float64)
    mean = (logp * weight[:, None]).sum(axis=0) / weight.sum()
    n2dll = -2.0 * (mean - mean.max())
    a, b, _ = np.polyfit(nodes, n2dll, 2)
    est = float(nodes[np.argmin(n2dll)]) if a <= 0.0 else float(np.clip(-b / (2 * a), nodes.min(), nodes.max()))
    return est if calibration is None else float(
        np.interp(est, calibration["raw_mass_gev"], calibration["calibrated_mass_gev"]))


def vertex(logp, grid, weight=None):
    """Raw -b/(2a) before clip/calibration; (nan, True) if the parabola is concave."""
    mean = logp.mean(axis=0) if weight is None else (logp * weight[:, None]).sum(axis=0) / weight.sum()
    a, b, _ = np.polyfit(np.asarray(grid, dtype=np.float64), -2.0 * (mean - mean.max()), 2)
    return (float("nan"), True) if a <= 0.0 else (float(-b / (2.0 * a)), False)


def calibration_from(anchors, grid):
    """The toy refuses to calibrate on non-monotonic anchors; that refusal is a result."""
    return {"raw_mass_gev": anchors, "calibrated_mass_gev": grid} if np.all(np.diff(anchors) > 0) else None


def record(rows, config, seed, dataset, truth, grid, logp, calibration, weight=None, **extra):
    raw, no_vertex = vertex(logp, grid, weight)
    rows.append({"config": config, "seed": seed, "dataset": dataset, "truth_mu": truth,
                 "raw_vertex": raw, "no_vertex": no_vertex,
                 "clipped_mu": profile(logp, grid, calibration, weight=weight),
                 "monotonic": calibration is not None,
                 "grid_min": min(grid), "grid_max": max(grid), **extra})


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {path} ({len(rows)} rows)", flush=True)


def eval_sets(msq, spec, calib_pool, eval_pool):
    calib_index, calib_labels = build_templates(calib_pool, msq, spec["templates"], N_CALIB,
                                                np.random.default_rng(7))
    sweeps = {(mu, s): resample(eval_pool, msq, mu, N_EVAL, np.random.default_rng(100 + s))
              for mu in spec["sweep"] for s in SEEDS}
    return calib_index, calib_labels, sweeps


def score(rows, config, seed, spec, logp, calib_logp, calib_labels, sweeps, obs_logp, obs_weight, **extra):
    """Anchors from the calib templates, then every sweep point and the observed set."""
    grid = spec["templates"]
    calibration = calibration_from(
        [profile(calib_logp[calib_labels == c], grid) for c in range(len(grid))], grid)
    for mu in spec["sweep"]:
        for s in SEEDS:
            record(rows, config, seed, "simulated", mu, grid, logp(mu, s), calibration, **extra)
    record(rows, config, seed, "observed", spec["truth"], grid, obs_logp, calibration,
           weight=obs_weight, **extra)
    return calibration


# ------------------------------------------------------------------ floor
def cmd_floor(args):
    """Exact posterior p(mu_k|x) from the matrix elements replaces the network, so any
    bias left is the estimator's own (parabola fitted to a non-quadratic likelihood)."""
    _, msq = load_pool()
    _, calib_pool, eval_pool = split_pool(len(msq))
    norm = {}

    def logpost(grid, index):
        for mu in grid:
            norm.setdefault(mu, float(np.mean(density_ratio(msq, mu))))
        s = np.stack([np.log(np.clip(density_ratio(msq[index], mu) / norm[mu], 1e-300, None))
                      for mu in grid], axis=1)
        return s - np.log(np.exp(s).sum(axis=1, keepdims=True))

    jobs = [(tuple(spec["templates"]), mu) for spec in GRIDS.values() for mu in spec["sweep"]]
    jobs += [(tuple(g), t) for g in GEOMETRY for t in (0.4, 2.5) if (tuple(g), t) not in jobs]
    rows = []
    for grid, truth in jobs:
        grid = list(grid)
        for s in range(args.seeds):
            rng = np.random.default_rng(500 + s)
            anchors = [profile(logpost(grid, resample(calib_pool, msq, m, N_CALIB, rng)), grid) for m in grid]
            index = resample(eval_pool, msq, truth, N_EVAL, np.random.default_rng(900 + s))
            record(rows, "perfect", s, "simulated", truth, grid, logpost(grid, index),
                   calibration_from(anchors, grid), grid_label="{" + ",".join(f"{g:g}" for g in grid) + "}")
        print(f"[floor] {grid} truth {truth}", flush=True)
    write(rows, Path(args.out or RESULTS / "floor.csv"))


# ------------------------------------------------------------------ toy MLP
def cmd_mlp(args):
    spec = GRIDS[args.grid]
    grid = spec["templates"]
    features, msq = load_pool()
    train_pool, calib_pool, eval_pool = split_pool(len(msq))
    obs_features, obs_weight = load_observed(spec["observed"])
    calib_index, calib_labels, sweeps = eval_sets(msq, spec, calib_pool, eval_pool)
    store = Path(args.members or CACHE / f"members_grid{args.grid}")
    store.mkdir(parents=True, exist_ok=True)

    for seed in range(args.n_members):
        target = store / f"member_{seed:02d}.npz"
        if target.exists():
            continue
        rng = np.random.default_rng(1000 + seed)
        torch.manual_seed(2000 + seed)
        draw, labels = build_templates(train_pool, msq, grid, N_PER_CLASS, rng)
        m = train_mask(draw, rng, len(features))
        xt, yt = features[draw[m]].astype(np.float32), labels[m].astype(np.int64)
        xv, yv = features[draw[~m]].astype(np.float32), labels[~m].astype(np.int64)
        scaler = Standardizer.fit(xt)
        model = TemplateDiscriminator(len(FEATURES), len(grid), HIDDEN)
        hist = train_discriminator(model, scaler.transform(xt), yt, scaler.transform(xv), yv,
                                   TOY, DEVICE, rng, f"{args.grid}-m{seed:02d}")
        evals = {"calib": features[calib_index], "observed": obs_features,
                 **{f"sweep_{mu}_s{s}": features[i] for (mu, s), i in sweeps.items()}}
        out = {k: predict_log_probabilities(model, scaler.transform(v.astype(np.float32)), DEVICE, 1024
                                            ).astype(np.float32) for k, v in evals.items()}
        np.savez_compressed(target, loss=np.asarray(hist["loss"]),
                            accuracy=np.asarray(hist["validation_accuracy"]), **out)

    members = [np.load(p) for p in sorted(store.glob("member_*.npz"))]
    rows, rng = [], np.random.default_rng(0)
    for k in K_VALUES:
        if k == 1:
            combos = [(i,) for i in range(min(len(members), N_DRAWS))]
        else:
            allc = list(itertools.combinations(range(len(members)), k))
            combos = [allc[i] for i in rng.choice(len(allc), size=min(N_DRAWS, len(allc)), replace=False)]
        for n, subset in enumerate(combos):
            avg = lambda key: np.mean([members[i][key] for i in subset], axis=0)  # noqa: E731
            score(rows, f"MLP K={k}", n, spec, lambda mu, s: avg(f"sweep_{mu}_s{s}"),
                  avg("calib"), calib_labels, sweeps, avg("observed"), obs_weight)
        print(f"[mlp K={k}] {len(combos)} draws from {len(members)} members", flush=True)
    write(rows, Path(args.out or RESULTS / f"mlp_grid{args.grid}.csv"))


# ------------------------------------------------------------------ EveNet heads
class TokenHead(torch.nn.Module):
    """Head on the frozen event token; hidden=None is a linear probe."""

    def __init__(self, dim, hidden, n_classes=3):
        super().__init__()
        self.net = (torch.nn.Linear(dim, n_classes) if hidden is None else torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, n_classes)))

    def forward(self, x=None, x_mask=None, event_token=None):
        return {"classification/EVENT": self.net(event_token)}


def evenet_inputs(features):
    """Toy features -> EveNet schema.  Per lepton [energy, pt, eta, phi, isBTag=0,
    isLepton=1, charge=0]; charge is ZERO-FILLED (not in this dataset).  Globals
    [met=0, met_phi=0, nLepton=4, nbJet=0, nJet=0, HT, HT_lep, M_all, M_leps, M_bjets=0];
    met=0 is correct for a fully visible 4l final state."""
    n = len(features)
    x = np.zeros((n, 4, 7), dtype=np.float32)
    for i in range(4):
        x[:, i, :4] = features[:, [4 * i + 3, 4 * i, 4 * i + 1, 4 * i + 2]]
        x[:, i, 5] = 1.0
    ht = sum(features[:, 4 * i] for i in range(4))
    g = np.zeros((n, 10), dtype=np.float32)
    g[:, 2], g[:, 5], g[:, 6], g[:, 7], g[:, 8] = 4.0, ht, ht, features[:, 16], features[:, 16]
    return x, g, np.ones((n, 4), dtype=bool)


def evenet_embeddings(features, obs_features, train_pool, grid_name):
    """Frozen pretrained encoder run once; cached.  Needs the evenet package and, at
    this size, a GPU.  Returns (pool_cache, obs_cache, backbone) as (emb, mask, token)."""
    from evenet_lite import EvenetLiteClassifier
    from evenet_lite.callbacks import EvenetLiteNormalizer
    clf = EvenetLiteClassifier(class_labels=["lo", "mid", "hi"], device="auto", pretrained=True,
                               global_input_dim=10, sequential_input_dim=7, n_ensemble=1)
    single = clf.model.models[0].to(DEVICE)
    for p in single.backbone.parameters():
        p.requires_grad = False
    single.backbone.eval()
    trainable = sum(p.numel() for p in clf.model.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in single.Classification.parameters()), "backbone not frozen"
    x, g, mask = evenet_inputs(features)
    sub = train_pool[:200000]
    norm = EvenetLiteNormalizer(EvenetLiteClassifier.DEFAULT_NORMALIZATION_RULES)
    norm.fit({"x": x[sub], "globals": g[sub], "mask": mask[sub]}, EvenetLiteClassifier.DEFAULT_FEATURE_NAMES)

    @torch.no_grad()
    def run(x, g, mask):
        out = []
        for i in range(0, len(x), 4096):
            b = norm.transform({k: torch.from_numpy(v[i:i + 4096]) for k, v in
                                (("x", x), ("globals", g), ("mask", mask))})
            e, m, t = single.backbone(b["x"].float().to(DEVICE), b["mask"].to(DEVICE), b["globals"].float().to(DEVICE))
            out.append((e.half().cpu(), m.cpu(), t.half().cpu()))
        return tuple(torch.cat(z) for z in zip(*out))
    return run(x, g, mask), run(*evenet_inputs(obs_features)), single


def cmd_heads(args):
    """No ensembling: each seed is its own single model; the question is whether the
    pretrained inductive bias replaces ensemble training.  Head trained with the toy's
    regime (AdamW 3e-3, wd 1e-4, batch 2048, 100 epochs, no early stopping)."""
    spec = GRIDS[args.grid]
    grid = spec["templates"]
    features, msq = load_pool()
    train_pool, calib_pool, eval_pool = split_pool(len(msq))
    obs_features, obs_weight = load_observed(spec["observed"])
    calib_index, calib_labels, sweeps = eval_sets(msq, spec, calib_pool, eval_pool)
    emb = CACHE / "embeddings"
    token, obs_token = torch.load(emb / "token_pool.pt"), torch.load(emb / f"token_obs_{args.grid}.pt")
    full = None

    rows = []
    for name in args.head.split(","):
        if name == "full" and full is None:  # needs per-object embeddings: not cached, re-run encoder
            pool_cache, obs_cache, single = evenet_embeddings(features, obs_features, train_pool, args.grid)
            full = (pool_cache, obs_cache, single)
        for seed in [int(s) for s in args.seeds.split(",")]:
            torch.manual_seed(2000 + seed)
            if name == "full":
                head = copy.deepcopy(full[2].Classification)
                torch.manual_seed(2000 + seed)
                for m in head.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
                head = head.to(DEVICE)

                def forward(idx, cache=full[0]):
                    e, msk, t = cache
                    return torch.cat([head(x=e[j].float().to(DEVICE), x_mask=msk[j].to(DEVICE),
                                           event_token=t[j].float().to(DEVICE))["classification/EVENT"]
                                      for j in np.array_split(idx, max(1, len(idx) // 4096))])
            else:
                head = TokenHead(token.shape[-1], HEAD_SPECS[name])

                def forward(idx, cache=(None, None, token)):
                    return head.net(cache[2][idx].float())
            rng = np.random.default_rng(1000 + seed)
            draw, labels = build_templates(train_pool, msq, grid, N_PER_CLASS, rng)
            m = train_mask(draw, rng, len(features))
            tr_idx, tr_y = draw[m], torch.from_numpy(labels[m]).long()
            opt = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.0001)
            for _ in range(100):
                order, losses = rng.permutation(len(tr_idx)), []
                for i in range(0, len(order), 2048):
                    sel = order[i:i + 2048]
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(forward(tr_idx[sel]), tr_y[sel].to(DEVICE))
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
            head.eval()

            def logp(idx, cache=None):
                with torch.no_grad():
                    out = forward(idx) if cache is None else forward(idx, cache)
                    return torch.log_softmax(out, dim=1).cpu().numpy().astype(np.float64)
            with torch.no_grad():
                acc = float((forward(draw[~m]).argmax(1).cpu().numpy() == labels[~m]).mean())
            obs_cache = full[1] if name == "full" else (None, None, obs_token)
            final = float(np.mean(losses))
            cal = score(rows, name, seed, spec, lambda mu, s: logp(sweeps[(mu, s)]),
                        logp(calib_index), calib_labels, sweeps,
                        logp(np.arange(len(obs_features)), obs_cache), obs_weight,
                        loss=final, nats=float(np.log(3) - final), accuracy=acc)
            print(f"[{name} seed {seed}] loss={final:.4f} acc={acc:.4f} monotonic={cal is not None}", flush=True)
    write(rows, Path(args.out or RESULTS / f"heads_grid{args.grid}.csv"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["floor", "mlp", "heads"])
    p.add_argument("--grid", choices=sorted(GRIDS), default="L")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--head", default="linear,tiny,small")
    p.add_argument("--n-members", type=int, default=16)
    p.add_argument("--members")
    p.add_argument("--out")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.command == "floor":
        a.seeds = len(a.seeds.split(","))
    {"floor": cmd_floor, "mlp": cmd_mlp, "heads": cmd_heads}[a.command](a)
