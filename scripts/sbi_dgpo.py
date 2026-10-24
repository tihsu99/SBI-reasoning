"""SBI-guided DGPO with the moment-curve estimator, compared with the pipeline's toy estimator,
truth-reward DGPO and plain supervised reconstruction.

One run (one seed) does, with config/toy.yaml physics and the pipeline's own models:
  1. simulate the 400/500/600 GeV templates, and at every truth mass two independent unlabeled
     datasets: one to adapt on, one to evaluate on;
  2. train the visible and full MLP profilers and the conditional flow (pipeline functions), and
     the moment-curve estimators: visible-only, and full-event anchored on simulated neutrinos;
  3. post-train the flow:
       cond_sft         a flow conditioned on the parent mass, trained on the templates and run at
                        the visible-only moment-curve estimate (no DGPO; the proposal's
                        template-conditioned baseline)
       truth_dgpo       one model, per-event truth reward on template simulation (NeurIPS-style)
       sbi_toy          per dataset, reward -(m_full(reco) - m_visible(data))^2, toy estimators
       sbi_new          the same reward with moment-curve estimators
       sbi_new_shifted  sbi_new aimed at m_visible - delta, where delta is the reconstructed-peak
                        offset that sbi_new produces on fresh template simulation (known mass)
  4. evaluate every reconstruction, plus the analytic baseline and true neutrinos, on the held-out
     dataset at each truth mass.

Only simulation truth is used for training and calibration; target datasets are unlabeled.

    python scripts/sbi_dgpo.py --seed 0                    # full run, ~10 min on a T4
    python scripts/sbi_dgpo.py --seed 0 --truths 450 --n-template 20000 --n-real 20000 \
        --profiler-epochs 10 --flow-epochs 10 --dgpo-iterations 200   # quick check
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sbi_reasoning.models import (  # noqa: E402
    ConditionalFlow, Standardizer, TemplateDiscriminator, predict_log_probabilities, sample_flow,
    train_dgpo, train_discriminator, train_flow)
from sbi_reasoning.moment_curve import MomentCurve, full_features, visible_features  # noqa: E402
from sbi_reasoning.physics import (  # noqa: E402
    baseline_neutrino_momentum, fit_mass_peak, generate_events, reconstruct_parent_masses)
from sbi_reasoning.pipeline import _split_templates, independent_dgpo_seed, profile_mass_estimate  # noqa: E402

HIST_MASS = np.linspace(0.0, 1000.0, 101)  # reconstructed parent mass, both parents per event
HIST_RESID = np.linspace(-1500.0, 1500.0, 121)  # neutrino pz residual, both neutrinos


def log(message):
    print(f"[sbi-dgpo {time.strftime('%H:%M:%S')}] {message}", flush=True)


class ToyEstimator:
    """The pipeline's estimate: MLP log-probs -> quadratic vertex over the templates -> calibration."""

    def __init__(self, profiler, scaler, calibration, masses, device):
        self.profiler, self.scaler, self.calibration, self.masses, self.device = (
            profiler, scaler, calibration, masses, device)

    def estimate(self, obs, nu=None):
        x = obs if nu is None else np.concatenate([obs, nu.reshape(len(obs), -1)], axis=1)
        logp = predict_log_probabilities(self.profiler, self.scaler.transform(x), self.device, 8192)
        return profile_mass_estimate(logp, self.masses, self.calibration)[0]


class Reward:
    """What train_dgpo's `estimator` hook expects: a visible-only target and a candidate estimate."""

    def __init__(self, target, candidate, shift=0.0):
        self.target_fn, self.candidate_fn, self.shift = target, candidate, shift

    def target(self, obs):
        return self.target_fn(obs) - self.shift

    def candidate(self, obs, nu):
        return self.candidate_fn(obs, nu)


def train_truth_dgpo(policy, reference, conditions, targets, dgpo, sampling, iterations, device, rng):
    """NeurIPS-style post-training on simulation: K candidates per event, reward -||nu_hat - nu||^2
    in standardised units, within-event normalised and clipped advantages, shared timestep and
    noise per event, the DGPO preference loss per event plus the velocity KL term."""
    reference = copy.deepcopy(reference).to(device).eval().requires_grad_(False)
    policy.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(dgpo["learning_rate"]))
    batch, group = int(dgpo["batch_size"]), int(dgpo["group_size"])
    beta, kl_beta, t_min = float(dgpo["beta"]), float(dgpo["kl_beta"]), float(dgpo["minimum_timestep"])
    for iteration in range(iterations):
        index = rng.integers(0, len(conditions), size=batch)
        condition = torch.from_numpy(conditions[index]).to(device).repeat(group, 1)
        truth = torch.from_numpy(targets[index]).to(device)
        with torch.no_grad():
            candidates = sample_flow(policy, condition, int(sampling["ode_steps"]),
                                     float(sampling["output_clip_sigma"])).view(group, batch, -1)
            reward = -((candidates - truth[None]) ** 2).sum(-1)
            advantage = ((reward - reward.mean(0)) / (reward.std(0, unbiased=False) + 1e-6)).clamp(-3.0, 3.0)
        policy.train()
        flat = candidates.reshape(group * batch, -1)
        t = (t_min + (1.0 - t_min) * torch.rand((batch, 1), device=device)).repeat(group, 1)
        noise = torch.randn((batch, flat.shape[1]), device=device).repeat(group, 1)
        noisy = (1.0 - t) * flat + t * noise
        velocity = policy(noisy, t, condition)
        with torch.no_grad():
            reference_velocity = reference(noisy, t, condition)
        loss_model = ((velocity - (noise - flat)) ** 2).mean(1).view(group, batch)
        loss_reference = ((reference_velocity - (noise - flat)) ** 2).mean(1).view(group, batch)
        preference = -F.logsigmoid(-beta * (advantage * (loss_model - loss_reference)).mean(0)).mean()
        loss = preference + kl_beta * F.mse_loss(velocity, reference_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(dgpo["gradient_clip"]))
        optimizer.step()
        if iteration % 200 == 0 or iteration == iterations - 1:
            log(f"truth-dgpo {iteration + 1}/{iterations} reward={float(reward.mean()):.3f}")


def mass_condition(observables, condition_scaler, mass):
    """Visible conditions plus the scaled parent mass, for the mass-conditioned flow."""
    column = np.full((len(observables), 1), (mass - 500.0) / 100.0, dtype=np.float32)
    return np.concatenate([condition_scaler.transform(observables), column], axis=1)


@torch.no_grad()
def reconstruct(flow, observables, condition_scaler, target_scaler, sampling, noise, device, mass=None):
    conditions = (condition_scaler.transform(observables) if mass is None
                  else mass_condition(observables, condition_scaler, mass))
    out = []
    for s in range(0, len(conditions), 8192):
        out.append(sample_flow(flow, torch.from_numpy(conditions[s:s + 8192]).to(device),
                               int(sampling["ode_steps"]), float(sampling["output_clip_sigma"]),
                               torch.from_numpy(noise[s:s + 8192]).to(device)).cpu().numpy())
    return target_scaler.inverse(np.concatenate(out))


def evaluate(obs, nu_true, nu_hat, truth, evaluation, estimators):
    """Reconstructed parent-mass peak (the primary metric), its width, distribution and per-event
    metrics against the true neutrinos, and the full-event estimates on the reconstruction."""
    leptons = obs[:, :8].reshape(-1, 2, 4)
    masses_hat = reconstruct_parent_masses(leptons, nu_hat.reshape(-1, 2, 3), 0.0).ravel()
    masses_true = reconstruct_parent_masses(leptons, nu_true.reshape(-1, 2, 3), 0.0).ravel()
    peak, width = fit_mass_peak(masses_hat, int(evaluation["histogram_bins"]), tuple(evaluation["mass_range_gev"]))
    residual = nu_hat.reshape(len(obs), -1) - nu_true.reshape(len(obs), -1)
    return {
        "peak": peak, "peak_bias": peak - truth, "resolution": width,
        "median_bias": float(np.median(masses_hat)) - truth,
        "geo_mean_bias": float(np.exp(np.mean(np.log(masses_hat[masses_hat > 0])))) - truth,
        "w1_mass_vs_true_nu": float(wasserstein_distance(masses_hat, masses_true)),
        "nu_rmse": float(np.sqrt(np.mean(residual**2))),
        "nu_mae": float(np.mean(np.abs(residual))),
        "nu_mean_residual_rms": float(np.sqrt(np.mean(residual.mean(0) ** 2))),
        **{f"{name}_bias": float(fn(obs, nu_hat)) - truth for name, fn in estimators.items()},
        "mass_hist": np.histogram(masses_hat, bins=HIST_MASS)[0].tolist(),
        "pz_residual_hist": np.histogram(residual[:, 2::3].ravel(), bins=HIST_RESID)[0].tolist(),
    }


def run(args):
    start = time.time()
    config = yaml.safe_load((ROOT / "config" / "toy.yaml").read_text())
    physics, data, sampling, evaluation = config["physics"], config["data"], config["sampling"], config["evaluation"]
    masses = [float(m) for m in data["template_masses_gev"]]
    grid = data["real_masses_gev"]
    truths = args.truths or [float(m) for m in np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"])]
    n_template = args.n_template or int(data["template_events_per_mass"])
    n_real = args.n_real or int(data["real_events_per_mass"])
    training = {**config["training"]}
    training["profiler_epochs"] = args.profiler_epochs or training["profiler_epochs"]
    training["flow_epochs"] = args.flow_epochs or training["flow_epochs"]
    dgpo = {**config["dgpo"], "iterations": args.dgpo_iterations or int(config["dgpo"]["iterations_per_pseudo_data"])}
    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def simulate(mass, n, stream):
        """Independent event streams: 1 templates, 2 adaptation, 3 evaluation, 4/5 offset templates."""
        return generate_events(mass, n, physics, np.random.default_rng([seed, stream, int(round(mass * 10))]))

    templates = {m: simulate(m, n_template, 1) for m in masses}
    adapt = {t: simulate(t, n_real, 2) for t in truths}
    held_out = {t: simulate(t, n_real, 3) for t in truths}
    log(f"simulated {n_template}/template and {n_real}/dataset at {len(truths)} masses on {device}")

    # the pipeline's profilers and flow, trained as in pipeline.run_training
    rng = np.random.default_rng(seed + 1)
    torch.manual_seed(seed + 1)
    train, validation = _split_templates(templates, float(data["validation_fraction"]), rng)
    condition_scaler = Standardizer.fit(train["observables"])
    target_scaler = Standardizer.fit(train["neutrino_target"])
    full_train = np.concatenate([train["observables"], train["neutrino_target"]], axis=1)
    full_validation = np.concatenate([validation["observables"], validation["neutrino_target"]], axis=1)
    full_scaler = Standardizer.fit(full_train)
    hidden = [int(h) for h in config["model"]["hidden_dims"]]
    visible_profiler = TemplateDiscriminator(10, 3, hidden)
    full_profiler = TemplateDiscriminator(16, 3, hidden)
    train_discriminator(visible_profiler, condition_scaler.transform(train["observables"]), train["labels"],
                        condition_scaler.transform(validation["observables"]), validation["labels"],
                        training, device, rng, "visible-profiler")
    train_discriminator(full_profiler, full_scaler.transform(full_train), train["labels"],
                        full_scaler.transform(full_validation), validation["labels"], training, device, rng,
                        "full-profiler")
    toy = {}
    for name, profiler, scaler, x in (("visible", visible_profiler, condition_scaler, validation["observables"]),
                                      ("full", full_profiler, full_scaler, full_validation)):
        logp = predict_log_probabilities(profiler, scaler.transform(x), device, 8192)
        anchors = [profile_mass_estimate(logp[validation["labels"] == c], masses)[0] for c in range(3)]
        calibration = {"raw_mass_gev": anchors, "calibrated_mass_gev": masses} if np.all(np.diff(anchors) > 0) else None
        toy[name] = ToyEstimator(profiler, scaler, calibration, masses, device)
    flow = ConditionalFlow(6, 10, hidden, int(config["model"]["time_embedding_dim"]))
    train_flow(flow, condition_scaler.transform(train["observables"]),
               target_scaler.transform(train["neutrino_target"]), training, device, rng)
    flow.to(device)
    cond_flow = ConditionalFlow(6, 11, hidden, int(config["model"]["time_embedding_dim"]))
    train_flow(cond_flow, np.concatenate([mass_condition(train["observables"][train["labels"] == c], condition_scaler, m)
                                          for c, m in enumerate(masses)]),
               target_scaler.transform(np.concatenate([train["neutrino_target"][train["labels"] == c]
                                                       for c in range(3)])), training, device, rng)
    cond_flow.to(device)

    # moment-curve estimators, cross-fitted on all template events
    obs = np.concatenate([templates[m]["observables"] for m in masses])
    nu = np.concatenate([templates[m]["neutrino_target"] for m in masses])
    labels = np.concatenate([np.full(n_template, c, dtype=np.int64) for c in range(3)])
    visible_mc = MomentCurve(visible_features, masses, 50.0).fit((obs,), labels, args.folds, args.linear_epochs, device, seed)
    full_mc = MomentCurve(full_features, masses, 100.0).fit((obs, nu), labels, args.folds, args.linear_epochs, device, seed + 7)
    log("trained profilers, flow and moment-curve estimators")

    def sbi_dgpo(reward, dataset, mass, scenario):
        """Fine-tune a fresh copy of the flow on one unlabeled dataset (`mass` only labels the log)."""
        torch.manual_seed(scenario)
        policy = copy.deepcopy(flow)
        train_dgpo(policy, flow, visible_profiler, full_profiler, {mass: dataset}, condition_scaler, target_scaler,
                   full_scaler, masses, {}, dgpo, sampling, 8192, device, np.random.default_rng(scenario),
                   estimator=reward)
        return policy

    rewards = {"sbi_toy": Reward(toy["visible"].estimate, toy["full"].estimate),
               "sbi_new": Reward(visible_mc.estimate, full_mc.estimate)}

    # offset of sbi_new's reconstructed peak, measured on fresh simulation at the template masses
    offsets = []
    for m in masses:
        policy = sbi_dgpo(rewards["sbi_new"], simulate(m, n_real, 4), m, independent_dgpo_seed(seed + 1000, m))
        d = simulate(m, n_real, 5)
        noise = np.random.default_rng([seed, 6, int(m)]).standard_normal((n_real, 6)).astype(np.float32)
        nu_hat = reconstruct(policy, d["observables"], condition_scaler, target_scaler, sampling, noise, device)
        offsets.append(evaluate(d["observables"], d["neutrino_target"], nu_hat, m, evaluation, {})["peak_bias"])
    delta = float(np.mean(offsets))
    rewards["sbi_new_shifted"] = Reward(visible_mc.estimate, full_mc.estimate, shift=delta)
    log(f"sbi_new peak offset on template simulation: {np.round(offsets, 2).tolist()} -> delta {delta:+.2f} GeV")

    flows = {"sft": {t: flow for t in truths}}
    truth_policy = copy.deepcopy(flow)
    train_truth_dgpo(truth_policy, flow, condition_scaler.transform(train["observables"]),
                     target_scaler.transform(train["neutrino_target"]), dgpo, sampling, dgpo["iterations"],
                     device, np.random.default_rng([seed, 99]))
    flows["truth_dgpo"] = {t: truth_policy for t in truths}
    for name, reward in rewards.items():
        flows[name] = {t: sbi_dgpo(reward, adapt[t], t, independent_dgpo_seed(seed, t)) for t in truths}
        log(f"{name} adapted at {len(truths)} masses")

    rows = []
    estimators = {"sbi_full_new": full_mc.estimate, "sbi_full_toy": toy["full"].estimate}
    for t in truths:
        d = held_out[t]
        noise = np.random.default_rng([seed, 5, int(t)]).standard_normal((n_real, 6)).astype(np.float32)
        recos = {"baseline": baseline_neutrino_momentum(d["observables"]).reshape(n_real, -1),
                 "oracle": d["neutrino_target"]}
        for name, per_mass in flows.items():
            recos[name] = reconstruct(per_mass[t], d["observables"], condition_scaler, target_scaler, sampling, noise, device)
        m_visible = visible_mc.estimate(d["observables"])
        recos["cond_sft"] = reconstruct(cond_flow, d["observables"], condition_scaler, target_scaler, sampling, noise,
                                        device, mass=m_visible)
        visible = {"visible_new_bias": m_visible - t,
                   "visible_toy_bias": toy["visible"].estimate(d["observables"]) - t}
        for name, nu_hat in recos.items():
            rows.append({"seed": seed, "truth": t, "method": name, **visible,
                         **evaluate(d["observables"], d["neutrino_target"], nu_hat, t, evaluation, estimators)})
        log(f"truth {t:g}: " + ", ".join(f"{r['method']} {r['peak_bias']:+.1f}" for r in rows if r["truth"] == t))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"seed": seed, "args": vars(args), "delta": delta, "template_offsets": offsets, "rows": rows,
              "wall_s": time.time() - start}
    (out / f"sbi_dgpo_s{seed}.json").write_text(json.dumps(result))
    log(f"wrote {out / f'sbi_dgpo_s{seed}.json'} in {result['wall_s']:.0f} s")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "outputs" / "sbi_dgpo"))
    p.add_argument("--truths", type=lambda s: [float(x) for x in s.split(",")], default=None,
                   help="comma-separated truth masses (default: the config's 400-600 grid)")
    p.add_argument("--n-template", type=int, default=None, help="events per template (default: config)")
    p.add_argument("--n-real", type=int, default=None, help="events per dataset (default: config)")
    p.add_argument("--profiler-epochs", type=int, default=None)
    p.add_argument("--flow-epochs", type=int, default=None)
    p.add_argument("--dgpo-iterations", type=int, default=None)
    p.add_argument("--linear-epochs", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
