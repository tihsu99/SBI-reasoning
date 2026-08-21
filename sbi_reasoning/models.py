from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MetricLogger = Callable[[dict[str, float]], None]


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if mixed_precision not in dtypes:
        raise ValueError(
            "mixed_precision must be one of: none, bf16, fp16"
        )
    return torch.autocast(device_type="cuda", dtype=dtypes[mixed_precision])


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean, "scale": self.scale}

    @classmethod
    def from_dict(cls, values: dict[str, np.ndarray]) -> "Standardizer":
        return cls(mean=np.asarray(values["mean"]), scale=np.asarray(values["scale"]))


class DenseNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.SiLU()])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TemplateDiscriminator(DenseNetwork):
    pass


class MassRatioEstimator(DenseNetwork):
    def __init__(self, feature_dim: int, hidden_dims: list[int]):
        super().__init__(feature_dim + 1, 1, hidden_dims)


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        hidden_dims: list[int],
        time_embedding_dim: int,
    ):
        super().__init__()
        if time_embedding_dim % 2 != 0:
            raise ValueError("time_embedding_dim must be even")
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        self.hidden_dims = list(hidden_dims)
        self.time_embedding_dim = time_embedding_dim
        self.velocity = DenseNetwork(
            target_dim + condition_dim + time_embedding_dim,
            target_dim,
            hidden_dims,
        )

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_embedding_dim // 2
        frequencies = torch.exp(
            torch.linspace(
                0.0,
                math.log(1000.0),
                half_dim,
                device=timestep.device,
                dtype=timestep.dtype,
            )
        )
        angles = timestep * frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self, noisy_target: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if timestep.ndim == 1:
            timestep = timestep[:, None]
        features = torch.cat(
            [noisy_target, condition, self._time_embedding(timestep)], dim=-1
        )
        return self.velocity(features)

    def architecture(self) -> dict[str, Any]:
        return {
            "target_dim": self.target_dim,
            "condition_dim": self.condition_dim,
            "hidden_dims": self.hidden_dims,
            "time_embedding_dim": self.time_embedding_dim,
        }


def iter_batches(
    event_count: int, batch_size: int, rng: np.random.Generator
) -> list[np.ndarray]:
    order = rng.permutation(event_count)
    return [order[start : start + batch_size] for start in range(0, event_count, batch_size)]


def train_discriminator(
    model: TemplateDiscriminator,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    name: str,
    metric_logger: MetricLogger | None = None,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["profiler_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.to(device)
    history = {"loss": [], "validation_accuracy": []}
    epochs = int(config["profiler_epochs"])
    batch_size = int(config["batch_size"])
    mixed_precision = str(config.get("mixed_precision", "none"))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and mixed_precision == "fp16"
    )
    best_accuracy = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in iter_batches(len(train_features), batch_size, rng):
            features = torch.from_numpy(train_features[indices]).to(device)
            labels = torch.from_numpy(train_labels[indices]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, mixed_precision):
                loss = F.cross_entropy(model(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_logits = model(torch.from_numpy(validation_features).to(device))
            predictions = validation_logits.argmax(dim=1).cpu().numpy()
        accuracy = float(np.mean(predictions == validation_labels))
        history["loss"].append(float(np.mean(losses)))
        history["validation_accuracy"].append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
        if metric_logger is not None:
            metric_logger(
                {
                    f"train/{name}/loss": history["loss"][-1],
                    f"train/{name}/validation_accuracy": accuracy,
                    f"train/{name}/epoch": float(epoch + 1),
                }
            )
        print(
            f"[{name}] epoch {epoch + 1:03d}/{epochs:03d} "
            f"loss={history['loss'][-1]:.4f} validation_accuracy={accuracy:.4f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_ratio_estimator(
    model: MassRatioEstimator,
    train_features: np.ndarray,
    train_masses: np.ndarray,
    validation_features: np.ndarray,
    validation_masses: np.ndarray,
    mass_scaler: Standardizer,
    config: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    name: str,
    metric_logger: MetricLogger | None = None,
) -> dict[str, list[float]]:
    """Train a likelihood-to-evidence ratio estimator with joint/product pairs."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["profiler_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.to(device)
    history = {"loss": [], "validation_accuracy": []}
    epochs = int(config["profiler_epochs"])
    batch_size = int(config["batch_size"])
    mixed_precision = str(config.get("mixed_precision", "none"))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and mixed_precision == "fp16"
    )
    all_train_masses = np.asarray(train_masses, dtype=np.float32).reshape(-1, 1)
    all_validation_masses = np.asarray(validation_masses, dtype=np.float32).reshape(
        -1, 1
    )
    best_accuracy = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in iter_batches(len(train_features), batch_size, rng):
            features = train_features[indices]
            matched_mass = all_train_masses[indices]
            marginal_mass = all_train_masses[
                rng.integers(0, len(all_train_masses), size=len(indices))
            ]
            positive = np.concatenate(
                [features, mass_scaler.transform(matched_mass)], axis=1
            )
            negative = np.concatenate(
                [features, mass_scaler.transform(marginal_mass)], axis=1
            )
            inputs = torch.from_numpy(np.concatenate([positive, negative])).to(device)
            labels = torch.cat(
                [
                    torch.ones(len(indices), device=device),
                    torch.zeros(len(indices), device=device),
                ]
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, mixed_precision):
                logits = model(inputs).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_marginal = all_validation_masses[
            rng.integers(0, len(all_validation_masses), size=len(all_validation_masses))
        ]
        positive = np.concatenate(
            [
                validation_features,
                mass_scaler.transform(all_validation_masses),
            ],
            axis=1,
        )
        negative = np.concatenate(
            [validation_features, mass_scaler.transform(validation_marginal)], axis=1
        )
        with torch.no_grad():
            positive_logits = model(torch.from_numpy(positive).to(device)).squeeze(1)
            negative_logits = model(torch.from_numpy(negative).to(device)).squeeze(1)
            accuracy = 0.5 * (
                float((positive_logits > 0.0).float().mean().cpu())
                + float((negative_logits < 0.0).float().mean().cpu())
            )
        history["loss"].append(float(np.mean(losses)))
        history["validation_accuracy"].append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
        if metric_logger is not None:
            metric_logger(
                {
                    f"train/{name}/loss": history["loss"][-1],
                    f"train/{name}/validation_accuracy": accuracy,
                    f"train/{name}/epoch": float(epoch + 1),
                }
            )
        print(
            f"[{name}] epoch {epoch + 1:03d}/{epochs:03d} "
            f"loss={history['loss'][-1]:.4f} validation_accuracy={accuracy:.4f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_flow(
    model: ConditionalFlow,
    conditions: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    metric_logger: MetricLogger | None = None,
) -> list[float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["flow_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.to(device)
    epochs = int(config["flow_epochs"])
    batch_size = int(config["batch_size"])
    mixed_precision = str(config.get("mixed_precision", "none"))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and mixed_precision == "fp16"
    )
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in iter_batches(len(conditions), batch_size, rng):
            condition = torch.from_numpy(conditions[indices]).to(device)
            target = torch.from_numpy(targets[indices]).to(device)
            noise = torch.randn_like(target)
            timestep = torch.rand((len(indices), 1), device=device)
            noisy_target = (1.0 - timestep) * target + timestep * noise
            target_velocity = noise - target
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, mixed_precision):
                loss = F.mse_loss(
                    model(noisy_target, timestep, condition), target_velocity
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        if metric_logger is not None:
            metric_logger(
                {
                    "train/flow/loss": history[-1],
                    "train/flow/epoch": float(epoch + 1),
                }
            )
        print(
            f"[flow] epoch {epoch + 1:03d}/{epochs:03d} loss={history[-1]:.4f}",
            flush=True,
        )
    return history


@torch.no_grad()
def sample_flow(
    model: ConditionalFlow,
    conditions: torch.Tensor,
    steps: int,
    clip_sigma: float,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    samples = (
        torch.randn(
            (conditions.shape[0], model.target_dim),
            device=conditions.device,
            dtype=conditions.dtype,
        )
        if initial_noise is None
        else initial_noise.clone()
    )
    step_size = 1.0 / steps
    for step in range(steps):
        time_value = 1.0 - step * step_size
        timestep = torch.full(
            (conditions.shape[0], 1), time_value, device=conditions.device
        )
        samples = samples - step_size * model(samples, timestep, conditions)
        samples.clamp_(-clip_sigma, clip_sigma)
    return samples


@torch.no_grad()
def predict_log_probabilities(
    model: TemplateDiscriminator,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        chunks.append(F.log_softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def predict_log_ratios(
    model: MassRatioEstimator,
    features: np.ndarray,
    hypothesis_masses: list[float],
    mass_scaler: Standardizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate log likelihood-to-evidence ratios on a mass scan."""
    model.eval()
    masses = mass_scaler.transform(
        np.asarray(hypothesis_masses, dtype=np.float32).reshape(-1, 1)
    )
    chunks = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        mass = torch.from_numpy(masses).to(device)
        event_count = len(batch)
        hypothesis_count = len(mass)
        repeated_features = batch[:, None, :].expand(
            event_count, hypothesis_count, batch.shape[1]
        )
        repeated_masses = mass[None, :, :].expand(event_count, hypothesis_count, 1)
        inputs = torch.cat([repeated_features, repeated_masses], dim=2).reshape(
            event_count * hypothesis_count, -1
        )
        chunks.append(model(inputs).reshape(event_count, hypothesis_count).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def predict_profile_scores(
    model: nn.Module,
    features: np.ndarray,
    profiler_type: str,
    hypothesis_masses: list[float],
    mass_scaler: Standardizer | None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate either supported mass-profiler representation."""
    if profiler_type == "template_classifier":
        if not isinstance(model, TemplateDiscriminator):
            raise TypeError("template_classifier requires TemplateDiscriminator")
        return predict_log_probabilities(model, features, device, batch_size)
    if profiler_type == "mass_parameterized_ratio":
        if not isinstance(model, MassRatioEstimator):
            raise TypeError("mass_parameterized_ratio requires MassRatioEstimator")
        if mass_scaler is None:
            raise ValueError("mass_parameterized_ratio requires a mass scaler")
        return predict_log_ratios(
            model,
            features,
            hypothesis_masses,
            mass_scaler,
            device,
            batch_size,
        )
    raise ValueError(f"Unknown profiler_type: {profiler_type}")


def profile_mass_vertex(profile: torch.Tensor, template_masses: list[float]) -> float:
    masses = np.asarray(template_masses, dtype=np.float64)
    values = profile.detach().cpu().numpy().astype(np.float64)
    negative_two_delta_log_likelihood = -2.0 * (values - values.max())
    minimum = int(np.argmin(negative_two_delta_log_likelihood))
    if minimum == 0 or minimum == len(masses) - 1:
        return float(masses[minimum])
    local = slice(minimum - 1, minimum + 2)
    quadratic, linear, _ = np.polyfit(
        masses[local], negative_two_delta_log_likelihood[local], 2
    )
    if quadratic <= 0.0:
        return float(masses[np.argmin(negative_two_delta_log_likelihood)])
    return float(np.clip(-linear / (2.0 * quadratic), masses.min(), masses.max()))


def calibrate_profile_estimate(
    estimate_gev: float,
    calibration: dict[str, list[float]] | None,
) -> float:
    if calibration is None:
        return estimate_gev
    return float(
        np.interp(
            estimate_gev,
            np.asarray(calibration["raw_mass_gev"], dtype=np.float64),
            np.asarray(calibration["calibrated_mass_gev"], dtype=np.float64),
        )
    )


def dgpo_group_loss(
    model_losses: torch.Tensor,
    reference_losses: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Equation 17 with group-size normalization absorbed into beta."""
    log_ratio = model_losses - reference_losses
    preference_logit = -beta * torch.mean(advantages * log_ratio)
    return -F.logsigmoid(preference_logit)


def profile_estimator_reward(
    candidate_estimate_gev: float,
    target_estimate_gev: float,
    scale_gev: float,
) -> float:
    if scale_gev <= 0.0:
        raise ValueError("Profile estimator scale must be positive")
    return -((candidate_estimate_gev - target_estimate_gev) / scale_gev) ** 2


def train_dgpo(
    policy: ConditionalFlow,
    reference: ConditionalFlow,
    observed_profiler: nn.Module,
    full_profiler: nn.Module,
    real_datasets: dict[float, dict[str, np.ndarray]],
    condition_scaler: Standardizer,
    target_scaler: Standardizer,
    full_scaler: Standardizer,
    mass_scaler: Standardizer | None,
    profile_masses: list[float],
    profiler_type: str,
    profile_calibration: dict[str, dict[str, list[float]]],
    dgpo_config: dict[str, Any],
    sampling_config: dict[str, Any],
    evaluation_batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    mixed_precision: str = "none",
    metric_logger: MetricLogger | None = None,
) -> tuple[dict[str, list[float]], str]:
    reference = copy.deepcopy(reference).to(device).eval()
    reference.requires_grad_(False)
    observed_profiler.eval().requires_grad_(False)
    full_profiler.eval().requires_grad_(False)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(dgpo_config["learning_rate"])
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and mixed_precision == "fp16"
    )

    observed_calibration = profile_calibration.get("observable")
    full_calibration = profile_calibration.get("visible_plus_truth_invisible")
    target_estimates: dict[float, float] = {}
    for mass, dataset in real_datasets.items():
        standardized = condition_scaler.transform(dataset["observables"])
        log_probabilities = predict_profile_scores(
            observed_profiler,
            standardized,
            profiler_type,
            profile_masses,
            mass_scaler,
            device,
            evaluation_batch_size,
        )
        raw_estimate = profile_mass_vertex(
            torch.from_numpy(log_probabilities).to(device).mean(dim=0),
            profile_masses,
        )
        target_estimates[mass] = calibrate_profile_estimate(
            raw_estimate,
            observed_calibration,
        )

    iterations = int(dgpo_config["iterations"])
    batch_size = int(dgpo_config["batch_size"])
    group_size = int(dgpo_config["group_size"])
    ode_steps = int(sampling_config["ode_steps"])
    clip_sigma = float(sampling_config["output_clip_sigma"])
    estimator_scale_gev = float(dgpo_config["profile_estimator_scale_gev"])
    masses = np.asarray(sorted(real_datasets), dtype=np.float64)
    history = {
        "loss": [],
        "reward_mean": [],
        "reward_spread": [],
        "target_estimate_gev": [],
        "candidate_estimate_mean_gev": [],
        "estimator_gap_gev": [],
    }
    gap_tolerance = float(dgpo_config["estimator_gap_tolerance_gev"])
    gap_patience = int(dgpo_config["estimator_gap_patience"])
    spread_threshold = float(dgpo_config["reward_spread_threshold"])
    spread_patience = int(dgpo_config["reward_spread_patience"])
    minimum_iterations = int(dgpo_config["minimum_iterations"])
    gap_streak = 0
    low_spread_streak = 0
    stop_reason = "maximum_iterations"

    for iteration in range(iterations):
        if str(dgpo_config["condition_sampling"]) == "balanced":
            mass = float(masses[iteration % len(masses)])
        else:
            mass = float(rng.choice(masses))
        dataset = real_datasets[mass]
        indices = rng.integers(0, len(dataset["observables"]), size=batch_size)
        observations_raw = dataset["observables"][indices]
        conditions_np = condition_scaler.transform(observations_raw)
        conditions = torch.from_numpy(conditions_np).to(device)
        repeated_conditions = conditions.repeat(group_size, 1)

        with torch.no_grad():
            candidate_standardized = sample_flow(
                policy,
                repeated_conditions,
                ode_steps,
                clip_sigma,
            ).view(group_size, batch_size, -1)
            candidate_raw = target_scaler.inverse(
                candidate_standardized.cpu().numpy().reshape(group_size * batch_size, -1)
            ).reshape(group_size, batch_size, -1)
            rewards = []
            candidate_estimates = []
            for group_index in range(group_size):
                full_raw = np.concatenate(
                    [observations_raw, candidate_raw[group_index]], axis=1
                )
                full_features = full_scaler.transform(full_raw)
                candidate_profile = torch.from_numpy(
                    predict_profile_scores(
                        full_profiler,
                        full_features,
                        profiler_type,
                        profile_masses,
                        mass_scaler,
                        device,
                        evaluation_batch_size,
                    )
                ).to(device).mean(dim=0)
                raw_candidate_estimate = profile_mass_vertex(
                    candidate_profile,
                    profile_masses,
                )
                candidate_estimate = calibrate_profile_estimate(
                    raw_candidate_estimate,
                    full_calibration,
                )
                candidate_estimates.append(candidate_estimate)
                rewards.append(
                    profile_estimator_reward(
                        candidate_estimate,
                        target_estimates[mass],
                        estimator_scale_gev,
                    )
                )
            rewards_tensor = torch.tensor(rewards, device=device, dtype=conditions.dtype)
            reward_std = rewards_tensor.std(unbiased=False)
            advantages = (rewards_tensor - rewards_tensor.mean()) / (reward_std + 1e-6)
            estimator_gap = abs(float(np.mean(candidate_estimates)) - target_estimates[mass])

        candidate_flat = candidate_standardized.detach().reshape(
            group_size * batch_size, -1
        )
        timestep_base = torch.rand((batch_size, 1), device=device)
        minimum_timestep = float(dgpo_config["minimum_timestep"])
        timestep_base = minimum_timestep + (1.0 - minimum_timestep) * timestep_base
        timestep = timestep_base.repeat(group_size, 1)
        shared_noise = torch.randn((batch_size, policy.target_dim), device=device)
        noise = shared_noise.repeat(group_size, 1)
        noisy_target = (1.0 - timestep) * candidate_flat + timestep * noise
        target_velocity = noise - candidate_flat

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, mixed_precision):
            model_velocity = policy(noisy_target, timestep, repeated_conditions)
            with torch.no_grad():
                reference_velocity = reference(
                    noisy_target, timestep, repeated_conditions
                )
            model_losses = ((model_velocity - target_velocity) ** 2).mean(dim=1)
            reference_losses = ((reference_velocity - target_velocity) ** 2).mean(dim=1)
            model_group_losses = model_losses.view(group_size, batch_size).mean(dim=1)
            reference_group_losses = reference_losses.view(group_size, batch_size).mean(dim=1)
            preference_loss = dgpo_group_loss(
                model_group_losses,
                reference_group_losses,
                advantages,
                float(dgpo_config["beta"]),
            )
            kl_loss = F.mse_loss(model_velocity, reference_velocity)
            loss = preference_loss + float(dgpo_config["kl_beta"]) * kl_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float(dgpo_config["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()

        history["loss"].append(float(loss.detach().cpu()))
        history["reward_mean"].append(float(rewards_tensor.mean().cpu()))
        history["reward_spread"].append(float(reward_std.cpu()))
        history["target_estimate_gev"].append(target_estimates[mass])
        history["candidate_estimate_mean_gev"].append(
            float(np.mean(candidate_estimates))
        )
        history["estimator_gap_gev"].append(estimator_gap)
        gap_streak = gap_streak + 1 if estimator_gap <= gap_tolerance else 0
        low_spread_streak = (
            low_spread_streak + 1
            if float(reward_std.cpu()) <= spread_threshold
            else 0
        )
        if metric_logger is not None:
            metric_logger(
                {
                    "train/dgpo/loss": history["loss"][-1],
                    "train/dgpo/reward_mean": history["reward_mean"][-1],
                    "train/dgpo/reward_spread": history["reward_spread"][-1],
                    "train/dgpo/target_estimate_gev": history[
                        "target_estimate_gev"
                    ][-1],
                    "train/dgpo/candidate_estimate_mean_gev": history[
                        "candidate_estimate_mean_gev"
                    ][-1],
                    "train/dgpo/estimator_gap_gev": estimator_gap,
                    "train/dgpo/iteration": float(iteration + 1),
                    "train/dgpo/mass_gev": mass,
                }
            )
        if (iteration + 1) % int(dgpo_config["log_every"]) == 0 or iteration == 0:
            print(
                f"[dgpo] iteration {iteration + 1:04d}/{iterations:04d} mass={mass:.1f} "
                f"loss={history['loss'][-1]:.4f} "
                f"reward={history['reward_mean'][-1]:.4f} "
                f"spread={history['reward_spread'][-1]:.4f} "
                f"target={history['target_estimate_gev'][-1]:.1f} GeV "
                f"candidate={history['candidate_estimate_mean_gev'][-1]:.1f} GeV "
                f"gap={estimator_gap:.1f} GeV",
                flush=True,
            )
        if iteration + 1 >= minimum_iterations:
            if gap_streak >= gap_patience:
                stop_reason = "estimator_gap"
            elif low_spread_streak >= spread_patience:
                stop_reason = "low_reward_spread"
            if stop_reason != "maximum_iterations":
                print(
                    f"[dgpo] early stop mass={mass:.1f} reason={stop_reason} "
                    f"iterations={iteration + 1}",
                    flush=True,
                )
                break
    return history, stop_reason
