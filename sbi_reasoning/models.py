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


def normalized_profile(log_probabilities: torch.Tensor) -> torch.Tensor:
    profile = log_probabilities.mean(dim=0)
    return profile - profile.max()


def profile_mass_vertex(profile: torch.Tensor, template_masses: list[float]) -> float:
    masses = np.asarray(template_masses, dtype=np.float64)
    values = profile.detach().cpu().numpy().astype(np.float64)
    negative_two_delta_log_likelihood = -2.0 * (values - values.max())
    quadratic, linear, _ = np.polyfit(masses, negative_two_delta_log_likelihood, 2)
    if quadratic <= 0.0:
        return float(masses[np.argmin(negative_two_delta_log_likelihood)])
    return float(np.clip(-linear / (2.0 * quadratic), masses.min(), masses.max()))


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


def train_dgpo(
    policy: ConditionalFlow,
    reference: ConditionalFlow,
    observed_profiler: TemplateDiscriminator,
    full_profiler: TemplateDiscriminator,
    real_datasets: dict[float, dict[str, np.ndarray]],
    condition_scaler: Standardizer,
    target_scaler: Standardizer,
    full_scaler: Standardizer,
    template_masses: list[float],
    dgpo_config: dict[str, Any],
    sampling_config: dict[str, Any],
    evaluation_batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    mixed_precision: str = "none",
    metric_logger: MetricLogger | None = None,
) -> dict[str, list[float]]:
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

    target_profile_masses: dict[float, float] = {}
    for mass, dataset in real_datasets.items():
        standardized = condition_scaler.transform(dataset["observables"])
        log_probabilities = predict_log_probabilities(
            observed_profiler, standardized, device, evaluation_batch_size
        )
        target_profile = torch.from_numpy(log_probabilities).to(device).mean(dim=0)
        target_profile_masses[mass] = profile_mass_vertex(target_profile, template_masses)

    iterations = int(dgpo_config["iterations"])
    batch_size = int(dgpo_config["batch_size"])
    group_size = int(dgpo_config["group_size"])
    ode_steps = int(sampling_config["ode_steps"])
    clip_sigma = float(sampling_config["output_clip_sigma"])
    masses = np.asarray(sorted(real_datasets), dtype=np.float64)
    history = {"loss": [], "reward_mean": [], "reward_spread": []}

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
            for group_index in range(group_size):
                full_raw = np.concatenate(
                    [observations_raw, candidate_raw[group_index]], axis=1
                )
                full_features = torch.from_numpy(full_scaler.transform(full_raw)).to(device)
                candidate_profile = normalized_profile(
                    F.log_softmax(full_profiler(full_features), dim=1)
                )
                candidate_mass = profile_mass_vertex(candidate_profile, template_masses)
                profile_difference = (
                    candidate_mass - target_profile_masses[mass]
                ) / float(dgpo_config["profile_mass_scale_gev"])
                rewards.append(-(profile_difference**2))
            rewards_tensor = torch.tensor(rewards, device=device, dtype=conditions.dtype)
            reward_std = rewards_tensor.std(unbiased=False)
            advantages = (rewards_tensor - rewards_tensor.mean()) / (reward_std + 1e-6)

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
        if metric_logger is not None:
            metric_logger(
                {
                    "train/dgpo/loss": history["loss"][-1],
                    "train/dgpo/reward_mean": history["reward_mean"][-1],
                    "train/dgpo/reward_spread": history["reward_spread"][-1],
                    "train/dgpo/iteration": float(iteration + 1),
                    "train/dgpo/mass_gev": mass,
                }
            )
        if (iteration + 1) % int(dgpo_config["log_every"]) == 0 or iteration == 0:
            print(
                f"[dgpo] iteration {iteration + 1:04d}/{iterations:04d} mass={mass:.1f} "
                f"loss={history['loss'][-1]:.4f} "
                f"reward={history['reward_mean'][-1]:.4f} "
                f"spread={history['reward_spread'][-1]:.4f}",
                flush=True,
            )
    return history
