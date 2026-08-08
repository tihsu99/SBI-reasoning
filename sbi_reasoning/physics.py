from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit


def _isotropic_directions(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    cos_theta = rng.uniform(-1.0, 1.0, size=shape)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1
    )


def _boost(four_vector: np.ndarray, beta: np.ndarray) -> np.ndarray:
    energy = four_vector[..., :1]
    momentum = four_vector[..., 1:]
    beta2 = np.sum(beta**2, axis=-1, keepdims=True)
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    beta_dot_p = np.sum(beta * momentum, axis=-1, keepdims=True)
    safe_beta2 = np.where(beta2 > 0.0, beta2, 1.0)
    factor = ((gamma - 1.0) * beta_dot_p / safe_beta2) + gamma * energy
    boosted_energy = gamma * (energy + beta_dot_p)
    boosted_momentum = momentum + factor * beta
    return np.concatenate([boosted_energy, boosted_momentum], axis=-1)


def sample_parent_energy(
    parent_mass_gev: float,
    event_count: int,
    energy_config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a two-sided exponential energy, conditioned on the mass shell."""
    distribution = str(energy_config["distribution"])
    if distribution != "double_exponential":
        raise ValueError(f"Unsupported parent-energy distribution: {distribution}")
    center = float(energy_config["center_gev"])
    scale = float(energy_config["scale_gev"])
    if scale <= 0.0:
        raise ValueError("Parent-energy scale must be positive")

    if parent_mass_gev < center:
        lower_cdf = 0.5 * np.exp((parent_mass_gev - center) / scale)
    else:
        lower_cdf = 1.0 - 0.5 * np.exp(-(parent_mass_gev - center) / scale)
    probability = rng.uniform(lower_cdf, 1.0, size=event_count)
    energy = np.where(
        probability < 0.5,
        center + scale * np.log(2.0 * probability),
        center - scale * np.log(2.0 * (1.0 - probability)),
    )
    return np.maximum(energy, parent_mass_gev)


def generate_events(
    parent_mass_gev: float,
    event_count: int,
    physics: dict[str, Any],
    rng: np.random.Generator,
    visible_energy_scale_shift: float = 0.0,
) -> dict[str, np.ndarray]:
    """Generate back-to-back parent pairs followed by isotropic two-body decays."""
    lepton_mass = float(physics["lepton_mass_gev"])
    neutrino_mass = float(physics["neutrino_mass_gev"])
    if parent_mass_gev <= lepton_mass + neutrino_mass:
        raise ValueError("Parent mass must exceed the sum of daughter masses")

    parent_energy = sample_parent_energy(
        parent_mass_gev,
        event_count,
        physics["parent_energy"],
        rng,
    )
    parent_p = np.sqrt(np.maximum(parent_energy**2 - parent_mass_gev**2, 0.0))
    parent_direction = _isotropic_directions(rng, (event_count,))
    parent_momentum = parent_p[:, None] * parent_direction
    parent_momentum = np.stack([parent_momentum, -parent_momentum], axis=1)
    parent_beta = parent_momentum / parent_energy[:, None, None]

    mass2 = parent_mass_gev**2
    lambda_term = (
        mass2 - (lepton_mass + neutrino_mass) ** 2
    ) * (mass2 - (lepton_mass - neutrino_mass) ** 2)
    decay_p = np.sqrt(lambda_term) / (2.0 * parent_mass_gev)
    lepton_energy = np.sqrt(decay_p**2 + lepton_mass**2)
    neutrino_energy = np.sqrt(decay_p**2 + neutrino_mass**2)

    lepton_direction = _isotropic_directions(rng, (event_count, 2))
    lepton_rest = np.concatenate(
        [
            np.full((event_count, 2, 1), lepton_energy),
            decay_p * lepton_direction,
        ],
        axis=-1,
    )
    neutrino_rest = np.concatenate(
        [
            np.full((event_count, 2, 1), neutrino_energy),
            -decay_p * lepton_direction,
        ],
        axis=-1,
    )
    leptons_truth = _boost(lepton_rest, parent_beta)
    neutrinos_truth = _boost(neutrino_rest, parent_beta)

    scale = rng.normal(
        1.0,
        float(physics["lepton_relative_resolution"]),
        size=(event_count, 2, 1),
    )
    nominal_lepton_reco_p = leptons_truth[..., 1:] * scale
    lepton_reco_p = nominal_lepton_reco_p * (1.0 + visible_energy_scale_shift)
    lepton_reco_e = np.sqrt(np.sum(lepton_reco_p**2, axis=-1, keepdims=True) + lepton_mass**2)
    leptons_reco = np.concatenate([lepton_reco_e, lepton_reco_p], axis=-1)

    met = neutrinos_truth[:, :, 1:3].sum(axis=1)
    met += rng.normal(
        0.0, float(physics["met_resolution_gev"]), size=(event_count, 2)
    )
    met -= visible_energy_scale_shift * nominal_lepton_reco_p[..., :2].sum(axis=1)
    observables = np.concatenate([leptons_reco.reshape(event_count, -1), met], axis=1)
    neutrino_target = neutrinos_truth[..., 1:].reshape(event_count, -1)

    return {
        "observables": observables.astype(np.float32),
        "neutrino_target": neutrino_target.astype(np.float32),
        "leptons_truth": leptons_truth.astype(np.float32),
        "leptons_reco": leptons_reco.astype(np.float32),
        "neutrinos_truth": neutrinos_truth.astype(np.float32),
        "parent_energy_gev": parent_energy.astype(np.float32),
        "parent_mass_gev": np.full(event_count, parent_mass_gev, dtype=np.float32),
        "visible_energy_scale_shift": np.full(
            event_count, visible_energy_scale_shift, dtype=np.float32
        ),
    }


def baseline_neutrino_momentum(observables: np.ndarray) -> np.ndarray:
    """Split MET according to lepton transverse momentum and set neutrino pz to zero."""
    leptons = observables[:, :8].reshape(-1, 2, 4)
    met = observables[:, 8:10]
    lepton_pt = np.linalg.norm(leptons[..., 1:3], axis=-1)
    weights = lepton_pt / np.maximum(lepton_pt.sum(axis=1, keepdims=True), 1e-8)
    transverse = weights[..., None] * met[:, None, :]
    pz = np.zeros((*transverse.shape[:2], 1), dtype=transverse.dtype)
    return np.concatenate([transverse, pz], axis=-1)


def reconstruct_parent_masses(
    leptons_reco: np.ndarray,
    neutrino_momentum: np.ndarray,
    neutrino_mass_gev: float,
) -> np.ndarray:
    neutrino_momentum = neutrino_momentum.reshape(-1, 2, 3)
    neutrino_energy = np.sqrt(
        np.sum(neutrino_momentum**2, axis=-1) + neutrino_mass_gev**2
    )
    total_energy = leptons_reco[..., 0] + neutrino_energy
    total_momentum = leptons_reco[..., 1:] + neutrino_momentum
    mass2 = total_energy**2 - np.sum(total_momentum**2, axis=-1)
    return np.sqrt(np.maximum(mass2, 0.0))


def _gaussian_with_offset(
    x: np.ndarray, amplitude: float, mean: float, sigma: float, offset: float
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset


def fit_mass_peak(
    masses: np.ndarray, bins: int, mass_range: tuple[float, float]
) -> tuple[float, float]:
    """Fit a Gaussian peak; fall back to median and central 68% width."""
    values = np.asarray(masses, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    low, high = mass_range
    values = values[(values >= low) & (values <= high)]
    if values.size < 20:
        return float("nan"), float("nan")

    counts, edges = np.histogram(values, bins=bins, range=mass_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centers[np.argmax(counts)])
    q16, q84 = np.quantile(values, [0.16, 0.84])
    initial_sigma = max(float(0.5 * (q84 - q16)), (high - low) / bins)
    fit_mask = np.abs(centers - mode) <= 2.0 * initial_sigma
    try:
        parameters, _ = curve_fit(
            _gaussian_with_offset,
            centers[fit_mask],
            counts[fit_mask],
            p0=[float(counts.max()), mode, initial_sigma, float(counts.min())],
            bounds=(
                [0.0, low, (high - low) / (2.0 * bins), 0.0],
                [np.inf, high, high - low, np.inf],
            ),
            maxfev=20000,
        )
        return float(parameters[1]), abs(float(parameters[2]))
    except (RuntimeError, ValueError):
        return float(np.median(values)), initial_sigma
