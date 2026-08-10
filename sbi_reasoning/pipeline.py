from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .models import (
    ConditionalFlow,
    Standardizer,
    TemplateDiscriminator,
    predict_log_probabilities,
    sample_flow,
    train_dgpo,
    train_discriminator,
    train_flow,
)
from .physics import (
    baseline_neutrino_momentum,
    fit_mass_peak,
    generate_events,
    reconstruct_parent_masses,
)
from .plotting import (
    plot_jes_likelihood_diagnostics,
    plot_mass_jes_likelihood_grid,
    plot_mass_jes_profile_scatter,
    plot_mass_jes_reconstruction_heatmaps,
    plot_dataset_sanity,
    plot_final_benchmark,
    plot_mass_spectra,
    plot_momentum_reconstruction_diagnostics,
    plot_sbi_likelihood_diagnostics,
    plot_sbi_input_distributions,
    plot_training_history,
)
from .tracking import ExperimentTracker


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "seed",
        "output_dir",
        "data",
        "experiments",
        "physics",
        "model",
        "training",
        "sampling",
        "dgpo",
        "evaluation",
        "figures",
        "logging",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def expand_mass_grid(specification: list[float] | dict[str, float]) -> list[float]:
    if isinstance(specification, list):
        return [float(value) for value in specification]
    start = float(specification["start"])
    stop = float(specification["stop"])
    step = float(specification["step"])
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count + 1)]


def enabled_experiments(config: dict[str, Any]) -> set[str]:
    enabled = set(
        config["experiments"].get(
            "enabled", ["mass", "jes", "mass_jes_grid"]
        )
    )
    allowed = {"mass", "jes", "mass_jes_grid"}
    unknown = enabled - allowed
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    supported = [{"mass"}, allowed]
    if enabled not in supported:
        raise ValueError(
            "experiments.enabled must be [mass] or "
            "[mass, jes, mass_jes_grid]"
        )
    return enabled


def mass_tag(mass: float) -> str:
    return f"{mass:g}".replace(".", "p")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_dataset(path: Path, dataset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def dataset_path(output_dir: Path, sample_type: str, mass: float) -> Path:
    return output_dir / "data" / sample_type / f"mass_{mass_tag(mass)}.npz"


def signed_tag(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    return f"{sign}{mass_tag(abs(100.0 * value))}pct"


def independent_dgpo_seed(base_seed: int, mass_gev: float) -> int:
    """Derive an order-independent random seed for one pseudo-data mass."""
    mass_key = int(round(1000.0 * mass_gev))
    return int(np.random.SeedSequence([base_seed, mass_key]).generate_state(1)[0])


def hypothesis_dataset_path(
    output_dir: Path,
    sample_type: str,
    mass: float,
    jes_shift: float,
) -> Path:
    return (
        output_dir
        / "data"
        / sample_type
        / f"mass_{mass_tag(mass)}_jes_{signed_tag(jes_shift)}.npz"
    )


def run_generation(
    config: dict[str, Any], tracker: ExperimentTracker | None = None
) -> None:
    start_time = time.perf_counter()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    template_masses = [float(value) for value in config["data"]["template_masses_gev"]]
    real_masses = expand_mass_grid(config["data"]["real_masses_gev"])
    enabled = enabled_experiments(config)
    systematics_enabled = bool(enabled & {"jes", "mass_jes_grid"})
    jes_shifts = (
        [float(value) for value in config["experiments"]["jes_template_shifts"]]
        if systematics_enabled
        else []
    )
    grid_specs = (
        [(mass, shift) for mass in template_masses for shift in jes_shifts]
        if systematics_enabled
        else []
    )
    jes_real_specs = (
        [
            (
                float(
                    config["experiments"]["jet_energy_scale"]["parent_mass_gev"]
                ),
                float(shift),
            )
            for shift in config["experiments"]["jet_energy_scale"]["real_shifts"]
        ]
        if "jes" in enabled
        else []
    )
    grid_surface_specs = (
        [
            (float(point["mass_gev"]), float(point["jes_shift"]))
            for point in config["experiments"]["mass_jes_grid"]["benchmark_points"]
        ]
        if "mass_jes_grid" in enabled
        else []
    )
    grid_profile_specs = (
        [
            (mass, float(jes_shift))
            for mass in real_masses
            for jes_shift in config["experiments"]["mass_jes_grid"][
                "evaluation_jes_shifts"
            ]
        ]
        if "mass_jes_grid" in enabled
        else []
    )
    grid_real_specs = sorted(set(grid_surface_specs + grid_profile_specs))
    total_datasets = (
        len(template_masses)
        + len(real_masses)
        + len(grid_specs)
        + len(jes_real_specs)
        + len(grid_real_specs)
    )
    seed_sequence = np.random.SeedSequence(int(config["seed"]))
    child_seeds = iter(seed_sequence.spawn(total_datasets))
    template_datasets: dict[float, dict[str, np.ndarray]] = {}

    print("[generate] starting template generation", flush=True)
    for mass in template_masses:
        dataset = generate_events(
            mass,
            int(config["data"]["template_events_per_mass"]),
            config["physics"],
            np.random.default_rng(next(child_seeds)),
        )
        save_dataset(dataset_path(output_dir, "templates", mass), dataset)
        template_datasets[mass] = dataset
        print(f"[generate] template mass={mass:g} GeV events={len(dataset['observables'])}", flush=True)

    print("[generate] starting unlabeled real-data generation", flush=True)
    for mass in real_masses:
        dataset = generate_events(
            mass,
            int(config["data"]["real_events_per_mass"]),
            config["physics"],
            np.random.default_rng(next(child_seeds)),
        )
        save_dataset(dataset_path(output_dir, "real", mass), dataset)
        print(f"[generate] real mass={mass:g} GeV events={len(dataset['observables'])}", flush=True)

    if grid_specs:
        print("[generate] starting mass x JES hypothesis templates", flush=True)
    for mass, jes_shift in grid_specs:
        dataset = generate_events(
            mass,
            int(config["data"]["template_events_per_mass"]),
            config["physics"],
            np.random.default_rng(next(child_seeds)),
            visible_energy_scale_shift=jes_shift,
        )
        save_dataset(
            hypothesis_dataset_path(output_dir, "hypothesis_templates", mass, jes_shift),
            dataset,
        )
        print(
            f"[generate] grid template mass={mass:g} GeV JES={jes_shift:+.0%}",
            flush=True,
        )

    if jes_real_specs:
        print("[generate] starting JES pseudo-data", flush=True)
    for mass, jes_shift in jes_real_specs:
        dataset = generate_events(
            mass,
            int(config["data"]["real_events_per_mass"]),
            config["physics"],
            np.random.default_rng(next(child_seeds)),
            visible_energy_scale_shift=jes_shift,
        )
        save_dataset(
            hypothesis_dataset_path(output_dir, "jes_real", mass, jes_shift), dataset
        )

    if grid_real_specs:
        print("[generate] starting mass x JES benchmark pseudo-data", flush=True)
    for mass, jes_shift in grid_real_specs:
        dataset = generate_events(
            mass,
            int(config["data"]["real_events_per_mass"]),
            config["physics"],
            np.random.default_rng(next(child_seeds)),
            visible_energy_scale_shift=jes_shift,
        )
        save_dataset(
            hypothesis_dataset_path(output_dir, "grid_real", mass, jes_shift), dataset
        )

    _, figure_paths = plot_dataset_sanity(template_datasets, config, output_dir)
    _, input_figure_paths = plot_sbi_input_distributions(
        template_datasets, config, output_dir
    )
    figure_paths.extend(input_figure_paths)
    if tracker is not None:
        tracker.log_figures(figure_paths, "generation")
    print(f"[generate] finished in {time.perf_counter() - start_time:.1f} s", flush=True)


def _split_templates(
    datasets: dict[float, dict[str, np.ndarray]],
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_parts: dict[str, list[np.ndarray]] = {
        "observables": [],
        "neutrino_target": [],
        "labels": [],
    }
    validation_parts = copy.deepcopy(train_parts)
    for label, mass in enumerate(sorted(datasets)):
        dataset = datasets[mass]
        order = rng.permutation(len(dataset["observables"]))
        validation_count = max(1, int(round(validation_fraction * len(order))))
        partitions = {
            "validation": order[:validation_count],
            "train": order[validation_count:],
        }
        for name, indices in partitions.items():
            destination = validation_parts if name == "validation" else train_parts
            destination["observables"].append(dataset["observables"][indices])
            destination["neutrino_target"].append(dataset["neutrino_target"][indices])
            destination["labels"].append(np.full(len(indices), label, dtype=np.int64))
    train = {key: np.concatenate(values) for key, values in train_parts.items()}
    validation = {key: np.concatenate(values) for key, values in validation_parts.items()}
    return train, validation


def _load_all_datasets(
    output_dir: Path, sample_type: str, masses: list[float]
) -> dict[float, dict[str, np.ndarray]]:
    datasets = {}
    for mass in masses:
        path = dataset_path(output_dir, sample_type, mass)
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset {path}; run the generate stage first")
        datasets[mass] = load_dataset(path)
    return datasets


def _load_hypothesis_datasets(
    output_dir: Path,
    sample_type: str,
    specifications: list[tuple[float, float]],
) -> dict[tuple[float, float], dict[str, np.ndarray]]:
    datasets = {}
    for mass, jes_shift in specifications:
        path = hypothesis_dataset_path(output_dir, sample_type, mass, jes_shift)
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset {path}; run the generate stage first")
        datasets[(mass, jes_shift)] = load_dataset(path)
    return datasets


def run_training(
    config: dict[str, Any], tracker: ExperimentTracker | None = None
) -> None:
    start_time = time.perf_counter()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(str(config["training"]["device"]))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        device_name = torch.cuda.get_device_name(device)
        memory_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(
            f"[train] device={device} name={device_name} memory={memory_gib:.1f} GiB "
            f"mixed_precision={config['training']['mixed_precision']}",
            flush=True,
        )
    else:
        print(f"[train] device={device}", flush=True)
    np_rng = np.random.default_rng(int(config["seed"]) + 1)
    torch.manual_seed(int(config["seed"]) + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]) + 1)
    metric_logger = tracker.log if tracker is not None else None

    template_masses = [float(value) for value in config["data"]["template_masses_gev"]]
    real_masses = expand_mass_grid(config["data"]["real_masses_gev"])
    systematics_enabled = bool(
        enabled_experiments(config) & {"jes", "mass_jes_grid"}
    )
    template_datasets = _load_all_datasets(output_dir, "templates", template_masses)
    real_datasets = _load_all_datasets(output_dir, "real", real_masses)
    train, validation = _split_templates(
        template_datasets,
        float(config["data"]["validation_fraction"]),
        np_rng,
    )

    condition_scaler = Standardizer.fit(train["observables"])
    target_scaler = Standardizer.fit(train["neutrino_target"])
    full_train_raw = np.concatenate(
        [train["observables"], train["neutrino_target"]], axis=1
    )
    full_validation_raw = np.concatenate(
        [validation["observables"], validation["neutrino_target"]], axis=1
    )
    full_scaler = Standardizer.fit(full_train_raw)
    train_conditions = condition_scaler.transform(train["observables"])
    validation_conditions = condition_scaler.transform(validation["observables"])
    train_targets = target_scaler.transform(train["neutrino_target"])
    full_train = full_scaler.transform(full_train_raw)
    full_validation = full_scaler.transform(full_validation_raw)

    hidden_dims = [int(value) for value in config["model"]["hidden_dims"]]
    class_count = len(template_masses)
    observed_profiler = TemplateDiscriminator(
        train_conditions.shape[1], class_count, hidden_dims
    )
    full_profiler = TemplateDiscriminator(full_train.shape[1], class_count, hidden_dims)
    print("[train] fitting observable-only template discriminator", flush=True)
    observed_history = train_discriminator(
        observed_profiler,
        train_conditions,
        train["labels"],
        validation_conditions,
        validation["labels"],
        config["training"],
        device,
        np_rng,
        "observable-profiler",
        metric_logger,
    )
    print("[train] fitting observable+truth template discriminator", flush=True)
    full_history = train_discriminator(
        full_profiler,
        full_train,
        train["labels"],
        full_validation,
        validation["labels"],
        config["training"],
        device,
        np_rng,
        "full-profiler",
        metric_logger,
    )
    classifier_validation_rows: list[dict[str, float | str]] = []
    profile_calibration: dict[str, dict[str, list[float]]] = {}
    for profiler_name, profiler, features in [
        ("observable", observed_profiler, validation_conditions),
        ("visible_plus_truth_invisible", full_profiler, full_validation),
    ]:
        log_probabilities = predict_log_probabilities(
            profiler,
            features,
            device,
            int(config["evaluation"]["batch_size"]),
        )
        predictions = log_probabilities.argmax(axis=1)
        raw_anchor_masses = []
        for truth_index, truth_mass in enumerate(template_masses):
            selected = validation["labels"] == truth_index
            raw_anchor_masses.append(
                profile_mass_estimate(
                    log_probabilities[selected], template_masses
                )[0]
            )
            for prediction_index, prediction_mass in enumerate(template_masses):
                count = int(np.sum(predictions[selected] == prediction_index))
                classifier_validation_rows.append(
                    {
                        "profiler": profiler_name,
                        "truth_mass_gev": truth_mass,
                        "predicted_mass_gev": prediction_mass,
                        "event_count": count,
                        "fraction": count / max(int(np.sum(selected)), 1),
                    }
                )
        if np.all(np.diff(raw_anchor_masses) > 0.0):
            profile_calibration[profiler_name] = {
                "raw_mass_gev": [float(value) for value in raw_anchor_masses],
                "calibrated_mass_gev": template_masses,
            }
        else:
            message = (
                f"Cannot calibrate non-monotonic {profiler_name} profile anchors: "
                f"{raw_anchor_masses}"
            )
            if bool(config["evaluation"].get("profile_fail_on_invalid", True)):
                raise RuntimeError(message)
            print(f"[profile-warning] {message}", flush=True)
    with (output_dir / "profile_classifier_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(classifier_validation_rows[0])
        )
        writer.writeheader()
        writer.writerows(classifier_validation_rows)
    calibration_rows = [
        {
            "profiler": profiler_name,
            "raw_profile_mass_gev": raw_mass,
            "calibrated_mass_gev": calibrated_mass,
        }
        for profiler_name, calibration in profile_calibration.items()
        for raw_mass, calibrated_mass in zip(
            calibration["raw_mass_gev"], calibration["calibrated_mass_gev"]
        )
    ]
    with (output_dir / "profile_calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "profiler",
                "raw_profile_mass_gev",
                "calibrated_mass_gev",
            ],
        )
        writer.writeheader()
        writer.writerows(calibration_rows)
    chance_accuracy = 1.0 / len(template_masses)
    minimum_margin = float(
        config["evaluation"].get("profile_minimum_accuracy_above_chance", 0.05)
    )
    for name, history in [
        ("observable", observed_history),
        ("visible+truth-invisible", full_history),
    ]:
        final_accuracy = float(history["validation_accuracy"][-1])
        if final_accuracy < chance_accuracy + minimum_margin:
            message = (
                f"{name} validation accuracy={final_accuracy:.3f} is below "
                f"chance+margin={chance_accuracy + minimum_margin:.3f}"
            )
            if bool(config["evaluation"].get("profile_fail_on_invalid", True)):
                raise RuntimeError(message)
            print(f"[profile-warning] {message}", flush=True)

    hypothesis_grid = None
    hypothesis_histories: dict[str, dict[str, list[float]]] = {}
    if systematics_enabled:
        jes_shifts = [
            float(value) for value in config["experiments"]["jes_template_shifts"]
        ]
        hypothesis_coordinates = sorted(
            [(mass, shift) for mass in template_masses for shift in jes_shifts]
        )
        hypothesis_datasets = _load_hypothesis_datasets(
            output_dir, "hypothesis_templates", hypothesis_coordinates
        )
        hypothesis_train, hypothesis_validation = _split_templates(
            hypothesis_datasets,
            float(config["data"]["validation_fraction"]),
            np_rng,
        )
        hypothesis_condition_scaler = Standardizer.fit(
            hypothesis_train["observables"]
        )
        hypothesis_full_train_raw = np.concatenate(
            [hypothesis_train["observables"], hypothesis_train["neutrino_target"]],
            axis=1,
        )
        hypothesis_full_validation_raw = np.concatenate(
            [
                hypothesis_validation["observables"],
                hypothesis_validation["neutrino_target"],
            ],
            axis=1,
        )
        hypothesis_full_scaler = Standardizer.fit(hypothesis_full_train_raw)
        hypothesis_train_conditions = hypothesis_condition_scaler.transform(
            hypothesis_train["observables"]
        )
        hypothesis_validation_conditions = hypothesis_condition_scaler.transform(
            hypothesis_validation["observables"]
        )
        hypothesis_full_train = hypothesis_full_scaler.transform(
            hypothesis_full_train_raw
        )
        hypothesis_full_validation = hypothesis_full_scaler.transform(
            hypothesis_full_validation_raw
        )
        hypothesis_class_count = len(hypothesis_coordinates)
        hypothesis_observed_profiler = TemplateDiscriminator(
            hypothesis_train_conditions.shape[1], hypothesis_class_count, hidden_dims
        )
        hypothesis_full_profiler = TemplateDiscriminator(
            hypothesis_full_train.shape[1], hypothesis_class_count, hidden_dims
        )
        print("[train] fitting visible mass x JES discriminator", flush=True)
        hypothesis_observed_history = train_discriminator(
            hypothesis_observed_profiler,
            hypothesis_train_conditions,
            hypothesis_train["labels"],
            hypothesis_validation_conditions,
            hypothesis_validation["labels"],
            config["training"],
            device,
            np_rng,
            "mass-jes-visible-profiler",
            metric_logger,
        )
        print("[train] fitting visible+truth mass x JES discriminator", flush=True)
        hypothesis_full_history = train_discriminator(
            hypothesis_full_profiler,
            hypothesis_full_train,
            hypothesis_train["labels"],
            hypothesis_full_validation,
            hypothesis_validation["labels"],
            config["training"],
            device,
            np_rng,
            "mass-jes-full-profiler",
            metric_logger,
        )
        hypothesis_histories = {
            "hypothesis_observed_profiler": hypothesis_observed_history,
            "hypothesis_full_profiler": hypothesis_full_history,
        }
        hypothesis_grid = {
            "coordinates": hypothesis_coordinates,
            "condition_scaler": hypothesis_condition_scaler.as_dict(),
            "full_scaler": hypothesis_full_scaler.as_dict(),
            "observed_profiler_state": hypothesis_observed_profiler.cpu().state_dict(),
            "full_profiler_state": hypothesis_full_profiler.cpu().state_dict(),
        }

    flow = ConditionalFlow(
        target_dim=train_targets.shape[1],
        condition_dim=train_conditions.shape[1],
        hidden_dims=hidden_dims,
        time_embedding_dim=int(config["model"]["time_embedding_dim"]),
    )
    print("[train] fitting conditional rectified flow", flush=True)
    flow_history = train_flow(
        flow,
        train_conditions,
        train_targets,
        config["training"],
        device,
        np_rng,
        metric_logger,
    )
    sft_state = copy.deepcopy(flow.state_dict())
    reference = copy.deepcopy(flow)
    if config["dgpo"].get("scope") != "independent_per_pseudo_data":
        raise ValueError(
            "dgpo.scope must be independent_per_pseudo_data"
        )
    iterations_per_pseudo_data = int(
        config["dgpo"]["iterations_per_pseudo_data"]
    )
    dgpo_config = {**config["dgpo"], "iterations": iterations_per_pseudo_data}
    dgpo_states: dict[float, dict[str, torch.Tensor]] = {}
    dgpo_scenario_seeds: dict[float, int] = {}
    dgpo_history: dict[str, list[float]] = {
        "loss": [],
        "reward_mean": [],
        "reward_spread": [],
        "mass_gev": [],
    }
    print(
        "[train] starting independent DGPO fine-tuning for each pseudo-data mass",
        flush=True,
    )
    for mass, dataset in sorted(real_datasets.items()):
        scenario_seed = independent_dgpo_seed(int(config["seed"]), mass)
        scenario_rng = np.random.default_rng(scenario_seed)
        torch.manual_seed(scenario_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(scenario_seed)
        print(
            f"[train] DGPO pseudo-data mass={mass:g} GeV seed={scenario_seed}",
            flush=True,
        )
        policy = copy.deepcopy(reference)
        scenario_history = train_dgpo(
            policy,
            reference,
            observed_profiler,
            full_profiler,
            {mass: dataset},
            condition_scaler,
            target_scaler,
            full_scaler,
            dgpo_config,
            config["sampling"],
            int(config["evaluation"]["batch_size"]),
            device,
            scenario_rng,
            str(config["training"].get("mixed_precision", "none")),
            metric_logger,
        )
        dgpo_states[float(mass)] = {
            key: value.cpu() for key, value in policy.state_dict().items()
        }
        dgpo_scenario_seeds[float(mass)] = scenario_seed
        for metric in ["loss", "reward_mean", "reward_spread"]:
            dgpo_history[metric].extend(scenario_history[metric])
        dgpo_history["mass_gev"].extend(
            [float(mass)] * len(scenario_history["loss"])
        )

    checkpoint = {
        "template_masses": template_masses,
        "hidden_dims": hidden_dims,
        "time_embedding_dim": int(config["model"]["time_embedding_dim"]),
        "condition_scaler": condition_scaler.as_dict(),
        "target_scaler": target_scaler.as_dict(),
        "full_scaler": full_scaler.as_dict(),
        "observed_profiler_state": observed_profiler.cpu().state_dict(),
        "full_profiler_state": full_profiler.cpu().state_dict(),
        "flow_sft_state": {key: value.cpu() for key, value in sft_state.items()},
        "flow_dgpo_states": dgpo_states,
        "dgpo_scenario_seeds": dgpo_scenario_seeds,
        "dgpo_scope": "independent_per_pseudo_data",
        "profile_calibration": profile_calibration,
        "history": {
            "observed_profiler": observed_history,
            "full_profiler": full_history,
            "flow": flow_history,
            "dgpo": dgpo_history,
            **hypothesis_histories,
        },
        "hypothesis_grid": hypothesis_grid,
    }
    torch.save(checkpoint, output_dir / "models.pt")
    history_rows = []
    for component, history in checkpoint["history"].items():
        metric_series = history if isinstance(history, dict) else {"loss": history}
        for metric, values in metric_series.items():
            for step, value in enumerate(values, start=1):
                history_rows.append(
                    {
                        "component": component,
                        "metric": metric,
                        "step": step,
                        "value": float(value),
                    }
                )
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    figure_paths = plot_training_history(
        observed_history,
        full_history,
        flow_history,
        dgpo_history,
        config["figures"],
        output_dir,
    )
    if tracker is not None:
        tracker.log_figures(figure_paths, "training")
    print(f"[train] finished in {time.perf_counter() - start_time:.1f} s", flush=True)


def profile_mass_estimate(
    log_probabilities: np.ndarray,
    template_masses: list[float],
    calibration: dict[str, list[float]] | None = None,
) -> tuple[float, float]:
    masses = np.asarray(template_masses, dtype=np.float64)
    profile = log_probabilities.mean(axis=0, dtype=np.float64)
    negative_two_delta_log_likelihood = -2.0 * (profile - profile.max())
    quadratic, linear, _ = np.polyfit(masses, negative_two_delta_log_likelihood, 2)
    if quadratic <= 0.0:
        estimate = float(masses[np.argmin(negative_two_delta_log_likelihood)])
        uncertainty = float("nan")
    else:
        estimate = float(
            np.clip(-linear / (2.0 * quadratic), masses.min(), masses.max())
        )
        uncertainty = float(1.0 / np.sqrt(quadratic * len(log_probabilities)))
    if calibration is not None:
        raw_anchors = np.asarray(calibration["raw_mass_gev"], dtype=np.float64)
        calibrated_anchors = np.asarray(
            calibration["calibrated_mass_gev"], dtype=np.float64
        )
        segment = int(
            np.clip(np.searchsorted(raw_anchors, estimate) - 1, 0, len(raw_anchors) - 2)
        )
        slope = (
            calibrated_anchors[segment + 1] - calibrated_anchors[segment]
        ) / (raw_anchors[segment + 1] - raw_anchors[segment])
        estimate = float(np.interp(estimate, raw_anchors, calibrated_anchors))
        uncertainty *= abs(float(slope))
    return estimate, uncertainty


def profile_quality_metrics(
    log_probabilities: np.ndarray,
    template_masses: list[float],
    truth_mass_gev: float,
    profile_name: str,
    calibration: dict[str, list[float]] | None = None,
) -> dict[str, float | str]:
    masses = np.asarray(template_masses, dtype=np.float64)
    scores = negative_two_delta_mean_log_score(log_probabilities)
    quadratic, _, _ = np.polyfit(masses, scores, 2)
    estimate, uncertainty = profile_mass_estimate(
        log_probabilities, template_masses, calibration
    )
    boundary = np.isclose(estimate, masses.min()) or np.isclose(
        estimate, masses.max()
    )
    return {
        "profile": profile_name,
        "truth_mass_gev": truth_mass_gev,
        "estimated_mass_gev": estimate,
        "bias_gev": estimate - truth_mass_gev,
        "uncertainty_gev": uncertainty,
        "quadratic_curvature_per_gev2": float(quadratic),
        "score_dynamic_range": float(scores.max() - scores.min()),
        "convex": float(quadratic > 0.0),
        "boundary_estimate": float(boundary),
    }


def negative_two_delta_mean_log_score(log_probabilities: np.ndarray) -> np.ndarray:
    profile = log_probabilities.mean(axis=0, dtype=np.float64)
    return -2.0 * (profile - profile.max())


def profile_vertex_from_scores(
    coordinates: list[float], scores: np.ndarray
) -> float:
    values = np.asarray(coordinates, dtype=np.float64)
    quadratic, linear, _ = np.polyfit(values, scores, 2)
    if quadratic <= 0.0:
        return float(values[np.argmin(scores)])
    return float(np.clip(-linear / (2.0 * quadratic), values.min(), values.max()))


def scores_to_hypothesis_surface(
    scores: np.ndarray,
    hypothesis_coordinates: list[tuple[float, float]],
    template_masses: list[float],
    template_shifts: list[float],
) -> np.ndarray:
    surface = np.empty((len(template_shifts), len(template_masses)))
    for index, (mass, shift) in enumerate(hypothesis_coordinates):
        mass_index = template_masses.index(mass)
        shift_index = template_shifts.index(shift)
        surface[shift_index, mass_index] = scores[index]
    return surface


def _load_models(
    checkpoint_path: Path, device: torch.device
) -> tuple[
    dict[str, Any],
    Standardizer,
    Standardizer,
    Standardizer,
    TemplateDiscriminator,
    TemplateDiscriminator,
    ConditionalFlow,
    dict[float, ConditionalFlow],
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    condition_scaler = Standardizer.from_dict(checkpoint["condition_scaler"])
    target_scaler = Standardizer.from_dict(checkpoint["target_scaler"])
    full_scaler = Standardizer.from_dict(checkpoint["full_scaler"])
    hidden_dims = checkpoint["hidden_dims"]
    class_count = len(checkpoint["template_masses"])
    observed_profiler = TemplateDiscriminator(
        len(condition_scaler.mean), class_count, hidden_dims
    )
    full_profiler = TemplateDiscriminator(len(full_scaler.mean), class_count, hidden_dims)
    observed_profiler.load_state_dict(checkpoint["observed_profiler_state"])
    full_profiler.load_state_dict(checkpoint["full_profiler_state"])
    architecture = {
        "target_dim": len(target_scaler.mean),
        "condition_dim": len(condition_scaler.mean),
        "hidden_dims": hidden_dims,
        "time_embedding_dim": checkpoint["time_embedding_dim"],
    }
    sft_flow = ConditionalFlow(**architecture)
    sft_flow.load_state_dict(checkpoint["flow_sft_state"])
    dgpo_flows: dict[float, ConditionalFlow] = {}
    if "flow_dgpo_states" in checkpoint:
        for mass, state in checkpoint["flow_dgpo_states"].items():
            model = ConditionalFlow(**architecture)
            model.load_state_dict(state)
            dgpo_flows[float(mass)] = model.to(device).eval()
    else:
        raise RuntimeError(
            "Checkpoint contains the obsolete shared DGPO model; rerun training "
            "to create one flow_dgpo_states entry per pseudo-data mass"
        )
    return (
        checkpoint,
        condition_scaler,
        target_scaler,
        full_scaler,
        observed_profiler.to(device).eval(),
        full_profiler.to(device).eval(),
        sft_flow.to(device).eval(),
        dgpo_flows,
    )


def _dgpo_flow_for_mass(
    flows: dict[float, ConditionalFlow], mass: float
) -> ConditionalFlow:
    if mass in flows:
        return flows[mass]
    available = sorted(flows)
    raise KeyError(
        f"No independent DGPO checkpoint for {mass:g} GeV; "
        f"available masses are {available}"
    )


def _load_hypothesis_models(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[
    list[tuple[float, float]],
    Standardizer,
    Standardizer,
    TemplateDiscriminator,
    TemplateDiscriminator,
]:
    grid = checkpoint["hypothesis_grid"]
    coordinates = [
        (float(coordinate[0]), float(coordinate[1]))
        for coordinate in grid["coordinates"]
    ]
    condition_scaler = Standardizer.from_dict(grid["condition_scaler"])
    full_scaler = Standardizer.from_dict(grid["full_scaler"])
    hidden_dims = checkpoint["hidden_dims"]
    observed = TemplateDiscriminator(
        len(condition_scaler.mean), len(coordinates), hidden_dims
    )
    full = TemplateDiscriminator(len(full_scaler.mean), len(coordinates), hidden_dims)
    observed.load_state_dict(grid["observed_profiler_state"])
    full.load_state_dict(grid["full_profiler_state"])
    return (
        coordinates,
        condition_scaler,
        full_scaler,
        observed.to(device).eval(),
        full.to(device).eval(),
    )


def _sample_dataset(
    model: ConditionalFlow,
    conditions: np.ndarray,
    target_scaler: Standardizer,
    sampling_config: dict[str, Any],
    batch_size: int,
    device: torch.device,
    initial_noise: np.ndarray,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(conditions), batch_size):
        stop = start + batch_size
        condition = torch.from_numpy(conditions[start:stop]).to(device)
        noise = torch.from_numpy(initial_noise[start:stop]).to(device)
        generated = sample_flow(
            model,
            condition,
            int(sampling_config["ode_steps"]),
            float(sampling_config["output_clip_sigma"]),
            noise,
        )
        chunks.append(generated.cpu().numpy())
    return target_scaler.inverse(np.concatenate(chunks))


def run_evaluation(
    config: dict[str, Any], tracker: ExperimentTracker | None = None
) -> None:
    start_time = time.perf_counter()
    output_dir = Path(config["output_dir"])
    device = choose_device(str(config["training"]["device"]))
    systematics_enabled = bool(
        enabled_experiments(config) & {"jes", "mass_jes_grid"}
    )
    checkpoint_path = output_dir / "models.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing {checkpoint_path}; run the train stage first")
    (
        checkpoint,
        condition_scaler,
        target_scaler,
        full_scaler,
        observed_profiler,
        full_profiler,
        sft_flow,
        dgpo_flows,
    ) = _load_models(checkpoint_path, device)
    calibrations = checkpoint.get("profile_calibration", {})
    observed_calibration = calibrations.get("observable")
    full_calibration = calibrations.get("visible_plus_truth_invisible")
    real_masses = expand_mass_grid(config["data"]["real_masses_gev"])
    real_datasets = _load_all_datasets(output_dir, "real", real_masses)
    batch_size = int(config["evaluation"]["batch_size"])
    bins = int(config["evaluation"]["histogram_bins"])
    mass_range = tuple(float(value) for value in config["evaluation"]["mass_range_gev"])
    neutrino_mass = float(config["physics"]["neutrino_mass_gev"])
    rng = np.random.default_rng(int(config["seed"]) + 2)
    metrics: list[dict[str, float | str]] = []
    spectra: dict[float, dict[str, np.ndarray]] = {}
    profile_rows = []
    likelihood_profiles: dict[float, dict[str, np.ndarray]] = {}
    profile_quality_rows: list[dict[str, float | str]] = []
    momentum_diagnostic: dict[str, np.ndarray] | None = None
    momentum_diagnostic_mass = float(
        config["figures"]["momentum_diagnostic_mass_gev"]
    )

    print(f"[evaluate] device={device}", flush=True)
    for mass, dataset in sorted(real_datasets.items()):
        conditions = condition_scaler.transform(dataset["observables"])
        initial_noise = rng.normal(size=(len(conditions), len(target_scaler.mean))).astype(np.float32)
        sft_target = _sample_dataset(
            sft_flow,
            conditions,
            target_scaler,
            config["sampling"],
            batch_size,
            device,
            initial_noise,
        )
        dgpo_target = _sample_dataset(
            _dgpo_flow_for_mass(dgpo_flows, mass),
            conditions,
            target_scaler,
            config["sampling"],
            batch_size,
            device,
            initial_noise,
        )
        method_targets = {
            "baseline": baseline_neutrino_momentum(dataset["observables"]),
            "sft": sft_target,
            "dgpo": dgpo_target,
            "oracle": dataset["neutrinos_truth"][..., 1:],
        }
        if np.isclose(mass, momentum_diagnostic_mass):
            momentum_diagnostic = {
                "truth": dataset["neutrino_target"],
                **method_targets,
            }
        spectra[mass] = {}
        for method, target in method_targets.items():
            reconstructed = reconstruct_parent_masses(
                dataset["leptons_reco"], target, neutrino_mass
            ).ravel()
            spectra[mass][method] = reconstructed
            peak, resolution = fit_mass_peak(reconstructed, bins, mass_range)
            metrics.append(
                {
                    "method": method,
                    "truth_mass_gev": mass,
                    "peak_gev": peak,
                    "resolution_gev": resolution,
                    "bias_gev": peak - mass,
                    "relative_precision_percent": 100.0 * resolution / peak,
                }
            )

        observed_log_probabilities = predict_log_probabilities(
            observed_profiler, conditions, device, batch_size
        )
        raw_observed_mass, _ = profile_mass_estimate(
            observed_log_probabilities, checkpoint["template_masses"]
        )
        observed_mass, observed_uncertainty = profile_mass_estimate(
            observed_log_probabilities,
            checkpoint["template_masses"],
            observed_calibration,
        )
        truth_full_features = full_scaler.transform(
            np.concatenate(
                [dataset["observables"], dataset["neutrino_target"]], axis=1
            )
        )
        truth_full_log_probabilities = predict_log_probabilities(
            full_profiler, truth_full_features, device, batch_size
        )
        raw_truth_full_mass, _ = profile_mass_estimate(
            truth_full_log_probabilities, checkpoint["template_masses"]
        )
        truth_full_mass, truth_full_uncertainty = profile_mass_estimate(
            truth_full_log_probabilities,
            checkpoint["template_masses"],
            full_calibration,
        )
        likelihood_profiles[mass] = {
            "visible": negative_two_delta_mean_log_score(
                observed_log_probabilities
            ),
            "visible_truth": negative_two_delta_mean_log_score(
                truth_full_log_probabilities
            ),
        }
        profile_quality_rows.extend(
            [
                profile_quality_metrics(
                    observed_log_probabilities,
                    checkpoint["template_masses"],
                    mass,
                    "observable",
                    observed_calibration,
                ),
                profile_quality_metrics(
                    truth_full_log_probabilities,
                    checkpoint["template_masses"],
                    mass,
                    "visible_plus_truth_invisible",
                    full_calibration,
                ),
            ]
        )
        full_profile_results = {}
        raw_full_profile_results = {}
        for method, target in [("sft", sft_target), ("dgpo", dgpo_target)]:
            full_features = full_scaler.transform(
                np.concatenate([dataset["observables"], target], axis=1)
            )
            full_log_probabilities = predict_log_probabilities(
                full_profiler, full_features, device, batch_size
            )
            raw_full_profile_results[method] = profile_mass_estimate(
                full_log_probabilities, checkpoint["template_masses"]
            )[0]
            full_profile_results[method] = profile_mass_estimate(
                full_log_probabilities,
                checkpoint["template_masses"],
                full_calibration,
            )
            profile_quality_rows.append(
                profile_quality_metrics(
                    full_log_probabilities,
                    checkpoint["template_masses"],
                    mass,
                    method,
                    full_calibration,
                )
            )
        profile_rows.append(
            {
                "truth_mass_gev": mass,
                "raw_observed_profile_mass_gev": raw_observed_mass,
                "observed_profile_mass_gev": observed_mass,
                "observed_profile_uncertainty_gev": observed_uncertainty,
                "raw_truth_full_profile_mass_gev": raw_truth_full_mass,
                "truth_full_profile_mass_gev": truth_full_mass,
                "truth_full_profile_uncertainty_gev": truth_full_uncertainty,
                "raw_sft_full_profile_mass_gev": raw_full_profile_results["sft"],
                "sft_full_profile_mass_gev": full_profile_results["sft"][0],
                "sft_full_profile_uncertainty_gev": full_profile_results["sft"][1],
                "raw_dgpo_full_profile_mass_gev": raw_full_profile_results["dgpo"],
                "dgpo_full_profile_mass_gev": full_profile_results["dgpo"][0],
                "dgpo_full_profile_uncertainty_gev": full_profile_results["dgpo"][1],
            }
        )
        dgpo_metric = metrics[-2]
        print(
            f"[evaluate] mass={mass:g} GeV dgpo_peak={dgpo_metric['peak_gev']:.2f} "
            f"resolution={dgpo_metric['resolution_gev']:.2f} "
            f"bias={dgpo_metric['bias_gev']:+.2f}",
            flush=True,
        )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output_dir / "profile_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    with (output_dir / "profile_quality_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_quality_rows[0]))
        writer.writeheader()
        writer.writerows(profile_quality_rows)
    interior_min = min(float(value) for value in checkpoint["template_masses"])
    interior_max = max(float(value) for value in checkpoint["template_masses"])
    observable_interior = [
        row
        for row in profile_quality_rows
        if row["profile"] == "observable"
        and interior_min < float(row["truth_mass_gev"]) < interior_max
    ]
    boundary_fraction = (
        float(
            np.mean(
                [float(row["boundary_estimate"]) for row in observable_interior]
            )
        )
        if observable_interior
        else float("nan")
    )
    nonconvex_fraction = (
        float(
            np.mean([1.0 - float(row["convex"]) for row in observable_interior])
        )
        if observable_interior
        else float("nan")
    )
    print(
        f"[profile-check] observable interior boundary_fraction={boundary_fraction:.3f} "
        f"nonconvex_fraction={nonconvex_fraction:.3f}",
        flush=True,
    )
    if momentum_diagnostic is None:
        raise ValueError(
            f"Momentum diagnostic mass {momentum_diagnostic_mass:g} GeV "
            "is not present in data.real_masses_gev"
        )

    if not systematics_enabled:
        summary = {}
        for method in ["baseline", "sft", "dgpo", "oracle"]:
            selected = [row for row in metrics if row["method"] == method]
            summary[method] = {
                "mean_absolute_bias_gev": float(
                    np.mean([abs(float(row["bias_gev"])) for row in selected])
                ),
                "mean_resolution_gev": float(
                    np.mean([float(row["resolution_gev"]) for row in selected])
                ),
                "mean_relative_precision_percent": float(
                    np.mean(
                        [
                            float(row["relative_precision_percent"])
                            for row in selected
                        ]
                    )
                ),
            }
        summary["profile_alignment"] = {
            "sft_mean_absolute_difference_gev": float(
                np.mean(
                    [
                        abs(
                            row["sft_full_profile_mass_gev"]
                            - row["observed_profile_mass_gev"]
                        )
                        for row in profile_rows
                    ]
                )
            ),
            "dgpo_mean_absolute_difference_gev": float(
                np.mean(
                    [
                        abs(
                            row["dgpo_full_profile_mass_gev"]
                            - row["observed_profile_mass_gev"]
                        )
                        for row in profile_rows
                    ]
                )
            ),
        }
        summary["profile_quality"] = {
            "observable_interior_boundary_fraction": boundary_fraction,
            "observable_interior_nonconvex_fraction": nonconvex_fraction,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        figure_paths = []
        figure_paths.extend(
            plot_momentum_reconstruction_diagnostics(
                momentum_diagnostic,
                momentum_diagnostic_mass,
                config,
                output_dir,
            )
        )
        figure_paths.extend(plot_mass_spectra(spectra, config, output_dir))
        figure_paths.extend(
            plot_final_benchmark(metrics, profile_rows, config, output_dir)
        )
        _, likelihood_paths = plot_sbi_likelihood_diagnostics(
            likelihood_profiles,
            profile_rows,
            checkpoint["history"],
            checkpoint["template_masses"],
            config,
            output_dir,
        )
        figure_paths.extend(likelihood_paths)
        if tracker is not None:
            tracker.log_figures(figure_paths, "evaluation")
            tracker.update_summary(
                {
                    "dgpo_mean_absolute_bias_gev": summary["dgpo"][
                        "mean_absolute_bias_gev"
                    ],
                    "dgpo_mean_resolution_gev": summary["dgpo"][
                        "mean_resolution_gev"
                    ],
                    "dgpo_profile_alignment_gev": summary["profile_alignment"][
                        "dgpo_mean_absolute_difference_gev"
                    ],
                }
            )
        print(
            f"[evaluate] mass-only workflow finished in "
            f"{time.perf_counter() - start_time:.1f} s",
            flush=True,
        )
        return

    (
        hypothesis_coordinates,
        hypothesis_condition_scaler,
        hypothesis_full_scaler,
        hypothesis_observed_profiler,
        hypothesis_full_profiler,
    ) = _load_hypothesis_models(checkpoint, device)
    template_masses = [float(value) for value in checkpoint["template_masses"]]
    template_shifts = [
        float(value) for value in config["experiments"]["jes_template_shifts"]
    ]
    jes_mass = float(config["experiments"]["jet_energy_scale"]["parent_mass_gev"])
    jes_real_specs = [
        (jes_mass, float(shift))
        for shift in config["experiments"]["jet_energy_scale"]["real_shifts"]
    ]
    jes_datasets = _load_hypothesis_datasets(output_dir, "jes_real", jes_real_specs)
    jes_indices = [
        index
        for index, (mass, _) in enumerate(hypothesis_coordinates)
        if np.isclose(mass, jes_mass)
    ]
    jes_profiles: dict[float, dict[str, np.ndarray]] = {}
    jes_estimate_rows: list[dict[str, float]] = []
    for (_, truth_shift), dataset in sorted(jes_datasets.items()):
        visible_features = hypothesis_condition_scaler.transform(dataset["observables"])
        truth_full_features = hypothesis_full_scaler.transform(
            np.concatenate(
                [dataset["observables"], dataset["neutrino_target"]], axis=1
            )
        )
        visible_log = predict_log_probabilities(
            hypothesis_observed_profiler, visible_features, device, batch_size
        )[:, jes_indices]
        full_log = predict_log_probabilities(
            hypothesis_full_profiler, truth_full_features, device, batch_size
        )[:, jes_indices]
        visible_scores = negative_two_delta_mean_log_score(visible_log)
        full_scores = negative_two_delta_mean_log_score(full_log)
        jes_profiles[truth_shift] = {
            "visible": visible_scores,
            "visible_truth": full_scores,
        }
        jes_estimate_rows.append(
            {
                "truth_jes_shift": truth_shift,
                "visible_profile_jes_shift": profile_vertex_from_scores(
                    template_shifts, visible_scores
                ),
                "truth_full_profile_jes_shift": profile_vertex_from_scores(
                    template_shifts, full_scores
                ),
            }
        )
    with (output_dir / "jes_profile_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jes_estimate_rows[0]))
        writer.writeheader()
        writer.writerows(jes_estimate_rows)

    grid_specs = [
        (float(point["mass_gev"]), float(point["jes_shift"]))
        for point in config["experiments"]["mass_jes_grid"]["benchmark_points"]
    ]
    grid_datasets = _load_hypothesis_datasets(output_dir, "grid_real", grid_specs)
    grid_surfaces: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for coordinate, dataset in grid_datasets.items():
        visible_features = hypothesis_condition_scaler.transform(dataset["observables"])
        truth_full_features = hypothesis_full_scaler.transform(
            np.concatenate(
                [dataset["observables"], dataset["neutrino_target"]], axis=1
            )
        )
        visible_scores = negative_two_delta_mean_log_score(
            predict_log_probabilities(
                hypothesis_observed_profiler, visible_features, device, batch_size
            )
        )
        full_scores = negative_two_delta_mean_log_score(
            predict_log_probabilities(
                hypothesis_full_profiler, truth_full_features, device, batch_size
            )
        )
        surfaces = {}
        for name, scores in [("visible", visible_scores), ("visible_truth", full_scores)]:
            surfaces[name] = scores_to_hypothesis_surface(
                scores,
                hypothesis_coordinates,
                template_masses,
                template_shifts,
            )
        grid_surfaces[coordinate] = surfaces

    grid_profile_specs = [
        (mass, float(jes_shift))
        for mass in real_masses
        for jes_shift in config["experiments"]["mass_jes_grid"][
            "evaluation_jes_shifts"
        ]
    ]
    grid_profile_datasets = _load_hypothesis_datasets(
        output_dir, "grid_real", grid_profile_specs
    )
    grid_profile_rows: list[dict[str, float | str]] = []
    grid_reconstruction_rows: list[dict[str, float | str]] = []
    print("[evaluate] profiling the mass x JES real-data grid", flush=True)
    for (truth_mass, truth_shift), dataset in sorted(grid_profile_datasets.items()):
        nominal_conditions = condition_scaler.transform(dataset["observables"])
        initial_noise = rng.normal(
            size=(len(nominal_conditions), len(target_scaler.mean))
        ).astype(np.float32)
        sft_target = _sample_dataset(
            sft_flow,
            nominal_conditions,
            target_scaler,
            config["sampling"],
            batch_size,
            device,
            initial_noise,
        )
        dgpo_target = _sample_dataset(
            _dgpo_flow_for_mass(dgpo_flows, truth_mass),
            nominal_conditions,
            target_scaler,
            config["sampling"],
            batch_size,
            device,
            initial_noise,
        )
        reconstruction_targets = {
            "baseline": baseline_neutrino_momentum(dataset["observables"]),
            "sft": sft_target,
            "dgpo": dgpo_target,
            "oracle": dataset["neutrinos_truth"][..., 1:],
        }
        for method, target in reconstruction_targets.items():
            reconstructed = reconstruct_parent_masses(
                dataset["leptons_reco"], target, neutrino_mass
            ).ravel()
            peak, resolution = fit_mass_peak(reconstructed, bins, mass_range)
            grid_reconstruction_rows.append(
                {
                    "method": method,
                    "truth_mass_gev": truth_mass,
                    "truth_jes_shift": truth_shift,
                    "peak_gev": peak,
                    "bias_gev": peak - truth_mass,
                    "resolution_gev": resolution,
                }
            )
        visible_features = hypothesis_condition_scaler.transform(dataset["observables"])
        method_scores = {
            "visible": negative_two_delta_mean_log_score(
                predict_log_probabilities(
                    hypothesis_observed_profiler,
                    visible_features,
                    device,
                    batch_size,
                )
            )
        }
        for method, target in [("sft", sft_target), ("dgpo", dgpo_target)]:
            full_features = hypothesis_full_scaler.transform(
                np.concatenate([dataset["observables"], target], axis=1)
            )
            method_scores[method] = negative_two_delta_mean_log_score(
                predict_log_probabilities(
                    hypothesis_full_profiler, full_features, device, batch_size
                )
            )
        for method, scores in method_scores.items():
            surface = scores_to_hypothesis_surface(
                scores,
                hypothesis_coordinates,
                template_masses,
                template_shifts,
            )
            profiled_mass = profile_vertex_from_scores(
                template_masses, surface.min(axis=0)
            )
            profiled_jes_shift = profile_vertex_from_scores(
                template_shifts, surface.min(axis=1)
            )
            grid_profile_rows.append(
                {
                    "method": method,
                    "truth_mass_gev": truth_mass,
                    "truth_jes_shift": truth_shift,
                    "profiled_mass_gev": profiled_mass,
                    "bias_gev": profiled_mass - truth_mass,
                    "profiled_jes_shift": profiled_jes_shift,
                    "jes_bias": profiled_jes_shift - truth_shift,
                }
            )
        print(
            f"[evaluate] grid mass={truth_mass:g} GeV JES={truth_shift:+.0%}",
            flush=True,
        )
    with (output_dir / "mass_jes_profile_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid_profile_rows[0]))
        writer.writeheader()
        writer.writerows(grid_profile_rows)
    with (output_dir / "mass_jes_reconstruction_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(grid_reconstruction_rows[0])
        )
        writer.writeheader()
        writer.writerows(grid_reconstruction_rows)

    summary = {}
    for method in ["baseline", "sft", "dgpo", "oracle"]:
        selected = [row for row in metrics if row["method"] == method]
        summary[method] = {
            "mean_absolute_bias_gev": float(np.mean([abs(float(row["bias_gev"])) for row in selected])),
            "mean_resolution_gev": float(np.mean([float(row["resolution_gev"]) for row in selected])),
            "mean_relative_precision_percent": float(
                np.mean([float(row["relative_precision_percent"]) for row in selected])
            ),
        }
    summary["profile_alignment"] = {
        "sft_mean_absolute_difference_gev": float(
            np.mean(
                [
                    abs(row["sft_full_profile_mass_gev"] - row["observed_profile_mass_gev"])
                    for row in profile_rows
                ]
            )
        ),
        "dgpo_mean_absolute_difference_gev": float(
            np.mean(
                [
                    abs(row["dgpo_full_profile_mass_gev"] - row["observed_profile_mass_gev"])
                    for row in profile_rows
                ]
            )
        ),
    }
    summary["profile_quality"] = {
        "observable_interior_boundary_fraction": boundary_fraction,
        "observable_interior_nonconvex_fraction": nonconvex_fraction,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    figure_paths = []
    figure_paths.extend(
        plot_momentum_reconstruction_diagnostics(
            momentum_diagnostic,
            momentum_diagnostic_mass,
            config,
            output_dir,
        )
    )
    figure_paths.extend(plot_mass_spectra(spectra, config, output_dir))
    figure_paths.extend(plot_final_benchmark(metrics, profile_rows, config, output_dir))
    _, likelihood_paths = plot_sbi_likelihood_diagnostics(
        likelihood_profiles,
        profile_rows,
        checkpoint["history"],
        template_masses,
        config,
        output_dir,
    )
    figure_paths.extend(likelihood_paths)
    _, jes_paths = plot_jes_likelihood_diagnostics(
        jes_profiles,
        jes_estimate_rows,
        checkpoint["history"],
        template_shifts,
        config,
        output_dir,
    )
    figure_paths.extend(jes_paths)
    _, grid_paths = plot_mass_jes_likelihood_grid(
        grid_surfaces,
        template_masses,
        template_shifts,
        config,
        output_dir,
    )
    figure_paths.extend(grid_paths)
    figure_paths.extend(
        plot_mass_jes_profile_scatter(
            grid_profile_rows,
            config,
            output_dir,
        )
    )
    figure_paths.extend(
        plot_mass_jes_reconstruction_heatmaps(
            grid_reconstruction_rows,
            config,
            output_dir,
        )
    )
    if tracker is not None:
        tracker.log_figures(figure_paths, "evaluation")
        tracker.update_summary(
            {
                "dgpo_mean_absolute_bias_gev": summary["dgpo"]["mean_absolute_bias_gev"],
                "dgpo_mean_resolution_gev": summary["dgpo"]["mean_resolution_gev"],
                "dgpo_profile_alignment_gev": summary["profile_alignment"]["dgpo_mean_absolute_difference_gev"],
            }
        )
    print(f"[evaluate] finished in {time.perf_counter() - start_time:.1f} s", flush=True)


def run_pipeline(config: dict[str, Any], stage: str) -> None:
    tracker = ExperimentTracker.initialize(config, stage)
    try:
        if stage in {"generate", "all"}:
            run_generation(config, tracker)
        if stage in {"train", "all"}:
            run_training(config, tracker)
        if stage in {"evaluate", "all"}:
            run_evaluation(config, tracker)
    finally:
        tracker.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the data-profile-guided neutrino reconstruction toy model"
    )
    parser.add_argument("--config", type=Path, default=Path("config/toy.yaml"))
    parser.add_argument(
        "--stage", choices=["generate", "train", "evaluate", "all"], default="all"
    )
    arguments = parser.parse_args()
    print(f"[startup] config={arguments.config} stage={arguments.stage}", flush=True)
    config = load_config(arguments.config)
    run_pipeline(config, arguments.stage)


if __name__ == "__main__":
    main()
