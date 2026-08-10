from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

from .physics import fit_mass_peak, reconstruct_parent_masses


COLORS = {
    "neutral": "#767676",
    "dark": "#272727",
    "blue": "#0F4D92",
    "light_blue": "#3775BA",
    "red": "#B64342",
    "teal": "#42949E",
    "orange": "#E69F00",
    "light": "#CFCECE",
}

METHOD_COLORS = {
    "baseline": COLORS["neutral"],
    "sft": COLORS["light_blue"],
    "dgpo": COLORS["red"],
    "oracle": COLORS["blue"],
}

METHOD_LABELS = {
    "baseline": "MET split",
    "sft": "Conditional flow",
    "dgpo": "Flow + DGPO",
    "oracle": "Detector oracle",
}


def _shade_color(
    base_color: str, value: float, minimum: float, maximum: float
) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(base_color))
    if np.isclose(maximum, minimum):
        return tuple(base)
    position = (value - minimum) / (maximum - minimum)
    if position <= 0.5:
        white_fraction = 0.58 * (1.0 - 2.0 * position)
        shaded = (1.0 - white_fraction) * base + white_fraction
    else:
        black_fraction = 0.28 * (2.0 * position - 1.0)
        shaded = (1.0 - black_fraction) * base
    return tuple(shaded)


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _clean_axes(axes: list[Any] | np.ndarray) -> None:
    for axis in np.asarray(axes, dtype=object).ravel():
        if not axis.get_visible():
            continue
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0)


def _panel_labels(axes: list[Any] | np.ndarray) -> None:
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes, dtype=object).ravel()):
        if axis.get_visible():
            axis.text(
                -0.14,
                1.05,
                label,
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
            )


def save_publication_figure(
    figure: Figure,
    stem: Path,
    figure_config: dict[str, Any],
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension in figure_config.get("formats", ["svg", "pdf", "tiff", "png"]):
        path = stem.with_suffix(f".{extension}")
        options: dict[str, Any] = {"bbox_inches": "tight"}
        if extension in {"tif", "tiff"}:
            options.update(
                dpi=int(figure_config.get("raster_dpi", 600)),
                pil_kwargs={"compression": "tiff_lzw"},
            )
        elif extension == "png":
            options["dpi"] = int(figure_config.get("preview_dpi", 300))
        figure.savefig(path, **options)
        saved.append(path)
    plt.close(figure)
    return saved


def plot_dataset_sanity(
    datasets: dict[float, dict[str, np.ndarray]],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, float]], list[Path]]:
    apply_nature_style()
    figure_config = config["figures"]
    width = float(figure_config.get("width_inches", 7.2))
    figure, axes = plt.subplots(2, 2, figsize=(width, 0.72 * width))
    axes = np.asarray(axes)
    masses = sorted(datasets)
    palette = [COLORS["blue"], COLORS["teal"], COLORS["red"]]
    colors = {
        mass: palette[index % len(palette)] for index, mass in enumerate(masses)
    }
    bins = int(config["evaluation"]["histogram_bins"])
    mass_range = tuple(float(value) for value in config["evaluation"]["mass_range_gev"])
    neutrino_mass = float(config["physics"]["neutrino_mass_gev"])
    rows: list[dict[str, float]] = []

    sampling_is_momentum = "parent_momentum" in config["physics"]
    sampling_key = (
        "parent_momentum_gev" if sampling_is_momentum else "parent_energy_gev"
    )
    sampling_label = (
        "Parent momentum magnitude (GeV)"
        if sampling_is_momentum
        else "Parent energy (GeV)"
    )
    sampling_config_key = "parent_momentum" if sampling_is_momentum else "parent_energy"
    sampling_values = np.concatenate(
        [dataset[sampling_key] for dataset in datasets.values()]
    )
    sampling_high = float(np.quantile(sampling_values, 0.995))
    sampling_low = 0.0 if sampling_is_momentum else min(masses)
    for mass in masses:
        dataset = datasets[mass]
        axes[0, 0].hist(
            dataset[sampling_key],
            bins=70,
            range=(sampling_low, sampling_high),
            density=True,
            histtype="step",
            color=colors[mass],
            label=f"{mass:g} GeV",
        )
    center = float(config["physics"][sampling_config_key]["center_gev"])
    axes[0, 0].axvline(center, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    axes[0, 0].set(xlabel=sampling_label, ylabel="Probability density")
    mass_handles, mass_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        mass_handles,
        mass_labels,
        title="Parent mass",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(masses),
    )

    closure_by_mass: dict[float, np.ndarray] = {}
    for mass in masses:
        dataset = datasets[mass]
        closure = reconstruct_parent_masses(
            dataset["leptons_truth"],
            dataset["neutrinos_truth"][..., 1:],
            neutrino_mass,
        ).ravel() - mass
        closure_by_mass[mass] = closure
    closure_limit = max(
        0.01,
        float(np.quantile(np.abs(np.concatenate(list(closure_by_mass.values()))), 0.999)),
    )
    for mass in masses:
        axes[0, 1].hist(
            closure_by_mass[mass],
            bins=60,
            range=(-1.1 * closure_limit, 1.1 * closure_limit),
            density=True,
            histtype="step",
            color=colors[mass],
            label=f"{mass:g} GeV",
        )
    axes[0, 1].axvline(0.0, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    axes[0, 1].set(xlabel="Truth mass closure (GeV)", ylabel="Probability density")

    oracle_points = []
    for mass in masses:
        dataset = datasets[mass]
        oracle = reconstruct_parent_masses(
            dataset["leptons_reco"],
            dataset["neutrinos_truth"][..., 1:],
            neutrino_mass,
        ).ravel()
        peak, resolution = fit_mass_peak(oracle, bins, mass_range)
        oracle_points.append((mass, peak, resolution))
        axes[1, 0].hist(
            oracle,
            bins=bins,
            range=mass_range,
            density=True,
            histtype="step",
            color=colors[mass],
            label=f"{mass:g} GeV",
        )
        rows.append(
            {
                "truth_mass_gev": mass,
                "event_count": float(len(dataset["observables"])),
                "parent_energy_median_gev": float(np.median(dataset["parent_energy_gev"])),
                "parent_momentum_median_gev": float(
                    np.median(dataset["parent_momentum_gev"])
                ),
                "truth_closure_mean_gev": float(np.mean(closure_by_mass[mass])),
                "truth_closure_rms_gev": float(np.std(closure_by_mass[mass])),
                "oracle_peak_gev": peak,
                "oracle_resolution_gev": resolution,
            }
        )
    axes[1, 0].set(xlabel="Oracle reconstructed mass (GeV)", ylabel="Probability density")

    truth = np.asarray([point[0] for point in oracle_points])
    peak = np.asarray([point[1] for point in oracle_points])
    resolution = np.asarray([point[2] for point in oracle_points])
    axes[1, 1].errorbar(
        truth,
        peak,
        yerr=resolution,
        color=COLORS["blue"],
        marker="o",
        markersize=3.5,
        capsize=2,
        label="Peak $\\pm$ Gaussian $\\sigma$",
    )
    padding = 0.05 * (max(masses) - min(masses))
    limits = (min(masses) - padding, max(masses) + padding)
    axes[1, 1].plot(limits, limits, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        xlim=limits,
        ylim=limits,
        xlabel="Truth parent mass (GeV)",
        ylabel="Oracle peak (GeV)",
    )
    axes[1, 1].legend(loc="upper left")
    axes[0, 0].text(
        0.98,
        0.96,
        f"Double exponential; centre = {center:g} GeV",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    axes[1, 0].text(
        0.02,
        0.96,
        f"$n={len(next(iter(datasets.values()))['observables']):,}$ events per mass",
        transform=axes[1, 0].transAxes,
        va="top",
        fontsize=6.5,
    )
    _clean_axes(axes)
    _panel_labels(axes)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.90, wspace=0.32, hspace=0.38)
    paths = save_publication_figure(
        figure, output_dir / "dataset_truth_mass_sanity", figure_config
    )
    _write_rows(output_dir / "dataset_sanity.csv", rows)
    return rows, paths


def plot_training_history(
    observed_history: dict[str, list[float]],
    full_history: dict[str, list[float]],
    flow_history: list[float],
    dgpo_history: dict[str, list[float]],
    figure_config: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    apply_nature_style()
    width = float(figure_config.get("width_inches", 7.2))
    figure, axes = plt.subplots(2, 2, figsize=(width, 0.63 * width))
    axes[0, 0].plot(observed_history["loss"], color=COLORS["blue"], label="Visible")
    axes[0, 0].plot(full_history["loss"], color=COLORS["red"], label="Visible + invisible")
    axes[0, 0].set(ylabel="Cross-entropy", xlabel="Epoch")
    method_handles, method_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        method_handles,
        method_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    axes[0, 1].plot(observed_history["validation_accuracy"], color=COLORS["blue"])
    axes[0, 1].plot(full_history["validation_accuracy"], color=COLORS["red"])
    axes[0, 1].set(ylabel="Validation accuracy", xlabel="Epoch", ylim=(0.0, 1.02))
    axes[1, 0].plot(flow_history, color=COLORS["light_blue"])
    axes[1, 0].set(ylabel="Flow-matching loss", xlabel="Epoch")
    rewards = np.asarray(dgpo_history["reward_mean"], dtype=np.float64)
    scenario_masses = np.asarray(
        dgpo_history.get("mass_gev", np.zeros(len(rewards))), dtype=np.float64
    )
    unique_masses = list(dict.fromkeys(scenario_masses.tolist()))
    for scenario_index, mass in enumerate(unique_masses):
        indices = np.flatnonzero(scenario_masses == mass)
        axes[1, 1].plot(
            indices,
            rewards[indices],
            color=COLORS["red"],
            alpha=0.75,
            linewidth=0.9,
        )
        if scenario_index > 0:
            axes[1, 1].axvline(
                indices[0] - 0.5,
                color=COLORS["light"],
                linewidth=0.7,
            )
        label_stride = max(1, int(np.ceil(len(unique_masses) / 9)))
        if scenario_index % label_stride == 0:
            axes[1, 1].text(
                float(indices.mean()),
                0.98,
                f"{mass:g}",
                transform=axes[1, 1].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=5.5,
            )
    axes[1, 1].set(
        ylabel="DGPO profile reward",
        xlabel="Independent DGPO iteration",
        title="Pseudo-data mass (GeV)",
    )
    _clean_axes(axes)
    _panel_labels(axes)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.90, wspace=0.32, hspace=0.38)
    return save_publication_figure(figure, output_dir / "training_history", figure_config)


def _quadratic_curve(template_masses: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dense_mass = np.linspace(template_masses.min(), template_masses.max(), 300)
    coefficients = np.polyfit(template_masses, values, 2)
    dense_values = np.polyval(coefficients, dense_mass)
    dense_values -= dense_values.min()
    return dense_mass, dense_values


def _profile_vertex(coordinates: np.ndarray, scores: np.ndarray) -> float:
    quadratic, linear, _ = np.polyfit(coordinates, scores, 2)
    if quadratic <= 0.0:
        return float(coordinates[np.argmin(scores)])
    return float(
        np.clip(
            -linear / (2.0 * quadratic), coordinates.min(), coordinates.max()
        )
    )


def plot_sbi_likelihood_diagnostics(
    profiles: dict[float, dict[str, np.ndarray]],
    profile_rows: list[dict[str, float]],
    histories: dict[str, Any],
    template_masses: list[float],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, float | str]], list[Path]]:
    apply_nature_style()
    figure_config = config["figures"]
    selected = [float(value) for value in figure_config["likelihood_masses_gev"]]
    missing = [mass for mass in selected if mass not in profiles]
    if missing:
        raise ValueError(f"Likelihood diagnostic masses are missing from real data: {missing}")
    width = float(figure_config.get("width_inches", 7.2))
    figure = plt.figure(figsize=(width, 0.75 * width))
    grid = figure.add_gridspec(2, len(selected), height_ratios=[1.0, 1.25], hspace=0.48, wspace=0.45)
    accuracy_axis = figure.add_subplot(grid[0, :2])
    estimate_axis = figure.add_subplot(grid[0, 2:])
    likelihood_axes = [figure.add_subplot(grid[1, index]) for index in range(len(selected))]
    axes = [accuracy_axis, estimate_axis, *likelihood_axes]

    observed_history = histories["observed_profiler"]
    full_history = histories["full_profiler"]
    accuracy_axis.plot(
        np.arange(1, len(observed_history["validation_accuracy"]) + 1),
        observed_history["validation_accuracy"],
        color=COLORS["blue"],
        label="Visible",
    )
    accuracy_axis.plot(
        np.arange(1, len(full_history["validation_accuracy"]) + 1),
        full_history["validation_accuracy"],
        color=COLORS["red"],
        label="Visible + truth invisible",
    )
    accuracy_axis.axhline(
        1.0 / len(template_masses),
        color=COLORS["dark"],
        linestyle="--",
        linewidth=0.8,
        label="Random classifier",
    )
    accuracy_axis.set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0.0, 1.02))

    truth = np.asarray([row["truth_mass_gev"] for row in profile_rows])
    visible = np.asarray([row["observed_profile_mass_gev"] for row in profile_rows])
    full_truth = np.asarray([row["truth_full_profile_mass_gev"] for row in profile_rows])
    estimate_axis.plot(truth, visible, marker="o", markersize=2.5, color=COLORS["blue"], label="Visible")
    estimate_axis.plot(truth, full_truth, marker="o", markersize=2.5, color=COLORS["red"], label="Visible + truth invisible")
    padding = 0.05 * (max(template_masses) - min(template_masses))
    limits = (min(template_masses) - padding, max(template_masses) + padding)
    estimate_axis.plot(limits, limits, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    estimate_axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="Truth parent mass (GeV)",
        ylabel="Calibrated profile estimate (GeV)",
    )

    template_array = np.asarray(template_masses, dtype=np.float64)
    csv_rows: list[dict[str, float | str]] = []
    for index, (mass, axis) in enumerate(zip(selected, likelihood_axes)):
        for method, color, label in [
            ("visible", COLORS["blue"], "Visible"),
            ("visible_truth", COLORS["red"], "Visible + truth invisible"),
        ]:
            values = profiles[mass][method]
            dense_mass, dense_values = _quadratic_curve(template_array, values)
            curvature = float(np.polyfit(template_array, values, 2)[0])
            axis.plot(
                dense_mass,
                dense_values,
                color=color,
                linestyle="-" if curvature > 0.0 else ":",
                label=label if index == 0 else None,
            )
            axis.plot(template_array, values, linestyle="none", marker="o", markersize=2.5, color=color)
            for template_mass, value in zip(template_array, values):
                csv_rows.append(
                    {
                        "truth_mass_gev": mass,
                        "profile": method,
                        "template_mass_gev": float(template_mass),
                        "negative_two_delta_mean_log_score": float(value),
                    }
                )
        axis.axvline(mass, color=COLORS["dark"], linestyle="--", linewidth=0.8)
        axis.set_title(f"Truth {mass:g} GeV")
        axis.set_xlabel("Mass (GeV)")
        if index == 0:
            axis.set_ylabel(r"$-2\Delta\langle\log p\rangle$")
        else:
            axis.tick_params(labelleft=False)
    method_handles, method_labels = accuracy_axis.get_legend_handles_labels()
    figure.legend(
        method_handles,
        method_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )
    figure.text(
        0.99,
        0.02,
        "Markers: templates; solid/dotted curves: convex/non-convex quadratic interpolation",
        ha="right",
        fontsize=6.5,
    )
    estimate_axis.text(
        0.98,
        0.04,
        f"$n={int(config['data']['real_events_per_mass']):,}$ events per mass",
        transform=estimate_axis.transAxes,
        ha="right",
        fontsize=6.5,
    )
    _clean_axes(axes)
    _panel_labels([accuracy_axis, estimate_axis, likelihood_axes[0]])
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.90)
    paths = save_publication_figure(
        figure, output_dir / "sbi_likelihood_diagnostics", figure_config
    )
    _write_rows(output_dir / "sbi_likelihood_profiles.csv", csv_rows)
    return csv_rows, paths


def plot_mass_spectra(
    spectra: dict[float, dict[str, np.ndarray]],
    config: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    apply_nature_style()
    figure_config = config["figures"]
    selected = [float(value) for value in figure_config["likelihood_masses_gev"]]
    width = float(figure_config.get("width_inches", 7.2))
    columns = 3
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width, 0.31 * width * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    bins = int(config["evaluation"]["histogram_bins"])
    mass_range = tuple(float(value) for value in config["evaluation"]["mass_range_gev"])
    for index, (axis, mass) in enumerate(zip(axes, selected)):
        for method in ["baseline", "sft", "dgpo", "oracle"]:
            axis.hist(
                spectra[mass][method],
                bins=bins,
                range=mass_range,
                density=True,
                histtype="step",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method] if index == 0 else None,
            )
        axis.axvline(mass, color=COLORS["dark"], linestyle="--", linewidth=0.9)
        axis.set_title(f"Truth {mass:g} GeV")
        axis.set_xlabel("Reconstructed mass (GeV)")
        axis.set_ylabel("Probability density")
    for axis in axes[len(selected):]:
        axis.set_visible(False)
    method_handles, method_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        method_handles,
        method_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
    )
    _clean_axes(axes)
    _panel_labels(axes)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.88, wspace=0.35, hspace=0.43)
    return save_publication_figure(
        figure, output_dir / "reconstructed_mass_spectra", figure_config
    )


def plot_jes_likelihood_diagnostics(
    profiles: dict[float, dict[str, np.ndarray]],
    estimate_rows: list[dict[str, float]],
    histories: dict[str, Any],
    template_shifts: list[float],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, float | str]], list[Path]]:
    apply_nature_style()
    figure_config = config["figures"]
    truth_shifts = sorted(profiles)
    width = float(figure_config.get("width_inches", 7.2))
    figure = plt.figure(figsize=(width, 0.72 * width))
    grid = figure.add_gridspec(
        2,
        len(truth_shifts),
        height_ratios=[1.0, 1.2],
        hspace=0.50,
        wspace=0.45,
    )
    accuracy_columns = max(1, len(truth_shifts) // 2)
    accuracy_axis = figure.add_subplot(grid[0, :accuracy_columns])
    estimate_axis = figure.add_subplot(grid[0, accuracy_columns:])
    profile_axes = [
        figure.add_subplot(grid[1, index]) for index in range(len(truth_shifts))
    ]
    axes = [accuracy_axis, estimate_axis, *profile_axes]

    visible_history = histories["hypothesis_observed_profiler"]
    full_history = histories["hypothesis_full_profiler"]
    accuracy_axis.plot(
        np.arange(1, len(visible_history["validation_accuracy"]) + 1),
        visible_history["validation_accuracy"],
        color=COLORS["blue"],
        label="Visible",
    )
    accuracy_axis.plot(
        np.arange(1, len(full_history["validation_accuracy"]) + 1),
        full_history["validation_accuracy"],
        color=COLORS["red"],
        label="Visible + truth invisible",
    )
    accuracy_axis.set(xlabel="Epoch", ylabel="Grid validation accuracy", ylim=(0.0, 1.02))

    truth = 100.0 * np.asarray([row["truth_jes_shift"] for row in estimate_rows])
    visible = 100.0 * np.asarray([row["visible_profile_jes_shift"] for row in estimate_rows])
    full = 100.0 * np.asarray([row["truth_full_profile_jes_shift"] for row in estimate_rows])
    estimate_axis.plot(truth, visible, marker="o", markersize=2.7, color=COLORS["blue"], label="Visible")
    estimate_axis.plot(truth, full, marker="o", markersize=2.7, color=COLORS["red"], label="Visible + truth invisible")
    limits = (100.0 * min(template_shifts), 100.0 * max(template_shifts))
    estimate_axis.plot(limits, limits, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    estimate_axis.set(xlim=limits, ylim=limits, xlabel="Truth JES shift (%)", ylabel="Profile estimate (%)")

    template_percent = 100.0 * np.asarray(template_shifts)
    csv_rows: list[dict[str, float | str]] = []
    for index, (truth_shift, axis) in enumerate(zip(truth_shifts, profile_axes)):
        for method, color, label in [
            ("visible", COLORS["blue"], "Visible"),
            ("visible_truth", COLORS["red"], "Visible + truth invisible"),
        ]:
            values = profiles[truth_shift][method]
            dense_shift, dense_values = _quadratic_curve(template_percent, values)
            axis.plot(dense_shift, dense_values, color=color, label=label if index == 0 else None)
            axis.plot(template_percent, values, linestyle="none", marker="o", markersize=2.5, color=color)
            for template_shift, value in zip(template_shifts, values):
                csv_rows.append(
                    {
                        "truth_jes_shift": truth_shift,
                        "profile": method,
                        "template_jes_shift": template_shift,
                        "negative_two_delta_mean_log_score": float(value),
                    }
                )
        axis.axvline(100.0 * truth_shift, color=COLORS["dark"], linestyle="--", linewidth=0.8)
        axis.set_title(f"Truth {100.0 * truth_shift:+.0f}%")
        axis.set_xlabel("JES shift (%)")
        if index == 0:
            axis.set_ylabel(r"$-2\Delta\langle\log p\rangle$")
        else:
            axis.tick_params(labelleft=False)
    method_handles, method_labels = accuracy_axis.get_legend_handles_labels()
    figure.legend(
        method_handles,
        method_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    figure.text(
        0.99,
        0.02,
        "Markers: evaluated templates; curves: quadratic interpolation",
        ha="right",
        fontsize=6.5,
    )
    _clean_axes(axes)
    _panel_labels([accuracy_axis, estimate_axis, profile_axes[0]])
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.90)
    paths = save_publication_figure(
        figure, output_dir / "jes_likelihood_diagnostics", figure_config
    )
    _write_rows(output_dir / "jes_likelihood_profiles.csv", csv_rows)
    return csv_rows, paths


def plot_mass_jes_likelihood_grid(
    surfaces: dict[tuple[float, float], dict[str, np.ndarray]],
    template_masses: list[float],
    template_shifts: list[float],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, float | str]], list[Path]]:
    apply_nature_style()
    figure_config = config["figures"]
    points = list(surfaces)
    width = float(figure_config.get("width_inches", 7.2))
    figure, axes = plt.subplots(2, len(points), figsize=(width, 0.68 * width), squeeze=False)
    maximum = max(float(np.max(values[method])) for values in surfaces.values() for method in ["visible", "visible_truth"])
    mass_edges = _bin_edges(np.asarray(template_masses, dtype=np.float64))
    shift_percent = 100.0 * np.asarray(template_shifts, dtype=np.float64)
    shift_edges = _bin_edges(shift_percent)
    csv_rows: list[dict[str, float | str]] = []
    estimate_rows: list[dict[str, float | str]] = []
    image = None
    for column, ((truth_mass, truth_shift), values) in enumerate(surfaces.items()):
        for row, (method, row_label) in enumerate(
            [("visible", "Visible"), ("visible_truth", "Visible + truth invisible")]
        ):
            axis = axes[row, column]
            surface = values[method]
            image = axis.pcolormesh(
                mass_edges,
                shift_edges,
                surface,
                cmap="Blues",
                vmin=0.0,
                vmax=maximum,
                shading="flat",
            )
            for shift_index, shift in enumerate(shift_percent):
                for mass_index, mass in enumerate(template_masses):
                    value = float(surface[shift_index, mass_index])
                    axis.text(
                        mass,
                        shift,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if value > 0.55 * maximum else COLORS["dark"],
                    )
                    csv_rows.append(
                        {
                            "truth_mass_gev": truth_mass,
                            "truth_jes_shift": truth_shift,
                            "profile": method,
                            "template_mass_gev": float(mass),
                            "template_jes_shift": float(template_shifts[shift_index]),
                            "negative_two_delta_mean_log_score": value,
                        }
                    )
            mass_array = np.asarray(template_masses, dtype=np.float64)
            shift_array = np.asarray(template_shifts, dtype=np.float64)
            estimated_mass = _profile_vertex(mass_array, surface.min(axis=0))
            estimated_shift = _profile_vertex(shift_array, surface.min(axis=1))
            axis.plot(
                truth_mass,
                100.0 * truth_shift,
                marker="*",
                markersize=7,
                markerfacecolor=COLORS["red"],
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            axis.plot(
                estimated_mass,
                100.0 * estimated_shift,
                marker="D",
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=COLORS["dark"],
                markeredgewidth=0.9,
            )
            estimate_rows.append(
                {
                    "truth_mass_gev": truth_mass,
                    "truth_jes_shift": truth_shift,
                    "profile": method,
                    "estimated_mass_gev": estimated_mass,
                    "estimated_jes_shift": estimated_shift,
                }
            )
            axis.set_xticks(template_masses)
            axis.set_yticks(shift_percent)
            axis.set_xlabel("Mass hypothesis (GeV)")
            if column == 0:
                axis.set_ylabel(f"{row_label}\nJES hypothesis (%)")
            else:
                axis.tick_params(labelleft=False)
            estimate_label = (
                f"Estimate ({estimated_mass:.0f} GeV, "
                f"{100.0 * estimated_shift:+.1f}%)"
            )
            if row == 0:
                axis.set_title(
                    f"Truth ({truth_mass:g} GeV, {100.0 * truth_shift:+.0f}%)\n"
                    f"{estimate_label}"
                )
            else:
                axis.set_title(estimate_label)
    if image is not None:
        colorbar_axis = figure.add_axes([0.89, 0.17, 0.018, 0.64])
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(r"$-2\Delta\langle\log p\rangle$")
        colorbar.outline.set_linewidth(0.8)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                marker="*",
                markersize=7,
                markerfacecolor=COLORS["red"],
                markeredgecolor="white",
                markeredgewidth=0.5,
                label="Pseudo-data truth",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="D",
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=COLORS["dark"],
                markeredgewidth=0.9,
                label="Profile estimate",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    _clean_axes(axes)
    _panel_labels(axes)
    figure.subplots_adjust(left=0.10, right=0.86, bottom=0.10, top=0.87, wspace=0.26, hspace=0.52)
    paths = save_publication_figure(
        figure, output_dir / "mass_jes_likelihood_grid", figure_config
    )
    _write_rows(output_dir / "mass_jes_likelihood_grid.csv", csv_rows)
    _write_rows(output_dir / "mass_jes_profile_estimates.csv", estimate_rows)
    return csv_rows, paths


def plot_mass_jes_profile_scatter(
    rows: list[dict[str, float | str]],
    config: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    apply_nature_style()
    figure_config = config["figures"]
    width = float(figure_config.get("width_inches", 7.2))
    figure, (profile_axis, bias_axis) = plt.subplots(
        1,
        2,
        figsize=(width, 0.56 * width),
        gridspec_kw={"width_ratios": [2.25, 1.0]},
    )
    methods = ["visible", "sft", "dgpo"]
    labels = {
        "visible": "Visible-only profile",
        "sft": "Full profile: conditional flow",
        "dgpo": "Full profile: flow + DGPO",
    }
    markers = {"visible": "o", "sft": "^", "dgpo": "s"}
    method_colors = {
        "visible": COLORS["blue"],
        "sft": COLORS["orange"],
        "dgpo": COLORS["red"],
    }
    shifts = np.asarray([float(row["truth_jes_shift"]) for row in rows])
    shift_minimum = float(shifts.min())
    shift_maximum = float(shifts.max())
    truth_masses = np.asarray([float(row["truth_mass_gev"]) for row in rows])
    padding = 0.05 * (truth_masses.max() - truth_masses.min())
    limits = (truth_masses.min() - padding, truth_masses.max() + padding)
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        profile_axis.scatter(
            [float(row["truth_mass_gev"]) for row in selected],
            [float(row["profiled_mass_gev"]) for row in selected],
            c=[
                _shade_color(
                    method_colors[method],
                    float(row["truth_jes_shift"]),
                    shift_minimum,
                    shift_maximum,
                )
                for row in selected
            ],
            marker=markers[method],
            s=18,
            linewidths=0.45,
            edgecolors="white",
            alpha=0.92,
        )
    profile_axis.plot(limits, limits, color=COLORS["dark"], linestyle="--", linewidth=1.0, label="Truth")
    profile_axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="Truth parent mass (GeV)",
        ylabel="Profiled parent mass (GeV)",
    )
    unique_shifts = sorted({float(value) for value in shifts})
    shade_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.3,
            markerfacecolor=_shade_color(
                COLORS["neutral"], shift, shift_minimum, shift_maximum
            ),
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=f"{100.0 * shift:+.0f}%",
        )
        for shift in unique_shifts
    ]
    profile_axis.legend(
        handles=shade_handles,
        title="Truth JES shade",
        loc="upper left",
        ncol=len(shade_handles),
        handletextpad=0.2,
        columnspacing=0.6,
    )

    mass_minimum = float(truth_masses.min())
    mass_maximum = float(truth_masses.max())

    def mass_marker_size(mass: float) -> float:
        if np.isclose(mass_minimum, mass_maximum):
            return 20.0
        return 13.0 + 17.0 * (mass - mass_minimum) / (
            mass_maximum - mass_minimum
        )

    for method in methods:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["truth_mass_gev"]),
            reverse=True,
        )
        bias_axis.scatter(
            [100.0 * float(row["truth_jes_shift"]) for row in selected],
            [100.0 * float(row["profiled_jes_shift"]) for row in selected],
            c=[
                _shade_color(
                    method_colors[method],
                    float(row["truth_jes_shift"]),
                    shift_minimum,
                    shift_maximum,
                )
                for row in selected
            ],
            marker=markers[method],
            s=[mass_marker_size(float(row["truth_mass_gev"])) for row in selected],
            linewidths=0.45,
            edgecolors="white",
            alpha=0.92,
        )
    jes_limits = (100.0 * shifts.min(), 100.0 * shifts.max())
    bias_axis.plot(
        jes_limits,
        jes_limits,
        color=COLORS["dark"],
        linestyle="--",
        linewidth=1.0,
    )
    bias_axis.set(
        xlim=jes_limits,
        ylim=jes_limits,
        xlabel="Truth JES shift (%)",
        ylabel="Profiled JES shift (%)",
    )
    size_masses = [mass_minimum, 0.5 * (mass_minimum + mass_maximum), mass_maximum]
    size_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=np.sqrt(mass_marker_size(mass)),
            markerfacecolor=COLORS["light"],
            markeredgecolor=COLORS["dark"],
            markeredgewidth=0.5,
            label=f"{mass:.0f}",
        )
        for mass in size_masses
    ]
    bias_axis.legend(
        handles=size_handles,
        title="Truth mass (GeV)",
        loc="lower right",
    )
    method_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=markers[method],
            markersize=4.5,
            markerfacecolor=method_colors[method],
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=labels[method],
        )
        for method in methods
    ]
    method_handles.append(
        Line2D(
            [],
            [],
            color=COLORS["dark"],
            linestyle="--",
            linewidth=1.0,
            label="Truth",
        )
    )
    figure.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
    )
    profile_axis.text(
        0.99,
        0.03,
        f"$n={int(config['data']['real_events_per_mass']):,}$ events per grid point",
        transform=profile_axis.transAxes,
        ha="right",
        fontsize=6.5,
    )
    _clean_axes([profile_axis, bias_axis])
    _panel_labels([profile_axis, bias_axis])
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.88, wspace=0.34)
    return save_publication_figure(
        figure, output_dir / "mass_jes_profile_scatter", figure_config
    )


def plot_momentum_reconstruction_diagnostics(
    targets: dict[str, np.ndarray],
    truth_mass_gev: float,
    config: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Compare predicted and truth neutrino momentum for the mass-only trial."""
    apply_nature_style()
    figure_config = config["figures"]
    methods = ["baseline", "sft", "dgpo"]
    components = [r"$p_x$", r"$p_y$", r"$p_z$"]
    component_colors = [COLORS["blue"], COLORS["orange"], COLORS["red"]]
    truth = np.asarray(targets["truth"]).reshape(-1, 3)
    predictions = {
        method: np.asarray(targets[method]).reshape(-1, 3) for method in methods
    }
    point_count = min(
        len(truth), int(figure_config.get("momentum_scatter_points", 2000))
    )
    selected_indices = np.unique(
        np.linspace(0, len(truth) - 1, point_count, dtype=int)
    )

    plotted_values = [truth[selected_indices]] + [
        predictions[method][selected_indices] for method in methods
    ]
    lower, upper = np.quantile(np.concatenate(plotted_values), [0.005, 0.995])
    limit = max(abs(float(lower)), abs(float(upper)), 1.0)
    limits = (-limit, limit)

    width = float(figure_config.get("width_inches", 7.2))
    figure, axes = plt.subplots(2, 2, figsize=(width, 0.67 * width))
    scatter_axes = axes.ravel()[:3]
    metric_axis = axes.ravel()[3]
    sample_rows: list[dict[str, float | int | str]] = []
    metric_rows: list[dict[str, float | str]] = []

    for method, axis in zip(methods, scatter_axes):
        prediction = predictions[method]
        for component_index, (component, color) in enumerate(
            zip(components, component_colors)
        ):
            axis.scatter(
                truth[selected_indices, component_index],
                prediction[selected_indices, component_index],
                s=4.0,
                alpha=0.28,
                linewidths=0.0,
                color=color,
                label=component,
                rasterized=True,
            )
            residual = prediction[:, component_index] - truth[:, component_index]
            metric_rows.append(
                {
                    "method": method,
                    "component": component.replace("$", ""),
                    "bias_gev": float(np.mean(residual)),
                    "resolution_gev": float(np.std(residual)),
                    "rmse_gev": float(np.sqrt(np.mean(residual**2))),
                }
            )
            for entry_index in selected_indices:
                sample_rows.append(
                    {
                        "method": method,
                        "entry_index": int(entry_index),
                        "component": component.replace("$", ""),
                        "truth_momentum_gev": float(
                            truth[entry_index, component_index]
                        ),
                        "reconstructed_momentum_gev": float(
                            prediction[entry_index, component_index]
                        ),
                    }
                )
        axis.plot(
            limits,
            limits,
            color=COLORS["dark"],
            linestyle="--",
            linewidth=0.9,
        )
        axis.set(
            xlim=limits,
            ylim=limits,
            xlabel="Truth neutrino momentum (GeV)",
            ylabel="Reconstructed momentum (GeV)",
            title=METHOD_LABELS[method],
        )
    scatter_axes[0].legend(
        title="Component",
        loc="upper left",
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.7,
    )

    positions = np.arange(len(components), dtype=np.float64)
    offsets = np.linspace(-0.18, 0.18, len(methods))
    for offset, method in zip(offsets, methods):
        selected = [row for row in metric_rows if row["method"] == method]
        metric_axis.errorbar(
            positions + offset,
            [float(row["bias_gev"]) for row in selected],
            yerr=[float(row["resolution_gev"]) for row in selected],
            marker="o",
            markersize=3.0,
            capsize=2.0,
            linewidth=1.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    metric_axis.axhline(
        0.0, color=COLORS["dark"], linestyle="--", linewidth=0.9
    )
    metric_axis.set(
        xticks=positions,
        xticklabels=components,
        xlabel="Neutrino momentum component",
        ylabel=r"Residual bias $\pm\sigma$ (GeV)",
        title="Momentum residuals",
    )
    metric_axis.legend(loc="upper left")
    figure.text(
        0.99,
        0.015,
        (
            f"Truth parent mass = {truth_mass_gev:g} GeV; "
            f"$n={len(truth) // 2:,}$ events"
        ),
        ha="right",
        fontsize=6.5,
    )
    _clean_axes(axes)
    _panel_labels(axes)
    figure.subplots_adjust(
        left=0.09, right=0.99, bottom=0.12, top=0.93, wspace=0.31, hspace=0.39
    )
    _write_rows(output_dir / "momentum_reconstruction_samples.csv", sample_rows)
    _write_rows(output_dir / "momentum_reconstruction_metrics.csv", metric_rows)
    return save_publication_figure(
        figure, output_dir / "momentum_reconstruction_diagnostics", figure_config
    )


def plot_mass_jes_reconstruction_heatmaps(
    rows: list[dict[str, float | str]],
    config: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Plot reconstructed mass bias and resolution over the truth parameter grid."""
    apply_nature_style()
    figure_config = config["figures"]
    methods = list(
        figure_config.get(
            "reconstruction_heatmap_methods",
            ["baseline", "sft", "dgpo", "oracle"],
        )
    )
    unknown_methods = [method for method in methods if method not in METHOD_LABELS]
    if unknown_methods:
        raise ValueError(f"Unknown reconstruction heatmap methods: {unknown_methods}")

    masses = np.asarray(
        sorted({float(row["truth_mass_gev"]) for row in rows}), dtype=np.float64
    )
    shifts = np.asarray(
        sorted({float(row["truth_jes_shift"]) for row in rows}), dtype=np.float64
    )
    mass_edges = _bin_edges(masses)
    shift_percent = 100.0 * shifts
    shift_edges = _bin_edges(shift_percent)

    matrices: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        lookup = {
            (float(row["truth_mass_gev"]), float(row["truth_jes_shift"])): row
            for row in method_rows
        }
        missing = [
            (mass, shift)
            for shift in shifts
            for mass in masses
            if (float(mass), float(shift)) not in lookup
        ]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} reconstruction grid points for {method}"
            )
        matrices[method] = {
            metric: np.asarray(
                [
                    [
                        float(lookup[(float(mass), float(shift))][metric])
                        for mass in masses
                    ]
                    for shift in shifts
                ]
            )
            for metric in ["bias_gev", "resolution_gev"]
        }

    finite_bias = np.concatenate(
        [matrices[method]["bias_gev"].ravel() for method in methods]
    )
    bias_limit = max(float(np.nanmax(np.abs(finite_bias))), 1.0)
    finite_resolution = np.concatenate(
        [matrices[method]["resolution_gev"].ravel() for method in methods]
    )
    resolution_limit = max(float(np.nanmax(finite_resolution)), 1.0)

    width = float(figure_config.get("width_inches", 7.2))
    figure = plt.figure(figsize=(width, 0.86 * width))
    outer_grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.035],
        wspace=0.30,
        hspace=0.20,
    )
    heatmap_grid = outer_grid[0, :].subgridspec(
        len(methods), 2, wspace=0.30, hspace=0.20
    )
    axes = np.empty((len(methods), 2), dtype=object)
    bias_image = None
    resolution_image = None
    for row_index, method in enumerate(methods):
        bias_axis = figure.add_subplot(heatmap_grid[row_index, 0])
        resolution_axis = figure.add_subplot(heatmap_grid[row_index, 1])
        axes[row_index] = [bias_axis, resolution_axis]
        bias_image = bias_axis.pcolormesh(
            mass_edges,
            shift_edges,
            matrices[method]["bias_gev"],
            cmap="RdBu_r",
            norm=matplotlib.colors.TwoSlopeNorm(
                vmin=-bias_limit, vcenter=0.0, vmax=bias_limit
            ),
            shading="flat",
            rasterized=True,
        )
        resolution_image = resolution_axis.pcolormesh(
            mass_edges,
            shift_edges,
            matrices[method]["resolution_gev"],
            cmap="Blues",
            vmin=0.0,
            vmax=resolution_limit,
            shading="flat",
            rasterized=True,
        )
        bias_axis.set_ylabel(f"{METHOD_LABELS[method]}\nTruth JES shift (%)")
        if row_index == 0:
            bias_axis.set_title("Reconstructed mass bias")
            resolution_axis.set_title("Gaussian mass resolution")
        if row_index == len(methods) - 1:
            bias_axis.set_xlabel("Truth parent mass (GeV)")
            resolution_axis.set_xlabel("Truth parent mass (GeV)")
        else:
            bias_axis.tick_params(labelbottom=False)
            resolution_axis.tick_params(labelbottom=False)
        resolution_axis.tick_params(labelleft=False)

    tick_count = min(7, len(masses))
    tick_indices = np.unique(
        np.linspace(0, len(masses) - 1, tick_count, dtype=int)
    )
    for axis in axes.ravel():
        axis.set_xticks(masses[tick_indices])
        axis.set_yticks(shift_percent)

    bias_colorbar_axis = figure.add_subplot(outer_grid[1, 0])
    resolution_colorbar_axis = figure.add_subplot(outer_grid[1, 1])
    if bias_image is not None:
        bias_colorbar = figure.colorbar(
            bias_image, cax=bias_colorbar_axis, orientation="horizontal"
        )
        bias_colorbar.set_label("Peak bias (GeV)")
        bias_colorbar.outline.set_linewidth(0.8)
    if resolution_image is not None:
        resolution_colorbar = figure.colorbar(
            resolution_image,
            cax=resolution_colorbar_axis,
            orientation="horizontal",
        )
        resolution_colorbar.set_label(r"Gaussian $\sigma$ (GeV)")
        resolution_colorbar.outline.set_linewidth(0.8)

    figure.text(
        0.99,
        0.012,
        (
            f"Peak bias = fitted peak - truth mass; "
            f"$n={int(config['data']['real_events_per_mass']):,}$ events per grid point"
        ),
        ha="right",
        fontsize=6.5,
    )
    _clean_axes(axes)
    _panel_labels(axes)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.94)
    return save_publication_figure(
        figure, output_dir / "mass_jes_reconstruction_heatmaps", figure_config
    )


def _bin_edges(centres: np.ndarray) -> np.ndarray:
    if len(centres) < 2:
        return np.asarray([centres[0] - 0.5, centres[0] + 0.5])
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(
        [[centres[0] - (midpoints[0] - centres[0])], midpoints, [centres[-1] + (centres[-1] - midpoints[-1])]]
    )


def plot_final_benchmark(
    metrics: list[dict[str, float | str]],
    profile_rows: list[dict[str, float]],
    config: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    apply_nature_style()
    figure_config = config["figures"]
    width = float(figure_config.get("width_inches", 7.2))
    figure = plt.figure(figsize=(width, 0.78 * width))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.38)
    peak_axis = figure.add_subplot(grid[0, :])
    bias_axis = figure.add_subplot(grid[1, 0])
    resolution_axis = figure.add_subplot(grid[1, 1])
    profile_axis = figure.add_subplot(grid[1, 2])
    axes = [peak_axis, bias_axis, resolution_axis, profile_axis]
    methods = ["baseline", "sft", "dgpo", "oracle"]
    truth_values = sorted({float(row["truth_mass_gev"]) for row in metrics})
    spacing = float(np.min(np.diff(truth_values))) if len(truth_values) > 1 else 1.0
    offsets = np.linspace(-0.18, 0.18, len(methods)) * spacing
    for offset, method in zip(offsets, methods):
        rows = [row for row in metrics if row["method"] == method]
        truth = np.asarray([float(row["truth_mass_gev"]) for row in rows])
        peak = np.asarray([float(row["peak_gev"]) for row in rows])
        resolution = np.asarray([float(row["resolution_gev"]) for row in rows])
        bias = np.asarray([float(row["bias_gev"]) for row in rows])
        peak_axis.errorbar(
            truth + offset,
            peak,
            yerr=resolution,
            marker="o",
            markersize=2.4,
            capsize=1.3,
            linewidth=0.9,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        bias_axis.plot(truth, bias, marker="o", markersize=2.1, color=METHOD_COLORS[method])
        resolution_axis.plot(truth, resolution, marker="o", markersize=2.1, color=METHOD_COLORS[method])
    padding = 0.05 * (max(truth_values) - min(truth_values))
    limits = (min(truth_values) - padding, max(truth_values) + padding)
    peak_axis.plot(limits, limits, color=COLORS["dark"], linestyle="--", linewidth=1.0, label="Unbiased")
    peak_axis.set(xlim=limits, xlabel="Truth parent mass (GeV)", ylabel="Reconstructed peak $\\pm$ Gaussian $\\sigma$ (GeV)")
    peak_axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    bias_axis.axhline(0.0, color=COLORS["dark"], linestyle="--", linewidth=0.9)
    bias_axis.set(xlabel="Truth mass (GeV)", ylabel="Bias (GeV)")
    resolution_axis.set(xlabel="Truth mass (GeV)", ylabel="Resolution $\\sigma$ (GeV)")

    truth = np.asarray([row["truth_mass_gev"] for row in profile_rows])
    profile_axis.plot(truth, [row["observed_profile_mass_gev"] for row in profile_rows], color=COLORS["blue"], marker="o", markersize=2.0, label="Visible")
    profile_axis.plot(truth, [row["sft_full_profile_mass_gev"] for row in profile_rows], color=COLORS["light_blue"], marker="o", markersize=2.0, label="SFT full")
    profile_axis.plot(truth, [row["dgpo_full_profile_mass_gev"] for row in profile_rows], color=COLORS["red"], marker="o", markersize=2.0, label="DGPO full")
    profile_axis.plot(limits, limits, color=COLORS["dark"], linestyle="--", linewidth=0.9)
    profile_axis.set(
        xlabel="Truth mass (GeV)",
        ylabel="Calibrated profile estimate (GeV)",
        xlim=limits,
        ylim=limits,
    )
    profile_axis.legend(loc="upper left")
    peak_axis.text(
        0.99,
        0.04,
        f"$n={int(config['data']['real_events_per_mass']):,}$ events per mass",
        transform=peak_axis.transAxes,
        ha="right",
        fontsize=6.5,
    )
    _clean_axes(axes)
    _panel_labels(axes)
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.88)
    return save_publication_figure(figure, output_dir / "final_benchmark", figure_config)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
