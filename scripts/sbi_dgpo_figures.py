"""Tables and figures from scripts/sbi_dgpo.py runs.

    python scripts/sbi_dgpo_figures.py outputs/sbi_dgpo results/sbi_dgpo

Reads every sbi_dgpo_s*.json in the input directory and writes:
  summary.csv      per method, each metric's |value| averaged over truth masses: mean and sd over seeds
  fig_spectra.png  reconstructed parent-mass spectra at three truth masses and the neutrino pz
                   residual at 500 GeV (lowest seed)
  fig_peak.png     reconstructed parent-mass peak and its bias vs truth, mean +- sd over seeds
  fig_metrics.png  neutrino RMSE, W1 to the true-neutrino mass spectrum, mean-residual drift and the
                   moment-curve full-event estimate on the reconstruction, vs truth
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [  # (key, label, colour, marker); the colours form one validated categorical palette
    ("sft", "Supervised flow, no post-training", "#2a78d6", "o"),
    ("truth_dgpo", "DGPO, truth reward (NeurIPS-style)", "#eb6834", "s"),
    ("sbi_new", "SBI-DGPO, moment-curve estimators", "#1baf7a", "^"),
    ("sbi_toy", "SBI-DGPO, toy estimators", "#eda100", "D"),
    ("sbi_new_shifted", "SBI-DGPO, moment curve + offset-shifted target", "#6250d6", "v"),
    ("cond_sft", "Mass-conditioned flow at the SBI mass estimate (no DGPO)", "#e34948", "P"),
    ("oracle", "True neutrinos", "#0b0b0b", ""),
]
METRICS = ["peak_bias", "resolution", "median_bias", "geo_mean_bias", "w1_mass_vs_true_nu", "nu_rmse",
           "nu_mae", "nu_mean_residual_rms", "sbi_full_new_bias"]
HIST_MASS = np.linspace(0.0, 1000.0, 101)
HIST_RESID = np.linspace(-1500.0, 1500.0, 121)
GRAY = "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY, "legend.frameon": False})


def table(rows, seeds, method, metric):
    """array [seed, truth] of one metric, and the truths."""
    truths = sorted({r["truth"] for r in rows})
    lookup = {(r["seed"], r["truth"]): r[metric] for r in rows if r["method"] == method}
    return np.array([[lookup[(s, t)] for t in truths] for s in seeds]), np.array(truths)


def spread(values):
    return values.std(0, ddof=1) if len(values) > 1 else None


def write_summary(rows, seeds, out):
    summary = []
    for method in [m[0] for m in METHODS] + ["baseline"]:
        entry = {"method": method, "seeds": len(seeds)}
        for metric in METRICS:
            per_seed = np.mean(np.abs(table(rows, seeds, method, metric)[0]), axis=1)
            entry[metric] = float(per_seed.mean())
            entry[f"{metric}_sd"] = float(per_seed.std(ddof=1)) if len(seeds) > 1 else float("nan")
        peaks, truths = table(rows, seeds, method, "peak")
        fits = [np.polyfit(truths, p, 1) for p in peaks]
        entry["signed_peak_bias"] = float((peaks - truths).mean())
        entry["slope"] = float(np.mean([f[0] for f in fits]))
        entry["nonlinearity"] = float(np.mean([np.std(p - np.polyval(f, truths)) for p, f in zip(peaks, fits)]))
        summary.append(entry)
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"{'method':16} peak bias   |bias|  slope  nonlin  width  ν RMSE    W1  drift  ({len(seeds)} seeds)")
    for e in summary:
        print(f"{e['method']:16} {e['signed_peak_bias']:+8.1f}  {e['peak_bias']:5.1f}±{e['peak_bias_sd']:3.1f}  "
              f"{e['slope']:5.2f}  {e['nonlinearity']:5.1f}  {e['resolution']:5.1f}  {e['nu_rmse']:6.1f}  "
              f"{e['w1_mass_vs_true_nu']:5.1f}  {e['nu_mean_residual_rms']:5.1f}")


def fig_peak(rows, seeds, out):
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    offsets = np.linspace(-5, 5, len(METHODS))
    for (key, label, colour, marker), dx in zip(METHODS, offsets):
        peaks, truths = table(rows, seeds, key, "peak")
        fmt, style = (marker + "-") if marker else "--", dict(color=colour, lw=1.4, ms=4, elinewidth=1, capsize=0)
        axes[0].errorbar(truths + dx, peaks.mean(0), yerr=spread(peaks), fmt=fmt, label=label, **style)
        axes[1].errorbar(truths + dx, (peaks - truths).mean(0), yerr=spread(peaks - truths), fmt=fmt, **style)
    axes[0].plot([390, 610], [390, 610], color=GRAY, lw=0.8, ls=":", label="peak = truth")
    axes[0].set_ylabel("Reconstructed parent-mass peak (GeV)")
    axes[1].axhline(0, color=GRAY, lw=0.8, ls=":")
    axes[1].set_ylabel("Peak − truth (GeV)")
    axes[1].set_xlabel("Truth parent mass (GeV)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.text(0.5, 0.005, f"Held-out pseudo-datasets of 100k events; mean and sd over {len(seeds)} full re-runs.",
             ha="center", fontsize=6.5, color=GRAY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "fig_peak.png", dpi=200)
    plt.close(fig)


def fig_metrics(rows, seeds, out):
    panels = [("nu_rmse", "Neutrino momentum RMSE (GeV)"),
              ("w1_mass_vs_true_nu", "W1(reco mass, true-ν mass) (GeV)"),
              ("nu_mean_residual_rms", "Mean-residual drift (GeV)"),
              ("sbi_full_new_bias", "Moment-curve full-event estimate − truth (GeV)")]
    offsets = np.linspace(-5, 5, len(METHODS))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for ax, (metric, title) in zip(axes.flat, panels):
        for (key, label, colour, marker), dx in zip(METHODS, offsets):
            if key == "oracle" and metric != "sbi_full_new_bias":
                continue
            values, truths = table(rows, seeds, key, metric)
            ax.errorbar(truths + dx, values.mean(0), yerr=spread(values), fmt=(marker + "-") if marker else "--",
                        color=colour, lw=1.2, ms=3.5, elinewidth=1, capsize=0, label=label)
        ax.set_title(title, fontsize=8)
        if metric == "sbi_full_new_bias":
            ax.axhline(0, color=GRAY, lw=0.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("Truth parent mass (GeV)")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig_metrics.png", dpi=200)
    plt.close(fig)


def fig_spectra(rows, seed, out, truths=(425.0, 500.0, 575.0)):
    runs = {(r["truth"], r["method"]): r for r in rows if r["seed"] == seed}
    centres = 0.5 * (HIST_MASS[1:] + HIST_MASS[:-1])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for ax, t in zip(axes.flat[:3], truths):
        oracle = np.asarray(runs[(t, "oracle")]["mass_hist"], dtype=float)
        ax.fill_between(centres, oracle / oracle.sum(), step="mid", color="#d8d7d2", lw=0, label="True neutrinos")
        top = 0.0
        for key, label, colour, _ in METHODS[:-1]:
            r = runs[(t, key)]
            h = np.asarray(r["mass_hist"], dtype=float)
            top = max(top, (h / h.sum()).max())
            ax.step(centres, h / h.sum(), where="mid", color=colour, lw=1.2,
                    label=f"{label}: peak {r['peak_bias']:+.0f}, width {r['resolution']:.0f}")
        ax.axvline(t, color=GRAY, lw=0.8, ls=":")
        ax.set(xlim=(150, 850), yticks=[], xlabel="Reconstructed parent mass (GeV)")
        ax.set_ylim(0, top * 1.9)  # the true-neutrino spike is truncated
        ax.set_title(f"Truth {t:.0f} GeV (true-ν spectrum truncated)", fontsize=8)
        ax.legend(fontsize=5.2, loc="upper right")
    ax = axes[1, 1]
    centres = 0.5 * (HIST_RESID[1:] + HIST_RESID[:-1])
    for key, label, colour, _ in METHODS[:-1]:
        r = runs[(500.0, key)]
        h = np.asarray(r["pz_residual_hist"], dtype=float)
        ax.step(centres, h / h.sum(), where="mid", color=colour, lw=1.2, label=f"{label}: RMSE {r['nu_rmse']:.0f}")
    ax.axvline(0, color=GRAY, lw=0.8, ls=":")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.set(yticks=[], xlabel="pz residual (GeV)")
    ax.set_title("Truth 500 GeV: neutrino pz residual (reco − true)", fontsize=8)
    ax.legend(fontsize=5.2, loc="upper right")
    fig.text(0.5, 0.005, f"One run (seed {seed}), held-out 100k-event datasets, both parents per event. "
             "Legend: peak bias and fitted width in GeV.", ha="center", fontsize=6.5, color=GRAY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "fig_spectra.png", dpi=200)
    plt.close(fig)


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(p.read_text()) for p in sorted(source.glob("sbi_dgpo_s*.json"))]
    rows = [row for run in runs for row in run["rows"]]
    seeds = sorted(run["seed"] for run in runs)
    print("template offsets delta per seed:", [round(run["delta"], 2) for run in runs])
    write_summary(rows, seeds, out)
    fig_peak(rows, seeds, out)
    fig_metrics(rows, seeds, out)
    if {425.0, 500.0, 575.0} <= {r["truth"] for r in rows}:
        fig_spectra(rows, seeds[0], out)


if __name__ == "__main__":
    main()
