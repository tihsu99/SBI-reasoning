"""Three summary figures for the hZZ mu study, read from results/*.csv.

Every panel: y = predicted mu (observed pseudo-data unless noted), dashed line =
true mu, dotted line = perfect-analytic-classifier floor, grey lines = grid edges.
Error bars are the spread across seeds / ensemble draws.
"""
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
REGIONS = [("L", 0.4, (0.3, 2.3)), ("H", 2.5, (1.0, 3.0))]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def rows(name):
    with open(R / name) as f:
        return list(csv.DictReader(f))


def floor_at(grid_label, truth):
    v = [float(r["clipped_mu"]) for r in rows("floor.csv")
         if r["grid_label"] == grid_label and abs(float(r["truth_mu"]) - truth) < 1e-9]
    return np.mean(v), np.std(v, ddof=1)


FLOOR = {"L": floor_at("{0.3,1.3,2.3}", 0.4)[0], "H": floor_at("{1,2,3}", 2.5)[0]}


def guides(ax, truth, edges, floor):
    for i, e in enumerate(edges):
        ax.axhline(e, color="0.75", lw=0.8, label="grid edge (point here = clipped)" if i == 0 else None)
    ax.axhline(truth, color="k", ls="--", lw=1, label="true mu")
    if floor is not None:
        ax.axhline(floor, color="k", ls=":", lw=1, label="perfect classifier")
    ax.set_ylim(min(edges) - 0.4, max(edges) + 0.4)


def save(fig, name, ax):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(R / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_ensemble():
    data = rows("mlp_ensemble16.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    for ax, (g, truth, edges) in zip(axes, REGIONS):
        sel = sorted((int(r["K"]), float(r["recovered_mu"]), float(r["spread"])) for r in data
                     if r["grid"] == g and r["dataset"] == "observed")
        k, mu, sd = zip(*sel)
        ax.errorbar(range(len(k)), mu, yerr=sd, fmt="o", color=OKABE[0], capsize=3)
        ax.set_xticks(range(len(k)), [f"K={x}" for x in k])
        guides(ax, truth, edges, FLOOR[g])
        ax.set(title=f"MLP ensemble, true mu = {truth}", xlabel="ensemble size", ylabel="predicted mu")
    save(fig, "fig1_ensemble", axes[0])


def fig_evenet():
    heads = rows("heads_scan_gpu.csv")
    mlp = rows("mlp_ensemble16.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    for ax, (g, truth, edges) in zip(axes, REGIONS):
        labels, mu, sd, col = [], [], [], []
        for k in ("1", "8"):
            r = next(r for r in mlp if r["grid"] == g and r["dataset"] == "observed" and r["K"] == k)
            labels.append(f"MLP\nK={k}"); mu.append(float(r["recovered_mu"]))
            sd.append(float(r["spread"])); col.append(OKABE[0])
        for i, h in enumerate(("linear", "tiny", "small", "full")):
            v = [float(r["recovered_mu"]) for r in heads
                 if r["grid"] == g and r["head"] == h and r["dataset"] == "observed"]
            labels.append(f"EveNet\n{h}"); mu.append(np.mean(v))
            sd.append(np.std(v, ddof=1)); col.append(OKABE[1 + i])
        for x, (m, s, c) in enumerate(zip(mu, sd, col)):
            ax.errorbar(x, m, yerr=s, fmt="o", color=c, capsize=3)
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        guides(ax, truth, edges, FLOOR[g])
        ax.set(title=f"MLP vs EveNet heads, true mu = {truth}", ylabel="predicted mu")
    save(fig, "fig2_evenet", axes[0])


def fig_templates():
    grids = ["{0,1,3}", "{0,1,2}", "{0.3,1.3,2.3}", "{0.5,1.5,2.5}", "{1,2,3}"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    for ax, truth in zip(axes, (0.4, 2.5)):
        for x, gl in enumerate(grids):
            nodes = [float(t) for t in gl.strip("{}").split(",")]
            ax.plot([x] * 3, nodes, "_", color="0.6", ms=14, mew=1.5)
            m, s = floor_at(gl, truth)
            inside = min(nodes) <= truth <= max(nodes)
            ax.errorbar(x, m, yerr=s, fmt="o", capsize=3, color=OKABE[0],
                        mfc=OKABE[0] if inside else "white")
        ax.axhline(truth, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(grids)), grids, fontsize=7)
        ax.set(title=f"template grids, true mu = {truth}", xlabel="template values",
               ylabel="predicted mu (perfect classifier)", ylim=(-0.2, 3.3))
    axes[0].plot([], [], "_", color="0.6", ms=10, label="template value")
    axes[0].plot([], [], "o", color=OKABE[0], label="predicted (truth inside grid)")
    axes[0].plot([], [], "o", color=OKABE[0], mfc="white", label="predicted (truth outside grid)")
    axes[0].plot([], [], "k--", lw=1, label="true mu")
    save(fig, "fig3_templates", axes[0])


if __name__ == "__main__":
    fig_ensemble()
    fig_evenet()
    fig_templates()
