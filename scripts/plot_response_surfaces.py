"""Phase 3b: response-surface heatmaps per (task, model) per PROTOCOL_V2 §3.

Reads results/cell_stats_diagonal.csv and results/cell_regimes.csv and emits:

  figures/response_surface_<task>_<model>.png   — 7-panel (N x R) heatmap of
                                                  median test_rel_l2 per
                                                  capacity slot, R=48 masked
  figures/regime_map_<task>_<model>.png         — 7-panel (N x R) regime
                                                  label map per capacity
  figures/sensitivity_distributions.png         — pooled |S_N|, |S_R|, |S_C|
                                                  histograms with the
                                                  calibrated thresholds

Figures are gitignored; this script is the canonical regenerator.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAG_CSV = REPO_ROOT / "results" / "cell_stats_diagonal.csv"
REGIMES_CSV = REPO_ROOT / "results" / "cell_regimes.csv"
CALIB_JSON = REPO_ROOT / "results" / "regime_calibration.json"
FIGURES_DIR = REPO_ROOT / "figures"

CAPACITY_ORDER = [
    "tiny",
    "small",
    "small-med",
    "medium",
    "med-large",
    "large",
    "xlarge",
]

REGIME_COLORS = {
    "data_limited": "#4daf4a",
    "resolution_sensitive": "#377eb8",
    "capacity_limited": "#984ea3",
    "saturated": "#ffff33",
    "mixed": "#999999",
    "unstable": "#e41a1c",
    "ambiguous": "#dddddd",
    "data_bug_r48": "#000000",
}
REGIME_LABEL_ORDER = list(REGIME_COLORS.keys())


def plot_response_surface(diag: pd.DataFrame, task: str, model: str) -> None:
    sub = diag[(diag["task"] == task) & (diag["model_name"] == model)]
    if sub.empty:
        return
    caps_present = [c for c in CAPACITY_ORDER if c in sub["capacity_name"].unique()]
    if not caps_present:
        return
    n_panels = len(caps_present)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3.2), sharey=True)
    if n_panels == 1:
        axes = [axes]
    # Mask R=48 by setting median to NaN for plotting.
    plot_sub = sub.copy()
    plot_sub.loc[plot_sub["is_r48"], "median_test_rel_l2"] = np.nan
    err_min = max(plot_sub["median_test_rel_l2"].min(skipna=True), 1e-4)
    err_max = plot_sub["median_test_rel_l2"].max(skipna=True)
    if not np.isfinite(err_min) or not np.isfinite(err_max) or err_min >= err_max:
        plt.close(fig)
        return
    norm = mcolors.LogNorm(vmin=err_min, vmax=err_max)
    for ax, cap in zip(axes, caps_present):
        cell = plot_sub[plot_sub["capacity_name"] == cap]
        pivot = cell.pivot(
            index="dataset_size", columns="resolution", values="median_test_rel_l2"
        )
        pivot = pivot.sort_index().sort_index(axis=1)
        if pivot.empty:
            ax.set_title(f"{cap}\n(no cells)", fontsize=8)
            continue
        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap="viridis",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [str(int(c)) for c in pivot.columns], rotation=90, fontsize=6
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(i)) for i in pivot.index], fontsize=6)
        ax.set_title(cap, fontsize=8)
        ax.set_xlabel("R", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("N", fontsize=7)
    fig.suptitle(
        f"Response surface — {task} / {model}  (median test_rel_l2, log scale; R=48 masked)",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("median test_rel_l2", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    out = FIGURES_DIR / f"response_surface_{task}_{model}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_regime_map(regimes: pd.DataFrame, task: str, model: str) -> None:
    sub = regimes[(regimes["task"] == task) & (regimes["model_name"] == model)]
    if sub.empty:
        return
    caps_present = [c for c in CAPACITY_ORDER if c in sub["capacity_name"].unique()]
    if not caps_present:
        return
    label_to_idx = {lbl: i for i, lbl in enumerate(REGIME_LABEL_ORDER)}
    n_panels = len(caps_present)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3.2), sharey=True)
    if n_panels == 1:
        axes = [axes]
    cmap = mcolors.ListedColormap([REGIME_COLORS[lbl] for lbl in REGIME_LABEL_ORDER])
    bounds = list(range(len(REGIME_LABEL_ORDER) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    for ax, cap in zip(axes, caps_present):
        cell = sub[sub["capacity_name"] == cap].copy()
        cell["regime_idx"] = cell["regime"].map(label_to_idx).fillna(-1).astype(int)
        pivot = cell.pivot(
            index="dataset_size", columns="resolution", values="regime_idx"
        )
        pivot = pivot.sort_index().sort_index(axis=1)
        if pivot.empty:
            ax.set_title(f"{cap}\n(no cells)", fontsize=8)
            continue
        ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [str(int(c)) for c in pivot.columns], rotation=90, fontsize=6
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(i)) for i in pivot.index], fontsize=6)
        ax.set_title(cap, fontsize=8)
        ax.set_xlabel("R", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("N", fontsize=7)
    fig.suptitle(
        f"Regime map — {task} / {model}  (PROTOCOL_V2 §6 labels)",
        fontsize=10,
    )
    # Custom legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[lbl], label=lbl)
        for lbl in REGIME_LABEL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.05),
    )
    out = FIGURES_DIR / f"regime_map_{task}_{model}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_distributions(regimes: pd.DataFrame, calib: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, axis_name, col in zip(
        axes,
        ["|S_N|", "|S_R|", "|S_C|"],
        ["s_n", "s_r", "s_c"],
    ):
        vals = regimes[col].abs().dropna()
        ax.hist(vals, bins=60, color="#4477aa", alpha=0.85)
        ax.axvline(calib["THRESH_LOW"], color="#999999", linestyle="--", linewidth=1)
        ax.axvline(calib["THRESH_HIGH"], color="#222222", linestyle="--", linewidth=1)
        ax.set_xlabel(axis_name, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 5)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"{axis_name}  median={vals.median():.3f}",
            fontsize=8,
        )
    axes[0].set_ylabel("# cells", fontsize=8)
    fig.suptitle(
        "Local sensitivity distributions (PROTOCOL §6.1)\n"
        f"THRESH_LOW={calib['THRESH_LOW']:.3f}, "
        f"THRESH_HIGH={calib['THRESH_HIGH']:.3f}",
        fontsize=10,
    )
    out = FIGURES_DIR / "sensitivity_distributions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    diag = pd.read_csv(DIAG_CSV)
    regimes = pd.read_csv(REGIMES_CSV)
    calib = json.loads(CALIB_JSON.read_text())

    task_model_pairs = (
        diag[["task", "model_name"]].drop_duplicates().values.tolist()
    )
    print(f"Plotting response surfaces and regime maps for {len(task_model_pairs)} (task, model) pairs.")
    for task, model in task_model_pairs:
        plot_response_surface(diag, task, model)
        plot_regime_map(regimes, task, model)
        print(f"  {task} / {model}")
    plot_sensitivity_distributions(regimes, calib)
    print("Wrote sensitivity_distributions.png")


if __name__ == "__main__":
    main()
