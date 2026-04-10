"""Phase 5: convergence and instability heatmaps per PROTOCOL_V2 §8.

Reads results/cell_stats_diagonal.csv and emits:

  figures/divergence_heatmap_<task>_<model>.png
  figures/seed_cv_heatmap_<task>_<model>.png

These are gitignored but regenerable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAG_CSV = REPO_ROOT / "results" / "cell_stats_diagonal.csv"
FIGURES_DIR = REPO_ROOT / "figures"
CAPACITY_ORDER = [
    "tiny", "small", "small-med", "medium", "med-large", "large", "xlarge",
]


def heatmap_panels(
    diag: pd.DataFrame,
    task: str,
    model: str,
    value_col: str,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    log_scale: bool = False,
    out_name: str = "",
) -> None:
    sub = diag[(diag["task"] == task) & (diag["model_name"] == model)]
    if sub.empty:
        return
    caps = [c for c in CAPACITY_ORDER if c in sub["capacity_name"].unique()]
    n_panels = len(caps)
    if not n_panels:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3.2), sharey=True)
    if n_panels == 1:
        axes = [axes]
    norm = (
        mcolors.LogNorm(vmin=max(vmin, 1e-4), vmax=vmax)
        if log_scale
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    for ax, cap in zip(axes, caps):
        cell = sub[sub["capacity_name"] == cap]
        pivot = cell.pivot(
            index="dataset_size", columns="resolution", values=value_col
        ).sort_index().sort_index(axis=1)
        if pivot.empty:
            ax.set_visible(False)
            continue
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", norm=norm, cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=90, fontsize=6)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(i)) for i in pivot.index], fontsize=6)
        ax.set_title(cap, fontsize=8)
        ax.set_xlabel("R", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("N", fontsize=7)
    fig.suptitle(f"{title} — {task} / {model}", fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(value_col, fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    out = FIGURES_DIR / f"{out_name}_{task}_{model}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    diag = pd.read_csv(DIAG_CSV)
    pairs = diag[["task", "model_name"]].drop_duplicates().values.tolist()
    for task, model in pairs:
        heatmap_panels(
            diag, task, model,
            value_col="divergence_rate",
            title="Divergence rate",
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            out_name="divergence_heatmap",
        )
        heatmap_panels(
            diag, task, model,
            value_col="seed_cv_test_rel_l2",
            title="Seed CV (test_rel_l2)",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            out_name="seed_cv_heatmap",
        )
        print(f"  {task} / {model}")
    print("Done.")


if __name__ == "__main__":
    main()
