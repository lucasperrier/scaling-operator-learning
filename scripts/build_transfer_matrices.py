"""Phase 4: cross-resolution transfer matrices per PROTOCOL_V2 §7.

Reads results/cell_stats_crossres.csv and results/cell_stats_diagonal.csv.
Emits:

  results/transfer_scores.csv       — one row per (task, model, capacity, N)
                                       slice with the 5 fragility scores
  results/transfer_matrices.json    — full per-slice transfer matrices (K[i,j])
  figures/transfer_matrix_<task>_<model>.png — faceted transfer-matrix heatmaps

The transfer matrix K[i,j] per slice is built from the median_test_rel_l2 of
each cross-res cell. The diagonal K[i,i] comes from the diagonal training stats
(not from the cross-res records, which exclude R_train == R_eval).

Fragility scores per slice (PROTOCOL §7.2):
  mean_degradation        Mean of K[i,j] / K[i,i] over off-diagonal cells
  worst_degradation       Max of K[i,j] / K[i,i]
  worst_pair              The (R_train, R_eval) that achieved the maximum
  train_low_eval_high     Mean degradation for R_eval > R_train
  train_high_eval_low     Mean degradation for R_eval < R_train
  transfer_asymmetry      train_high_eval_low / train_low_eval_high
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
CROSS_CSV = REPO_ROOT / "results" / "cell_stats_crossres.csv"
DIAG_CSV = REPO_ROOT / "results" / "cell_stats_diagonal.csv"
OUT_SCORES = REPO_ROOT / "results" / "transfer_scores.csv"
OUT_MATRICES = REPO_ROOT / "results" / "transfer_matrices.json"
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


def build_slice_matrix(
    cross_slice: pd.DataFrame,
    diag_df: pd.DataFrame,
    task: str,
    model: str,
    capacity: str,
    n: int,
) -> tuple[pd.DataFrame | None, dict | None]:
    """Build the 4x5 transfer matrix for one (T, M, C, N) slice.

    Returns (matrix_df, scores_dict) or (None, None) if data is insufficient.
    """
    # Off-diagonal values from cross-res
    if cross_slice.empty:
        return None, None

    r_train_vals = sorted(cross_slice["resolution_train"].unique())
    r_eval_vals = sorted(cross_slice["resolution_eval"].unique())

    # Pivot: rows = R_train, cols = R_eval, values = median_test_rel_l2
    pivot = cross_slice.pivot(
        index="resolution_train",
        columns="resolution_eval",
        values="median_test_rel_l2",
    ).sort_index().sort_index(axis=1)

    # Diagonal entries from diag_df
    diag_sub = diag_df[
        (diag_df["task"] == task)
        & (diag_df["model_name"] == model)
        & (diag_df["capacity_name"] == capacity)
        & (diag_df["dataset_size"] == n)
        & (~diag_df["is_r48"])
    ]
    diag_by_r = diag_sub.set_index("resolution")["median_test_rel_l2"]

    # Build full matrix (add diagonal entries where R_train == R_eval)
    all_r = sorted(set(r_train_vals) | set(r_eval_vals))
    full = pd.DataFrame(index=r_train_vals, columns=all_r, dtype=float)
    for rt in r_train_vals:
        for re in all_r:
            if rt == re:
                full.loc[rt, re] = diag_by_r.get(rt, np.nan)
            elif re in pivot.columns:
                full.loc[rt, re] = pivot.loc[rt, re] if rt in pivot.index else np.nan
    full.index.name = "R_train"
    full.columns.name = "R_eval"

    # Compute fragility scores (PROTOCOL §7.2)
    degradation_ratios = []
    low_high_ratios = []
    high_low_ratios = []
    worst_deg = 0.0
    worst_pair = ("", "")

    for rt in r_train_vals:
        diag_val = full.loc[rt, rt] if rt in full.columns else np.nan
        if np.isnan(diag_val) or diag_val <= 0:
            continue
        for re in all_r:
            if re == rt:
                continue
            val = full.loc[rt, re]
            if np.isnan(val):
                continue
            deg = val / diag_val
            degradation_ratios.append(deg)
            if deg > worst_deg:
                worst_deg = deg
                worst_pair = (int(rt), int(re))
            if re > rt:
                low_high_ratios.append(deg)
            elif re < rt:
                high_low_ratios.append(deg)

    if not degradation_ratios:
        return full, None

    mean_deg = float(np.mean(degradation_ratios))
    mean_lh = float(np.mean(low_high_ratios)) if low_high_ratios else float("nan")
    mean_hl = float(np.mean(high_low_ratios)) if high_low_ratios else float("nan")
    asymmetry = (mean_hl / mean_lh) if (mean_lh > 0 and not np.isnan(mean_lh)) else float("nan")

    scores = {
        "task": task,
        "model_name": model,
        "capacity_name": capacity,
        "dataset_size": n,
        "n_off_diagonal": len(degradation_ratios),
        "mean_degradation": mean_deg,
        "worst_degradation": float(worst_deg),
        "worst_pair_r_train": worst_pair[0],
        "worst_pair_r_eval": worst_pair[1],
        "train_low_eval_high": mean_lh,
        "train_high_eval_low": mean_hl,
        "transfer_asymmetry": asymmetry,
    }
    return full, scores


def plot_transfer_matrices(all_matrices: dict, scores_df: pd.DataFrame) -> None:
    """Plot one figure per (task, model) with panels faceted by capacity and N."""
    task_model_pairs = (
        scores_df[["task", "model_name"]].drop_duplicates().values.tolist()
    )
    for task, model in task_model_pairs:
        sub = scores_df[
            (scores_df["task"] == task) & (scores_df["model_name"] == model)
        ]
        caps = [c for c in CAPACITY_ORDER if c in sub["capacity_name"].unique()]
        n_vals = sorted(sub["dataset_size"].unique())
        # Pick 3 representative N values (min, median, max) to keep panels manageable.
        if len(n_vals) > 3:
            rep_n = [n_vals[0], n_vals[len(n_vals) // 2], n_vals[-1]]
        else:
            rep_n = n_vals
        # Pick 3 representative capacities.
        if len(caps) > 3:
            rep_caps = [caps[0], caps[len(caps) // 2], caps[-1]]
        else:
            rep_caps = caps
        n_rows = len(rep_caps)
        n_cols = len(rep_n)
        if n_rows == 0 or n_cols == 0:
            continue
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.3 * n_cols, 2.8 * n_rows), squeeze=False
        )
        for ri, cap in enumerate(rep_caps):
            for ci, n in enumerate(rep_n):
                ax = axes[ri][ci]
                key = f"{task}|{model}|{cap}|{n}"
                mat = all_matrices.get(key)
                if mat is None:
                    ax.set_visible(False)
                    continue
                mdf = pd.DataFrame(mat)
                mdf.index = mdf.index.astype(int)
                mdf.columns = mdf.columns.astype(int)
                im = ax.imshow(
                    mdf.values,
                    aspect="auto",
                    origin="lower",
                    cmap="RdYlGn_r",
                    vmin=max(0.001, np.nanmin(mdf.values)),
                    vmax=np.nanmax(mdf.values),
                )
                ax.set_xticks(range(len(mdf.columns)))
                ax.set_xticklabels(mdf.columns, fontsize=6, rotation=90)
                ax.set_yticks(range(len(mdf.index)))
                ax.set_yticklabels(mdf.index, fontsize=6)
                ax.set_title(f"{cap}, N={n}", fontsize=7)
                if ci == 0:
                    ax.set_ylabel("R_train", fontsize=7)
                if ri == n_rows - 1:
                    ax.set_xlabel("R_eval", fontsize=7)
        fig.suptitle(
            f"Transfer matrix — {task} / {model}  (median test_rel_l2)",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
        cbar.set_label("median test_rel_l2", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        out = FIGURES_DIR / f"transfer_matrix_{task}_{model}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    cross = pd.read_csv(CROSS_CSV)
    diag = pd.read_csv(DIAG_CSV)

    slice_cols = ["task", "model_name", "capacity_name", "dataset_size"]
    scores_rows = []
    matrices = {}

    for keys, g in cross.groupby(slice_cols):
        task, model, cap, n = keys
        mat, scores = build_slice_matrix(g, diag, task, model, cap, int(n))
        if scores is not None:
            scores_rows.append(scores)
        if mat is not None:
            key = f"{task}|{model}|{cap}|{n}"
            matrices[key] = mat.to_dict()

    scores_df = pd.DataFrame(scores_rows)
    scores_df.to_csv(OUT_SCORES, index=False)
    OUT_MATRICES.write_text(json.dumps(matrices, indent=1))
    print(f"Wrote {OUT_SCORES}: {len(scores_df)} slices with scores.")
    print(f"Wrote {OUT_MATRICES}: {len(matrices)} matrices.")

    # Summary statistics
    print()
    print("Transfer score summary (PROTOCOL §7.2):")
    print(f"  mean_degradation:  median={scores_df['mean_degradation'].median():.3f}, mean={scores_df['mean_degradation'].mean():.3f}")
    print(f"  worst_degradation: median={scores_df['worst_degradation'].median():.3f}, mean={scores_df['worst_degradation'].mean():.3f}")
    print(f"  train_low_eval_high: median={scores_df['train_low_eval_high'].median():.3f}")
    print(f"  train_high_eval_low: median={scores_df['train_high_eval_low'].median():.3f}")
    print(f"  transfer_asymmetry: median={scores_df['transfer_asymmetry'].median():.3f}")
    print()
    print("Per (task, model) median transfer scores:")
    print(
        scores_df.groupby(["task", "model_name"])[
            ["mean_degradation", "worst_degradation", "transfer_asymmetry"]
        ]
        .median()
        .round(3)
        .to_string()
    )
    print()

    # Worst 10 slices by worst_degradation
    print("Top 10 slices by worst_degradation:")
    print(
        scores_df.nlargest(10, "worst_degradation")[
            ["task", "model_name", "capacity_name", "dataset_size",
             "worst_degradation", "worst_pair_r_train", "worst_pair_r_eval",
             "transfer_asymmetry"]
        ].to_string(index=False)
    )

    # Figures
    plot_transfer_matrices(matrices, scores_df)
    print("\nFigures written.")


if __name__ == "__main__":
    main()
