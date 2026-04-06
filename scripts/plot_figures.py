"""Generate all figures for the operator-learning scaling paper.

Figures:
  1. Data scaling: E vs N at fixed R, across models
  2. Resolution scaling: E vs R at fixed N, across models
  3. Capacity scaling: E vs D at fixed N and R
  4. Exponent comparison: α, β, γ across models (bar chart with CIs)
  5. Cross-resolution transfer matrix (heatmap)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaling_operator_learning.analysis import fit_power_law

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_COLORS = {
    "mlp_baseline": "#1f77b4",
    "deeponet": "#ff7f0e",
    "fno": "#2ca02c",
}
MODEL_MARKERS = {
    "mlp_baseline": "o",
    "deeponet": "s",
    "fno": "^",
}


# ── Helpers ──────────────────────────────────────────────────────────────
def _plot_scaling_curve(ax, X, E, E_err, label, color, marker, ls="-", lw=1.2):
    ax.errorbar(X, E, yerr=E_err, fmt=marker, color=color, markersize=4,
                capsize=2, linewidth=0.8, label=label, zorder=3)
    fit = fit_power_law(X.astype(float), E.astype(float))
    if fit is not None:
        X_smooth = np.geomspace(X.min(), X.max(), 100)
        E_pred = fit["E_inf"] + fit["a"] * X_smooth ** (-fit["alpha"])
        ax.plot(X_smooth, E_pred, color=color, ls=ls, lw=lw, alpha=0.6)


def _setup_loglog(ax, xlabel="", ylabel="Test error"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, which="both")


# ── Figure 1: Data scaling at fixed R ───────────────────────────────────
def fig1_data_scaling(df: pd.DataFrame, out_dir: Path):
    resolutions = sorted(df["resolution"].unique()) if "resolution" in df.columns else [None]
    show_res = resolutions[:3]
    models = [m for m in ["mlp_baseline", "deeponet", "fno"] if m in df["model_name"].unique()]
    cap_target = "medium" if "medium" in df["capacity_name"].unique() else df["capacity_name"].iloc[0]

    fig, axes = plt.subplots(1, len(show_res), figsize=(4 * len(show_res), 3.5), sharey=True)
    if len(show_res) == 1:
        axes = [axes]

    for ax, R in zip(axes, show_res):
        sub = df[df["capacity_name"] == cap_target]
        if R is not None:
            sub = sub[sub["resolution"] == R]
        for model in models:
            msub = sub[sub["model_name"] == model].sort_values("dataset_size")
            if msub.empty:
                continue
            N = msub["dataset_size"].values
            E = msub["test_rel_l2_mean"].values
            E_err = msub.get("test_rel_l2_stderr", pd.Series(np.zeros(len(msub)))).values
            _plot_scaling_curve(ax, N, E, E_err, model, MODEL_COLORS.get(model, "gray"),
                                MODEL_MARKERS.get(model, "o"))
        title = f"R={R}" if R is not None else ""
        ax.set_title(title)
        _setup_loglog(ax, xlabel="N (training samples)",
                      ylabel="Relative L2 error" if ax == axes[0] else "")
        ax.legend(loc="upper right")

    fig.suptitle(f"Data Scaling (capacity: {cap_target})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_data_scaling.pdf")
    fig.savefig(out_dir / "fig1_data_scaling.png")
    plt.close(fig)
    print("  Fig 1: Data scaling saved")


# ── Figure 2: Resolution scaling at fixed N ─────────────────────────────
def fig2_resolution_scaling(df: pd.DataFrame, out_dir: Path):
    if "resolution" not in df.columns:
        print("  Fig 2: No resolution column, skipping")
        return
    models = [m for m in ["mlp_baseline", "deeponet", "fno"] if m in df["model_name"].unique()]
    cap_target = "medium" if "medium" in df["capacity_name"].unique() else df["capacity_name"].iloc[0]
    N_vals = sorted(df["dataset_size"].unique())
    show_N = [N_vals[len(N_vals) // 4], N_vals[len(N_vals) // 2], N_vals[-1]]
    show_N = sorted(set(show_N))

    fig, axes = plt.subplots(1, len(show_N), figsize=(4 * len(show_N), 3.5), sharey=True)
    if len(show_N) == 1:
        axes = [axes]

    for ax, N in zip(axes, show_N):
        sub = df[(df["capacity_name"] == cap_target) & (df["dataset_size"] == N)]
        for model in models:
            msub = sub[sub["model_name"] == model].sort_values("resolution")
            if msub.empty:
                continue
            R = msub["resolution"].values
            E = msub["test_rel_l2_mean"].values
            E_err = msub.get("test_rel_l2_stderr", pd.Series(np.zeros(len(msub)))).values
            _plot_scaling_curve(ax, R, E, E_err, model, MODEL_COLORS.get(model, "gray"),
                                MODEL_MARKERS.get(model, "o"))
        ax.set_title(f"N={N}")
        _setup_loglog(ax, xlabel="R (resolution)",
                      ylabel="Relative L2 error" if ax == axes[0] else "")
        ax.legend(loc="upper right")

    fig.suptitle(f"Resolution Scaling (capacity: {cap_target})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_resolution_scaling.pdf")
    fig.savefig(out_dir / "fig2_resolution_scaling.png")
    plt.close(fig)
    print("  Fig 2: Resolution scaling saved")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--grouped-csv", required=True)
    parser.add_argument("--fits-json", default=None)
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.grouped_csv)
    fits = json.loads(Path(args.fits_json).read_text()) if args.fits_json else None

    print("Generating figures...")
    fig1_data_scaling(df, out_dir)
    fig2_resolution_scaling(df, out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
