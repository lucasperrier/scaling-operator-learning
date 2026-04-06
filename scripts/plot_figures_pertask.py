"""Generate per-task figures for the paper — more informative than the all-tasks-mixed versions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaling_operator_learning.analysis import fit_power_law

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

MODEL_COLORS = {"mlp_baseline": "#1f77b4", "deeponet": "#ff7f0e", "fno": "#2ca02c"}
MODEL_LABELS = {"mlp_baseline": "MLP", "deeponet": "DeepONet", "fno": "FNO"}
MODEL_MARKERS = {"mlp_baseline": "o", "deeponet": "s", "fno": "^"}
TASK_LABELS = {"burgers_operator": "Burgers", "darcy": "Darcy", "diffusion": "Diffusion"}


def _plot_with_fit(ax, X, E, E_err, label, color, marker):
    ax.errorbar(X, E, yerr=E_err, fmt=marker, color=color, markersize=4,
                capsize=2, linewidth=0.8, label=label, zorder=3)
    fit = fit_power_law(X.astype(float), E.astype(float))
    if fit is not None and fit["alpha"] < 4.9:
        X_s = np.geomspace(X.min(), X.max(), 100)
        E_p = fit["E_inf"] + fit["a"] * X_s ** (-fit["alpha"])
        ax.plot(X_s, E_p, color=color, ls="--", lw=1.0, alpha=0.6)
        return fit["alpha"]
    return None


def fig_data_scaling_per_task(df, out_dir):
    """3-panel figure: one per task, E vs N at fixed R=128, medium capacity."""
    tasks = sorted(df["task"].unique())
    models = [m for m in ["mlp_baseline", "deeponet", "fno"] if m in df["model_name"].unique()]
    cap = "medium"
    R_val = 128

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.5), sharey=False)
    for ax, task in zip(axes, tasks):
        sub = df[(df["task"] == task) & (df["capacity_name"] == cap) & (df["resolution"] == R_val)]
        for model in models:
            msub = sub[sub["model_name"] == model].sort_values("dataset_size")
            if msub.empty:
                continue
            N = msub["dataset_size"].values
            E = msub["test_rel_l2_mean"].values
            E_err = msub["test_rel_l2_stderr"].fillna(0).values
            alpha = _plot_with_fit(ax, N, E, E_err, MODEL_LABELS[model], MODEL_COLORS[model], MODEL_MARKERS[model])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N (training samples)")
        ax.set_ylabel("Relative L2 error" if ax == axes[0] else "")
        ax.set_title(f"{TASK_LABELS[task]} (R={R_val})")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right")
    fig.suptitle("Data Scaling (medium capacity)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_data_scaling_per_task.pdf")
    fig.savefig(out_dir / "fig_data_scaling_per_task.png")
    plt.close(fig)
    print("  Per-task data scaling saved")


def fig_resolution_scaling_per_task(df, out_dir):
    """3-panel: E vs R at fixed N=1000, medium capacity."""
    tasks = sorted(df["task"].unique())
    models = [m for m in ["mlp_baseline", "deeponet", "fno"] if m in df["model_name"].unique()]
    cap = "medium"
    N_val = 1000

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.5), sharey=False)
    for ax, task in zip(axes, tasks):
        sub = df[(df["task"] == task) & (df["capacity_name"] == cap) & (df["dataset_size"] == N_val)]
        for model in models:
            msub = sub[sub["model_name"] == model].sort_values("resolution")
            if msub.empty:
                continue
            R = msub["resolution"].values
            E = msub["test_rel_l2_mean"].values
            E_err = msub["test_rel_l2_stderr"].fillna(0).values
            _plot_with_fit(ax, R, E, E_err, MODEL_LABELS[model], MODEL_COLORS[model], MODEL_MARKERS[model])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("R (resolution)")
        ax.set_ylabel("Relative L2 error" if ax == axes[0] else "")
        ax.set_title(f"{TASK_LABELS[task]} (N={N_val})")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best")
    fig.suptitle("Resolution Scaling (medium capacity)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_resolution_scaling_per_task.pdf")
    fig.savefig(out_dir / "fig_resolution_scaling_per_task.png")
    plt.close(fig)
    print("  Per-task resolution scaling saved")


def fig_capacity_scaling_per_task(df, out_dir):
    """3-panel: E vs D at fixed N=1000, R=128."""
    tasks = sorted(df["task"].unique())
    models = [m for m in ["mlp_baseline", "deeponet", "fno"] if m in df["model_name"].unique()]
    N_val = 1000
    R_val = 128

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.5), sharey=False)
    for ax, task in zip(axes, tasks):
        sub = df[(df["task"] == task) & (df["dataset_size"] == N_val) & (df["resolution"] == R_val)]
        for model in models:
            msub = sub[sub["model_name"] == model].sort_values("parameter_count")
            if msub.empty:
                continue
            D = msub["parameter_count"].values
            E = msub["test_rel_l2_mean"].values
            E_err = msub["test_rel_l2_stderr"].fillna(0).values
            _plot_with_fit(ax, D, E, E_err, MODEL_LABELS[model], MODEL_COLORS[model], MODEL_MARKERS[model])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("D (parameters)")
        ax.set_ylabel("Relative L2 error" if ax == axes[0] else "")
        ax.set_title(f"{TASK_LABELS[task]} (N={N_val}, R={R_val})")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best")
    fig.suptitle("Capacity Scaling", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_capacity_scaling_per_task.pdf")
    fig.savefig(out_dir / "fig_capacity_scaling_per_task.png")
    plt.close(fig)
    print("  Per-task capacity scaling saved")


def fig_3d_exponents(fits, out_dir):
    """Grouped bar chart: α, β, γ from the full 3D fits, per task × model."""
    recs = fits["full_3d_fits"]
    tasks = sorted(set(r["task"] for r in recs))
    models = ["mlp_baseline", "deeponet", "fno"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    params = [("alpha", r"$\alpha$ (data)"), ("beta", r"$\beta$ (capacity)"), ("gamma", r"$\gamma$ (resolution)")]

    for ax, (param, ylabel) in zip(axes, params):
        x = np.arange(len(tasks))
        width = 0.25
        for i, model in enumerate(models):
            vals, ci_lo, ci_hi = [], [], []
            for task in tasks:
                rec = next((r for r in recs if r["task"] == task and r["model_name"] == model), None)
                if rec and param in rec:
                    v = rec[param]
                    b = rec.get("bootstrap", {})
                    vals.append(v)
                    ci_lo.append(max(0, v - b.get(f"{param}_ci_lo", v)))
                    ci_hi.append(max(0, b.get(f"{param}_ci_hi", v) - v))
                else:
                    vals.append(0); ci_lo.append(0); ci_hi.append(0)
            ax.bar(x + i * width, vals, width, label=MODEL_LABELS[model],
                   color=MODEL_COLORS[model], yerr=[ci_lo, ci_hi], capsize=3)
        ax.set_xticks(x + width)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(r"3D Scaling Exponents: $E(N,D,R) = E_\infty + a N^{-\alpha} + b D^{-\beta} + c R^{-\gamma}$",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_3d_exponents.pdf")
    fig.savefig(out_dir / "fig_3d_exponents.png")
    plt.close(fig)
    print("  3D exponents saved")


def fig_cross_res_per_task(full_df, out_dir):
    """Cross-resolution heatmaps per task for DeepONet and FNO."""
    xr = full_df[full_df["eval_resolution"].notna()]
    tasks = sorted(xr["task"].unique())
    models = ["deeponet", "fno"]

    fig, axes = plt.subplots(len(tasks), len(models), figsize=(5 * len(models), 4 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]

    for row, task in enumerate(tasks):
        tdf = xr[xr["task"] == task]
        for col, model in enumerate(models):
            ax = axes[row][col]
            mdf = tdf[tdf["model_name"] == model]
            if mdf.empty:
                ax.set_title(f"{TASK_LABELS[task]} — {MODEL_LABELS[model]} (no data)")
                continue
            pivot = mdf.pivot_table(values="test_rel_l2_mean", index="resolution",
                                    columns="eval_resolution", aggfunc="mean")
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)
            im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([int(c) for c in pivot.columns], rotation=45)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([int(i) for i in pivot.index])
            ax.set_xlabel(r"$R_{\mathrm{eval}}$")
            ax.set_ylabel(r"$R_{\mathrm{train}}$")
            ax.set_title(f"{TASK_LABELS[task]} — {MODEL_LABELS[model]}")
            # Annotate cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                fontsize=7, color="white" if v > pivot.values[~np.isnan(pivot.values)].mean() else "black")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Cross-Resolution Transfer", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cross_res_per_task.pdf")
    fig.savefig(out_dir / "fig_cross_res_per_task.png")
    plt.close(fig)
    print("  Cross-res per-task saved")


def fig_data_scaling_allR(df, out_dir):
    """Per-task: 4 panels (one per R), all models, medium capacity."""
    tasks = sorted(df["task"].unique())
    models = [m for m in ["mlp_baseline", "deeponet", "fno"] if m in df["model_name"].unique()]
    cap = "medium"
    resolutions = sorted(df["resolution"].unique())

    for task in tasks:
        tdf = df[(df["task"] == task) & (df["capacity_name"] == cap)]
        fig, axes = plt.subplots(1, len(resolutions), figsize=(3.8 * len(resolutions), 3.3), sharey=True)
        for ax, R in zip(axes, resolutions):
            sub = tdf[tdf["resolution"] == R]
            for model in models:
                msub = sub[sub["model_name"] == model].sort_values("dataset_size")
                if msub.empty:
                    continue
                N = msub["dataset_size"].values
                E = msub["test_rel_l2_mean"].values
                E_err = msub["test_rel_l2_stderr"].fillna(0).values
                alpha = _plot_with_fit(ax, N, E, E_err, MODEL_LABELS[model], MODEL_COLORS[model], MODEL_MARKERS[model])
                if alpha:
                    ax.text(0.95, 0.05 + models.index(model) * 0.08,
                            f"α={alpha:.2f}", transform=ax.transAxes,
                            fontsize=7, ha="right", color=MODEL_COLORS[model])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("N")
            ax.set_ylabel("Rel. L2 error" if ax == axes[0] else "")
            ax.set_title(f"R={R}")
            ax.grid(True, alpha=0.3, which="both")
            if ax == axes[-1]:
                ax.legend(loc="upper right")
        fig.suptitle(f"{TASK_LABELS[task]} — Data Scaling (medium capacity)", y=1.02, fontsize=12)
        fig.tight_layout()
        name = f"fig_{task}_data_scaling_allR"
        fig.savefig(out_dir / f"{name}.pdf")
        fig.savefig(out_dir / f"{name}.png")
        plt.close(fig)
    print("  Per-task all-R data scaling saved")


def main():
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    df_train = pd.read_csv("runs/grouped_metrics_train.csv")
    df_full = pd.read_csv("runs/grouped_metrics.csv")
    fits = json.loads(Path("results/scaling_fits.json").read_text())

    print("Generating per-task figures...")
    fig_data_scaling_per_task(df_train, out_dir)
    fig_resolution_scaling_per_task(df_train, out_dir)
    fig_capacity_scaling_per_task(df_train, out_dir)
    fig_3d_exponents(fits, out_dir)
    fig_cross_res_per_task(df_full, out_dir)
    fig_data_scaling_allR(df_train, out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
