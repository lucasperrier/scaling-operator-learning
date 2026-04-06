"""Generate diagnostic figures for multi-law analysis.

For each task×model, show:
  1. Data scaling: E vs N with top-2 law fits overlaid
  2. Resolution scaling: E vs R
  3. Capacity scaling: E vs D
  4. Summary heatmap: which law wins where
  5. Akaike weight stacked bar chart
  6. Bootstrap law-selection stability
  7. Restricted-set comparison table
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ──── Re-define laws for plotting ────

def _power(X, E_inf, a, alpha):
    return E_inf + a * np.power(X, -alpha)

def _logarithmic(X, a, b):
    return a - b * np.log(X)

def _linear(X, a, b):
    return a + b * X

def _exponential(X, E_inf, a, alpha):
    return E_inf + a * np.exp(-alpha * X)

def _saturation(X, E_inf, a, X0, alpha):
    return E_inf + a * np.power(1.0 + X / X0, -alpha)

def _stretched_exp(X, E_inf, a, alpha, beta):
    return E_inf + a * np.exp(-alpha * np.power(X, beta))

LAW_FUNCS = {
    "power": _power,
    "logarithmic": _logarithmic,
    "linear": _linear,
    "exponential": _exponential,
    "saturation": _saturation,
    "stretched_exp": _stretched_exp,
}

LAW_COLORS = {
    "power": "#e41a1c",
    "logarithmic": "#377eb8",
    "linear": "#4daf4a",
    "exponential": "#984ea3",
    "saturation": "#ff7f00",
    "stretched_exp": "#a65628",
}

LAW_LABELS = {
    "power": r"$E_\infty + a X^{-\alpha}$",
    "logarithmic": r"$a - b \ln X$",
    "linear": r"$a + bX$",
    "exponential": r"$E_\infty + a e^{-\alpha X}$",
    "saturation": r"$E_\infty + a/(1+X/X_0)^\alpha$",
    "stretched_exp": r"$E_\infty + a e^{-\alpha X^\beta}$",
}

AXIS_LABELS = {"data": "Dataset size $N$", "resolution": "Resolution $R$", "capacity": "Parameters $D$"}
AXIS_XKEY = {"data": "dataset_size", "resolution": "resolution", "capacity": "parameter_count"}

TASKS = ["burgers_operator", "darcy", "diffusion"]
TASK_NAMES = {"burgers_operator": "Burgers", "darcy": "Darcy", "diffusion": "Diffusion"}
MODELS = ["deeponet", "fno", "mlp_baseline"]
MODEL_NAMES = {"deeponet": "DeepONet", "fno": "FNO", "mlp_baseline": "MLP"}


def eval_law(law_name, params, X_fine):
    """Evaluate a law on a fine grid for plotting."""
    try:
        if law_name == "power":
            return _power(X_fine, params["E_inf"], params["a"], params["alpha"])
        elif law_name == "logarithmic":
            return _logarithmic(X_fine, params["a"], params["b"])
        elif law_name == "linear":
            return _linear(X_fine, params["a"], params["b"])
        elif law_name == "exponential":
            # alpha was rescaled back to original units
            return _exponential(X_fine, params["E_inf"], params["a"], params["alpha"])
        elif law_name == "saturation":
            return _saturation(X_fine, params["E_inf"], params["a"], params["X0"], params["alpha"])
        elif law_name == "stretched_exp":
            X_norm = X_fine / params.get("X_normalizer", 1.0)
            return _stretched_exp(X_norm, params["E_inf"], params["a"], params["alpha"], params["beta"])
    except Exception:
        return None
    return None


def main():
    df = pd.read_csv("runs/grouped_metrics_train.csv")
    df["test_rel_l2_mean"] = pd.to_numeric(df["test_rel_l2_mean"], errors="coerce")
    with open("results/multilaw_fits.json") as f:
        results = json.load(f)

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    # Figure 1: Per-task×model, representative slice with overlay
    # ════════════════════════════════════════════════════════════════
    for axis in ["data", "resolution", "capacity"]:
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Multi-law fits — {axis.capitalize()} scaling", fontsize=16, y=0.98)

        fits_key = f"{axis}_fits"
        all_fits = results[fits_key]

        for i, task in enumerate(TASKS):
            for j, model in enumerate(MODELS):
                ax = axes[i, j]
                # Get data for a representative slice
                task_model_fits = [f for f in all_fits
                                   if f["task"] == task and f["model_name"] == model]
                if not task_model_fits:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                    continue

                # Pick middle slice (by R² to get a representative one)
                task_model_fits.sort(key=lambda f: f.get("best_r2", 0))
                mid_idx = len(task_model_fits) // 2
                best_fit = task_model_fits[mid_idx]

                # Get actual data for this slice
                xkey = AXIS_XKEY[axis]
                if axis == "data":
                    mask = (df["task"] == task) & (df["model_name"] == model) & \
                           (df["capacity_name"] == best_fit.get("capacity_name", "medium")) & \
                           (df["resolution"] == best_fit.get("resolution", 128))
                elif axis == "resolution":
                    mask = (df["task"] == task) & (df["model_name"] == model) & \
                           (df["capacity_name"] == best_fit.get("capacity_name", "medium")) & \
                           (df["dataset_size"] == best_fit.get("dataset_size", 1000))
                elif axis == "capacity":
                    mask = (df["task"] == task) & (df["model_name"] == model) & \
                           (df["dataset_size"] == best_fit.get("dataset_size", 1000)) & \
                           (df["resolution"] == best_fit.get("resolution", 128))

                sub = df[mask].sort_values(xkey)
                if len(sub) < 2:
                    ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
                    continue

                X = sub[xkey].values.astype(float)
                E = sub["test_rel_l2_mean"].values.astype(float)

                # Plot data
                ax.scatter(X, E, s=80, c="black", zorder=10, label="Data")

                # Plot winning law
                X_fine = np.linspace(X.min() * 0.9, X.max() * 1.1, 200)
                law = best_fit["best_law"]
                params = best_fit["best_params"]
                E_fit = eval_law(law, params, X_fine)
                if E_fit is not None:
                    ax.plot(X_fine, E_fit, color=LAW_COLORS.get(law, "red"), lw=2.5,
                            label=f"Winner: {LAW_LABELS.get(law, law)} (R²={best_fit['best_r2']:.3f})")

                # Collect law vote counts for this task×model
                law_votes = {}
                for f in task_model_fits:
                    l = f.get("best_law", "?")
                    law_votes[l] = law_votes.get(l, 0) + 1
                votes_str = ", ".join(f"{k}:{v}" for k, v in
                                      sorted(law_votes.items(), key=lambda x: -x[1])[:3])

                ax.set_xlabel(AXIS_LABELS[axis])
                ax.set_ylabel("Relative L2 error")
                if axis == "data":
                    ax.set_xscale("log")
                elif axis == "capacity":
                    ax.set_xscale("log")
                ax.set_title(f"{TASK_NAMES[task]} — {MODEL_NAMES[model]}\nVotes: {votes_str}", fontsize=10)
                ax.legend(fontsize=7, loc="best")
                ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = fig_dir / f"multilaw_{axis}_scaling.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # ════════════════════════════════════════════════════════════════
    # Figure 2: Law-selection heatmap
    # ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Winning law by AICc — across all slices", fontsize=14, y=1.02)

    summary = results.get("summary", [])
    all_laws = sorted(set(LAW_COLORS.keys()))
    law_to_idx = {l: i for i, l in enumerate(all_laws)}

    for ax_idx, axis in enumerate(["data", "resolution", "capacity"]):
        ax = axes[ax_idx]
        axis_summary = [s for s in summary if s["axis"] == axis]

        labels = []
        data_matrix = []
        for s in axis_summary:
            labels.append(f"{TASK_NAMES.get(s['task'], s['task'])}\n{MODEL_NAMES.get(s['model'], s['model'])}")
            row = [0] * len(all_laws)
            for law, count in s.get("law_counts", {}).items():
                if law in law_to_idx:
                    row[law_to_idx[law]] = count
            total = sum(row)
            if total > 0:
                row = [r / total for r in row]
            data_matrix.append(row)

        if not data_matrix:
            continue

        data_matrix = np.array(data_matrix)
        im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_laws)))
        ax.set_xticklabels([l.replace("_", "\n") for l in all_laws], fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{axis.capitalize()} axis", fontsize=12)

        # Annotate with percentages
        for i in range(len(labels)):
            for j in range(len(all_laws)):
                val = data_matrix[i, j]
                if val > 0.05:
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=7, color="white" if val > 0.5 else "black")

    plt.colorbar(im, ax=axes, label="Fraction of slices", shrink=0.8)
    plt.tight_layout()
    path = fig_dir / "multilaw_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ════════════════════════════════════════════════════════════════
    # Figure 3: E_inf (irreducible error) comparison
    # ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Irreducible error floors $E_\\infty$ from best data-scaling fits", fontsize=14)

    data_fits = results["data_fits"]
    for i, task in enumerate(TASKS):
        ax = axes[i]
        for j, model in enumerate(MODELS):
            e_infs = []
            for f in data_fits:
                if f["task"] == task and f["model_name"] == model:
                    bp = f.get("best_params", {})
                    if "E_inf" in bp and bp["E_inf"] > 0:
                        e_infs.append(bp["E_inf"])

            if e_infs:
                positions = [j]
                bp = ax.boxplot([e_infs], positions=positions, widths=0.6,
                               patch_artist=True)
                bp["boxes"][0].set_facecolor(["#e41a1c", "#377eb8", "#4daf4a"][j])
                bp["boxes"][0].set_alpha(0.7)

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS])
        ax.set_ylabel("$E_\\infty$")
        ax.set_title(TASK_NAMES[task])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = fig_dir / "multilaw_e_inf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ════════════════════════════════════════════════════════════════
    # Figure 4: Resolution — all slices overlay to show the trend (or lack thereof)
    # ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle("Resolution scaling — all slices overlaid (normalized)", fontsize=14, y=0.98)

    for i, task in enumerate(TASKS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            mdf = df[(df["task"] == task) & (df["model_name"] == model)]

            for (cap, N_val), sub in mdf.groupby(["capacity_name", "dataset_size"]):
                sub = sub.sort_values("resolution")
                R = sub["resolution"].values
                E = sub["test_rel_l2_mean"].values
                if len(E) < 2:
                    continue
                # Normalize to E(R_min) = 1
                E_norm = E / E[0] if E[0] > 0 else E
                ax.plot(R, E_norm, "o-", alpha=0.3, markersize=3, color="steelblue")

            ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
            ax.set_xlabel("Resolution $R$")
            ax.set_ylabel("$E(R) / E(R_{min})$")
            ax.set_title(f"{TASK_NAMES[task]} — {MODEL_NAMES[model]}")
            ax.grid(True, alpha=0.3)
            # Annotate: fraction where more R helps
            res_fits = [f for f in results["resolution_fits"]
                        if f["task"] == task and f["model_name"] == model]
            n_log = sum(1 for f in res_fits if f.get("best_law") == "logarithmic" and
                        f.get("best_params", {}).get("b", 0) > 0)
            n_lin_neg = sum(1 for f in res_fits if f.get("best_law") == "linear" and
                           f.get("best_params", {}).get("b", 0) < 0)
            n_helps = n_log + n_lin_neg
            n_total = len(res_fits)
            ax.text(0.02, 0.98, f"R helps: {n_helps}/{n_total}", transform=ax.transAxes,
                    va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / "multilaw_resolution_all_slices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ════════════════════════════════════════════════════════════════
    # Figure 5: Akaike weight stacked bar chart
    # ════════════════════════════════════════════════════════════════
    summary = results.get("summary", [])
    all_laws = sorted(set(LAW_COLORS.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Mean Akaike weights per task–model–axis", fontsize=14, y=1.02)

    for ax_idx, axis in enumerate(["data", "resolution", "capacity"]):
        ax = axes[ax_idx]
        axis_summary = [s for s in summary if s["axis"] == axis]
        if not axis_summary:
            continue

        labels = []
        weight_data = {law: [] for law in all_laws}
        for s in axis_summary:
            labels.append(f"{TASK_NAMES.get(s['task'], s['task'])}\n{MODEL_NAMES.get(s['model'], s['model'])}")
            maw = s.get("mean_akaike_weights", {})
            for law in all_laws:
                weight_data[law].append(maw.get(law, 0.0))

        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for law in all_laws:
            vals = np.array(weight_data[law])
            bars = ax.bar(x, vals, bottom=bottom, color=LAW_COLORS.get(law, "#999999"),
                          label=law.replace("_", " "), edgecolor="white", linewidth=0.5)
            # Hatch ambiguous cases (top-2 weights close)
            for i, s in enumerate(axis_summary):
                maw = s.get("mean_akaike_weights", {})
                sorted_w = sorted(maw.values(), reverse=True)
                if len(sorted_w) >= 2 and (sorted_w[0] - sorted_w[1]) < 0.15:
                    bars[i].set_hatch("//")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Mean Akaike weight")
        ax.set_title(f"{axis.capitalize()} axis", fontsize=12)
        ax.set_ylim(0, 1)
        if ax_idx == 2:
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    path = fig_dir / "multilaw_akaike_weights.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ════════════════════════════════════════════════════════════════
    # Figure 6: Bootstrap law-selection stability
    # ════════════════════════════════════════════════════════════════
    bootstrap = results.get("bootstrap_law_selection", [])
    if bootstrap:
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle("Bootstrap law-selection stability (B=500)", fontsize=14, y=1.02)

        for ax_idx, axis in enumerate(["data", "resolution", "capacity"]):
            ax = axes[ax_idx]
            axis_boot = [b for b in bootstrap if b["axis"] == axis]
            if not axis_boot:
                continue

            labels = []
            probs = []
            dom_laws = []
            for b in axis_boot:
                labels.append(f"{TASK_NAMES.get(b['task'], b['task'])}\n{MODEL_NAMES.get(b['model'], b['model'])}")
                probs.append(b.get("dominant_prob", 0))
                dom_laws.append(b.get("dominant_law", "?"))

            y = np.arange(len(labels))
            colors = []
            for p in probs:
                if p >= 0.8:
                    colors.append("#2ca02c")   # solid green — confident
                elif p >= 0.6:
                    colors.append("#ff7f0e")   # orange — moderate
                else:
                    colors.append("#d62728")   # red — ambiguous

            ax.barh(y, probs, color=colors, edgecolor="black", linewidth=0.5)
            for i, (p, law) in enumerate(zip(probs, dom_laws)):
                ax.text(p + 0.02, i, f"{law} ({p:.0%})", va="center", fontsize=8)

            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlim(0, 1.15)
            ax.set_xlabel("Bootstrap winner probability")
            ax.set_title(f"{axis.capitalize()} axis", fontsize=12)
            ax.axvline(0.8, color="green", ls="--", alpha=0.4, label=">80% confident")
            ax.axvline(0.6, color="orange", ls="--", alpha=0.4, label="60-80% moderate")
            if ax_idx == 0:
                ax.legend(fontsize=7, loc="lower right")

        plt.tight_layout()
        path = fig_dir / "multilaw_bootstrap_stability.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # ════════════════════════════════════════════════════════════════
    # Figure 7: Restricted-set comparison table (resolution axis)
    # ════════════════════════════════════════════════════════════════
    restricted_path = Path("results/multilaw_fits_restricted.json")
    if restricted_path.exists():
        with open(restricted_path) as f:
            restricted = json.load(f)

        full_summary = [s for s in summary if s["axis"] == "resolution"]
        rest_summary = restricted.get("summary", [])

        if full_summary and rest_summary:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle("Full (6-law) vs Restricted (3-law) — Resolution axis", fontsize=14)

            labels = []
            full_laws = []
            rest_laws = []
            for fs in full_summary:
                key = (fs["task"], fs["model"])
                rs = next((r for r in rest_summary
                           if r["task"] == fs["task"] and r["model"] == fs["model"]), None)
                if rs is None:
                    continue
                labels.append(f"{TASK_NAMES.get(fs['task'], fs['task'])}\n{MODEL_NAMES.get(fs['model'], fs['model'])}")
                full_laws.append(fs["dominant_law"])
                rest_laws.append(rs["dominant_law"])

            # Table-like figure
            cell_text = []
            cell_colors = []
            for fl, rl in zip(full_laws, rest_laws):
                match = "✓" if fl == rl else "✗"
                cell_text.append([fl, rl, match])
                color = "#d4edda" if fl == rl else "#f8d7da"
                cell_colors.append(["white", "white", color])

            table = ax.table(
                cellText=cell_text,
                rowLabels=labels,
                colLabels=["Full (6-law)", "Restricted (3-law)", "Agree?"],
                cellColours=cell_colors,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.8)
            ax.axis("off")

            plt.tight_layout()
            path = fig_dir / "multilaw_restricted_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")

    print("\nDone — all multi-law figures saved to figures/")


if __name__ == "__main__":
    main()
