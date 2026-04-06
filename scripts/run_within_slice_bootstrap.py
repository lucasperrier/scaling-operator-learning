"""Tier 2.2 — Within-slice data-level bootstrap.

For each slice, resample the n data points with replacement (B=500),
refit all 6 laws, reselect the AICc winner each time.  This propagates
fitting uncertainty into law-selection uncertainty, complementing the
existing inter-slice (vote-resampling) bootstrap.

Output: results/within_slice_bootstrap.json
"""
from __future__ import annotations

import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_multilaw_analysis import fit_all_laws, ALL_LAW_NAMES

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _within_slice_bootstrap(X, E, B=500, rng_seed=42):
    """Resample (X, E) with replacement B times, refit & reselect AICc winner.

    Returns:
        dict with per-law win fraction across bootstrap replicates.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(X)
    if n < 3:
        return {"winner_fracs": {}, "n_boot": B, "n_points": n}

    win_counts = defaultdict(int)
    n_success = 0

    for _ in range(B):
        idxs = rng.randint(0, n, size=n)
        X_b = X[idxs]
        E_b = E[idxs]
        # Skip degenerate resamples (all same X value)
        if len(np.unique(X_b)) < 2:
            continue
        fits = fit_all_laws(X_b, E_b)
        if fits:
            win_counts[fits[0]["law"]] += 1
            n_success += 1

    if n_success == 0:
        return {"winner_fracs": {}, "n_boot": B, "n_success": 0, "n_points": n}

    winner_fracs = {law: win_counts[law] / n_success for law in win_counts}
    dominant = max(winner_fracs, key=winner_fracs.get)

    return {
        "winner_fracs": winner_fracs,
        "n_boot": B,
        "n_success": n_success,
        "n_points": n,
        "dominant_law": dominant,
        "dominant_frac": winner_fracs[dominant],
    }


def main():
    t0 = time.time()
    df = pd.read_csv("runs/grouped_metrics_train.csv")
    df["test_rel_l2_mean"] = pd.to_numeric(df["test_rel_l2_mean"], errors="coerce")
    df = df.dropna(subset=["test_rel_l2_mean"])
    print(f"Loaded {len(df)} training groups")

    all_slice_results = []
    axis_summary = []

    axis_configs = [
        ("data", "dataset_size", ["capacity_name", "resolution"]),
        ("capacity", "parameter_count", ["dataset_size", "resolution"]),
        ("resolution", "resolution", ["capacity_name", "dataset_size"]),
    ]

    for task in sorted(df["task"].unique()):
        tdf = df[df["task"] == task]
        for model in sorted(tdf["model_name"].unique()):
            mdf = tdf[tdf["model_name"] == model]

            for axis, x_col, group_cols in axis_configs:
                slice_boots = []

                for keys, sub in mdf.groupby(group_cols):
                    sub = sub.sort_values(x_col)
                    X = sub[x_col].values.astype(float)
                    E = sub["test_rel_l2_mean"].values.astype(float)

                    boot = _within_slice_bootstrap(X, E, B=500)

                    # Also record the original AICc winner for comparison
                    orig_fits = fit_all_laws(X, E)
                    orig_winner = orig_fits[0]["law"] if orig_fits else None

                    rec = {
                        "task": task,
                        "model": model,
                        "axis": axis,
                        "slice_keys": {c: (v.item() if hasattr(v, "item") else v)
                                       for c, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))},
                        "original_winner": orig_winner,
                        **boot,
                    }
                    all_slice_results.append(rec)
                    slice_boots.append(rec)

                # Aggregate: mean within-slice winner fraction per law
                if slice_boots:
                    law_fracs = defaultdict(list)
                    for sb in slice_boots:
                        for law in ALL_LAW_NAMES:
                            law_fracs[law].append(sb.get("winner_fracs", {}).get(law, 0.0))

                    mean_fracs = {law: float(np.mean(vs)) for law, vs in law_fracs.items()}
                    dominant = max(mean_fracs, key=mean_fracs.get)

                    # How often does the original AICc winner survive within-slice bootstrap?
                    n_stable = 0
                    n_total = 0
                    for sb in slice_boots:
                        if sb.get("original_winner") and sb.get("dominant_law"):
                            n_total += 1
                            if sb["original_winner"] == sb["dominant_law"]:
                                n_stable += 1
                    stability = n_stable / n_total if n_total > 0 else 0.0

                    summary = {
                        "task": task,
                        "model": model,
                        "axis": axis,
                        "n_slices": len(slice_boots),
                        "mean_within_slice_fracs": mean_fracs,
                        "dominant_law": dominant,
                        "dominant_frac": mean_fracs[dominant],
                        "original_winner_stability": stability,
                        "n_stable": n_stable,
                        "n_total": n_total,
                    }
                    axis_summary.append(summary)

                    print(f"  {task:20s} {model:16s} {axis:12s}: "
                          f"ws_dominant={dominant:14s} ws_frac={mean_fracs[dominant]:.2f}  "
                          f"orig_stable={stability:.0%}")

    # Print summary table
    print(f"\n{'='*100}")
    print("WITHIN-SLICE BOOTSTRAP SUMMARY")
    print(f"{'='*100}")
    print(f"{'Task':<22s} {'Model':<16s} {'Axis':<12s} {'WS Dominant':<16s} "
          f"{'WS Frac':<10s} {'Orig Stable':<12s}")
    print("-" * 100)
    for s in axis_summary:
        print(f"{s['task']:<22s} {s['model']:<16s} {s['axis']:<12s} "
              f"{s['dominant_law']:<16s} {s['dominant_frac']:<10.2f} "
              f"{s['original_winner_stability']:<12.0%}")

    # Compare with inter-slice bootstrap
    print(f"\n{'='*100}")
    print("KEY INSIGHT: Within-slice vs inter-slice bootstrap comparison")
    print(f"{'='*100}")
    print("Within-slice bootstrap propagates fitting uncertainty from the n data")
    print("points per slice. If within-slice dominant fractions are much lower than")
    print("inter-slice bootstrap probabilities, the per-slice data is insufficient")
    print("to confidently distinguish law families.\n")
    for s in axis_summary:
        if s["dominant_frac"] < 0.5:
            print(f"  ⚠  {s['task']:20s} {s['model']:16s} {s['axis']:12s}: "
                  f"ws_frac={s['dominant_frac']:.2f} — law selection is highly uncertain at the slice level")

    # Save
    out = Path("results")
    out.mkdir(exist_ok=True)
    save = {
        "summary": axis_summary,
        "per_slice": all_slice_results,
    }
    with open(out / "within_slice_bootstrap.json", "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nSaved → results/within_slice_bootstrap.json")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
