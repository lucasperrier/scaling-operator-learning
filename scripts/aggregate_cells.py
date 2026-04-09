"""Phase 2b: per-cell aggregation per PROTOCOL_V2.md §4.

Reads runs/runs_aggregate.csv and emits two canonical CSVs that
downstream phases (3, 4, 5) read:

  results/cell_stats_diagonal.csv  — one row per (task, model, capacity, N, R_train)
  results/cell_stats_crossres.csv  — one row per (task, model, capacity, N, R_train, R_eval)
                                     joined to its diagonal parent's convergence info

This script does NOT apply the §5 inclusion/exclusion rules. It emits
flags (`is_r48`, `parent_converged`) so downstream phases can filter
according to their own purpose. The rules in PROTOCOL_V2 §5 are
*reading* rules, not *aggregation* rules — the cells still exist, the
appendix narrative still uses them.

What is computed (PROTOCOL_V2 §4.1, §4.2, §4.3):

Per diagonal cell:
  n_runs, n_success, n_failed
  convergence_rate, divergence_rate, instability_rate
  median_test_rel_l2, iqr_test_rel_l2, seed_cv_test_rel_l2,
  worst_test_rel_l2, best_test_rel_l2
  median_best_epoch, median_runtime_seconds
  parameter_count
  data_seeds_observed, train_seeds_observed (semicolon-joined)
  is_r48 (bool flag for the data-generator-bug exclusion)

Per cross-res cell:
  n_runs (success only — cross-res rows have no `status` other than success)
  median_test_rel_l2, iqr_test_rel_l2, seed_cv_test_rel_l2,
  worst_test_rel_l2, best_test_rel_l2
  parent_convergence_rate, parent_converged (joined from diagonal)
  parent_median_test_rel_l2 (joined; the K[i,i] baseline at R_eval=R_train)
  degradation_vs_diagonal (this cell's median / parent_median)
  is_r48_train (bool: R_train == 48; the parent had the data bug)

The two CSVs together are the canonical input for Phases 3, 4, 5.
No phase reads runs_aggregate.csv directly after this point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = REPO_ROOT / "runs" / "runs_aggregate.csv"
RESULTS_DIR = REPO_ROOT / "results"
DIAG_OUT = RESULTS_DIR / "cell_stats_diagonal.csv"
CROSS_OUT = RESULTS_DIR / "cell_stats_crossres.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_iqr(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def _safe_cv(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    mean = float(np.mean(values))
    if mean == 0.0:
        return float("nan")
    return float(np.std(values, ddof=1) / abs(mean))


def aggregate_diagonal(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, model, capacity, N, R_train) diagonal cell."""
    diag = runs[runs["eval_resolution"].isna()].copy()
    # Boolean coercions: pandas reads 'True'/'False' strings as object dtype.
    diag["diverged_bool"] = diag["diverged"].fillna(False).astype(bool)
    diag["nan_bool"] = diag["nan_detected"].fillna(False).astype(bool)
    diag["success_bool"] = diag["status"] == "success"
    # `instability` per protocol §4.1: nan OR diverged OR best_epoch == -1.
    best_epoch_minus1 = diag["best_epoch"].fillna(-1).astype(int) == -1
    diag["instability_bool"] = (
        diag["diverged_bool"] | diag["nan_bool"] | best_epoch_minus1
    )

    group_cols = [
        "task",
        "model_name",
        "capacity_name",
        "dataset_size",
        "resolution",
    ]
    rows = []
    for keys, g in diag.groupby(group_cols):
        n_runs = len(g)
        n_success = int(g["success_bool"].sum())
        n_failed = n_runs - n_success
        convergence_rate = n_success / n_runs if n_runs else 0.0
        divergence_rate = float(g["diverged_bool"].mean()) if n_runs else 0.0
        instability_rate = float(g["instability_bool"].mean()) if n_runs else 0.0

        successes = g[g["success_bool"]]
        if not successes.empty:
            errs = successes["test_rel_l2"].to_numpy(dtype=float)
            errs = errs[~np.isnan(errs)]
        else:
            errs = np.array([], dtype=float)
        median_err = float(np.median(errs)) if len(errs) else float("nan")
        iqr_err = _safe_iqr(errs)
        cv_err = _safe_cv(errs)
        worst_err = float(np.max(errs)) if len(errs) else float("nan")
        best_err = float(np.min(errs)) if len(errs) else float("nan")

        if not successes.empty:
            best_epoch_vals = successes["best_epoch"].dropna().to_numpy(dtype=float)
            runtime_vals = (
                successes["runtime_seconds"].dropna().to_numpy(dtype=float)
            )
        else:
            best_epoch_vals = np.array([], dtype=float)
            runtime_vals = np.array([], dtype=float)
        median_best_epoch = (
            float(np.median(best_epoch_vals)) if len(best_epoch_vals) else float("nan")
        )
        median_runtime = (
            float(np.median(runtime_vals)) if len(runtime_vals) else float("nan")
        )

        # parameter_count is constant within a cell (same model+capacity+R), but
        # take the max in case of any row inconsistency.
        param_count = int(g["parameter_count"].max())

        data_seeds = sorted(int(s) for s in g["data_seed"].dropna().unique())
        train_seeds = sorted(int(s) for s in g["train_seed"].dropna().unique())

        rows.append(
            {
                "task": keys[0],
                "model_name": keys[1],
                "capacity_name": keys[2],
                "dataset_size": int(keys[3]),
                "resolution": int(keys[4]),
                "parameter_count": param_count,
                "n_runs": n_runs,
                "n_success": n_success,
                "n_failed": n_failed,
                "convergence_rate": convergence_rate,
                "divergence_rate": divergence_rate,
                "instability_rate": instability_rate,
                "median_test_rel_l2": median_err,
                "iqr_test_rel_l2": iqr_err,
                "seed_cv_test_rel_l2": cv_err,
                "worst_test_rel_l2": worst_err,
                "best_test_rel_l2": best_err,
                "median_best_epoch": median_best_epoch,
                "median_runtime_seconds": median_runtime,
                "n_data_seeds": len(data_seeds),
                "n_train_seeds": len(train_seeds),
                "data_seeds_observed": ";".join(str(s) for s in data_seeds),
                "train_seeds_observed": ";".join(str(s) for s in train_seeds),
                "is_r48": int(keys[4]) == 48,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def aggregate_crossres(
    runs: pd.DataFrame, diag_stats: pd.DataFrame
) -> pd.DataFrame:
    """One row per cross-res cell, joined to its diagonal parent."""
    cross = runs[runs["eval_resolution"].notna()].copy()
    cross["eval_resolution"] = cross["eval_resolution"].astype(int)
    cross["success_bool"] = cross["status"] == "success"

    # Build a quick lookup of diagonal parents:
    #   parent_key = (task, model_name, capacity_name, dataset_size, resolution)
    # The cross-res "parent" is the diagonal training cell at the SAME R_train.
    diag_lookup = diag_stats.set_index(
        ["task", "model_name", "capacity_name", "dataset_size", "resolution"]
    )

    group_cols = [
        "task",
        "model_name",
        "capacity_name",
        "dataset_size",
        "resolution",
        "eval_resolution",
    ]
    rows = []
    for keys, g in cross.groupby(group_cols):
        successes = g[g["success_bool"]]
        if not successes.empty:
            errs = successes["test_rel_l2"].to_numpy(dtype=float)
            errs = errs[~np.isnan(errs)]
        else:
            errs = np.array([], dtype=float)
        median_err = float(np.median(errs)) if len(errs) else float("nan")
        iqr_err = _safe_iqr(errs)
        cv_err = _safe_cv(errs)
        worst_err = float(np.max(errs)) if len(errs) else float("nan")
        best_err = float(np.min(errs)) if len(errs) else float("nan")

        parent_key = (keys[0], keys[1], keys[2], keys[3], keys[4])
        try:
            parent_row = diag_lookup.loc[parent_key]
            parent_conv = float(parent_row["convergence_rate"])
            parent_median = float(parent_row["median_test_rel_l2"])
        except KeyError:
            parent_conv = float("nan")
            parent_median = float("nan")

        if (
            not np.isnan(median_err)
            and not np.isnan(parent_median)
            and parent_median != 0.0
        ):
            degradation = median_err / parent_median
        else:
            degradation = float("nan")

        rows.append(
            {
                "task": keys[0],
                "model_name": keys[1],
                "capacity_name": keys[2],
                "dataset_size": int(keys[3]),
                "resolution_train": int(keys[4]),
                "resolution_eval": int(keys[5]),
                "n_runs": len(g),
                "n_success": int(g["success_bool"].sum()),
                "median_test_rel_l2": median_err,
                "iqr_test_rel_l2": iqr_err,
                "seed_cv_test_rel_l2": cv_err,
                "worst_test_rel_l2": worst_err,
                "best_test_rel_l2": best_err,
                "parent_convergence_rate": parent_conv,
                "parent_converged": parent_conv == 1.0,
                "parent_median_test_rel_l2": parent_median,
                "degradation_vs_diagonal": degradation,
                "is_r48_train": int(keys[4]) == 48,
            }
        )
    sort_cols = [
        "task",
        "model_name",
        "capacity_name",
        "dataset_size",
        "resolution_train",
        "resolution_eval",
    ]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = pd.read_csv(RUNS_CSV)

    diag_stats = aggregate_diagonal(runs)
    cross_stats = aggregate_crossres(runs, diag_stats)

    diag_stats.to_csv(DIAG_OUT, index=False)
    cross_stats.to_csv(CROSS_OUT, index=False)

    # Print a small summary so the run output is informative without bloating
    # the report.
    print(f"Wrote {DIAG_OUT}: {len(diag_stats)} diagonal cells")
    print(f"Wrote {CROSS_OUT}: {len(cross_stats)} cross-res cells")
    print()
    print("Diagonal cell counts by task / model:")
    print(
        diag_stats.groupby(["task", "model_name"]).size().to_string()
    )
    print()
    n_r48 = int(diag_stats["is_r48"].sum())
    print(
        f"Diagonal cells flagged is_r48: {n_r48} "
        f"(should match VALIDATION_REPORT §5: 7 caps × 7 N values × 3 models = 147)"
    )
    n_unstable_main = int(
        ((diag_stats["divergence_rate"] > 0.10) & (~diag_stats["is_r48"])).sum()
    )
    print(
        f"Diagonal cells with divergence_rate > 10% AND not R=48: {n_unstable_main}"
    )
    n_cross_parent_failed = int((~cross_stats["parent_converged"]).sum())
    print(
        f"Cross-res cells whose diagonal parent did not fully converge: "
        f"{n_cross_parent_failed} / {len(cross_stats)} "
        f"({100 * n_cross_parent_failed / len(cross_stats):.1f}%)"
    )
    n_cross_r48 = int(cross_stats["is_r48_train"].sum())
    print(f"Cross-res cells flagged is_r48_train: {n_cross_r48}")


if __name__ == "__main__":
    main()
