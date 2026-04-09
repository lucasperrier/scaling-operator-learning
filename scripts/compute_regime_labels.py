"""Phase 3a: local sensitivities + regime labels per PROTOCOL_V2 §6.

Reads results/cell_stats_diagonal.csv and emits:

  results/cell_regimes.csv  — one row per diagonal cell with S_N, S_R,
                              S_C, primary regime label, and the diagnostic
                              fields used to derive it.
  results/regime_calibration.json — the empirically-calibrated thresholds
                              (THRESH_HIGH, THRESH_LOW, THRESH_DIV, THRESH_CV)
                              and the Phase 3 calibration trace (quantiles of
                              the empirical |S_N|, |S_R|, |S_C| distributions).

This script implements PROTOCOL_V2 §6.1 (local sensitivities) and §6.3
(label rules). It also performs the §6.2 calibration step:

- THRESH_DIV = 0.10 (locked from PROTOCOL §6.2; 0 main-grid cells exceed it)
- THRESH_CV  = 0.50 (locked from PROTOCOL §6.2; 81 main-grid cells exceed it)
- THRESH_HIGH and THRESH_LOW are calibrated empirically:
    THRESH_HIGH = 75th percentile of |S| across the three axes pooled
    THRESH_LOW  = 25th percentile of |S| across the three axes pooled
  These quantile choices are documented in regime_calibration.json so the
  decision is auditable.

R=48 cells are excluded from the sensitivity computation entirely (they
have no successful runs and would corrupt log-log differences). Fill-in
cells are excluded from labeling (regime = 'ambiguous').
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAG_CSV = REPO_ROOT / "results" / "cell_stats_diagonal.csv"
OUT_CSV = REPO_ROOT / "results" / "cell_regimes.csv"
OUT_JSON = REPO_ROOT / "results" / "regime_calibration.json"

# Capacity ordering matching src/scaling_operator_learning/models/__init__.py
# (canonical small → large order). parameter_count is the actual numerical
# coordinate used for the log-log derivative, but the ordering is needed to
# pick neighbors.
CAPACITY_ORDER = [
    "tiny",
    "small",
    "small-med",
    "medium",
    "med-large",
    "large",
    "xlarge",
]
CAPACITY_RANK = {name: i for i, name in enumerate(CAPACITY_ORDER)}


# ─────────────────────────────────────────────────────────────────────────────
# Local sensitivity computation (PROTOCOL_V2 §6.1)
# ─────────────────────────────────────────────────────────────────────────────


def _log_slope(x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> float:
    """Discrete log-log slope between two points. Returns NaN on bad input."""
    if (
        x_lo is None
        or x_hi is None
        or y_lo is None
        or y_hi is None
        or x_lo <= 0
        or x_hi <= 0
        or y_lo <= 0
        or y_hi <= 0
        or x_lo == x_hi
    ):
        return float("nan")
    return float((np.log(y_hi) - np.log(y_lo)) / (np.log(x_hi) - np.log(x_lo)))


def _neighbor_value(
    df: pd.DataFrame,
    key_cols: list[str],
    key_vals: tuple,
    axis_col: str,
    axis_val: float,
    target_col: str = "median_test_rel_l2",
) -> float | None:
    """Look up the cell at the given (key_cols, axis_val) and return target_col.

    Returns None if the cell does not exist or if its target value is NaN /
    non-positive.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in zip(key_cols, key_vals):
        mask &= df[col] == val
    mask &= df[axis_col] == axis_val
    sub = df[mask]
    if sub.empty:
        return None
    val = float(sub[target_col].iloc[0])
    if np.isnan(val) or val <= 0:
        return None
    return val


def compute_axis_sensitivity(
    df: pd.DataFrame,
    cell_row: pd.Series,
    fixed_cols: list[str],
    axis_col: str,
    axis_values_sorted: list[float],
    log_axis_value_fn,
) -> float:
    """Compute log-log local slope along `axis_col` at the given cell.

    Uses centered finite differences when both neighbors exist, one-sided
    when only one does, NaN when neither does. The log-axis-value function
    handles the case where the axis isn't the column itself (e.g. for
    capacity, the axis is parameter_count).
    """
    fixed_vals = tuple(cell_row[c] for c in fixed_cols)
    cur_axis = cell_row[axis_col]
    if cur_axis not in axis_values_sorted:
        return float("nan")
    idx = axis_values_sorted.index(cur_axis)

    cur_log_x = log_axis_value_fn(cell_row, cur_axis)
    cur_y = float(cell_row["median_test_rel_l2"])
    if np.isnan(cur_y) or cur_y <= 0:
        return float("nan")

    lo_axis = axis_values_sorted[idx - 1] if idx > 0 else None
    hi_axis = (
        axis_values_sorted[idx + 1] if idx < len(axis_values_sorted) - 1 else None
    )

    lo_y = (
        _neighbor_value(df, fixed_cols, fixed_vals, axis_col, lo_axis)
        if lo_axis is not None
        else None
    )
    hi_y = (
        _neighbor_value(df, fixed_cols, fixed_vals, axis_col, hi_axis)
        if hi_axis is not None
        else None
    )

    if lo_y is not None and hi_y is not None:
        # Centered: slope between (lo, hi).
        # For capacity, the log axis values come from the lo/hi *cells*, not
        # from a single fn(cell) call.
        lo_row = df[
            (df[fixed_cols[0]] == fixed_vals[0])
            & (df[fixed_cols[1]] == fixed_vals[1])
            & (df[fixed_cols[2]] == fixed_vals[2])
            & (df[fixed_cols[3]] == fixed_vals[3])
            & (df[axis_col] == lo_axis)
        ].iloc[0]
        hi_row = df[
            (df[fixed_cols[0]] == fixed_vals[0])
            & (df[fixed_cols[1]] == fixed_vals[1])
            & (df[fixed_cols[2]] == fixed_vals[2])
            & (df[fixed_cols[3]] == fixed_vals[3])
            & (df[axis_col] == hi_axis)
        ].iloc[0]
        lo_log_x = log_axis_value_fn(lo_row, lo_axis)
        hi_log_x = log_axis_value_fn(hi_row, hi_axis)
        if (
            lo_log_x is None
            or hi_log_x is None
            or np.isnan(lo_log_x)
            or np.isnan(hi_log_x)
            or lo_log_x == hi_log_x
        ):
            return float("nan")
        return float((np.log(hi_y) - np.log(lo_y)) / (hi_log_x - lo_log_x))

    if hi_y is not None:
        hi_row = df[
            (df[fixed_cols[0]] == fixed_vals[0])
            & (df[fixed_cols[1]] == fixed_vals[1])
            & (df[fixed_cols[2]] == fixed_vals[2])
            & (df[fixed_cols[3]] == fixed_vals[3])
            & (df[axis_col] == hi_axis)
        ].iloc[0]
        hi_log_x = log_axis_value_fn(hi_row, hi_axis)
        if hi_log_x is None or cur_log_x is None or hi_log_x == cur_log_x:
            return float("nan")
        return float((np.log(hi_y) - np.log(cur_y)) / (hi_log_x - cur_log_x))

    if lo_y is not None:
        lo_row = df[
            (df[fixed_cols[0]] == fixed_vals[0])
            & (df[fixed_cols[1]] == fixed_vals[1])
            & (df[fixed_cols[2]] == fixed_vals[2])
            & (df[fixed_cols[3]] == fixed_vals[3])
            & (df[axis_col] == lo_axis)
        ].iloc[0]
        lo_log_x = log_axis_value_fn(lo_row, lo_axis)
        if lo_log_x is None or cur_log_x is None or cur_log_x == lo_log_x:
            return float("nan")
        return float((np.log(cur_y) - np.log(lo_y)) / (cur_log_x - lo_log_x))

    return float("nan")


def compute_all_sensitivities(diag: pd.DataFrame) -> pd.DataFrame:
    """Add S_N, S_R, S_C columns to the diagonal stats dataframe.

    Operates only on cells that are not R=48 and have at least one successful
    seed (median_test_rel_l2 finite). The neighbor lookups are restricted to
    the same task / model / capacity for S_N and S_R, and to the same task /
    model / N / R for S_C.
    """
    diag = diag.copy()
    # Pre-filter: cells we're willing to compute sensitivity FOR.
    valid_mask = (~diag["is_r48"]) & (diag["median_test_rel_l2"].notna())

    s_n_vals = np.full(len(diag), np.nan)
    s_r_vals = np.full(len(diag), np.nan)
    s_c_vals = np.full(len(diag), np.nan)

    # Pre-build per-slice axis value lists for efficiency.
    n_axis_per_slice = {}
    r_axis_per_slice = {}
    c_axis_per_n_r_slice = {}
    for keys, g in diag[valid_mask].groupby(["task", "model_name", "capacity_name"]):
        n_axis_per_slice[keys] = sorted(int(n) for n in g["dataset_size"].unique())
        r_axis_per_slice[keys] = sorted(int(r) for r in g["resolution"].unique())
    for keys, g in diag[valid_mask].groupby(
        ["task", "model_name", "dataset_size", "resolution"]
    ):
        # Capacities present at this (task, model, N, R), ranked by name.
        caps_present = sorted(
            g["capacity_name"].unique(), key=lambda c: CAPACITY_RANK.get(c, 99)
        )
        c_axis_per_n_r_slice[keys] = caps_present

    log_n = lambda row, val: float(np.log(val))  # noqa: E731
    log_r = lambda row, val: float(np.log(val))  # noqa: E731
    log_c = lambda row, val: (
        float(np.log(row["parameter_count"])) if row["parameter_count"] > 0 else None
    )

    for i, row in diag.iterrows():
        if not valid_mask.iloc[i]:
            continue
        slice_key = (row["task"], row["model_name"], row["capacity_name"])
        n_axis = n_axis_per_slice.get(slice_key, [])
        r_axis = r_axis_per_slice.get(slice_key, [])
        if not n_axis or not r_axis:
            continue

        # S_N: vary N at fixed (task, model, capacity, R).
        s_n = compute_axis_sensitivity(
            diag,
            row,
            fixed_cols=["task", "model_name", "capacity_name", "resolution"],
            axis_col="dataset_size",
            axis_values_sorted=n_axis,
            log_axis_value_fn=log_n,
        )
        # S_R: vary R at fixed (task, model, capacity, N).
        s_r = compute_axis_sensitivity(
            diag,
            row,
            fixed_cols=["task", "model_name", "capacity_name", "dataset_size"],
            axis_col="resolution",
            axis_values_sorted=r_axis,
            log_axis_value_fn=log_r,
        )
        # S_C: vary capacity at fixed (task, model, N, R).
        c_slice_key = (
            row["task"],
            row["model_name"],
            row["dataset_size"],
            row["resolution"],
        )
        c_axis = c_axis_per_n_r_slice.get(c_slice_key, [])
        if c_axis and row["capacity_name"] in c_axis:
            s_c = compute_axis_sensitivity(
                diag,
                row,
                fixed_cols=["task", "model_name", "dataset_size", "resolution"],
                axis_col="capacity_name",
                axis_values_sorted=c_axis,
                log_axis_value_fn=log_c,
            )
        else:
            s_c = float("nan")

        s_n_vals[i] = s_n
        s_r_vals[i] = s_r
        s_c_vals[i] = s_c

    diag["s_n"] = s_n_vals
    diag["s_r"] = s_r_vals
    diag["s_c"] = s_c_vals
    return diag


# ─────────────────────────────────────────────────────────────────────────────
# Threshold calibration (PROTOCOL_V2 §6.2)
# ─────────────────────────────────────────────────────────────────────────────


def calibrate_thresholds(diag: pd.DataFrame) -> dict:
    """Empirical calibration: pool |S| across N, R, C axes; pick 25/75 pcts.

    Locked from PROTOCOL §6.2:
        THRESH_DIV = 0.10
        THRESH_CV  = 0.50
    Empirical:
        THRESH_LOW  = 25th percentile of |S| pooled
        THRESH_HIGH = 75th percentile of |S| pooled
    """
    s_pool = pd.concat(
        [diag["s_n"].abs(), diag["s_r"].abs(), diag["s_c"].abs()],
        ignore_index=True,
    ).dropna()
    quantiles = {
        "p10": float(s_pool.quantile(0.10)),
        "p25": float(s_pool.quantile(0.25)),
        "p50": float(s_pool.quantile(0.50)),
        "p75": float(s_pool.quantile(0.75)),
        "p90": float(s_pool.quantile(0.90)),
    }
    return {
        "THRESH_DIV": 0.10,
        "THRESH_CV": 0.50,
        "THRESH_LOW": quantiles["p25"],
        "THRESH_HIGH": quantiles["p75"],
        "calibration_method": "pooled |S_N|, |S_R|, |S_C| across all non-R48 main-grid cells; LOW = p25, HIGH = p75",
        "n_observations_pooled": int(len(s_pool)),
        "pooled_quantiles": quantiles,
        "per_axis_quantiles": {
            "abs_s_n": {
                "p25": float(diag["s_n"].abs().quantile(0.25)),
                "p50": float(diag["s_n"].abs().quantile(0.50)),
                "p75": float(diag["s_n"].abs().quantile(0.75)),
            },
            "abs_s_r": {
                "p25": float(diag["s_r"].abs().quantile(0.25)),
                "p50": float(diag["s_r"].abs().quantile(0.50)),
                "p75": float(diag["s_r"].abs().quantile(0.75)),
            },
            "abs_s_c": {
                "p25": float(diag["s_c"].abs().quantile(0.25)),
                "p50": float(diag["s_c"].abs().quantile(0.50)),
                "p75": float(diag["s_c"].abs().quantile(0.75)),
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Label assignment (PROTOCOL_V2 §6.3)
# ─────────────────────────────────────────────────────────────────────────────


def assign_label(row: pd.Series, thresh: dict) -> str:
    if row["is_r48"]:
        return "data_bug_r48"
    # Unstable takes precedence.
    div_unstable = (
        not np.isnan(row["divergence_rate"])
        and row["divergence_rate"] > thresh["THRESH_DIV"]
    )
    cv_unstable = (
        not np.isnan(row["seed_cv_test_rel_l2"])
        and row["seed_cv_test_rel_l2"] > thresh["THRESH_CV"]
    )
    inst_unstable = (
        not np.isnan(row["instability_rate"])
        and row["instability_rate"] > thresh["THRESH_DIV"]
    )
    if div_unstable or cv_unstable or inst_unstable:
        return "unstable"

    s_n, s_r, s_c = row["s_n"], row["s_r"], row["s_c"]
    if any(np.isnan(x) for x in (s_n, s_r, s_c)):
        return "ambiguous"

    abs_s_n, abs_s_r, abs_s_c = abs(s_n), abs(s_r), abs(s_c)
    high = thresh["THRESH_HIGH"]
    low = thresh["THRESH_LOW"]

    n_high = abs_s_n > high
    r_high = abs_s_r > high
    c_high = abs_s_c > high

    if abs_s_n < low and abs_s_r < low and abs_s_c < low:
        return "saturated"
    if n_high and not r_high and not c_high:
        return "data_limited"
    if r_high and not n_high and not c_high:
        return "resolution_sensitive"
    if c_high and not n_high and not r_high:
        return "capacity_limited"
    return "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# Top-level
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    diag = pd.read_csv(DIAG_CSV)
    print(f"Loaded {len(diag)} diagonal cells.")

    diag = compute_all_sensitivities(diag)
    print(
        f"Sensitivities computed: "
        f"S_N defined in {diag['s_n'].notna().sum()} cells, "
        f"S_R in {diag['s_r'].notna().sum()}, "
        f"S_C in {diag['s_c'].notna().sum()}."
    )

    thresh = calibrate_thresholds(diag)
    print()
    print("Calibrated thresholds (PROTOCOL §6.2):")
    print(f"  THRESH_DIV  = {thresh['THRESH_DIV']:.4f}  (locked)")
    print(f"  THRESH_CV   = {thresh['THRESH_CV']:.4f}  (locked)")
    print(f"  THRESH_LOW  = {thresh['THRESH_LOW']:.4f}  (empirical p25)")
    print(f"  THRESH_HIGH = {thresh['THRESH_HIGH']:.4f}  (empirical p75)")
    print()
    print("Pooled |S| quantiles:")
    for q, v in thresh["pooled_quantiles"].items():
        print(f"  {q}: {v:.4f}")
    print()
    print("Per-axis |S| medians:")
    for axis, qs in thresh["per_axis_quantiles"].items():
        print(f"  {axis}: p25={qs['p25']:.4f}  p50={qs['p50']:.4f}  p75={qs['p75']:.4f}")

    diag["regime"] = diag.apply(lambda r: assign_label(r, thresh), axis=1)
    print()
    print("Regime label counts (all cells):")
    print(diag["regime"].value_counts().to_string())
    print()
    print("Regime label counts (excluding data_bug_r48 and ambiguous):")
    main_grid = diag[~diag["regime"].isin(["data_bug_r48", "ambiguous"])]
    print(main_grid["regime"].value_counts().to_string())
    print()
    print("Per (task, model) regime counts:")
    print(
        main_grid.groupby(["task", "model_name"])["regime"]
        .value_counts()
        .unstack(fill_value=0)
        .to_string()
    )

    out_cols = [
        "task",
        "model_name",
        "capacity_name",
        "dataset_size",
        "resolution",
        "parameter_count",
        "n_runs",
        "n_success",
        "convergence_rate",
        "divergence_rate",
        "instability_rate",
        "seed_cv_test_rel_l2",
        "median_test_rel_l2",
        "s_n",
        "s_r",
        "s_c",
        "is_r48",
        "regime",
    ]
    diag[out_cols].to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(thresh, indent=2))
    print()
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
