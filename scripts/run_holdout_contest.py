"""Tier 2.1 — Held-out prediction contest.

For each task–model–axis, hold out ~20% of grid configurations from the
slice-level analysis, fit (a) classic power-law model and (b) multi-law
(AICc-best) on the remaining 80%, and evaluate out-of-sample prediction
error for both approaches.

Output: results/holdout_prediction_contest.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Re-use fitting infrastructure from the multilaw analysis
from run_multilaw_analysis import (
    fit_all_laws,
    _power,
    _logarithmic,
    _linear,
    _exponential,
    _saturation,
    _stretched_exp,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Law prediction functions ─────────────────────────────────────────

_LAW_FUNCS = {
    "power": _power,
    "logarithmic": _logarithmic,
    "linear": _linear,
    "exponential": _exponential,
    "saturation": _saturation,
    "stretched_exp": _stretched_exp,
}


def _predict(law_name: str, params: dict, X: np.ndarray) -> np.ndarray | None:
    """Predict E at X using a fitted law and its parameters."""
    func = _LAW_FUNCS.get(law_name)
    if func is None:
        return None

    # Build ordered parameter list from param names
    _pnames = {
        "power": ["E_inf", "a", "alpha"],
        "exponential": ["E_inf", "a", "alpha"],
        "logarithmic": ["a", "b"],
        "stretched_exp": ["E_inf", "a", "alpha", "beta"],
        "saturation": ["E_inf", "a", "X0", "alpha"],
        "linear": ["a", "b"],
    }

    names = _pnames[law_name]
    try:
        popt = [params[n] for n in names]
    except KeyError:
        return None

    # Handle normalised exponential / stretched_exp
    X_norm = params.get("X_normalizer")
    if law_name == "exponential" and X_norm:
        # The stored alpha is in original units;
        # the function expects un-normalised X,
        # but alpha was rescaled in run_multilaw_analysis.
        # Re-scale back for prediction:
        return params["E_inf"] + params["a"] * np.exp(-params["alpha"] * X)
    if law_name == "stretched_exp" and X_norm:
        Xn = X / X_norm
        return params["E_inf"] + params["a"] * np.exp(
            -params["alpha"] * np.power(Xn, params["beta"])
        )

    try:
        return func(X, *popt)
    except Exception:
        return None


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _rel_rmse(y_true, y_pred):
    """RMSE normalised by range of y_true."""
    rng = y_true.max() - y_true.min()
    if rng == 0:
        return float("inf")
    return _rmse(y_true, y_pred) / rng


# ── Held-out experiment for one axis ─────────────────────────────────

def _run_holdout_axis(df, task, model, axis, x_col, group_cols, holdout_frac=0.2, rng_seed=42):
    """Run the held-out prediction contest for one task × model × axis.

    Returns a dict with OOS metrics for (a) power-law only and (b) AICc-best multi-law.
    """
    rng = np.random.RandomState(rng_seed)

    mdf = df[(df["task"] == task) & (df["model_name"] == model)]
    if mdf.empty:
        return None

    # Collect all slices for this axis
    slices = []
    for keys, sub in mdf.groupby(group_cols):
        sub = sub.sort_values(x_col)
        X = sub[x_col].values.astype(float)
        E = sub["test_rel_l2_mean"].values.astype(float)
        if len(X) < 3:
            continue
        slices.append({"keys": keys, "X": X, "E": E})

    if len(slices) < 3:
        return None

    # Hold out ~20% of slices
    n_hold = max(1, int(len(slices) * holdout_frac))
    hold_idxs = set(rng.choice(len(slices), size=n_hold, replace=False))
    train_slices = [s for i, s in enumerate(slices) if i not in hold_idxs]
    test_slices = [s for i, s in enumerate(slices) if i in hold_idxs]

    if not train_slices or not test_slices:
        return None

    # Pool all training slice points
    X_train = np.concatenate([s["X"] for s in train_slices])
    E_train = np.concatenate([s["E"] for s in train_slices])

    # (a) Power-law only fit on pooled training data
    power_fits = fit_all_laws(X_train, E_train, candidate_laws=["power"])
    # (b) Multi-law AICc-best on pooled training data
    multi_fits = fit_all_laws(X_train, E_train)

    if not power_fits and not multi_fits:
        return None

    # Pool test points
    X_test = np.concatenate([s["X"] for s in test_slices])
    E_test = np.concatenate([s["E"] for s in test_slices])

    result = {
        "task": task,
        "model": model,
        "axis": axis,
        "n_train_slices": len(train_slices),
        "n_test_slices": len(test_slices),
        "n_train_points": len(X_train),
        "n_test_points": len(X_test),
    }

    # Evaluate power-law OOS
    if power_fits:
        pf = power_fits[0]
        pred = _predict(pf["law"], pf["params"], X_test)
        if pred is not None:
            result["power_law_oos_rmse"] = _rmse(E_test, pred)
            result["power_law_oos_rel_rmse"] = _rel_rmse(E_test, pred)
            result["power_law_train_r2"] = pf["r2"]
        else:
            result["power_law_oos_rmse"] = None
            result["power_law_oos_rel_rmse"] = None

    # Evaluate multi-law OOS
    if multi_fits:
        mf = multi_fits[0]  # AICc-best
        pred = _predict(mf["law"], mf["params"], X_test)
        if pred is not None:
            result["multilaw_best"] = mf["law"]
            result["multilaw_oos_rmse"] = _rmse(E_test, pred)
            result["multilaw_oos_rel_rmse"] = _rel_rmse(E_test, pred)
            result["multilaw_train_r2"] = mf["r2"]
        else:
            result["multilaw_best"] = mf["law"]
            result["multilaw_oos_rmse"] = None
            result["multilaw_oos_rel_rmse"] = None

    # Is multi-law better?
    p_rmse = result.get("power_law_oos_rel_rmse")
    m_rmse = result.get("multilaw_oos_rel_rmse")
    if p_rmse is not None and m_rmse is not None:
        result["multilaw_wins"] = m_rmse < p_rmse
        result["relative_improvement"] = (p_rmse - m_rmse) / max(p_rmse, 1e-12)
    else:
        result["multilaw_wins"] = None

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    df = pd.read_csv("runs/grouped_metrics_train.csv")
    df["test_rel_l2_mean"] = pd.to_numeric(df["test_rel_l2_mean"], errors="coerce")
    df = df.dropna(subset=["test_rel_l2_mean"])
    print(f"Loaded {len(df)} training groups")

    results = []

    axis_configs = [
        ("data", "dataset_size", ["capacity_name", "resolution"]),
        ("capacity", "parameter_count", ["dataset_size", "resolution"]),
        ("resolution", "resolution", ["capacity_name", "dataset_size"]),
    ]

    for task in sorted(df["task"].unique()):
        for model in sorted(df["model_name"].unique()):
            for axis, x_col, group_cols in axis_configs:
                r = _run_holdout_axis(df, task, model, axis, x_col, group_cols)
                if r:
                    results.append(r)
                    pw = r.get("power_law_oos_rel_rmse", "N/A")
                    ml = r.get("multilaw_oos_rel_rmse", "N/A")
                    winner = r.get("multilaw_best", "?")
                    wins = r.get("multilaw_wins", "?")
                    print(f"  {task:20s} {model:16s} {axis:12s}: "
                          f"power={pw:.4f}  multi({winner})={ml:.4f}  "
                          f"multi_wins={wins}" if isinstance(pw, float) else
                          f"  {task:20s} {model:16s} {axis:12s}: incomplete")

    # Summary
    n_total = len(results)
    n_wins = sum(1 for r in results if r.get("multilaw_wins") is True)
    n_loses = sum(1 for r in results if r.get("multilaw_wins") is False)
    n_na = n_total - n_wins - n_loses

    print(f"\n{'='*90}")
    print("HELD-OUT PREDICTION CONTEST SUMMARY")
    print(f"{'='*90}")
    print(f"Total comparisons: {n_total}")
    print(f"Multi-law wins: {n_wins}/{n_total} ({100*n_wins/max(n_total,1):.0f}%)")
    print(f"Power-law wins: {n_loses}/{n_total} ({100*n_loses/max(n_total,1):.0f}%)")
    print(f"Incomplete: {n_na}")

    # Per-axis breakdown
    for axis in ["data", "capacity", "resolution"]:
        ar = [r for r in results if r["axis"] == axis]
        aw = sum(1 for r in ar if r.get("multilaw_wins") is True)
        al = sum(1 for r in ar if r.get("multilaw_wins") is False)
        print(f"  {axis:12s}: multi={aw}, power={al}, n/a={len(ar)-aw-al}")

    # Save
    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "holdout_prediction_contest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → results/holdout_prediction_contest.json")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
