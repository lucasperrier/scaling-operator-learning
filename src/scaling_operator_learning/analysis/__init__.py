"""Scaling-law fitting utilities.

Fits power-law models of the form E = E_inf + a * X^{-alpha}
to aggregated experiment results, with bootstrap confidence intervals.

Extended for operator learning with three scaling axes: N, D, R.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Functional forms
# ---------------------------------------------------------------------------

def _power_law(X: np.ndarray, E_inf: float, a: float, alpha: float) -> np.ndarray:
    """E(X) = E_inf + a * X^{-alpha}"""
    return E_inf + a * np.power(X.astype(float), -alpha)


def _full_law(ND: np.ndarray, E_inf: float, a: float, alpha: float, b: float, beta: float) -> np.ndarray:
    """E(N,D) = E_inf + a*N^{-alpha} + b*D^{-beta}

    *ND* has shape (2, n) with ND[0]=N, ND[1]=D.
    """
    N = ND[0].astype(float)
    D = ND[1].astype(float)
    return E_inf + a * np.power(N, -alpha) + b * np.power(D, -beta)


def _full_3d_law(
    NDR: np.ndarray,
    E_inf: float, a: float, alpha: float, b: float, beta: float, c: float, gamma: float,
) -> np.ndarray:
    """E(N,D,R) = E_inf + a*N^{-alpha} + b*D^{-beta} + c*R^{-gamma}

    *NDR* has shape (3, n) with NDR[0]=N, NDR[1]=D, NDR[2]=R.
    """
    N = NDR[0].astype(float)
    D = NDR[1].astype(float)
    R = NDR[2].astype(float)
    return E_inf + a * np.power(N, -alpha) + b * np.power(D, -beta) + c * np.power(R, -gamma)


# ---------------------------------------------------------------------------
# Single curve fitting
# ---------------------------------------------------------------------------

def _safe_curve_fit(func, xdata, ydata, p0, bounds, maxfev: int = 20000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds, maxfev=maxfev)
            return popt, pcov
        except (RuntimeError, ValueError):
            return None


def fit_power_law(
    X: np.ndarray,
    E: np.ndarray,
) -> dict[str, Any] | None:
    """Fit E(X) = E_inf + a * X^{-alpha} to data."""
    if len(X) < 3:
        return None
    p0 = [E.min(), 1.0, 0.5]
    bounds = ([0, 0, 0.01], [np.inf, np.inf, 5.0])
    result = _safe_curve_fit(_power_law, X, E, p0, bounds)
    if result is None:
        return None
    popt, pcov = result
    return {"E_inf": popt[0], "a": popt[1], "alpha": popt[2], "pcov": pcov}


def fit_full_surface(
    N: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
) -> dict[str, Any] | None:
    """Fit E(N,D) = E_inf + a*N^{-alpha} + b*D^{-beta}."""
    if len(E) < 5:
        return None
    p0 = [E.min(), 1.0, 0.5, 1.0, 0.5]
    bounds = ([0, 0, 0.01, 0, 0.01], [np.inf, np.inf, 5.0, np.inf, 5.0])
    ND = np.vstack([N.astype(float), D.astype(float)])
    result = _safe_curve_fit(_full_law, ND, E, p0, bounds)
    if result is None:
        return None
    popt, pcov = result
    return {
        "E_inf": popt[0], "a": popt[1], "alpha": popt[2],
        "b": popt[3], "beta": popt[4], "pcov": pcov,
    }


def fit_full_volume(
    N: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    E: np.ndarray,
) -> dict[str, Any] | None:
    """Fit E(N,D,R) = E_inf + a*N^{-alpha} + b*D^{-beta} + c*R^{-gamma}."""
    if len(E) < 7:
        return None
    p0 = [E.min(), 1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
    bounds = (
        [0, 0, 0.01, 0, 0.01, 0, 0.01],
        [np.inf, np.inf, 5.0, np.inf, 5.0, np.inf, 5.0],
    )
    NDR = np.vstack([N.astype(float), D.astype(float), R.astype(float)])
    result = _safe_curve_fit(_full_3d_law, NDR, E, p0, bounds)
    if result is None:
        return None
    popt, pcov = result
    return {
        "E_inf": popt[0], "a": popt[1], "alpha": popt[2],
        "b": popt[3], "beta": popt[4], "c": popt[5], "gamma": popt[6],
        "pcov": pcov,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_fit(fit_func, *arrays, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    all_params: list[dict[str, Any]] = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sampled = [arr[idx] for arr in arrays]
        result = fit_func(*sampled)
        if result is not None:
            all_params.append({k: v for k, v in result.items() if k != "pcov"})
    return all_params


def _summarize_bootstrap(samples: list[dict], *, params: list[str]) -> dict[str, Any]:
    if not samples:
        return {"n_boot_success": 0}
    summary: dict[str, Any] = {"n_boot_success": len(samples)}
    for p in params:
        vals = np.array([s[p] for s in samples])
        summary[f"{p}_mean"] = float(np.mean(vals))
        summary[f"{p}_std"] = float(np.std(vals))
        summary[f"{p}_ci_lo"] = float(np.percentile(vals, 2.5))
        summary[f"{p}_ci_hi"] = float(np.percentile(vals, 97.5))
    return summary


def bootstrap_power_law(X, E, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    samples = _bootstrap_fit(fit_power_law, X, E, n_boot=n_boot, seed=seed)
    return _summarize_bootstrap(samples, params=["E_inf", "a", "alpha"])


def bootstrap_full(N, D, E, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    samples = _bootstrap_fit(fit_full_surface, N, D, E, n_boot=n_boot, seed=seed)
    return _summarize_bootstrap(samples, params=["E_inf", "a", "alpha", "b", "beta"])


def bootstrap_full_3d(N, D, R, E, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    samples = _bootstrap_fit(fit_full_volume, N, D, R, E, n_boot=n_boot, seed=seed)
    return _summarize_bootstrap(
        samples, params=["E_inf", "a", "alpha", "b", "beta", "c", "gamma"]
    )


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def run_scaling_analysis(
    grouped_df: pd.DataFrame,
    *,
    metric_col: str = "test_rel_l2_mean",
    capacity_col: str = "parameter_count",
    data_col: str = "dataset_size",
    resolution_col: str = "resolution",
    model_col: str = "model_name",
    capacity_name_col: str = "capacity_name",
    max_divergence_rate: float = 0.30,
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> dict[str, Any]:
    """Run all scaling fits and bootstraps for all models.

    Returns a dict with keys:
        data_fits, capacity_fits, resolution_fits, full_2d_fits, full_3d_fits.
    """
    df = grouped_df.copy()
    if "divergence_rate" in df.columns:
        df = df[df["divergence_rate"] <= max_divergence_rate]
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    has_resolution = resolution_col in df.columns

    capacity_fits: list[dict[str, Any]] = []
    data_fits: list[dict[str, Any]] = []
    resolution_fits: list[dict[str, Any]] = []
    full_2d_fits: list[dict[str, Any]] = []
    full_3d_fits: list[dict[str, Any]] = []

    for model_name, model_df in df.groupby(model_col):
        # --- Data scaling: for each (capacity, resolution), fit E vs N ---
        group_cols = [capacity_name_col]
        if has_resolution:
            group_cols.append(resolution_col)
        for keys, sub in model_df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            N = sub[data_col].values.astype(float)
            E = sub[metric_col].values.astype(float)
            point_fit = fit_power_law(N, E)
            boot = bootstrap_power_law(N, E, n_boot=n_boot, seed=boot_seed)
            rec = {"model_name": str(model_name), "n_points": len(N)}
            for col, val in zip(group_cols, keys):
                rec[col] = val if not hasattr(val, 'item') else val.item()
            if point_fit:
                rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
            rec["bootstrap"] = boot
            data_fits.append(rec)

        # --- Capacity scaling: for each (dataset_size, resolution), fit E vs D ---
        group_cols = [data_col]
        if has_resolution:
            group_cols.append(resolution_col)
        for keys, sub in model_df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            D = sub[capacity_col].values.astype(float)
            E = sub[metric_col].values.astype(float)
            point_fit = fit_power_law(D, E)
            boot = bootstrap_power_law(D, E, n_boot=n_boot, seed=boot_seed)
            rec = {"model_name": str(model_name), "n_points": len(D)}
            for col, val in zip(group_cols, keys):
                rec[col] = val if not hasattr(val, 'item') else val.item()
            if point_fit:
                rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
            rec["bootstrap"] = boot
            capacity_fits.append(rec)

        # --- Resolution scaling: for each (capacity, dataset_size), fit E vs R ---
        if has_resolution:
            for (cap, ds), sub in model_df.groupby([capacity_name_col, data_col]):
                R = sub[resolution_col].values.astype(float)
                E = sub[metric_col].values.astype(float)
                point_fit = fit_power_law(R, E)
                boot = bootstrap_power_law(R, E, n_boot=n_boot, seed=boot_seed)
                rec = {
                    "model_name": str(model_name),
                    "capacity_name": str(cap),
                    "dataset_size": int(ds),
                    "n_points": len(R),
                }
                if point_fit:
                    rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
                rec["bootstrap"] = boot
                resolution_fits.append(rec)

        # --- Full 2D surface: E(N, D) ---
        N_all = model_df[data_col].values.astype(float)
        D_all = model_df[capacity_col].values.astype(float)
        E_all = model_df[metric_col].values.astype(float)
        point_fit = fit_full_surface(N_all, D_all, E_all)
        boot = bootstrap_full(N_all, D_all, E_all, n_boot=n_boot, seed=boot_seed)
        rec = {"model_name": str(model_name), "n_points": len(E_all)}
        if point_fit:
            rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
        rec["bootstrap"] = boot
        full_2d_fits.append(rec)

        # --- Full 3D volume: E(N, D, R) ---
        if has_resolution:
            R_all = model_df[resolution_col].values.astype(float)
            point_fit = fit_full_volume(N_all, D_all, R_all, E_all)
            boot = bootstrap_full_3d(N_all, D_all, R_all, E_all, n_boot=n_boot, seed=boot_seed)
            rec = {"model_name": str(model_name), "n_points": len(E_all)}
            if point_fit:
                rec.update({k: v for k, v in point_fit.items() if k != "pcov"})
            rec["bootstrap"] = boot
            full_3d_fits.append(rec)

    return {
        "data_fits": data_fits,
        "capacity_fits": capacity_fits,
        "resolution_fits": resolution_fits,
        "full_2d_fits": full_2d_fits,
        "full_3d_fits": full_3d_fits,
    }

    return {
        "capacity_fits": capacity_fits,
        "data_fits": data_fits,
        "full_fits": full_fits,
    }
