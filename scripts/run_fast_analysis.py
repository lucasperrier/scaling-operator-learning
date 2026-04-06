"""Fast scaling analysis — point estimates for all slices, bootstrap only for key fits."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _power_law(X, E_inf, a, alpha):
    return E_inf + a * np.power(X.astype(float), -alpha)


def _full_3d(NDR, E_inf, a, alpha, b, beta, c, gamma):
    N, D, R = NDR[0].astype(float), NDR[1].astype(float), NDR[2].astype(float)
    return E_inf + a * N**(-alpha) + b * D**(-beta) + c * R**(-gamma)


def fit_pl(X, E):
    if len(X) < 3:
        return None
    try:
        popt, _ = curve_fit(
            _power_law, X.astype(float), E.astype(float),
            p0=[E.min(), 1.0, 0.5],
            bounds=([0, 0, 0.01], [np.inf, np.inf, 5.0]),
            maxfev=5000,
        )
        resid = E - _power_law(X, *popt)
        r2 = 1 - np.sum(resid**2) / np.sum((E - E.mean())**2)
        return {"E_inf": float(popt[0]), "a": float(popt[1]), "alpha": float(popt[2]), "r2": float(r2)}
    except (RuntimeError, ValueError):
        return None


def fit_3d(N, D, R, E):
    if len(E) < 7:
        return None
    try:
        popt, _ = curve_fit(
            _full_3d,
            np.vstack([N.astype(float), D.astype(float), R.astype(float)]),
            E.astype(float),
            p0=[E.min(), 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            bounds=([0, 0, 0.01, 0, 0.01, 0, 0.01],
                    [np.inf, np.inf, 5.0, np.inf, 5.0, np.inf, 5.0]),
            maxfev=10000,
        )
        resid = E - _full_3d(np.vstack([N, D, R]), *popt)
        r2 = 1 - np.sum(resid**2) / np.sum((E - E.mean())**2)
        return {
            "E_inf": float(popt[0]), "a": float(popt[1]), "alpha": float(popt[2]),
            "b": float(popt[3]), "beta": float(popt[4]),
            "c": float(popt[5]), "gamma": float(popt[6]), "r2": float(r2),
        }
    except (RuntimeError, ValueError):
        return None


def bootstrap_3d(N, D, R, E, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    n = len(E)
    results = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = fit_3d(N[idx], D[idx], R[idx], E[idx])
        if r:
            results.append(r)
    if not results:
        return {}
    summary = {}
    for p in ["alpha", "beta", "gamma", "E_inf"]:
        vals = np.array([r[p] for r in results])
        summary[f"{p}_mean"] = float(np.mean(vals))
        summary[f"{p}_std"] = float(np.std(vals))
        summary[f"{p}_ci_lo"] = float(np.percentile(vals, 2.5))
        summary[f"{p}_ci_hi"] = float(np.percentile(vals, 97.5))
    summary["n_boot_success"] = len(results)
    return summary


def main():
    t0 = time.time()
    df = pd.read_csv("runs/grouped_metrics_train.csv")
    df["test_rel_l2_mean"] = pd.to_numeric(df["test_rel_l2_mean"], errors="coerce")
    df = df.dropna(subset=["test_rel_l2_mean"])
    print(f"Loaded {len(df)} training groups")

    results = {
        "data_fits": [],
        "capacity_fits": [],
        "resolution_fits": [],
        "full_3d_fits": [],
    }

    for task in sorted(df["task"].unique()):
        tdf = df[df["task"] == task]
        print(f"\n=== Task: {task} ({len(tdf)} groups) ===")

        for model in sorted(tdf["model_name"].unique()):
            mdf = tdf[tdf["model_name"] == model]

            # ---- Data scaling: E vs N for each (capacity, R) ----
            for (cap, R), sub in mdf.groupby(["capacity_name", "resolution"]):
                sub = sub.sort_values("dataset_size")
                N = sub["dataset_size"].values
                E = sub["test_rel_l2_mean"].values
                fit = fit_pl(N, E)
                rec = {"task": task, "model_name": model, "capacity_name": str(cap),
                       "resolution": int(R), "n_points": len(N), "axis": "data"}
                if fit:
                    rec.update(fit)
                results["data_fits"].append(rec)

            # ---- Capacity scaling: E vs D for each (N, R) ----
            for (N_val, R), sub in mdf.groupby(["dataset_size", "resolution"]):
                sub = sub.sort_values("parameter_count")
                D = sub["parameter_count"].values
                E = sub["test_rel_l2_mean"].values
                fit = fit_pl(D, E)
                rec = {"task": task, "model_name": model, "dataset_size": int(N_val),
                       "resolution": int(R), "n_points": len(D), "axis": "capacity"}
                if fit:
                    rec.update(fit)
                results["capacity_fits"].append(rec)

            # ---- Resolution scaling: E vs R for each (capacity, N) ----
            for (cap, N_val), sub in mdf.groupby(["capacity_name", "dataset_size"]):
                sub = sub.sort_values("resolution")
                R = sub["resolution"].values
                E = sub["test_rel_l2_mean"].values
                fit = fit_pl(R, E)
                rec = {"task": task, "model_name": model, "capacity_name": str(cap),
                       "dataset_size": int(N_val), "n_points": len(R), "axis": "resolution"}
                if fit:
                    rec.update(fit)
                results["resolution_fits"].append(rec)

            # ---- Full 3D: E(N, D, R) per model per task ----
            N_all = mdf["dataset_size"].values.astype(float)
            D_all = mdf["parameter_count"].values.astype(float)
            R_all = mdf["resolution"].values.astype(float)
            E_all = mdf["test_rel_l2_mean"].values.astype(float)
            fit = fit_3d(N_all, D_all, R_all, E_all)
            boot = bootstrap_3d(N_all, D_all, R_all, E_all, n_boot=200)
            rec = {"task": task, "model_name": model, "n_points": len(E_all)}
            if fit:
                rec.update(fit)
            rec["bootstrap"] = boot
            results["full_3d_fits"].append(rec)

            n_data = sum(1 for r in results["data_fits"] if r.get("alpha") and r["model_name"] == model and r["task"] == task)
            n_cap = sum(1 for r in results["capacity_fits"] if r.get("alpha") and r["model_name"] == model and r["task"] == task)
            n_res = sum(1 for r in results["resolution_fits"] if r.get("alpha") and r["model_name"] == model and r["task"] == task)
            print(f"  {model}: data={n_data} cap={n_cap} res={n_res} 3D={'OK' if fit else 'FAIL'}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("FULL 3D FITS: E(N,D,R) = E_inf + a*N^(-α) + b*D^(-β) + c*R^(-γ)")
    print(f"{'='*60}")
    for rec in results["full_3d_fits"]:
        if "alpha" not in rec:
            print(f"  {rec['task']:20s} {rec['model_name']:16s}  FAILED")
            continue
        b = rec.get("bootstrap", {})
        print(f"  {rec['task']:20s} {rec['model_name']:16s}  "
              f"α={rec['alpha']:.3f} [{b.get('alpha_ci_lo', 0):.3f},{b.get('alpha_ci_hi', 0):.3f}]  "
              f"β={rec['beta']:.3f} [{b.get('beta_ci_lo', 0):.3f},{b.get('beta_ci_hi', 0):.3f}]  "
              f"γ={rec['gamma']:.3f} [{b.get('gamma_ci_lo', 0):.3f},{b.get('gamma_ci_hi', 0):.3f}]  "
              f"R²={rec['r2']:.4f}")

    # ---- Data scaling summary ----
    print(f"\nDATA SCALING EXPONENTS (α) — median across slices:")
    for task in sorted(df["task"].unique()):
        for model in sorted(df["model_name"].unique()):
            alphas = [r["alpha"] for r in results["data_fits"]
                      if r.get("alpha") and r["task"] == task and r["model_name"] == model]
            if alphas:
                print(f"  {task:20s} {model:16s}  α={np.median(alphas):.3f}  "
                      f"[{np.percentile(alphas, 25):.3f}, {np.percentile(alphas, 75):.3f}]  (n={len(alphas)})")

    print(f"\nRESOLUTION SCALING EXPONENTS (γ) — median across slices:")
    for task in sorted(df["task"].unique()):
        for model in sorted(df["model_name"].unique()):
            gammas = [r["alpha"] for r in results["resolution_fits"]
                      if r.get("alpha") and r["task"] == task and r["model_name"] == model]
            if gammas:
                print(f"  {task:20s} {model:16s}  γ={np.median(gammas):.3f}  "
                      f"[{np.percentile(gammas, 25):.3f}, {np.percentile(gammas, 75):.3f}]  (n={len(gammas)})")

    print(f"\nCAPACITY SCALING EXPONENTS (β) — median across slices:")
    for task in sorted(df["task"].unique()):
        for model in sorted(df["model_name"].unique()):
            betas = [r["alpha"] for r in results["capacity_fits"]
                      if r.get("alpha") and r["task"] == task and r["model_name"] == model]
            if betas:
                print(f"  {task:20s} {model:16s}  β={np.median(betas):.3f}  "
                      f"[{np.percentile(betas, 25):.3f}, {np.percentile(betas, 75):.3f}]  (n={len(betas)})")

    # Save
    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "scaling_fits.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved → results/scaling_fits.json")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
