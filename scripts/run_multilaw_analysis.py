"""Multi-law scaling analysis.

Instead of forcing power laws, fit multiple candidate functional forms per axis
and use AIC for model selection.  The winning law type and its parameters become
physics-interpretable diagnostics.

Candidate laws for E(X):
  1. power:       E = E_inf + a * X^{-alpha}         discretization / statistical rate
  2. exponential: E = E_inf + a * exp(-alpha * X)     spectral convergence
  3. log:         E = a - b * log(X)                  sub-algebraic decay
  4. stretched_exp: E = E_inf + a * exp(-alpha * X^beta)  interpolation
  5. saturation:  E = E_inf + a / (1 + X/X0)^alpha   with characteristic scale
  6. linear:      E = a + b * X                       (detects INVERSE trends)

For the 3D law we keep the additive form but allow each axis to use its
best 1D law rather than forcing all three to be power.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────── Candidate 1D laws ────────────────────

def _power(X, E_inf, a, alpha):
    """E = E_inf + a * X^{-alpha}"""
    return E_inf + a * np.power(X, -alpha)

def _exponential(X, E_inf, a, alpha):
    """E = E_inf + a * exp(-alpha * X)"""
    return E_inf + a * np.exp(-alpha * X)

def _logarithmic(X, a, b):
    """E = a - b * log(X)"""
    return a - b * np.log(X)

def _stretched_exp(X, E_inf, a, alpha, beta):
    """E = E_inf + a * exp(-alpha * X^beta)"""
    return E_inf + a * np.exp(-alpha * np.power(X, beta))

def _saturation(X, E_inf, a, X0, alpha):
    """E = E_inf + a / (1 + X/X0)^alpha"""
    return E_inf + a * np.power(1.0 + X / X0, -alpha)

def _linear(X, a, b):
    """E = a + b * X"""
    return a + b * X


# ──────────────────── AIC helpers ────────────────────

def aic(n, k, rss):
    """Akaike Information Criterion (corrected for small samples)."""
    if rss <= 0 or n <= k + 1:
        return np.inf
    aic_val = n * np.log(rss / n) + 2 * k
    # AICc correction for small samples
    if n - k - 1 > 0:
        aic_val += 2 * k * (k + 1) / (n - k - 1)
    return aic_val


def bic(n, k, rss):
    """Bayesian Information Criterion."""
    if rss <= 0 or n <= 0:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


def akaike_weights(aic_values):
    """Compute Akaike weights from a list of AIC values.

    w_i = exp(-Delta_i / 2) / sum_j exp(-Delta_j / 2)
    where Delta_i = AIC_i - AIC_min.
    """
    aic_arr = np.array(aic_values, dtype=float)
    valid = np.isfinite(aic_arr)
    if not np.any(valid):
        return np.zeros_like(aic_arr)
    aic_min = np.min(aic_arr[valid])
    deltas = aic_arr - aic_min
    # Guard against overflow: set invalid to large delta
    deltas[~valid] = 1e10
    raw = np.exp(-0.5 * deltas)
    total = raw.sum()
    if total == 0:
        return np.zeros_like(aic_arr)
    return raw / total


def r_squared(E_obs, E_pred):
    ss_res = np.sum((E_obs - E_pred) ** 2)
    ss_tot = np.sum((E_obs - E_obs.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ──────────────────── Fit one law ────────────────────

def try_fit(law_name, func, X, E, p0, bounds, maxfev=10000):
    """Attempt curve_fit; return dict with params, R², AIC, BIC or None."""
    n = len(E)
    k = len(p0)  # number of free parameters
    try:
        popt, _ = curve_fit(func, X, E, p0=p0, bounds=bounds, maxfev=maxfev)
        E_pred = func(X, *popt)
        rss = float(np.sum((E - E_pred) ** 2))
        r2 = r_squared(E, E_pred)
        aic_val = aic(n, k, rss)
        bic_val = bic(n, k, rss)
        return {
            "law": law_name,
            "params": {name: float(v) for name, v in zip(_param_names(law_name), popt)},
            "n_params": k,
            "r2": float(r2),
            "rss": rss,
            "aic": float(aic_val),
            "bic": float(bic_val),
        }
    except (RuntimeError, ValueError, OverflowError):
        return None


def _param_names(law):
    return {
        "power": ["E_inf", "a", "alpha"],
        "exponential": ["E_inf", "a", "alpha"],
        "logarithmic": ["a", "b"],
        "stretched_exp": ["E_inf", "a", "alpha", "beta"],
        "saturation": ["E_inf", "a", "X0", "alpha"],
        "linear": ["a", "b"],
    }[law]


# ──────────────────── Fit all candidates for one slice ────────────────────

def fit_all_laws(X, E, candidate_laws=None):
    """Fit all candidate laws.  Returns list of successful fits sorted by AIC.

    If candidate_laws is given, only fit those laws (e.g. ["linear", "logarithmic", "power"]
    for the restricted 2-parameter-family analysis).
    """
    ALL_LAWS = {"power", "exponential", "logarithmic", "stretched_exp", "saturation", "linear"}
    if candidate_laws is None:
        candidate_laws = ALL_LAWS
    else:
        candidate_laws = set(candidate_laws)

    X = X.astype(float)
    E = E.astype(float)
    n = len(X)
    if n < 3:
        return []

    E_min, E_max = E.min(), E.max()
    E_range = max(E_max - E_min, 1e-12)
    X_min, X_max = X.min(), X.max()
    X_range = max(X_max - X_min, 1e-12)

    candidates = []

    # 1. Power law: E = E_inf + a * X^{-alpha}
    if "power" in candidate_laws:
        candidates.append(try_fit(
            "power", _power, X, E,
            p0=[E_min * 0.9, E_range, 0.5],
            bounds=([0, 0, 0.001], [E_max * 10, np.inf, 10.0]),
        ))

    # 2. Exponential: E = E_inf + a * exp(-alpha * X)
    # Normalize X to avoid numerical overflow
    if "exponential" in candidate_laws:
        X_norm = X / X_max  # normalise for numerical stability
        def _exp_scaled(Xn, E_inf, a, alpha):
            return E_inf + a * np.exp(-alpha * Xn)
        fit_exp = try_fit(
            "exponential", _exp_scaled, X_norm, E,
            p0=[E_min * 0.9, E_range, 1.0],
            bounds=([0, 0, 0.0], [E_max * 10, np.inf, 50.0]),
        )
        if fit_exp:
            # Rescale alpha back to original X units
            fit_exp["params"]["alpha"] = fit_exp["params"]["alpha"] / X_max
            fit_exp["params"]["X_normalizer"] = float(X_max)
        candidates.append(fit_exp)

    # 3. Logarithmic: E = a - b * log(X)
    if "logarithmic" in candidate_laws and n >= 2:
        candidates.append(try_fit(
            "logarithmic", _logarithmic, X, E,
            p0=[E_max, E_range / max(np.log(X_max / X_min), 0.1)],
            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
        ))

    # 4. Stretched exponential: E = E_inf + a * exp(-alpha * X^beta)
    if "stretched_exp" in candidate_laws and n >= 4:
        X_norm2 = X / X_max
        def _strexp_scaled(Xn, E_inf, a, alpha, beta):
            return E_inf + a * np.exp(-alpha * np.power(Xn, beta))
        fit_se = try_fit(
            "stretched_exp", _strexp_scaled, X_norm2, E,
            p0=[E_min * 0.9, E_range, 1.0, 0.5],
            bounds=([0, 0, 0.0, 0.01], [E_max * 10, np.inf, 50.0, 5.0]),
        )
        if fit_se:
            fit_se["params"]["X_normalizer"] = float(X_max)
        candidates.append(fit_se)

    # 5. Saturation: E = E_inf + a / (1 + X/X0)^alpha
    if "saturation" in candidate_laws and n >= 4:
        candidates.append(try_fit(
            "saturation", _saturation, X, E,
            p0=[E_min * 0.9, E_range, X.mean(), 1.0],
            bounds=([0, 0, X_min * 0.01, 0.01], [E_max * 10, np.inf, X_max * 100, 10.0]),
        ))

    # 6. Linear: E = a + b * X  (detects INVERSE relationship)
    if "linear" in candidate_laws:
        candidates.append(try_fit(
            "linear", _linear, X, E,
            p0=[E.mean(), 0.0],
            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
        ))

    # Filter None, sort by AIC
    valid = [c for c in candidates if c is not None]
    valid.sort(key=lambda c: c["aic"])
    return valid


# ──────────────────── Physics interpretation ────────────────────

def interpret_winner(axis, law, params, r2):
    """Return a short physics interpretation of the winning law."""
    if r2 < 0.5:
        return "no_clear_trend"

    if law == "power":
        alpha = params.get("alpha", 0)
        E_inf = params.get("E_inf", 0)
        if axis == "data":
            if alpha < 0.2:
                return "very_slow_data_scaling"
            elif alpha < 0.6:
                return "moderate_statistical_rate"
            else:
                return "fast_statistical_rate"
        elif axis == "resolution":
            if alpha < 0.3:
                return "near_resolution_invariant"
            elif alpha < 1.0:
                return "low_order_discretization"
            elif alpha < 2.5:
                return "high_order_discretization"
            else:
                return "spectral_like_convergence"
        elif axis == "capacity":
            if alpha < 0.2:
                return "near_capacity_invariant"
            else:
                return "algebraic_capacity_scaling"

    elif law == "exponential":
        return f"exponential_decay_({axis})"

    elif law == "logarithmic":
        b = params.get("b", 0)
        if b < 0:
            return f"log_growth_({axis})"  # error INCREASES with X
        return f"logarithmic_improvement_({axis})"

    elif law == "linear":
        b = params.get("b", 0)
        if b > 0:
            return f"error_increases_with_{axis}"  # pathological!
        elif b < 0:
            return f"linear_improvement_({axis})"
        return f"flat_({axis})"

    elif law == "stretched_exp":
        return f"stretched_exponential_({axis})"

    elif law == "saturation":
        return f"saturating_({axis})"

    return "unclassified"


# ──────────────────── Helpers ────────────────────

# Restricted candidate laws for low-n slices (2-parameter families only)
RESTRICTED_LAWS = ["linear", "logarithmic", "power"]

ALL_LAW_NAMES = ["exponential", "linear", "logarithmic", "power", "saturation", "stretched_exp"]


def _build_slice_record(task, model, axis, slice_keys, fits, n_points,
                        candidate_laws=None):
    """Build a per-slice record from fits, including Akaike weights and BIC."""
    if not fits:
        return None

    winner = fits[0]
    rec = {"task": task, "model_name": model, "n_points": n_points, "axis": axis}
    rec.update(slice_keys)

    rec["best_law"] = winner["law"]
    rec["best_r2"] = winner["r2"]
    rec["best_aic"] = winner["aic"]
    rec["best_bic"] = winner["bic"]
    rec["best_params"] = winner["params"]

    # Delta AIC to next-best different law
    other = [f for f in fits[1:] if f["law"] != winner["law"]]
    rec["delta_aic_2nd"] = other[0]["aic"] - winner["aic"] if other else float("inf")

    # BIC winner
    bic_sorted = sorted(fits, key=lambda f: f["bic"])
    rec["bic_winner"] = bic_sorted[0]["law"]
    rec["aicc_bic_agree"] = (rec["best_law"] == rec["bic_winner"])

    # Akaike weights (full weight vector)
    aic_vals = [f["aic"] for f in fits]
    weights = akaike_weights(aic_vals)
    rec["akaike_weights"] = {f["law"]: float(w) for f, w in zip(fits, weights)}
    rec["winner_akaike_weight"] = float(weights[0])

    # Top-2 weight gap (for ambiguity detection)
    if len(weights) > 1:
        sorted_w = sorted(weights, reverse=True)
        rec["top2_weight_gap"] = float(sorted_w[0] - sorted_w[1])
    else:
        rec["top2_weight_gap"] = 1.0

    rec["all_fits"] = fits
    return rec


def _aggregate_axis(per_slice_records, task, model, axis):
    """Compute summary statistics for one task×model×axis. Returns summary dict."""
    winners = [r for r in per_slice_records if r is not None]
    if not winners:
        return None

    # Law vote counts
    law_counts = {}
    for r in winners:
        law_counts[r["best_law"]] = law_counts.get(r["best_law"], 0) + 1
    dominant_law = max(law_counts, key=law_counts.get)
    median_r2 = float(np.median([r["best_r2"] for r in winners]))

    # Mean Akaike weights per law
    weight_accum = {}
    for r in winners:
        for law, w in r["akaike_weights"].items():
            weight_accum.setdefault(law, []).append(w)
    mean_akaike_weights = {law: float(np.mean(ws)) for law, ws in weight_accum.items()}

    # BIC concordance rate
    n_agree = sum(1 for r in winners if r.get("aicc_bic_agree", False))
    bic_concordance = n_agree / len(winners) if winners else 0.0

    # Ambiguity detection: cases where top-2 weights are within 15%
    n_ambiguous = sum(1 for r in winners if r.get("top2_weight_gap", 1.0) < 0.15)

    # Exponents
    exponents = []
    for r in winners:
        if r["best_law"] == dominant_law:
            exponents.append(r["best_params"].get("alpha", r["best_params"].get("b", None)))
    med_exp = float(np.median([e for e in exponents if e is not None])) if exponents else None

    interpretation = interpret_winner(axis, dominant_law,
        {"alpha": med_exp} if med_exp else {}, median_r2)

    return {
        "task": task, "model": model, "axis": axis,
        "dominant_law": dominant_law,
        "law_counts": law_counts,
        "mean_akaike_weights": mean_akaike_weights,
        "median_r2": median_r2,
        "median_exponent": med_exp,
        "n_slices": len(winners),
        "n_ambiguous": n_ambiguous,
        "bic_concordance": float(bic_concordance),
        "interpretation": interpretation,
    }


def _bootstrap_law_selection(slice_records, B=500, rng_seed=42):
    """Bootstrap law-selection stability.

    Resample slices with replacement B times, recompute vote winner each time.
    Returns dict with winner probabilities.
    """
    rng = np.random.RandomState(rng_seed)
    valid = [r for r in slice_records if r is not None]
    n = len(valid)
    if n == 0:
        return {"winner_probs": {}, "n_bootstrap": B}

    win_counts = {}
    for _ in range(B):
        idxs = rng.randint(0, n, size=n)
        boot_laws = [valid[i]["best_law"] for i in idxs]
        law_counts = {}
        for l in boot_laws:
            law_counts[l] = law_counts.get(l, 0) + 1
        boot_winner = max(law_counts, key=law_counts.get)
        win_counts[boot_winner] = win_counts.get(boot_winner, 0) + 1

    winner_probs = {law: count / B for law, count in win_counts.items()}
    return {
        "winner_probs": winner_probs,
        "n_bootstrap": B,
        "dominant_law": max(winner_probs, key=winner_probs.get),
        "dominant_prob": max(winner_probs.values()),
    }


# ──────────────────── Main ────────────────────

def main():
    t0 = time.time()
    df = pd.read_csv("runs/grouped_metrics_train.csv")
    df["test_rel_l2_mean"] = pd.to_numeric(df["test_rel_l2_mean"], errors="coerce")
    df = df.dropna(subset=["test_rel_l2_mean"])
    print(f"Loaded {len(df)} training groups")

    all_results = {
        "data_fits": [],
        "capacity_fits": [],
        "resolution_fits": [],
    }
    # Restricted-set results for resolution axis
    restricted_results = {"resolution_fits": []}

    summary_table = []
    bootstrap_results = []

    for task in sorted(df["task"].unique()):
        tdf = df[df["task"] == task]
        for model in sorted(tdf["model_name"].unique()):
            mdf = tdf[tdf["model_name"] == model]

            # ════════ Data scaling: E vs N ════════
            axis = "data"
            per_slice = []
            for (cap, R), sub in mdf.groupby(["capacity_name", "resolution"]):
                sub = sub.sort_values("dataset_size")
                X = sub["dataset_size"].values
                E = sub["test_rel_l2_mean"].values
                fits = fit_all_laws(X, E)
                rec = _build_slice_record(task, model, axis,
                    {"capacity_name": str(cap), "resolution": int(R)},
                    fits, len(X))
                if rec:
                    all_results["data_fits"].append(rec)
                    per_slice.append(rec)

            summary = _aggregate_axis(per_slice, task, model, axis)
            if summary:
                summary_table.append(summary)
                boot = _bootstrap_law_selection(per_slice)
                boot.update({"task": task, "model": model, "axis": axis})
                bootstrap_results.append(boot)

            # ════════ Capacity scaling: E vs D ════════
            axis = "capacity"
            per_slice = []
            for (N_val, R), sub in mdf.groupby(["dataset_size", "resolution"]):
                sub = sub.sort_values("parameter_count")
                X = sub["parameter_count"].values
                E = sub["test_rel_l2_mean"].values
                fits = fit_all_laws(X, E)
                rec = _build_slice_record(task, model, axis,
                    {"dataset_size": int(N_val), "resolution": int(R)},
                    fits, len(X))
                if rec:
                    all_results["capacity_fits"].append(rec)
                    per_slice.append(rec)

            summary = _aggregate_axis(per_slice, task, model, axis)
            if summary:
                summary_table.append(summary)
                boot = _bootstrap_law_selection(per_slice)
                boot.update({"task": task, "model": model, "axis": axis})
                bootstrap_results.append(boot)

            # ════════ Resolution scaling: E vs R ════════
            axis = "resolution"
            per_slice = []
            per_slice_restricted = []
            for (cap, N_val), sub in mdf.groupby(["capacity_name", "dataset_size"]):
                sub = sub.sort_values("resolution")
                X = sub["resolution"].values
                E = sub["test_rel_l2_mean"].values
                slice_keys = {"capacity_name": str(cap), "dataset_size": int(N_val)}

                # Full 6-law fit
                fits = fit_all_laws(X, E)
                rec = _build_slice_record(task, model, axis, slice_keys, fits, len(X))
                if rec:
                    all_results["resolution_fits"].append(rec)
                    per_slice.append(rec)

                # Restricted 2-param families (for n=4 resolution slices)
                fits_r = fit_all_laws(X, E, candidate_laws=RESTRICTED_LAWS)
                rec_r = _build_slice_record(task, model, axis, slice_keys, fits_r, len(X),
                                            candidate_laws=RESTRICTED_LAWS)
                if rec_r:
                    restricted_results["resolution_fits"].append(rec_r)
                    per_slice_restricted.append(rec_r)

            summary = _aggregate_axis(per_slice, task, model, axis)
            if summary:
                summary_table.append(summary)
                boot = _bootstrap_law_selection(per_slice)
                boot.update({"task": task, "model": model, "axis": axis})
                bootstrap_results.append(boot)

            print(f"  {task:20s} {model:16s} — done")

    # ──── Print summary ────
    print(f"\n{'='*90}")
    print("MULTI-LAW MODEL SELECTION SUMMARY (with Akaike weights & BIC)")
    print(f"{'='*90}")
    print(f"{'Task':<22s} {'Model':<16s} {'Axis':<12s} {'Winner':<16s} {'R²':<8s} "
          f"{'BIC conc.':<10s} {'Boot. prob':<10s} {'Law votes'}")
    print("-" * 140)
    for row in summary_table:
        votes = ", ".join(f"{k}:{v}" for k, v in sorted(row["law_counts"].items(), key=lambda x: -x[1]))
        # Find bootstrap result
        boot_row = [b for b in bootstrap_results
                    if b["task"] == row["task"] and b["model"] == row["model"] and b["axis"] == row["axis"]]
        boot_str = f"{boot_row[0]['dominant_prob']:.0%}" if boot_row else "N/A"
        print(f"{row['task']:<22s} {row['model']:<16s} {row['axis']:<12s} "
              f"{row['dominant_law']:<16s} {row['median_r2']:<8.3f} "
              f"{row['bic_concordance']:<10.0%} {boot_str:<10s} {votes}")

    # ──── Ambiguity report ────
    print(f"\n{'='*90}")
    print("AMBIGUITY REPORT (slices where top-2 Akaike weights differ by <15%)")
    print(f"{'='*90}")
    for row in summary_table:
        if row["n_ambiguous"] > 0:
            print(f"  {row['task']:20s} {row['model']:16s} {row['axis']:12s}: "
                  f"{row['n_ambiguous']}/{row['n_slices']} ambiguous slices")

    # ──── BIC concordance ────
    print(f"\n{'='*90}")
    print("BIC CONCORDANCE (fraction where AICc and BIC agree on winner)")
    print(f"{'='*90}")
    for row in summary_table:
        print(f"  {row['task']:20s} {row['model']:16s} {row['axis']:12s}: "
              f"{row['bic_concordance']:.0%}")

    # ──── Bootstrap stability ────
    print(f"\n{'='*90}")
    print("BOOTSTRAP LAW-SELECTION STABILITY (B=500)")
    print(f"{'='*90}")
    for b in bootstrap_results:
        probs = ", ".join(f"{k}:{v:.0%}" for k, v in
                          sorted(b["winner_probs"].items(), key=lambda x: -x[1])[:3])
        print(f"  {b['task']:20s} {b['model']:16s} {b['axis']:12s}: {probs}")

    # ──── Restricted vs full (resolution) ────
    print(f"\n{'='*90}")
    print("RESTRICTED (3-law) vs FULL (6-law) — Resolution axis")
    print(f"{'='*90}")
    for task in sorted(df["task"].unique()):
        for model in sorted(df["model_name"].unique()):
            full_recs = [r for r in all_results["resolution_fits"]
                         if r["task"] == task and r["model_name"] == model]
            rest_recs = [r for r in restricted_results["resolution_fits"]
                         if r["task"] == task and r["model_name"] == model]
            if not full_recs or not rest_recs:
                continue
            # Count changes
            n_same, n_diff = 0, 0
            for fr, rr in zip(full_recs, rest_recs):
                if fr["best_law"] == rr["best_law"]:
                    n_same += 1
                else:
                    n_diff += 1
            full_summary = _aggregate_axis(full_recs, task, model, "resolution")
            rest_summary = _aggregate_axis(rest_recs, task, model, "resolution")
            full_winner = full_summary["dominant_law"] if full_summary else "?"
            rest_winner = rest_summary["dominant_law"] if rest_summary else "?"
            print(f"  {task:20s} {model:16s}: full={full_winner:14s} restricted={rest_winner:14s}  "
                  f"agree={n_same}/{n_same+n_diff}")

    # ──── E_inf (irreducible error) analysis ────
    print(f"\n{'='*90}")
    print("IRREDUCIBLE ERROR FLOORS (E_inf) — from best per-slice fits")
    print(f"{'='*90}")
    for axis_name in ["data", "resolution", "capacity"]:
        print(f"\n--- {axis_name.upper()} axis ---")
        axis_fits = all_results[f"{axis_name}_fits"]
        for task in sorted(df["task"].unique()):
            for model in sorted(df["model_name"].unique()):
                e_infs = []
                for f in axis_fits:
                    if f["task"] == task and f["model_name"] == model:
                        bp = f.get("best_params", {})
                        if "E_inf" in bp:
                            e_infs.append(bp["E_inf"])
                if e_infs:
                    print(f"  {task:20s} {model:16s}: "
                          f"median E_inf = {np.median(e_infs):.4f}  "
                          f"[{np.percentile(e_infs, 25):.4f}, {np.percentile(e_infs, 75):.4f}]")

    # ──── Save ────
    out = Path("results")
    out.mkdir(exist_ok=True)

    # Save detailed results (without all_fits to keep file size manageable)
    save_results = {}
    for key in all_results:
        save_results[key] = []
        for rec in all_results[key]:
            r = {k: v for k, v in rec.items() if k != "all_fits"}
            save_results[key].append(r)
    save_results["summary"] = summary_table
    save_results["bootstrap_law_selection"] = bootstrap_results

    with open(out / "multilaw_fits.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nSaved → results/multilaw_fits.json")

    # Save restricted-set results
    save_restricted = {}
    for key in restricted_results:
        save_restricted[key] = []
        for rec in restricted_results[key]:
            r = {k: v for k, v in rec.items() if k != "all_fits"}
            save_restricted[key].append(r)
    # Build restricted summary
    restricted_summary = []
    for task in sorted(df["task"].unique()):
        for model in sorted(df["model_name"].unique()):
            rest_recs = [r for r in restricted_results["resolution_fits"]
                         if r["task"] == task and r["model_name"] == model]
            s = _aggregate_axis(rest_recs, task, model, "resolution")
            if s:
                restricted_summary.append(s)
    save_restricted["summary"] = restricted_summary

    with open(out / "multilaw_fits_restricted.json", "w") as f:
        json.dump(save_restricted, f, indent=2, default=str)
    print(f"Saved → results/multilaw_fits_restricted.json")

    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
