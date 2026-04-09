#!/usr/bin/env python3
"""
Independent data analysis of scaling operator learning experiments.
No reliance on paper framing — pure data exploration from a blank slate.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────
print("=" * 80)
print("INDEPENDENT ANALYSIS: SCALING OPERATOR LEARNING")
print("=" * 80)

df_grouped = pd.read_csv("runs/grouped_metrics.csv")
df_raw = pd.read_csv("runs/runs_aggregate.csv")

print(f"\nGrouped data: {len(df_grouped)} rows")
print(f"Raw runs: {len(df_raw)} rows")
print(f"\nTasks: {sorted(df_raw['task'].unique())}")
print(f"Models: {sorted(df_raw['model_name'].unique())}")
print(f"Capacities: {sorted(df_raw['capacity_name'].unique())}")
print(f"Dataset sizes (N): {sorted(df_raw['dataset_size'].unique())}")
print(f"Resolutions (R): {sorted(df_raw['resolution'].unique())}")
print(f"Eval resolutions: {sorted(df_raw['eval_resolution'].dropna().unique())}")

# ─────────────────────────────────────────────────────────────────────
# 2. DATA COMPLETENESS & HEALTH
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: DATA HEALTH & COMPLETENESS")
print("=" * 80)

# Status distribution
print("\n--- Run Status ---")
print(df_raw['status'].value_counts())

# Divergence rates by model
print("\n--- Divergence Rate by Model ---")
div_by_model = df_raw.groupby('model_name').agg(
    total=('status', 'count'),
    diverged=('diverged', lambda x: x.sum() if x.notna().any() else 0),
    success=('status', lambda x: (x == 'success').sum())
)
div_by_model['success_rate'] = div_by_model['success'] / div_by_model['total']
print(div_by_model)

# Divergence by model × task
print("\n--- Divergence Rate by Model × Task ---")
for task in sorted(df_raw['task'].unique()):
    for model in sorted(df_raw['model_name'].unique()):
        subset = df_raw[(df_raw['task'] == task) & (df_raw['model_name'] == model)]
        if len(subset) > 0:
            success_rate = (subset['status'] == 'success').mean()
            n = len(subset)
            print(f"  {task:25s} × {model:18s}: {success_rate:.3f} success ({n} runs)")

# ─────────────────────────────────────────────────────────────────────
# 3. BASIC PERFORMANCE LANDSCAPE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: PERFORMANCE LANDSCAPE")
print("=" * 80)

# Use grouped data (already averaged over seeds) for cleaner analysis
# Filter to eval_resolution == NaN or eval_resolution == resolution (same-res eval)
g = df_grouped.copy()
# Some rows have eval_resolution set — filter to same-resolution eval or no eval_resolution
g_same = g[g['eval_resolution'].isna() | (g['eval_resolution'] == g['resolution'])]

print(f"\nGrouped data (same-res eval): {len(g_same)} rows")

# Best achievable error per task × model (across all capacity, N, R)
print("\n--- Best Test Rel-L2 per Task × Model ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) > 0:
            best_row = subset.loc[subset['test_rel_l2_mean'].idxmin()]
            print(f"    {model:18s}: {best_row['test_rel_l2_mean']:.5f} "
                  f"(N={int(best_row['dataset_size'])}, R={int(best_row['resolution'])}, "
                  f"cap={best_row['capacity_name']}, params={int(best_row['parameter_count']):,})")

# ─────────────────────────────────────────────────────────────────────
# 4. MARGINAL EFFECTS: How does each axis affect error?
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: MARGINAL EFFECTS OF EACH AXIS")
print("=" * 80)

# 4a. Dataset size (N) effect - averaged over other axes
print("\n--- Effect of Dataset Size (N) [averaged over R, capacity] ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        by_N = subset.groupby('dataset_size')['test_rel_l2_mean'].mean()
        if len(by_N) >= 2:
            N_vals = by_N.index.values
            err_vals = by_N.values
            # Compute ratio of error at largest N to smallest N
            ratio = err_vals[-1] / err_vals[0]
            # Log-log slope (effective power-law exponent)
            slope, _, r, _, _ = stats.linregress(np.log(N_vals), np.log(err_vals))
            print(f"    {model:18s}: error drops {(1-ratio)*100:.1f}% from N={N_vals[0]} to N={N_vals[-1]}"
                  f" | log-log slope={slope:.3f} (R²={r**2:.3f})")

# 4b. Resolution (R) effect
print("\n--- Effect of Resolution (R) [averaged over N, capacity] ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        by_R = subset.groupby('resolution')['test_rel_l2_mean'].mean()
        if len(by_R) >= 2:
            R_vals = by_R.index.values
            err_vals = by_R.values
            ratio = err_vals[-1] / err_vals[0]
            slope, _, r, _, _ = stats.linregress(np.log(R_vals), np.log(err_vals))
            direction = "IMPROVES" if ratio < 1 else "WORSENS" if ratio > 1 else "FLAT"
            print(f"    {model:18s}: error changes {(ratio-1)*100:+.1f}% from R={R_vals[0]} to R={R_vals[-1]}"
                  f" ({direction}) | log-log slope={slope:.3f} (R²={r**2:.3f})")

# 4c. Capacity (parameter count) effect
print("\n--- Effect of Model Capacity [averaged over N, R] ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        by_D = subset.groupby('parameter_count')['test_rel_l2_mean'].mean().sort_index()
        if len(by_D) >= 2:
            D_vals = by_D.index.values
            err_vals = by_D.values
            ratio = err_vals[-1] / err_vals[0]
            slope, _, r, _, _ = stats.linregress(np.log(D_vals), np.log(err_vals))
            direction = "IMPROVES" if ratio < 1 else "WORSENS" if ratio > 1 else "FLAT"
            print(f"    {model:18s}: error changes {(ratio-1)*100:+.1f}% from {D_vals[0]:,.0f} to {D_vals[-1]:,.0f} params"
                  f" ({direction}) | log-log slope={slope:.3f} (R²={r**2:.3f})")

# ─────────────────────────────────────────────────────────────────────
# 5. INTERACTION EFFECTS: Do axes interact?
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5: INTERACTION EFFECTS BETWEEN AXES")
print("=" * 80)

# 5a. Does the N-scaling exponent change with R?
print("\n--- N-scaling exponent (log-log slope) at different resolutions ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        print(f"    {model}:")
        for R in sorted(subset['resolution'].unique()):
            sub_R = subset[subset['resolution'] == R]
            by_N = sub_R.groupby('dataset_size')['test_rel_l2_mean'].mean()
            if len(by_N) >= 3:
                slope, _, r, _, _ = stats.linregress(np.log(by_N.index.values), np.log(by_N.values))
                print(f"      R={R:>4d}: N-exponent = {slope:.3f} (R²={r**2:.3f})")

# 5b. Does the R-scaling exponent change with N?
print("\n--- R-scaling exponent (log-log slope) at different dataset sizes ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        print(f"    {model}:")
        for N in sorted(subset['dataset_size'].unique()):
            sub_N = subset[subset['dataset_size'] == N]
            by_R = sub_N.groupby('resolution')['test_rel_l2_mean'].mean()
            if len(by_R) >= 3:
                slope, _, r, _, _ = stats.linregress(np.log(by_R.index.values), np.log(by_R.values))
                print(f"      N={N:>5d}: R-exponent = {slope:.3f} (R²={r**2:.3f})")

# ─────────────────────────────────────────────────────────────────────
# 6. CROSS-RESOLUTION GENERALIZATION
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: CROSS-RESOLUTION GENERALIZATION")
print("=" * 80)

# Look at eval_resolution != resolution
g_cross = df_grouped[df_grouped['eval_resolution'].notna()].copy()
g_cross['eval_resolution'] = g_cross['eval_resolution'].astype(int)
g_cross['resolution'] = g_cross['resolution'].astype(int)
g_cross['res_ratio'] = g_cross['eval_resolution'] / g_cross['resolution']

if len(g_cross) > 0:
    print(f"\nCross-resolution data: {len(g_cross)} rows")
    
    # For each model, how does error change when eval Resolution differs from training?
    print("\n--- Error vs eval/train resolution ratio ---")
    for task in sorted(g_cross['task'].unique()):
        print(f"\n  Task: {task}")
        for model in sorted(g_cross['model_name'].unique()):
            subset = g_cross[(g_cross['task'] == task) & (g_cross['model_name'] == model)]
            if len(subset) == 0:
                continue
            by_ratio = subset.groupby('res_ratio')['test_rel_l2_mean'].mean()
            print(f"    {model}:")
            for ratio, err in by_ratio.items():
                marker = " <-- same res" if ratio == 1.0 else ""
                print(f"      eval/train = {ratio:.2f}: mean error = {err:.5f}{marker}")
else:
    print("\n  No cross-resolution evaluation data found.")

# ─────────────────────────────────────────────────────────────────────
# 7. EFFICIENCY ANALYSIS: PARETO FRONTIERS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 7: EFFICIENCY ANALYSIS")
print("=" * 80)

# 7a. Error vs compute (runtime)
print("\n--- Which model gives best error for given compute budget? ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        # Get rows with valid runtime
        valid = subset[subset['runtime_seconds_mean'].notna() & (subset['runtime_seconds_mean'] > 0)]
        if len(valid) > 0:
            mean_runtime = valid['runtime_seconds_mean'].mean()
            best_err = valid['test_rel_l2_mean'].min()
            median_err = valid['test_rel_l2_mean'].median()
            print(f"    {model:18s}: best={best_err:.5f}, median={median_err:.5f}, "
                  f"mean runtime={mean_runtime:.1f}s")

# 7b. Error per parameter (efficiency metric)
print("\n--- Parameter Efficiency: best error / log(param_count) ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        # For each capacity level, best error
        by_cap = subset.groupby(['capacity_name', 'parameter_count']).agg(
            best_err=('test_rel_l2_mean', 'min')
        ).reset_index().sort_values('parameter_count')
        print(f"    {model}:")
        for _, row in by_cap.iterrows():
            print(f"      {row['capacity_name']:>12s} ({row['parameter_count']:>10,.0f} params): "
                  f"best error = {row['best_err']:.5f}")

# ─────────────────────────────────────────────────────────────────────
# 8. VARIANCE ANALYSIS: How noisy are the estimates?
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 8: VARIANCE & NOISE ANALYSIS")
print("=" * 80)

# Coefficient of variation by axis
print("\n--- Coefficient of Variation of test_rel_l2 across seeds ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        valid = subset[subset['test_rel_l2_std'].notna() & (subset['test_rel_l2_mean'] > 0)]
        if len(valid) > 0:
            cv = (valid['test_rel_l2_std'] / valid['test_rel_l2_mean'])
            print(f"    {model:18s}: CV mean={cv.mean():.3f}, CV median={cv.median():.3f}, "
                  f"CV max={cv.max():.3f}")

# Which regimes are noisiest?
print("\n--- Noisiest regimes (highest CV) ---")
valid_cv = g_same[g_same['test_rel_l2_std'].notna() & (g_same['test_rel_l2_mean'] > 0)].copy()
valid_cv['cv'] = valid_cv['test_rel_l2_std'] / valid_cv['test_rel_l2_mean']
noisiest = valid_cv.nlargest(20, 'cv')[['task', 'model_name', 'capacity_name', 'dataset_size', 
                                          'resolution', 'test_rel_l2_mean', 'cv']]
print(noisiest.to_string())

# ─────────────────────────────────────────────────────────────────────
# 9. DIMINISHING RETURNS ANALYSIS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 9: DIMINISHING RETURNS & SATURATION")
print("=" * 80)

# For each task × model, compute marginal improvement from doubling N
print("\n--- Marginal improvement from doubling N ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        by_N = subset.groupby('dataset_size')['test_rel_l2_mean'].mean().sort_index()
        N_vals = by_N.index.values
        err_vals = by_N.values
        print(f"    {model}:")
        for i in range(len(N_vals) - 1):
            improvement = (err_vals[i] - err_vals[i+1]) / err_vals[i] * 100
            factor = N_vals[i+1] / N_vals[i]
            print(f"      N: {N_vals[i]:>5d} → {N_vals[i+1]:>5d} ({factor:.1f}×): "
                  f"error improves {improvement:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 10. LAW FITTING: INDEPENDENT POWER-LAW + ALTERNATIVES
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 10: INDEPENDENT SCALING LAW FITS")
print("=" * 80)

def power_law(x, e_inf, a, alpha):
    return e_inf + a * np.power(x, -alpha)

def exponential_decay(x, e_inf, a, alpha):
    return e_inf + a * np.exp(-alpha * x)

def logarithmic(x, a, b):
    return a - b * np.log(x)

def fit_and_compare(N_vals, err_vals, label=""):
    """Fit multiple functional forms and compare via AICc."""
    n = len(N_vals)
    if n < 4:
        return None
    
    results = {}
    
    # Power law
    try:
        popt, _ = curve_fit(power_law, N_vals, err_vals, p0=[0.01, 1.0, 0.5], 
                           maxfev=5000, bounds=([0, 0, 0.01], [np.inf, np.inf, 5.0]))
        pred = power_law(N_vals, *popt)
        ss_res = np.sum((err_vals - pred)**2)
        ss_tot = np.sum((err_vals - np.mean(err_vals))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        k = 3
        aic = n * np.log(ss_res / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
        results['power'] = {'params': popt, 'r2': r2, 'aicc': aicc, 
                           'e_inf': popt[0], 'exponent': popt[2]}
    except Exception:
        pass
    
    # Exponential
    try:
        popt, _ = curve_fit(exponential_decay, N_vals, err_vals, p0=[0.01, 1.0, 0.001],
                           maxfev=5000, bounds=([0, 0, 1e-8], [np.inf, np.inf, 1.0]))
        pred = exponential_decay(N_vals, *popt)
        ss_res = np.sum((err_vals - pred)**2)
        ss_tot = np.sum((err_vals - np.mean(err_vals))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        k = 3
        aic = n * np.log(ss_res / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
        results['exponential'] = {'params': popt, 'r2': r2, 'aicc': aicc, 
                                  'e_inf': popt[0]}
    except Exception:
        pass
    
    # Logarithmic
    try:
        popt, _ = curve_fit(logarithmic, N_vals, err_vals, p0=[1.0, 0.1], maxfev=5000)
        pred = logarithmic(N_vals, *popt)
        ss_res = np.sum((err_vals - pred)**2)
        ss_tot = np.sum((err_vals - np.mean(err_vals))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        k = 2
        aic = n * np.log(ss_res / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
        results['logarithmic'] = {'params': popt, 'r2': r2, 'aicc': aicc}
    except Exception:
        pass
    
    if not results:
        return None
    
    # Select best by AICc
    best_law = min(results, key=lambda k: results[k]['aicc'])
    return results, best_law

# Fit N-scaling for each task × model × resolution (medium capacity)
print("\n--- N-scaling law fits (medium capacity, per resolution) ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model) &
                        (g_same['capacity_name'] == 'medium')]
        if len(subset) == 0:
            continue
        print(f"    {model}:")
        for R in sorted(subset['resolution'].unique()):
            sub_R = subset[subset['resolution'] == R]
            by_N = sub_R.groupby('dataset_size')['test_rel_l2_mean'].mean().sort_index()
            result = fit_and_compare(by_N.index.values.astype(float), by_N.values)
            if result:
                results, best = result
                r2 = results[best]['r2']
                delta = ""
                if len(results) > 1:
                    aicc_sorted = sorted(results.items(), key=lambda x: x[1]['aicc'])
                    if len(aicc_sorted) >= 2:
                        delta = f", Δ(AICc)={aicc_sorted[1][1]['aicc'] - aicc_sorted[0][1]['aicc']:.1f}"
                extra = ""
                if best == 'power' and 'exponent' in results[best]:
                    extra = f", α={results[best]['exponent']:.3f}, E∞={results[best]['e_inf']:.5f}"
                elif best == 'exponential':
                    extra = f", E∞={results[best]['e_inf']:.5f}"
                print(f"      R={R:>4d}: best={best:>12s} (R²={r2:.4f}{delta}{extra})")

# Fit R-scaling for each task × model × N (medium capacity)
print("\n--- R-scaling law fits (medium capacity, per N) ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model) &
                        (g_same['capacity_name'] == 'medium')]
        if len(subset) == 0:
            continue
        print(f"    {model}:")
        for N in sorted(subset['dataset_size'].unique()):
            sub_N = subset[subset['dataset_size'] == N]
            by_R = sub_N.groupby('resolution')['test_rel_l2_mean'].mean().sort_index()
            if len(by_R) >= 3:
                result = fit_and_compare(by_R.index.values.astype(float), by_R.values)
                if result:
                    results, best = result
                    r2 = results[best]['r2']
                    # Check if error increases with R
                    if by_R.values[-1] > by_R.values[0]:
                        direction = " ⚠ ERROR INCREASES WITH R"
                    else:
                        direction = ""
                    print(f"      N={N:>5d}: best={best:>12s} (R²={r2:.4f}){direction}")

# ─────────────────────────────────────────────────────────────────────
# 11. ARCHITECTURE COMPARISON: WHICH MODEL WINS WHERE?
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 11: ARCHITECTURE COMPARISON — WHO WINS WHERE?")
print("=" * 80)

# Head-to-head at matched parameter count (approximately)
print("\n--- Model ranking at each (task, N, R) [using medium capacity] ---")
wins = {m: 0 for m in g_same['model_name'].unique()}
total_comparisons = 0

for task in sorted(g_same['task'].unique()):
    for N in sorted(g_same['dataset_size'].unique()):
        for R in sorted(g_same['resolution'].unique()):
            subset = g_same[(g_same['task'] == task) & 
                           (g_same['dataset_size'] == N) & 
                           (g_same['resolution'] == R) &
                           (g_same['capacity_name'] == 'medium')]
            if len(subset) >= 2:
                valid_subset = subset.dropna(subset=['test_rel_l2_mean'])
                if len(valid_subset) >= 2:
                    best_model = valid_subset.loc[valid_subset['test_rel_l2_mean'].idxmin(), 'model_name']
                    wins[best_model] += 1
                    total_comparisons += 1

print(f"\n  Total comparisons: {total_comparisons}")
for model, w in sorted(wins.items(), key=lambda x: -x[1]):
    pct = w / total_comparisons * 100 if total_comparisons > 0 else 0
    print(f"    {model:18s}: wins {w}/{total_comparisons} ({pct:.1f}%)")

# Where does each model dominate?
print("\n--- Regime analysis: where does each model excel? ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for N in sorted(g_same['dataset_size'].unique()):
        for R in sorted(g_same['resolution'].unique()):
            subset = g_same[(g_same['task'] == task) & 
                           (g_same['dataset_size'] == N) & 
                           (g_same['resolution'] == R) &
                           (g_same['capacity_name'] == 'medium')]
            if len(subset) >= 2:
                valid_subset2 = subset.dropna(subset=['test_rel_l2_mean'])
                if len(valid_subset2) >= 2:
                    ranked = valid_subset2.sort_values('test_rel_l2_mean')
                    winner = ranked.iloc[0]['model_name']
                    second = ranked.iloc[1]['model_name']
                    gap = ranked.iloc[1]['test_rel_l2_mean'] - ranked.iloc[0]['test_rel_l2_mean']
                    rel_gap = gap / ranked.iloc[0]['test_rel_l2_mean'] * 100
                    if rel_gap > 10:  # Only show decisive wins
                        print(f"    N={N:>5d}, R={R:>4d}: {winner} beats {second} by {rel_gap:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 12. MLP RESOLUTION CONFOUND
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 12: MLP RESOLUTION CONFOUND (CAPACITY GROWS WITH R)")
print("=" * 80)

for task in sorted(g_same['task'].unique()):
    mlp_data = g_same[(g_same['task'] == task) & (g_same['model_name'] == 'mlp_baseline')]
    if len(mlp_data) == 0:
        continue
    print(f"\n  Task: {task}")
    # How do params change with R for MLP?
    by_R_cap = mlp_data.groupby(['resolution', 'capacity_name'])['parameter_count'].first().reset_index()
    by_R_cap = by_R_cap.sort_values(['capacity_name', 'resolution'])
    for cap in sorted(by_R_cap['capacity_name'].unique()):
        sub = by_R_cap[by_R_cap['capacity_name'] == cap]
        params_list = [f"R={r}: {p:,}" for r, p in zip(sub['resolution'], sub['parameter_count'])]
        print(f"    {cap:>12s}: {' | '.join(params_list)}")
    
    # Compare mlp_baseline vs mlp_controlled if available
    mlp_ctrl = g_same[(g_same['task'] == task) & (g_same['model_name'] == 'mlp_controlled')]
    if len(mlp_ctrl) > 0:
        print(f"\n    MLP controlled (fixed params) vs baseline:")
        for R in sorted(mlp_data['resolution'].unique()):
            bl = mlp_data[(mlp_data['resolution'] == R) & (mlp_data['capacity_name'] == 'medium')]
            ctrl = mlp_ctrl[(mlp_ctrl['resolution'] == R) & (mlp_ctrl['capacity_name'] == 'medium')]
            if len(bl) > 0 and len(ctrl) > 0:
                bl_err = bl['test_rel_l2_mean'].mean()
                ctrl_err = ctrl['test_rel_l2_mean'].mean()
                print(f"      R={R:>4d}: baseline={bl_err:.5f} (params={bl['parameter_count'].iloc[0]:,}) | "
                      f"controlled={ctrl_err:.5f} (params={ctrl['parameter_count'].iloc[0]:,})")

# ─────────────────────────────────────────────────────────────────────
# 13. CORRELATION MATRIX OF AXES
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 13: RELATIVE IMPORTANCE OF SCALING AXES (via log-linear regression)")
print("=" * 80)

for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)].copy()
        if len(subset) < 10:
            continue
        # Log-linear model: log(E) = a0 + a1*log(N) + a2*log(R) + a3*log(D) + noise
        valid = subset[subset['test_rel_l2_mean'] > 0].copy()
        if len(valid) < 10:
            continue
        logE = np.log(valid['test_rel_l2_mean'].values)
        logN = np.log(valid['dataset_size'].values)
        logR = np.log(valid['resolution'].values)
        logD = np.log(valid['parameter_count'].values)
        
        X = np.column_stack([np.ones(len(logE)), logN, logR, logD])
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, logE, rcond=None)
            pred = X @ coeffs
            ss_res = np.sum((logE - pred)**2)
            ss_tot = np.sum((logE - np.mean(logE))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"    {model:18s}: log(E) = {coeffs[0]:.2f} + {coeffs[1]:.3f}·log(N) "
                  f"+ {coeffs[2]:.3f}·log(R) + {coeffs[3]:.3f}·log(D)  (R²={r2:.4f})")
            
            # Standardized coefficients (relative importance)
            stds = np.std(X[:, 1:], axis=0)
            std_coeffs = coeffs[1:] * stds / np.std(logE)
            total = np.sum(np.abs(std_coeffs))
            if total > 0:
                rel_imp = np.abs(std_coeffs) / total * 100
                print(f"                      Relative importance: N={rel_imp[0]:.1f}%, R={rel_imp[1]:.1f}%, D={rel_imp[2]:.1f}%")
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────
# 14. SURPRISING PATTERNS / ANOMALIES
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 14: ANOMALIES & SURPRISING PATTERNS")
print("=" * 80)

# Cases where more data hurts
print("\n--- Cases where more data INCREASES error (non-monotonic N-scaling) ---")
for task in sorted(g_same['task'].unique()):
    for model in sorted(g_same['model_name'].unique()):
        for cap in sorted(g_same['capacity_name'].unique()):
            for R in sorted(g_same['resolution'].unique()):
                subset = g_same[(g_same['task'] == task) & 
                               (g_same['model_name'] == model) &
                               (g_same['capacity_name'] == cap) &
                               (g_same['resolution'] == R)]
                by_N = subset.sort_values('dataset_size')
                if len(by_N) >= 3:
                    errs = by_N['test_rel_l2_mean'].values
                    Ns = by_N['dataset_size'].values
                    for i in range(len(errs) - 1):
                        if errs[i+1] > errs[i] * 1.05:  # 5% increase
                            print(f"  {task} / {model} / {cap} / R={R}: "
                                  f"error rises from N={Ns[i]} ({errs[i]:.5f}) to N={Ns[i+1]} ({errs[i+1]:.5f})")
                            break

# Cases where more resolution hurts significantly
print("\n--- Cases where higher R INCREASES error (with magnitude) ---")
for task in sorted(g_same['task'].unique()):
    for model in sorted(g_same['model_name'].unique()):
        for cap in ['medium']:  # Focus on medium capacity
            subset = g_same[(g_same['task'] == task) & 
                           (g_same['model_name'] == model) &
                           (g_same['capacity_name'] == cap)]
            by_R = subset.groupby('resolution')['test_rel_l2_mean'].mean().sort_index()
            if len(by_R) >= 2:
                R_vals = by_R.index.values
                err_vals = by_R.values
                if err_vals[-1] > err_vals[0] * 1.02:  # more than 2% increase
                    pct = (err_vals[-1] / err_vals[0] - 1) * 100
                    print(f"  {task:25s} / {model:18s}: R={R_vals[0]}→{R_vals[-1]} "
                          f"increases error by {pct:.1f}% ({err_vals[0]:.5f} → {err_vals[-1]:.5f})")

# ─────────────────────────────────────────────────────────────────────
# 15. TASK DIFFICULTY RANKING & CHARACTERISTICS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 15: TASK DIFFICULTY & DATA EFFICIENCY")
print("=" * 80)

# How many samples needed to reach a given error threshold?
thresholds = [0.1, 0.05, 0.02]
print("\n--- Minimum N to reach error threshold (best model, medium capacity) ---")
for task in sorted(g_same['task'].unique()):
    print(f"\n  Task: {task}")
    for thresh in thresholds:
        best_N = None
        best_model = None
        for model in sorted(g_same['model_name'].unique()):
            subset = g_same[(g_same['task'] == task) & 
                           (g_same['model_name'] == model) &
                           (g_same['capacity_name'] == 'medium')]
            below = subset[subset['test_rel_l2_mean'] < thresh]
            if len(below) > 0:
                min_N = below['dataset_size'].min()
                if best_N is None or min_N < best_N:
                    best_N = min_N
                    best_model = model
        if best_N is not None:
            print(f"    error < {thresh}: N={best_N} (model={best_model})")
        else:
            print(f"    error < {thresh}: NOT ACHIEVED")

# ─────────────────────────────────────────────────────────────────────
# 16. OPTIMIZER ABLATION (if data exists)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 16: OPTIMIZER ABLATION COMPARISON")
print("=" * 80)

import os
ablation_dirs = [d for d in os.listdir('.') if d.startswith('runs_ablation')]
if ablation_dirs:
    print(f"\nAblation directories found: {ablation_dirs}")
    for abl_dir in ablation_dirs:
        agg_path = os.path.join(abl_dir, 'grouped_metrics.csv') 
        if not os.path.exists(agg_path):
            # Look for metrics in subdirectories
            csv_files = []
            for root, dirs, files in os.walk(abl_dir):
                for f in files:
                    if f == 'metrics.json':
                        csv_files.append(os.path.join(root, f))
            print(f"  {abl_dir}: {len(csv_files)} metrics.json files found (not aggregated)")
        else:
            abl_df = pd.read_csv(agg_path)
            print(f"  {abl_dir}: {len(abl_df)} grouped rows")
else:
    print("\n  No ablation data found.")

# Check for optimizer info in raw data
optimizers = df_raw['optimizer'].dropna().unique()
schedulers = df_raw['scheduler'].dropna().unique()
if len(optimizers) > 0:
    print(f"\n  Optimizers in raw data: {optimizers}")
if len(schedulers) > 0:
    print(f"  Schedulers in raw data: {schedulers}")

# ─────────────────────────────────────────────────────────────────────
# 17. SUMMARY STATISTICS TABLE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 17: GRAND SUMMARY")
print("=" * 80)

print("\n--- Complete task × model × axis summary ---")
print(f"{'Task':25s} {'Model':18s} {'N-exponent':>11s} {'R-exponent':>11s} {'D-exponent':>11s} {'Best Error':>11s}")
print("-" * 90)

for task in sorted(g_same['task'].unique()):
    for model in sorted(g_same['model_name'].unique()):
        subset = g_same[(g_same['task'] == task) & (g_same['model_name'] == model)]
        if len(subset) == 0:
            continue
        
        # N exponent
        by_N = subset.groupby('dataset_size')['test_rel_l2_mean'].mean()
        if len(by_N) >= 3:
            n_slope, _, n_r, _, _ = stats.linregress(np.log(by_N.index.values), np.log(by_N.values))
        else:
            n_slope = float('nan')
        
        # R exponent
        by_R = subset.groupby('resolution')['test_rel_l2_mean'].mean()
        if len(by_R) >= 3:
            r_slope, _, r_r, _, _ = stats.linregress(np.log(by_R.index.values), np.log(by_R.values))
        else:
            r_slope = float('nan')
        
        # D exponent
        by_D = subset.groupby('parameter_count')['test_rel_l2_mean'].mean().sort_index()
        if len(by_D) >= 3:
            d_slope, _, d_r, _, _ = stats.linregress(np.log(by_D.index.values), np.log(by_D.values))
        else:
            d_slope = float('nan')
        
        best = subset['test_rel_l2_mean'].min()
        
        print(f"{task:25s} {model:18s} {n_slope:>11.3f} {r_slope:>11.3f} {d_slope:>11.3f} {best:>11.5f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
