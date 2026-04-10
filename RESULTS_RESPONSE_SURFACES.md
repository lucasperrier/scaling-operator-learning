# Results — Response surfaces (Phase 3)

**Date:** 2026-04-09
**Reads:** `results/cell_stats_diagonal.csv`, `results/cell_regimes.csv`, `results/regime_calibration.json`, `results/marginal_value_of_resolution.csv`
**Generates:** `figures/response_surface_<task>_<model>.png` and `figures/regime_map_<task>_<model>.png` via `scripts/plot_response_surfaces.py` (figures are gitignored — regenerable from the script).

This document interprets what the diagonal training-cell response surfaces look like when we apply PROTOCOL_V2 §6's sensitivity-based regime taxonomy. It is the first analysis that does *not* go through 1D law fits.

---

## §1 Threshold calibration (PROTOCOL §6.2)

Empirically calibrated from the pooled distribution of |S_N|, |S_R|, |S_C| across all non-R48 main-grid cells (n=9198 sensitivity observations across 3,227 valid cells). All thresholds are below.

| Threshold | Value | Source |
|---|---|---|
| `THRESH_DIV` | 0.10 | Locked in PROTOCOL §6.2 |
| `THRESH_CV` | 0.50 | Locked in PROTOCOL §6.2 |
| `THRESH_LOW` | **0.0365** | Empirical p25 of pooled \|S\| |
| `THRESH_HIGH` | **0.2266** | Empirical p75 of pooled \|S\| |

Pooled |S| quantiles for reference:

| p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|
| 0.013 | 0.037 | 0.094 | 0.227 | 0.501 |

Per-axis median |S| (which axis the data has the most response on, in absolute terms):

| Axis | p25 | p50 | p75 |
|---|---|---|---|
| `|S_N|` (data) | 0.056 | **0.130** | 0.264 |
| `|S_R|` (resolution) | 0.037 | 0.102 | 0.264 |
| `|S_C|` (capacity) | 0.026 | 0.065 | 0.157 |

**Reading.** The dataset axis carries the largest sensitivities on average, the resolution axis a comparable but slightly smaller signal, and the capacity axis the smallest. Capacity sensitivity being the lowest is consistent with the AUDIT.md observation that capacity slot names are not strict numerical coordinates and that "doubling capacity" is often a small change in parameter count for FNO.

The thresholds are *audit-friendly*: they are not hand-picked, they fall on quartile boundaries of the empirical distribution, and their derivation is reproducible from a single read of `results/cell_stats_diagonal.csv`. If anyone challenges them, the answer is "the 25th and 75th percentiles of how the actual data behaves in log-log neighbourhoods of every cell."

---

## §2 Headline regime distribution

After applying the §6.3 label rules with the calibrated thresholds, the 3,374 diagonal cells split as:

| Regime | Count | Share | Notes |
|---|---:|---:|---|
| `mixed` | 1,641 | 48.6% | Multiple axes simultaneously responsive; no single dominant axis. |
| `ambiguous` | 483 | 14.3% | Boundary cells where one of S_N, S_R, S_C is undefined (no neighbor). |
| `data_limited` | 398 | 11.8% | \|S_N\| > 0.227, others < 0.227. |
| `resolution_sensitive` | 387 | 11.5% | \|S_R\| > 0.227, others < 0.227. |
| `capacity_limited` | 186 | 5.5% | \|S_C\| > 0.227, others < 0.227. |
| `data_bug_r48` | 147 | 4.4% | All Burgers R=48 cells; reported in VALIDATION_REPORT §5b only. |
| `unstable` | 81 | 2.4% | divergence_rate > 0.10 OR seed_cv > 0.50 OR instability_rate > 0.10. |
| `saturated` | 51 | 1.5% | All three sensitivities below 0.037. |

**The 48.6% `mixed` fraction is the headline finding of the regime exercise.** It is *exactly* the kind of result a 1D-slice law-fit framing collapses away. Almost half of the cells in the operator-learning grid are simultaneously responsive to more than one of (data, resolution, capacity), and forcing each of those cells to commit to a single "winning law" along a single axis was — by construction — throwing away the joint information. This is the strongest empirical justification yet for the reframe.

---

## §3 Per (task, model) regime distribution

The same labels broken out by (task, model) — this is where the rewrite gets its per-architecture and per-task narrative.

|  | cap-lim | data-lim | mixed | res-sens | sat | unstable |
|---|---:|---:|---:|---:|---:|---:|
| **burgers / deeponet** | 36 | 3 | 233 | **117** | 3 | 0 |
| **burgers / fno** | 3 | **91** | 206 | 92 | 0 | 0 |
| **burgers / mlp_baseline** | 14 | 35 | 235 | 107 | 1 | 0 |
| **burgers / mlp_controlled** | 5 | 1 | 91 | 15 | 0 | 0 |
| **darcy / deeponet** | 34 | 52 | 157 | 15 | 6 | **44** |
| **darcy / fno** | 4 | 28 | 247 | 3 | **26** | 0 |
| **darcy / mlp_baseline** | 9 | **77** | 173 | 2 | 11 | **36** |
| **diffusion / deeponet** | 24 | 24 | 89 | 2 | 1 | 0 |
| **diffusion / fno** | 28 | 44 | 61 | 3 | 3 | 1 |
| **diffusion / mlp_baseline** | 17 | 41 | 69 | 13 | 0 | 0 |
| **diffusion / mlp_controlled** | 12 | 2 | 80 | 18 | 0 | 0 |

**Key per-(task, model) observations** that the rewrite should foreground:

1. **Burgers/FNO is the most data-limited model in the entire grid** (91 `data_limited` cells, by far the highest count). This contradicts the implicit "FNO is resolution-invariant" framing some prior work uses — in this experimental regime, FNO needs *more data*, not different resolution. The `resolution_sensitive` count for Burgers/FNO is also high (92), so the picture is "FNO needs both," but the data axis is the more pressing bottleneck.

2. **Burgers/DeepONet has the highest `resolution_sensitive` count** (117 cells) of any (task, model). This supports the "DeepONet is brittle to the choice of training resolution" reading.

3. **Burgers/`mlp_controlled` has only 15 `resolution_sensitive` cells out of 112** (13.4%), versus 117/602 (19.4%) for Burgers/DeepONet, 92/602 (15.3%) for Burgers/FNO, and 107/602 (17.8%) for Burgers/mlp_baseline. The capacity-controlled MLP design deliberately holds parameter count fixed across resolutions, and the regime-label distribution shows it works: it has materially fewer resolution-sensitive cells than its uncontrolled siblings. This deserves its own subsection in the rewrite (per AUDIT W2).

4. **Darcy/DeepONet has 44 `unstable` cells** and Darcy/mlp_baseline has 36 — together accounting for 80 of the 81 main-grid `unstable` cells. The instability story on the main grid is **almost entirely Darcy, almost entirely the non-FNO models**, and not Burgers at all (after excluding R=48). Darcy/FNO has zero unstable cells. This is a very specific finding the rewrite should report cleanly.

5. **Darcy/FNO has 26 `saturated` cells** (more than half the main-grid `saturated` count of 51). FNO on Darcy hits a regime where adding data, resolution, or capacity doesn't move error meaningfully. This is consistent with Darcy being a relatively easy task for FNO and is good news for the "regime maps tell you when to stop adding compute" framing.

6. **Diffusion is dominated by `mixed`** (60% across the four models combined), with very small `resolution_sensitive` counts (2–18 per model). This is consistent with the reviewer's concern that the Diffusion grid is sparse on N (5 values per model) and doesn't support strong axis-resolved claims. Diffusion remains a secondary case in the rewrite.

---

## §4 Marginal value of doubling R (per (task, model))

For each cell where both `R` and `2R` exist with successful runs, we compute the ratio `median_test_rel_l2(2R) / median_test_rel_l2(R)`. Ratio < 1 means doubling resolution helps; ratio > 1 means it hurts. Aggregated:

| Task / model | n pairs | median ratio | mean ratio |
|---|---:|---:|---:|
| burgers / deeponet | 427 | 0.956 | 0.997 |
| burgers / fno | 427 | **0.878** | 1.035 |
| burgers / mlp_baseline | 427 | 0.953 | 0.993 |
| burgers / mlp_controlled | 84 | 0.955 | **0.927** |
| darcy / deeponet | 231 | 0.989 | 1.051 |
| darcy / fno | 231 | 0.995 | 0.992 |
| darcy / mlp_baseline | 231 | 0.974 | 0.962 |
| diffusion / deeponet | 105 | 1.041 | 1.042 |
| diffusion / fno | 105 | 0.998 | 1.007 |
| diffusion / mlp_baseline | 105 | **1.095** | 1.129 |
| diffusion / mlp_controlled | 84 | **1.213** | **1.515** |

**Reading.** The "doubling R helps" story is *not* universal. It is most reliable for **Burgers/FNO** (median ratio 0.878 — 12% improvement from doubling R). It is *negative* for **Diffusion/mlp_controlled** (median 1.213 — doubling R makes error 21% worse on average) and for **Diffusion/mlp_baseline** (1.095 — 9.5% worse). Across the entire grid, **23.4% of doubling pairs have ratio > 1.05** — i.e. roughly one in four times you double the training resolution, the test error gets worse by at least 5%.

This single number (23.4%) is the strongest concrete refutation of the "more resolution is always better" implicit assumption that the law-fit framing was hiding.

### Per-transition breakdown (Burgers only, since it has the densest R grid)

| R_low → R_high | n | median ratio |
|---|---:|---:|
| 16 → 32 | 147 | **1.855** |
| 32 → 64 | 259 | 0.924 |
| 64 → 128 | 259 | **0.788** |
| 96 → 192 | 147 | 0.868 |
| 128 → 256 | 259 | 0.957 |
| 192 → 384 | 147 | 1.001 |
| 256 → 512 | 147 | 1.008 |

**Reading.** R=16 → R=32 has a *median ratio of 1.855*, meaning the typical Burgers cell gets nearly *twice as bad* when going from R=16 to R=32. The most plausible interpretation is that R=16 is so coarse that the test ground truth is itself heavily smoothed, making the test problem artificially easy; once the resolution increases enough to expose the actual non-trivial dynamics, the model has to actually solve it. R=64 → R=128 is the most genuinely-helpful transition (21% median improvement). Beyond R=128, marginal value of doubling R is essentially flat or zero (R=192→384 and R=256→512 are both within 1% of unity in the median).

**For the rewrite:** the practical design rule that falls out of this is "between R=64 and R=128 is the only transition with a reliable benefit of doubling resolution; below R=32 is misleading; above R=256 is wasted compute." That is exactly the kind of operational, response-surface-derived recommendation the reframe is built to support.

---

## §5 Best and worst cells (sanity check on the magnitudes)

Best 5 cells in the entire main grid (lowest median test_rel_l2, excluding R=48):

| task | model | capacity | N | R | median_rel_l2 | seed_cv |
|---|---|---|---:|---:|---:|---:|
| diffusion | fno | small-med | 5000 | 256 | **0.0021** | 0.059 |
| diffusion | fno | small | 1000 | 128 | 0.0021 | 0.083 |
| diffusion | fno | small | 5000 | 256 | 0.0023 | 0.315 |
| diffusion | fno | small-med | 1000 | 64 | 0.0023 | 0.150 |
| diffusion | fno | small | 5000 | 32 | 0.0024 | 0.107 |

**Notable:** the best 5 cells are all FNO on Diffusion at small to small-med capacity, *not* xlarge. This is consistent with the `capacity_limited` analysis above: more capacity is not always better.

Worst 5 cells (highest median test_rel_l2, excluding R=48 and unstable):

| task | model | capacity | N | R | median_rel_l2 | seed_cv |
|---|---|---|---:|---:|---:|---:|
| darcy | deeponet | xlarge | 3000 | 256 | **5.54** | 0.589 |
| darcy | mlp_baseline | small-med | 50 | 32 | 2.99 | 0.387 |
| darcy | mlp_baseline | medium | 50 | 32 | 2.94 | 0.363 |
| darcy | deeponet | small | 50 | 64 | 2.93 | 0.303 |
| darcy | mlp_baseline | small-med | 50 | 128 | 2.91 | 0.326 |

All worst cells are Darcy (consistent with the high `unstable` counts above). The xlarge-Darcy-DeepONet entry at N=3000 is particularly noteworthy: *more capacity and more data did not help*. This is `capacity_limited` in the wrong direction — Darcy/DeepONet at xlarge is over-parametrized for the task.

---

## §6 What the figures show

The plot script `scripts/plot_response_surfaces.py` produces three classes of figures (regenerable, gitignored):

1. **`figures/response_surface_<task>_<model>.png`** — 7-panel grid (one panel per capacity slot), each panel a (N, R) heatmap of `median_test_rel_l2` on a log color scale. R=48 columns are masked. These are the centerpiece visualizations the rewrite cites.

2. **`figures/regime_map_<task>_<model>.png`** — 7-panel grid in the same shape, each panel showing the regime label per (N, R) cell as a categorical color map. Reading these alongside the response-surface heatmaps lets the reader see *why* a particular cell got a particular label.

3. **`figures/sensitivity_distributions.png`** — three histograms of pooled |S_N|, |S_R|, |S_C| with the calibrated `THRESH_LOW` and `THRESH_HIGH` overlaid. This is the calibration evidence figure for the appendix.

To regenerate:

```bash
python3 scripts/aggregate_cells.py
python3 scripts/compute_regime_labels.py
python3 scripts/plot_response_surfaces.py
```

(The first two emit the canonical CSVs; the third reads them.)

---

## §7 What is intentionally NOT in this document

1. No claim that any particular axis "dominates" globally. The per-(task, model) tables show that the dominant axis varies — there is no universal answer.
2. No 1D law fits, no AICc, no exponent estimates, no functional form selection. Those go in `APPENDIX_LAW_FITS.md` (Phase 6) and serve as fragility evidence only.
3. No claim about Diffusion local slopes that requires more than 5 N values per model. Diffusion is reported alongside Burgers and Darcy but its sensitivity-axis claims are explicitly weaker.
4. No claim about Darcy at R > 256. Darcy's R range is {32, 64, 128, 256} and the protocol forbids extrapolating beyond it.
5. No claim about the R=48 cells. They are documented in VALIDATION_REPORT.md §5b only.

---

## §8 Phase 3 → Phase 4 handoff

The cell-level regime labels are now persisted in `results/cell_regimes.csv`. Phase 4 (cross-resolution transfer) reads `results/cell_stats_crossres.csv` joined to this regime label and asks: "for cells labeled `resolution_sensitive` in Phase 3, what does their cross-resolution transfer matrix look like, and is the fragility in the direction the regime label predicts?" That's the bridge from response-surface diagnostics to the cross-resolution transfer story, and it's what makes the two analyses mutually reinforcing rather than independent.
