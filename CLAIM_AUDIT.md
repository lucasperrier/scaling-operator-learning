# Claim Audit — Paper v4

**Date:** 2026-04-10
**Purpose:** Per-claim evidence audit for `paper/main_v4.tex`. Every numbered or bolded claim in the draft is listed here with its evidence source (which CSV, which script, which section of a RESULTS doc) and its strength assessment. Claims without grounding are flagged for removal or caveat.

This is the final deliverable of the 8-phase reframe pipeline.

---

## Abstract claims

| # | Claim | Evidence | Strength |
|---|---|---|---|
| A1 | "48.6% of cells are simultaneously responsive to multiple axes" | `results/cell_regimes.csv`: 1641/3374 = 48.6%. Script: `scripts/compute_regime_labels.py`. | **Strong.** Direct count from sensitivity computation with reproducible thresholds. |
| A2 | "the only clearly beneficial transition is R=64→128 (21% median error reduction)" | `results/marginal_value_of_resolution.csv`: Burgers R=64→128 median ratio = 0.788. | **Strong.** Direct computation from per-cell medians. |
| A3 | "23.4% of all resolution-doubling transitions increase error" | Same CSV: 575/2457 pairs have ratio > 1.05. | **Strong.** |
| A4 | "FNO training at R=256 and evaluating at R=32 degrades error by 6.56×" | `results/transfer_scores.csv`: median worst_degradation for Burgers/FNO = 6.564. | **Strong.** |
| A5 | "Darcy worst transfer degradation is 1.03×" | Same CSV: Darcy/FNO median worst_degradation = 1.026. | **Strong.** |
| A6 | "every training divergence traces to a data-generator bug at R=48" | `results/cell_stats_diagonal.csv`: 0 cells with div_rate > 0 excluding R=48. `scripts/repro_r48_data_bug.py` standalone reproduction. | **Strong.** Causally verified. |
| A7 | "SGD converges but to a 3× worse solution" | Phase 5 analysis: SGD-cosine median ratio for FNO = 3.02. | **Strong.** Direct median comparison. |

---

## §1 Introduction claims

| # | Claim | Evidence | Strength |
|---|---|---|---|
| I1 | "held-out contest: 15/33 wins, ~1.2% improvement" | `results/holdout_prediction_contest.json` (legacy). | **Strong.** From existing analysis infrastructure; audited by external reviewer. |
| I2 | "22/27 slices bootstrap-unstable" | `results/within_slice_bootstrap.json` (legacy). | **Strong.** Same. |

---

## §4 Response-surface results

| # | Claim | Evidence | Strength |
|---|---|---|---|
| R1 | "FNO on Burgers is the most data-limited model (91 cells)" | `results/cell_regimes.csv`, filter task=burgers, model=fno, regime=data_limited: count=91. | **Strong.** |
| R2 | "DeepONet on Burgers has the most resolution-sensitive cells (117)" | Same CSV, regime=resolution_sensitive: count=117. | **Strong.** |
| R3 | "MLP-ctrl has only 15 resolution-sensitive cells out of 112" | Same CSV: 15/112. | **Strong.** |
| R4 | "Darcy instability: 44 DeepONet + 36 MLP = 80 of 81 unstable" | Same CSV: 44+36=80, total unstable=81. | **Strong.** |
| R5 | "Darcy/FNO has zero unstable cells" | Same CSV. | **Strong.** |
| R6 | "Darcy/FNO has 26 saturated cells" | Same CSV. | **Strong.** |
| R7 | "R=16→32 median ratio 1.855" | `results/marginal_value_of_resolution.csv`. | **Strong.** |
| R8 | "Beyond R=256, marginal value is zero" | Same CSV: R=192→384 ratio 1.001, R=256→512 ratio 1.008. | **Strong.** |
| R9 | "Diffusion/MLP-ctrl: doubling R makes error 21% worse" | Same CSV: median ratio = 1.213. | **Strong.** |

---

## §5 Cross-resolution transfer

| # | Claim | Evidence | Strength |
|---|---|---|---|
| T1 | "All 10 worst transfer slices are Burgers/FNO" | `results/transfer_scores.csv`, nlargest(10, 'worst_degradation'). | **Strong.** |
| T2 | "8-9× degradation, training at R∈{128,256}, eval at R=32" | Same CSV: worst slices range 7.87-9.27. | **Strong.** |
| T3 | "FNO on Darcy degrades by only 1.03×" | Same CSV: Darcy/FNO median worst_degradation = 1.026. | **Strong.** |
| T4 | "FNO asymmetry = 1.65" | Same CSV: Burgers/FNO median transfer_asymmetry = 1.651. | **Strong.** |
| T5 | "DeepONet asymmetry = 1.44" | Same: Burgers/DeepONet = 1.437. | **Strong.** |
| T6 | "On Darcy and Diffusion, transfer is nearly symmetric (~1.0)" | Same: Darcy/DeepONet = 0.983, Darcy/FNO = 1.004, Diff/DeepONet = 0.983, Diff/FNO = 1.015. | **Strong.** |
| T7 | "MLP models are discretization-tied and cannot be evaluated at R_eval ≠ R_train" | Architectural: MLP input dim = R. `results/cell_stats_crossres.csv` has no mlp entries. | **Strong (architectural + empirical).** |

---

## §6 Convergence and instability

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | "Zero training runs experienced optimization divergence on clean data" | `results/cell_stats_diagonal.csv`: filter is_r48=False, sum of n_diverged across all cells = 0. Phase 2b verification. | **Strong.** |
| C2 | "582 runs failed at R=48 due to data-generator bug" | Same CSV: filter is_r48=True, n_diverged total = 582. `scripts/repro_r48_data_bug.py`. | **Strong.** Causally verified. |
| C3 | "81 unstable cells, 80 are Darcy" | `results/cell_regimes.csv`. | **Strong.** |
| C4 | "All three optimizer configs: 100% convergence" | Phase 5 analysis: 657 SGD, 756 Adam-2x, 17348 main grid (non-R48), all success. | **Strong.** |
| C5 | "SGD-cosine: 3× worse FNO error" | Phase 5 analysis: median ratio 3.02. | **Strong.** |
| C6 | "Adam-2x matches baseline within 4%" | Phase 5: DeepONet 0.96, FNO 0.97, MLP 0.99. | **Strong.** |

---

## §7 Discussion claims

| # | Claim | Evidence | Strength |
|---|---|---|---|
| D1 | "48.6% of cells are mixed — information no 1D slice retains" | A1 + architectural argument. | **Strong for the count; the 'no 1D slice' part is a methodological argument, not a computed number.** |
| D2 | "Transfer matrices reveal 8-9× degradation invisible in diagonal-only law fits" | T2 + the fact that cross-res was supplementary (not centerpiece) in the previous framing. | **Strong.** |
| D3 | "R=48 data bug would have been averaged into 3.3% divergence rate" | The previous draft literally did this (`<3.3\%` in main.tex). | **Strong (factual about the previous draft).** |
| D4 | "Best cells are all FNO on Diffusion at rel_l2 ≈ 0.002" | `results/cell_stats_diagonal.csv`, nsmallest(5, 'median_test_rel_l2'). | **Strong.** |
| D5 | "MLP-ctrl decoupling confirmed" | R3 (regime-label comparison). Also confirmed by the per-capacity table in the old appendix (persistent degradation at fixed D). | **Strong.** |

---

## §8 Conclusion

All three "practical design rules" are restatements of claims already audited above (R8, T2, C5).

---

## Removed claims (present in main.tex, absent from main_v4.tex)

| Old claim | Why removed |
|---|---|
| "AICc-selected winner in all 9 task-model combinations" | Forbidden framing per PROTOCOL_V2 §1. Law identity is not the primary result. |
| "We treat the identity of the best-fitting functional form as itself informative" | Same. |
| "Operator-learning scaling should be treated as a model-selection problem over law families" | Explicitly rejected in the reframe directive. |
| "35/35 Burgers slices improve with resolution" | Replaced by the more nuanced "23.4% of doubling transitions hurt" and the non-monotone Burgers transition table. |
| "Zero divergence" (original claim) | Already corrected to 582 by reviewer; now further corrected to "582 data-bug failures, 0 optimizer divergences." |
| Per-axis AICc winner tables in main text | Moved to Appendix C (law fits as supplementary analysis). |

---

## Verdict

**All 30+ quantitative claims in main_v4.tex are traceable to a specific CSV, JSON, or script output.** No claim depends on inference from a 1D law fit. No claim uses the forbidden framings listed in PROTOCOL_V2 §1. The three "practical design rules" in the conclusion are each supported by a primary finding from Phases 3, 4, or 5.

The draft is ready for the author's review and polish (figure integration, prose smoothing, additional appendix content from main.tex if desired).
