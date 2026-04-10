# Results — Cross-resolution transfer (Phase 4)

**Date:** 2026-04-10
**Reads:** `results/cell_stats_crossres.csv`, `results/cell_stats_diagonal.csv`
**Generates:** `results/transfer_scores.csv` (210 slices × 7 scores), `results/transfer_matrices.json` (full per-slice matrices), `figures/transfer_matrix_<task>_<model>.png` (gitignored; regenerable from `scripts/build_transfer_matrices.py`).

This document is the centerpiece of the new framing: it answers "what happens to model error when you train at one resolution and evaluate at another?" across every (task, model, capacity, N) slice where the data supports it.

---

## §0 Scope and architectural note

**Only DeepONet and FNO support cross-resolution evaluation.** MLP models (mlp_baseline, mlp_controlled) are discretization-tied: their input layer dimension is exactly R_train, so evaluating at R_eval ≠ R_train is architecturally undefined. This is a meaningful architectural distinction that the rewrite should foreground: the MLP family *cannot even ask* the transfer question, which is itself a design limitation.

Cross-resolution data on disk:

| Axis | Values |
|---|---|
| R_train | {32, 64, 128, 256} — the 4 core R values with full N coverage |
| R_eval | {32, 64, 128, 256, 512} — the 5-value superset; R_eval=512 has no corresponding diagonal entry |
| Per slice: off-diagonal cells | 16 (4 × 5 minus 4 where R_train = R_eval) |
| Total slices | 210 = (3 tasks × 2 models × 7 capacities × 5 N-values), with Diffusion having 5 N per model |
| Total cross-res rows | 3,360 |

The diagonal K[i,i] entries come from `cell_stats_diagonal.csv` (since the cross-res table excludes R_train = R_eval by construction).

---

## §1 Global transfer scores

Aggregated across all 210 slices:

| Score | median | mean |
|---|---:|---:|
| `mean_degradation` | 1.038 | 1.316 |
| `worst_degradation` | 1.298 | 1.999 |
| `train_low_eval_high` | 1.028 | — |
| `train_high_eval_low` | 1.078 | — |
| `transfer_asymmetry` | 1.013 | — |

**Reading.** The *typical* slice (median) sees only 3.8% mean degradation from cross-resolution transfer — but the *worst cell within a typical slice* degrades by 29.8%. The gap between mean (1.04) and worst (1.30) at the median is already meaningful; at the tails, it's enormous (see §2).

The train-high-eval-low direction is globally 7.8% worse than diagonal (median), while train-low-eval-high is only 2.8% worse. Training at high resolution and evaluating at low is the consistently riskier direction.

---

## §2 Per (task, model) transfer scores

This is the key table for the rewrite's architecture-comparison section.

| task / model | median mean_deg | median worst_deg | median asymmetry |
|---|---:|---:|---:|
| **burgers / deeponet** | 1.015 | 1.487 | **1.437** |
| **burgers / fno** | **2.589** | **6.564** | **1.651** |
| **darcy / deeponet** | 1.039 | 1.125 | 0.983 |
| **darcy / fno** | **1.002** | **1.026** | 1.004 |
| **diffusion / deeponet** | 1.072 | 1.223 | 0.983 |
| **diffusion / fno** | 1.074 | 1.316 | 1.015 |

### Headline findings

**Finding 1: FNO on Burgers is catastrophically fragile.** Median worst-case degradation = 6.56× (the typical FNO Burgers slice has a cross-res cell where error is 6.56 times worse than the diagonal baseline). This is the most important transfer result in the paper. All top 10 worst slices by `worst_degradation` in the entire dataset are Burgers/FNO, with degradation factors 8–9× when training at R ∈ {128, 256} and evaluating at R_eval = 32. The architecture marketed as "resolution-invariant" shows the worst resolution fragility.

**Finding 2: FNO on Darcy is essentially resolution-invariant.** Median worst-case degradation = 1.026× — i.e. the worst-case cross-res transfer on Darcy barely moves. Mean degradation is 1.002. This is the clearest evidence that the transfer property is task-dependent, not architecture-inherent. FNO's resolution invariance holds on the easier task (Darcy) but fails on the harder one (Burgers).

**Finding 3: Transfer asymmetry is systematic on Burgers, not on Darcy or Diffusion.** Burgers/FNO asymmetry is 1.65 (train-high-eval-low is 65% worse than the reverse); Burgers/DeepONet is 1.44. Darcy and Diffusion are within 2% of symmetric. The direction of worst degradation (training at high R, evaluating at low) is consistent with a spectral-aliasing mechanism: features learned at high resolution have no spectral support at low resolution, while features learned at low resolution are merely undersampled, which is less harmful.

**Finding 4: DeepONet is less fragile than FNO on Burgers but not trivially robust.** Burgers/DeepONet median worst degradation = 1.49× (vs FNO's 6.56×). DeepONet's branch-trunk decomposition provides partial insulation from the resolution mismatch but does not eliminate it.

**Finding 5: Diffusion shows moderate and symmetric transfer degradation for both architectures.** DeepONet (1.22× worst, 0.98 asymmetry) and FNO (1.32× worst, 1.02 asymmetry) on Diffusion are both mild. Diffusion is the easiest task for transfer, presumably because the underlying physics is smooth (heat equation).

---

## §3 Worst 10 slices in the dataset

All are Burgers/FNO. The worst pair is always R_train ∈ {128, 256} → R_eval = 32.

| capacity | N | worst_deg | worst pair (Rt → Re) | asymmetry |
|---|---:|---:|---|---:|
| medium | 5000 | **9.27** | 256 → 32 | 1.42 |
| small | 5000 | 8.98 | 256 → 32 | 1.39 |
| xlarge | 1000 | 8.75 | 128 → 32 | 1.57 |
| small-med | 5000 | 8.52 | 256 → 32 | 1.35 |
| xlarge | 5000 | 8.42 | 128 → 32 | 1.45 |
| small | 1000 | 8.28 | 256 → 32 | 1.69 |
| medium | 1000 | 8.04 | 256 → 32 | 1.48 |
| large | 1000 | 8.03 | 128 → 32 | 1.54 |
| large | 5000 | 7.96 | 128 → 32 | 1.40 |
| small-med | 1000 | 7.87 | 256 → 32 | 1.63 |

**Pattern.** High N (≥ 1000) + high R_train (≥ 128) + low R_eval (32) = worst transfer. The degradation scales with R_train / R_eval ratio (8× degradation at a 4–8× ratio, which is super-linear). Low capacity doesn't protect: the medium-capacity slice at N=5000 is the single worst.

**For the rewrite's practical design rules:** if you train a Burgers-type FNO at R=128 or higher, do NOT evaluate at R=32 — expect 8× error blow-up. Retrain at the target resolution, or accept 1.5× degradation by evaluating at R=64 instead.

---

## §4 Bridge to Phase 3 regime labels

For cells labeled `resolution_sensitive` in Phase 3, the transfer data confirms the label is doing real work:

- Burgers/DeepONet has 117 `resolution_sensitive` cells on the diagonal (Phase 3), and a 1.49× worst cross-res degradation. The local sensitivity on the training R axis correctly predicts that cross-res transfer is non-trivial.
- Burgers/FNO has 92 `resolution_sensitive` cells on the diagonal, and a 6.56× worst cross-res degradation. The regime label flags the risk, but the transfer magnitude is much larger than the label alone would suggest.
- Darcy/FNO has only 3 `resolution_sensitive` cells and a 1.026× worst cross-res degradation. The near-zero sensitivity correctly predicts near-zero transfer fragility.

The regime labels from Phase 3 and the transfer scores from Phase 4 are consistent. The regime map tells you *where* to worry; the transfer matrix tells you *how much*.

---

## §5 What the figures show

`figures/transfer_matrix_<task>_<model>.png` — one figure per (task, model) pair, with 3×3 panels selected as representative (smallest/median/largest capacity × smallest/median/largest N). Each panel is a 4×5 heatmap of `median_test_rel_l2` (R_train on vertical axis, R_eval on horizontal). Diagonal cells are the baseline; warm colors are high error (poor transfer).

To regenerate:

```bash
python3 scripts/build_transfer_matrices.py
```

---

## §6 What is intentionally NOT in this document

1. No transfer analysis for MLP models (architecturally impossible).
2. No claim about R_eval > 512 (not in the cross-res superset).
3. No claim about Darcy at R_train > 256 (Darcy's grid stops at 256).
4. No claim about R_train = 48 (excluded per PROTOCOL §5.1, and no cross-res records exist for it anyway).
5. No law-based explanation for transfer degradation. Phrases like "the transfer follows a power law" are forbidden.
