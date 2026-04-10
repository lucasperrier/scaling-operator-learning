# Results — Convergence and instability (Phase 5)

**Date:** 2026-04-10
**Reads:** `results/cell_stats_diagonal.csv`, `results/cell_regimes.csv`, `runs_ablation_sgd_cosine/`, `runs_ablation_adam_2x_patience/`
**Generates:** `figures/divergence_heatmap_*.png`, `figures/seed_cv_heatmap_*.png` via `scripts/plot_convergence.py` (gitignored, regenerable).

This document reports on three aspects of training stability in the new framing, repositioned from the previous draft's "robustness check" narrative.

---

## §1 Main-grid divergence: the R=48 data-generator bug is the only source

Across the entire non-R48 main grid (3,227 diagonal cells with successful runs), **zero cells** have `divergence_rate > 0`. The PROTOCOL §6.2 threshold `THRESH_DIV = 0.10` flags exactly zero cells for divergence-based instability on the main grid.

This is a stronger statement than the previous draft made. The previous draft said "582 training runs diverged (<3.3%)", implying that optimization divergence is a rare but real phenomenon in operator learning. The corrected statement is:

> **Zero Burgers training runs experienced optimization divergence on clean data.** All 582 failures occurred at a single resolution (R=48) due to NaN-contaminated input data from a spectral-solver bug in the data generator (`_spectral_solve` in `src/scaling_operator_learning/tasks/burgers.py`). Every other resolution (R ∈ {16, 32, 64, 96, 128, 192, 256, 384, 512}) has a 100% convergence rate across all models, capacities, and N values.

For Darcy and Diffusion, the diagonal convergence rate is also 100% across all cells. **There are no optimization divergences anywhere in the three-task, four-model, seven-capacity grid when the data generator is functioning correctly.**

See VALIDATION_REPORT.md §5b for the standalone reproduction of the R=48 bug.

---

## §2 Main-grid seed variability: the Darcy instability

While there are no *divergences* outside R=48, there *is* a seed-variability instability detected by `THRESH_CV = 0.50`:

| Condition | Count | Share |
|---|---:|---:|
| Non-R48 cells with seed_cv > 0.50 | 81 | 2.5% |
| Of which Darcy/DeepONet | 44 | |
| Of which Darcy/mlp_baseline | 36 | |
| Of which Diffusion/FNO | 1 | |
| Of which Burgers (any model) | 0 | |

**The seed-variability instability is concentrated in Darcy's non-FNO models.** Darcy/FNO has zero cells above the threshold. This is a task-model interaction: the Darcy task's log-normal permeability fields produce challenging training instances that DeepONet and MLP struggle with when the random seed is unfavorable, but that FNO handles robustly (possibly because FNO's spectral processing can handle the spatial frequency content of log-normal fields more naturally than the branch-trunk or fully-connected architectures).

Top 5 cells by seed_cv (excluding R=48):

| task | model | cap | N | R | seed_cv | median_rel_l2 |
|---|---|---|---:|---:|---:|---:|
| darcy | deeponet | med-large | 2000 | 256 | **1.54** | 0.78 |
| darcy | deeponet | large | 2000 | 64 | 1.18 | 0.93 |
| darcy | deeponet | large | 3000 | 64 | 1.04 | 2.73 |
| darcy | deeponet | xlarge | 2000 | 256 | 1.03 | 0.75 |
| darcy | deeponet | xlarge | 100 | 32 | 1.01 | 1.03 |

**For the rewrite:** the instability narrative in the new paper is: (a) zero optimization divergence on clean data; (b) a task-model-specific seed-sensitivity effect on Darcy (DeepONet and MLP, not FNO); (c) the 582 "divergences" are a data bug, not optimization failure. This is a much cleaner and more honest narrative than "582 runs diverged (<3.3%)."

---

## §3 Optimizer ablation: convergence rate and error comparison

The 657 SGD-cosine and 756 Adam-2x-patience ablation runs are all Burgers-only, medium capacity, three models (deeponet, fno, mlp_baseline). All cells used the same N and R range as the main grid (excluding R=48, which was not part of the ablation).

### 3.1 Convergence rate

| Optimizer config | Total runs | Success | Failure | Failure rate |
|---|---:|---:|---:|---:|
| **Main grid (Adam + ReduceLROnPlateau)** | 17,348 (non-R48) | 17,348 | 0 | **0.0%** |
| **SGD-cosine** | 657 | 657 | 0 | **0.0%** |
| **Adam-2x-patience** | 756 | 756 | 0 | **0.0%** |

**All three optimizer configurations achieve 100% convergence on clean Burgers data.** There is no optimization-limited convergence-failure regime in this experimental setup. The previous draft's "SGD+cosine as optimizer ablation for robustness" framing was misleading — it implied that optimizer choice might explain convergence failures, but there are no convergence failures to explain.

### 3.2 Error magnitude: optimizer choice matters for quality, not convergence

The convergence rate is the same, but the error quality is not.

| Model | SGD-cosine ratio | Adam-2x ratio |
|---|---:|---:|
| deeponet | **1.38×** | 0.96× |
| **fno** | **3.02×** | 0.97× |
| mlp_baseline | **1.43×** | 0.99× |

*Ratio = ablation median test_rel_l2 / main-grid (Adam+ReduceLROnPlateau) median test_rel_l2.*

**Reading.** SGD-cosine converges but to a substantially worse solution: 38% worse for DeepONet, **3× worse for FNO**, and 43% worse for MLP. Adam-2x-patience produces virtually the same error as Adam-standard (all ratios within 4% of 1.0, with a slight trend toward slightly better error).

**FNO is the most optimizer-sensitive model**: its spectral convolution layers require per-parameter learning-rate adaptation (Adam) to find a good solution, and the fixed learning-rate schedule of SGD fails to explore the loss landscape effectively. This is a concrete, experiment-grounded "practical design rule" for the rewrite: if you use FNO, use Adam (or similar adaptive optimizer); SGD will converge but to a 3× worse solution.

### 3.3 Repositioning from the previous draft

The previous draft called the optimizer ablation a "robustness check for the scaling law fits." In the new framing, it becomes evidence for two claims:

1. **Convergence is not optimizer-sensitive in this regime** (all three configs: 100%). This means the scaling analysis is not confounded by optimizer-dependent convergence failures.
2. **Error magnitude is optimizer-sensitive, especially for FNO** (3× worse with SGD). This is a practical design rule, not a scaling-law robustness check.

The rewrite should present the ablation in the §Convergence and Instability section with these two findings, not in a "Methods robustness" appendix.

---

## §4 Per-cell convergence and instability heatmaps

The plotting script `scripts/plot_convergence.py` (to be created in Phase 5b) emits:

- `figures/divergence_heatmap_<task>_<model>.png` — (N, R) heatmap of `divergence_rate` per capacity. On Burgers, this is a vertical stripe at R=48 and zeros everywhere else. On Darcy and Diffusion, it is all zeros.
- `figures/seed_cv_heatmap_<task>_<model>.png` — (N, R) heatmap of `seed_cv_test_rel_l2`. On Darcy/DeepONet and Darcy/mlp_baseline, this shows the regions of high seed variability.

These figures are gitignored and regenerable from the script. The main value is in the narrative above, not in the visual artifacts (since the divergence heatmap is trivially R=48 and the seed-CV heatmap is Darcy-specific).

---

## §5 Summary for the rewrite

The Phase 5 findings that the rewrite needs to incorporate:

1. **Zero optimization divergence on clean data.** The "582 diverged" figure from the previous draft is reattributed to the R=48 data-generator bug.
2. **Seed-variability instability is task/model-specific.** 81 cells exceed the THRESH_CV=0.50 threshold; 80 of them are Darcy (DeepONet and mlp_baseline, not FNO).
3. **Optimizer ablation confirms convergence robustness but reveals error sensitivity.** 100% convergence for all three configs; SGD-cosine → 3× worse FNO error, 1.4× worse DeepONet and MLP.
4. **Practical design rule:** "Use adaptive optimizers (Adam) for FNO; SGD converges but to a 3× worse solution."

---

## §6 What is intentionally NOT in this document

1. No analysis of the 582 R=48 runs as training failures. See VALIDATION_REPORT.md §5b.
2. No claim that the optimizer ablations' error differences follow a scaling law.
3. No claim about Darcy or Diffusion optimizer sensitivity (ablations are Burgers-only).
4. No claim about "optimization-limited regimes" in the 1D-slice law-fit sense.
