# Independent Data Analysis: Scaling Laws for Neural Operator Learning

**Date:** April 7, 2026  
**Data:** ~31,000 training runs across 3 PDE tasks, 4 architectures, 7 capacity levels, 7 dataset sizes, 4–10 resolutions, 9 seed combinations  
**Method:** Raw data analysis with no reliance on paper framing. All conclusions derived from `runs/grouped_metrics.csv` (6,734 aggregated rows) and `runs/runs_aggregate.csv` (31,370 individual runs).

---

## 1. Experimental Setup (as read from the data)

Three 1D PDE benchmarks map input functions to output functions:

| Task | Mapping | Character |
|------|---------|-----------|
| **Burgers** | Initial condition → solution at T=1 | Nonlinear advection, shock formation |
| **Darcy** | Permeability field → pressure solution | Linear elliptic, discontinuous coefficients |
| **Diffusion** | Initial condition → solution at T=1 | Linear, smooth, exact Fourier solution |

Four architectures are compared:

| Model | Type | Resolution dependence |
|-------|------|----------------------|
| **FNO** | Fourier Neural Operator | Resolution-invariant (spectral convolutions) |
| **DeepONet** | Branch-Trunk network | Resolution-invariant (pointwise evaluation) |
| **MLP baseline** | Feedforward MLP | Params grow with R (input/output dimension = R) |
| **MLP controlled** | Fixed-param MLP | Params held constant across R via width adjustment |

Three scaling axes are varied:
- **N** (dataset size): 50 to 5,000 samples
- **R** (discretization resolution): 16 to 512 grid points
- **D** (model capacity): 7 levels from ~2K to ~2.7M parameters

Each configuration is repeated over 3 data seeds × 3 training seeds = 9 runs. All models trained with Adam (lr=1e-3), early stopping (200-epoch patience), max 5,000 epochs. Metric: relative L2 error on held-out test set.

---

## 2. Architecture Comparison: FNO Wins Universally

At matched capacity (medium), FNO wins **143 out of 143 head-to-head comparisons** across every combination of task, dataset size, and resolution. There is no regime — low data, low resolution, any task — where DeepONet or MLP is competitive.

**Best achievable error per task × model (across all N, R, capacity):**

| Task | FNO | MLP baseline | DeepONet | MLP controlled |
|------|-----|-------------|----------|----------------|
| Burgers | **0.027** | 0.106 | 0.115 | 0.207 |
| Darcy | **0.471** | 0.685 | 0.743 | — |
| Diffusion | **0.002** | 0.018 | 0.098 | 0.016 |

The FNO advantage is not marginal. On Burgers at medium capacity, FNO beats the next-best model by 100–550% depending on N and R. On diffusion, FNO achieves 0.002 rel-L2 where DeepONet plateaus at 0.098 — a 49× gap.

**Implication:** For these 1D operator learning tasks, the architecture choice overwhelms all other scaling considerations.

---

## 3. Relative Importance of Scaling Axes

A log-linear regression $\log E = \beta_0 + \beta_N \log N + \beta_R \log R + \beta_D \log D$ quantifies how much each axis explains:

| Task | Model | N importance | R importance | D importance | Overall R² |
|------|-------|:-----------:|:-----------:|:-----------:|:----------:|
| Burgers | FNO | **69.6%** | 17.2% | 13.2% | 0.64 |
| Burgers | DeepONet | 43.6% | 22.2% | 34.2% | 0.49 |
| Burgers | MLP baseline | **66.1%** | 11.4% | 22.5% | 0.68 |
| Darcy | FNO | **85.2%** | 8.4% | 6.3% | 0.77 |
| Darcy | DeepONet | **92.8%** | 4.0% | 3.2% | 0.35 |
| Darcy | MLP baseline | **84.9%** | 5.1% | 10.0% | 0.83 |
| Diffusion | FNO | **82.0%** | 0.5% | 17.5% | 0.68 |
| Diffusion | MLP baseline | **82.1%** | 13.8% | 4.1% | 0.82 |
| Diffusion | MLP controlled | 12.2% | **51.8%** | 36.0% | 0.46 |

**Finding:** Dataset size (N) dominates in nearly every case, explaining 65–93% of error variance. Capacity (D) is a distant second. Resolution (R) is negligible or harmful for most model-task combinations.

The sole exception is `mlp_controlled` on diffusion, where R dominates — but in the *wrong direction* (error increases with R; see Section 5).

---

## 4. Data Scaling: Steep But Fast-Saturating

### 4a. Effective power-law exponents (log-log slopes, averaged over R and D)

| Task | FNO | MLP baseline | DeepONet |
|------|-----|-------------|----------|
| Burgers | −0.321 | −0.141 | −0.067 |
| Darcy | −0.090 | −0.328 | −0.145 |
| Diffusion | −0.352 | −0.518 | −0.101 |

FNO shows steep data scaling on Burgers and diffusion (exponents around −0.3 to −0.35), meaning a 10× increase in data roughly halves the error. MLP baseline scales fastest on diffusion (−0.518) but this conflates the capacity confound (see Section 5).

DeepONet has the weakest data scaling across the board, with exponents near −0.07 to −0.15. It learns slowly from additional samples.

### 4b. Diminishing returns

Marginal improvement from adding more data decays sharply:

**Diffusion / FNO:**
- N: 50→100 (2×): error improves 42%
- N: 100→500 (5×): error improves 59%
- N: 500→1000 (2×): error improves 13%
- N: 1000→5000 (5×): error improves **0.3%**

**Burgers / FNO (averaged over R):**
- N: 50→200 (4×): error improves ~50%
- N: 200→1000 (5×): error improves ~40%
- N: 1000→5000 (5×): error improves **~7%**

The data–error curve bends sharply. Past N≈1000, adding more data provides minimal benefit for FNO across all tasks, suggesting the error floor is dominated by architecture limitations or irreducible approximation error, not data starvation.

### 4c. The scaling law is not universal

Independent AICc-based model selection (fitting power-law, exponential, logarithmic) to each slice reveals:

- **Darcy:** Exponential decay wins for all models at all resolutions (fast early improvement, hard asymptotic floor)
- **Burgers:** Mixed — power law at some resolutions, exponential at others, with Δ(AICc) often <5 (inconclusive)
- **Diffusion:** Power law at low R, exponential at higher R for MLP/FNO

There is no single "scaling law" for operator learning. The functional form depends on the task, architecture, and resolution regime.

---

## 5. Resolution Is Not a Useful Scaling Axis

This is the central surprising finding. Rather than higher discretization resolution improving accuracy (as one might expect from numerical analysis), the data shows:

### 5a. Diffusion: higher R consistently *increases* error

| Model | R=32 error | R=256 error | Change |
|-------|-----------|-------------|--------|
| MLP baseline | 0.082 | 0.098 | **+20%** |
| MLP controlled | 0.021 | 0.026 | **+24%** |
| DeepONet | 0.209 | 0.243 | **+16%** |
| FNO | 0.008 | 0.008 | **+12%** |

This holds at every dataset size. The R-scaling exponent for MLP controlled on diffusion is **+0.638** (positive = error grows with resolution), with R²=0.96 — strong and reliable.

### 5b. Burgers: resolution helps for MLP/DeepONet, hurts for FNO

The N-scaling exponent for FNO on Burgers at R=16 is −0.402, but by R=512 it's −0.386, with the *absolute error increasing* 21% from R=16→R=512.

For MLP baseline on Burgers, resolution appears to help (R-exponent ≈ −0.2), but this is confounded: MLP parameters scale from ~20K at R=16 to ~148K at R=512. The "improvement" is just having a 7× larger model.

### 5c. Darcy: resolution is noise

DeepONet R-exponents fluctuate between −0.05 and +0.25 depending on N, with R² values averaging 0.3 — no signal. FNO shows R-exponents of −0.03 to +0.03. Resolution changes nothing meaningful for Darcy.

### 5d. The MLP controlled experiment confirms the confound

On Burgers, MLP controlled (fixed params across R) shows:
- R=32: error = 0.379
- R=128: error = 0.278
- R=256: error = 0.277

Some modest R benefit in Burgers (resolution captures shock structure), but on diffusion:
- R=32: error = 0.021
- R=256: error = 0.026

The "extra" grid points in diffusion are pure noise to the model — the solution's spectral content is captured at R=32.

---

## 6. Capacity Scaling Has Diminishing Returns

Effective D-exponents (log-log slopes across parameter count):

| Task | FNO | MLP baseline | DeepONet |
|------|-----|-------------|----------|
| Burgers | −0.037 | varied* | −0.066** |
| Darcy | −0.004 | −0.063 | −0.003 |
| Diffusion | −0.023 | −0.014 | −0.107 |

*MLP baseline D-exponent is confounded with R in the aggregation since param count varies with R.  
**DeepONet D-exponent has poor R² due to non-monotonicity.

Key observations:

1. **FNO on diffusion gets *worse* past small capacity.** Best error = 0.00213 at small (44K params), but 0.00307 at xlarge (2.7M params). The extra parameters overfit or interfere.

2. **FNO on Darcy:** Going from 7K to 2.7M parameters reduces error from 0.573 to 0.471 — a 388× increase in parameters for 18% error reduction.

3. **DeepONet on Darcy:** D-exponent = −0.003. Capacity is irrelevant. All capacity levels achieve roughly the same ~0.75–0.85 error.

---

## 7. Darcy Flow: A Pathological Case

The Darcy task stands out as anomalous across every analysis dimension:

- **High error floor:** Best achievable error is 0.471 (FNO), compared to 0.027 (Burgers) and 0.002 (diffusion). No model gets below 47% relative error.

- **Extreme variance:** DeepONet on Darcy has coefficient of variation up to **1.54** across seeds. Mean error values of 2.0–7.0 (rel-L2 > 1, i.e., worse than predicting the mean) appear frequently.

- **Pervasive non-monotonicity:** Dozens of cases where adding more data increases error — DeepONet at every capacity level, FNO at large/xlarge capacity, MLP baseline at multiple capacities. This is not noise: at specific slices (e.g., DeepONet/xlarge/R=128, N=1000→2000) error jumps from 0.80 to 1.88.

- **No axis helps reliably:** N-exponents are weak (−0.09 for FNO), R is noise (±0.03), D is negligible (−0.004). The log-linear R² for DeepONet is only 0.35, meaning 65% of error variance is unexplained by all three axes combined.

**Interpretation:** The discontinuous permeability coefficients in Darcy flow appear to create a fundamentally harder approximation problem for all three architectures. The nonlinear coefficient-to-solution map may require either much larger datasets, different architectures (e.g., attention-based), or problem-specific preprocessing (e.g., log-transformed coefficients).

---

## 8. DeepONet Saturation and Instability

DeepONet shows two distinct failure modes:

### 8a. Early saturation

On Burgers, DeepONet's N-exponent at R≥32 is only −0.04 to −0.07. Even going from N=50 to N=5000 (100×), error only drops ~25% (from ~0.50 to ~0.37 at medium capacity). By contrast, FNO drops ~85% over the same range.

On diffusion, DeepONet plateaus at ~0.10–0.22 regardless of N, while FNO reaches 0.002. The architecture appears to hit a representation bottleneck that more data cannot resolve.

### 8b. More data can hurt (overfitting to noise?)

On Burgers, DeepONet with large/xlarge capacity at N=2000→5000 consistently shows **5–15% error increases**. This pattern is systematic — appearing across multiple resolutions and capacity levels. Possible explanations: (a) larger training sets with fixed early-stopping patience may stop at worse local minima, or (b) the branch-trunk decomposition has limited expressiveness that is saturated before N=5000.

---

## 9. Cross-Resolution Transfer

For models trained at resolution R and evaluated at a different resolution:

**Burgers / FNO:**
| Eval/Train ratio | Mean error |
|:---------------:|:----------:|
| 0.12 | 0.443 |
| 0.25 | 0.369 |
| 0.50 | 0.305 |
| 1.0 (same) | ~0.25 |
| 2.0 | **0.252** |
| 4.0 | 0.278 |
| 16.0 | 0.432 |

FNO performs best at ~1–2× the training resolution, then degrades. Evaluating at much higher resolution (8–16×) returns error to untrained levels. This suggests FNO's learned spectral filters transfer modestly upward but not dramatically.

**Darcy / FNO:** Essentially flat across all ratios (0.675–0.696). The model's predictions are resolution-insensitive because the task itself has weak resolution dependence.

**Darcy / DeepONet:** Monotonically degrades from 1.36 (ratio=0.12) to 1.57 (ratio=16.0), suggesting the model becomes less reliable outside its training resolution.

---

## 10. Noise and Reproducibility

Coefficient of variation (std/mean of rel-L2 across 6–9 seed runs per configuration):

| Task | FNO | MLP baseline | DeepONet |
|------|:---:|:-----------:|:--------:|
| Burgers | 0.08 | 0.03 | 0.05 |
| Darcy | 0.07 | **0.25** | **0.25** |
| Diffusion | 0.17 | 0.07 | 0.11 |

Darcy is extremely noisy for DeepONet and MLP (CV=0.25), meaning a typical run's error varies by ±25% of the mean just from seed changes. This undermines any scaling law fit: with this noise level, detecting exponent differences of 0.1 requires far more repetitions.

The noisiest single configuration: DeepONet/med-large/N=2000/R=256 on Darcy, with CV=**1.54** (error ranges from near-zero to several times the mean).

---

## 11. Data Efficiency: Samples-to-Threshold

How many samples does the best model (at medium capacity) need to reach a given error threshold?

| Target error | Burgers | Diffusion | Darcy |
|:------------:|:-------:|:---------:|:-----:|
| < 0.10 | N=200 (FNO) | N=50 (FNO) | **Not achieved** |
| < 0.05 | N=500 (FNO) | N=50 (FNO) | **Not achieved** |
| < 0.02 | **Not achieved** | N=50 (FNO) | **Not achieved** |

FNO on diffusion reaches 0.02 rel-L2 with just 50 samples. The spectral structure of the heat equation perfectly matches FNO's Fourier-space inductive bias. Burgers requires 10–100× more data due to nonlinear shock dynamics. Darcy does not reach 10% error with any model/data/capacity combination in this study.

---

## 12. Summary of Actionable Findings

| Finding | Strength of evidence | Practical implication |
|---------|:-------------------:|----------------------|
| FNO dominates all comparisons | Very strong (143/143 wins) | Use FNO for 1D operator learning |
| Dataset size is the primary lever | Strong (65–93% variance explained) | Prioritize data collection over model tuning |
| Resolution does not help (often hurts) | Strong for diffusion, moderate for others | Use the lowest R that captures the physics; more grid points waste compute |
| Capacity has weak returns | Strong (exponents −0.003 to −0.1) | A modest model suffices; massive scaling is wasteful |
| No universal scaling law form | Moderate (AICc selection varies by regime) | Do not assume power-law scaling — validate per-task |
| DeepONet has early saturation | Strong (N-exponent −0.07 on Burgers) | DeepONet may have fundamental expressiveness limits for these tasks |
| Darcy is not well-served by any model | Strong (error floor 0.47, high variance) | Requires architectural innovation or task reformulation |
| Cross-resolution transfer is modest | Moderate (sweet spot at 1–2× ratio) | Don't count on training at low R and evaluating at high R |

---

## 13. Caveats and Limitations

1. **1D only.** All tasks are 1D PDEs. The resolution axis may be far more important in 2D/3D, where discretization error compounds with dimensionality.

2. **Fixed training protocol.** All models use Adam, same LR, same patience. Architecture-specific hyperparameter tuning could narrow gaps (e.g., DeepONet may benefit from different LR schedules).

3. **Limited resolution range for some tasks.** Darcy main runs only cover R=32–256 (4 points). Burgers extended runs cover 16–512 (10 points) but only for some capacity levels.

4. **Optimizer ablation data exists but is not aggregated.** 657 SGD+cosine and 756 Adam+2×patience runs exist in `runs_ablation_*/` but haven't been processed into grouped metrics. This limits conclusions about training protocol sensitivity.

5. **MLP controlled only available for Burgers and Diffusion.** The capacity confound in MLP baseline cannot be evaluated for Darcy.

6. **Metric is relative L2 only.** Different error norms (pointwise max, Sobolev, spectral) could change rankings, especially for tasks with localized features like Burgers shocks.
