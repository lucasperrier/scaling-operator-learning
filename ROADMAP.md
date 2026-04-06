# Roadmap — v3 Revision

Target: address all reviewer concerns, confront criterion sensitivity head-on, and position the multi-law framework against a traditional power-law baseline.

**Core methodological framing:** Classic scaling-law work is exponent-centric under a power-law prior; this paper is family-selection-centric under model uncertainty. The revision retains the multi-law framework as the main contribution, adds a traditional power-law analysis as benchmark baseline, and shows explicitly where the baseline breaks.

**Main vulnerability:** model selection at n = 5 is inherently fragile. Every experiment and text revision should make that fragility transparent and show that the main conclusions survive it.

---

## Tier 1: Pre-submission essentials (text + existing data)

These use existing data only — no new runs. Highest impact per effort.

### 1.1 Confront BIC table head-on
The current BIC concordance discussion is misleading. At n = 4–5, BIC's penalty k·ln(n) ≈ 1.6k is *weaker* than AICc's corrected penalty 2k + 2k(k+1)/(n−k−1). So BIC favors *more* complex models at these sample sizes — the opposite of its asymptotic reputation. The paper currently says "BIC penalises complexity more heavily" which is only true for large n.
- [x] Phase 1.1–1.4 already done: Akaike weights, bootstrap, restricted set, BIC concordance all implemented and in paper
- [x] Rewrite Appendix A.5 (BIC concordance) to explain the small-n penalty inversion explicitly
- [x] State plainly: "At n = 5, AICc is the more appropriate criterion (designed for small samples); BIC concordance is reported for transparency, not as a robustness check in the usual sense"
- [x] In the main text, qualify claims where AICc–BIC concordance is low: either "stable under AICc (BIC diverges due to small-n penalty inversion)" or "criterion-sensitive — see Appendix"

### 1.2 Fix aggregation-method inconsistency
- [x] Section 4.5 (multi-law fitting procedure, ~line 384) still says "We report the dominant law (most slice-level wins)" — this is the old vote-counting language
- [x] Replace with language consistent with Section 3.3: dominant law by mean Akaike weight, with vote counts as supporting detail
- [x] Audit full paper for any remaining "vote count → single winner" language

### 1.3 Soften interpretive claims
- [x] Section 6.1: "Logarithmic data scaling signals a high-dimensional learning problem" → frame as hypothesis/interpretation, not diagnostic fact. No dimensionality measurement or theoretical argument currently supports this as a firm conclusion
- [x] Section 6.3: "DeepONet appears to be a representational bottleneck in practice" → "performed worst in this study" / "appears to be a bottleneck in the settings studied (1D PDEs, three architectures)"
- [x] Audit Section 6 (architecture recommendations) for assertions that overgeneralise from 1D/3-architecture results

### 1.4 Add traditional power-law baseline framing to related work
- [x] Add a paragraph to Section 2 (Related Work) positioning the multi-law framework against classic methodology (Kaplan/Hoffmann/Hestness)
- [x] Key distinction: classic work assumes power laws and fits exponents; this paper selects among law families and treats law identity as informative
- [x] Frame the paper as showing *when* the traditional baseline succeeds and when it fails, not as discarding standard practice

**Deliverables:** ✅ revised `main.tex` (Appendix A.5, Section 4.5, Section 6, Section 2)

---

## Tier 2: New analysis on existing data (code + paper)

These require new code but use the existing 5,040 runs.

### 2.1 Held-out prediction contest (decisive new experiment)
This converts "which law fits better" from a model-selection question into a falsifiable prediction question. The single most important new analysis.
- [x] For each task–model–axis: hold out 20% of grid configurations (e.g., one capacity level + one resolution)
- [x] Fit (a) classic power-law model and (b) multi-law framework on the remaining 80%
- [x] Evaluate out-of-sample prediction error for both approaches
- [x] Report: does multi-law merely describe observed points better, or does it generalise better to unseen (N, D, R) settings?
- [x] Add results as a new subsection (Section 5.8 or Appendix)

### 2.2 Within-slice data-level bootstrap (methodological fix)
The current bootstrap resamples *slices* (which already have a fixed winner), not the data points within each slice. It measures inter-slice consensus but does not propagate fitting uncertainty.
- [x] For each slice: resample the n data points with replacement (B = 500), refit all 6 laws, reselect AICc winner each time
- [x] Report per-slice law-selection uncertainty (fraction of resamples each law wins)
- [x] Aggregate: mean within-slice winner probability per task–model–axis
- [x] This will almost certainly show wider uncertainty than vote-resampling — present honestly
- [x] Update bootstrap figures and tables to include both levels (vote-resampling for inter-slice consensus, data-level for within-slice uncertainty)

**Deliverables:** ✅ `scripts/run_holdout_contest.py`, `scripts/run_within_slice_bootstrap.py`, new Sections 5.8–5.9 in `main.tex` (placeholder tables pending data)

---

## Tier 3: New training runs (if targeting selective venue)

These require new runs and substantially extend the experimental scope.

### 3.1 Densify the data axis
The data axis (n = 5 points per slice) is the weakest link for law selection. Adding even 2 intermediate values substantially changes the model-selection landscape.
- [x] Add N ∈ {200, 2000} (or finer: {150, 300, 750, 2000, 3000}) to the dataset-size sweep
- [ ] Re-run law selection with the denser axis; check whether "logarithmic wins" stabilises or shifts
- [ ] Re-run BIC concordance on the denser axis — at n = 7+ the small-n penalty inversion becomes less severe
- [ ] Report: is the logarithmic vs power-law distinction criterion-robust once the axis is better sampled?

**Code:** `configs/*_dense_N.yaml` (3 configs, N = [50..5000] with 11 values)

### 3.2 MLP fixed-parameter-count experiment
The cleanest mechanistic test in the paper. "Resolution hurts" for MLP on Diffusion could be a confound with D ∝ R.
- [x] Design MLP variant where hidden width is adjusted to hold D approximately fixed across R ∈ {32, 64, 128, 256}
- [ ] Run on Diffusion task (where "resolution hurts" is strongest)
- [ ] If conclusions flip → the confound was real; if they hold → genuine resolution effect
- [ ] Report in a new subsection or appendix

**Code:** `src/scaling_operator_learning/models/mlp_controlled.py` — registered as `mlp_controlled`, param count holds ~33K across R=32..256

### 3.3 Widen resolution grid
With only n = 4 resolution values, the specific law on this axis is exploratory. Extending the range tests whether the directional trichotomy is stable.
- [x] Add R ∈ {16, 512} (or {16, 48, 96, 192, 384, 512}) to the resolution sweep
- [ ] Check whether Burgers/Darcy/Diffusion trichotomy survives
- [ ] Look for regime changes: does resolution help at first then saturate? Hurt only above a threshold?

**Code:** `configs/*_wide_R.yaml` (3 configs, R = [16..512] with 10 values)

### 3.4 Frequency-filtered diffusion experiments
Tests the causal mechanism behind "resolution hurts": is it extraneous high-frequency content?
- [x] Train at high R but low-pass filter inputs to match the output-relevant spectral content
- [ ] Compare filtered high-R training against raw high-R training on Diffusion
- [ ] If filtering fixes the degradation → strong support for "extraneous frequencies hurt" interpretation

**Code:** `src/scaling_operator_learning/tasks/diffusion_filtered.py` — registered as `diffusion_filtered` task; `configs/diffusion_filtered.yaml`

### 3.5 DeepONet interpolation ablation
- [x] Test native-resolution branch evaluation (no interpolation back to R_train) on a subset of runs
- [ ] Report whether cross-resolution transfer conclusions change

**Code:** `scripts/run_deeponet_ablation.py` — compares interpolation vs subsampling strategies

### 3.6 Early-stopping / optimiser sensitivity
- [x] Rerun a subset (e.g., Burgers, all models, medium capacity) with 2× patience and a different optimiser (SGD + cosine schedule)
- [ ] Check if law-selection winners shift

**Code:** `scripts/run_optimizer_ablation.py` — runs Adam-2×-patience and SGD-cosine variants; `train_one_run()` now supports `optimizer_type` and `scheduler_type` args

**Deliverables:** ✅ all code and configs ready; training runs pending GPU execution. New configs in `configs/*_dense_N.yaml`, `configs/*_wide_R.yaml`, `configs/diffusion_filtered.yaml`; new model `mlp_controlled`; new task `diffusion_filtered`; scripts `run_deeponet_ablation.py`, `run_optimizer_ablation.py`, `run_tier3_all.sh`; placeholder appendix sections in `main.tex`

---

## Tier 4: Future work (beyond current paper scope)

These are important questions the paper should mention but not attempt to answer in this revision.

### 4.1 Higher-dimensional PDEs
- 2D/3D PDE benchmarks — resolution effects amplified (R^d cost growth), may shift the balance
- Tests whether qualitative findings generalise beyond 1D

### 4.2 Newer architectures
- GNOT, operator transformers, Geo-FNO
- Question: does a stronger architecture collapse the log-vs-power distinction, or does law type remain task-controlled?

### 4.3 Joint heterogeneous-law 3D model
- Fit a model that can be logarithmic in N, linear in R, and flat in D simultaneously
- Bridge between axiswise fits and classic additive power law

### 4.4 Varying PDE smoothness systematically
- Family of tasks interpolating between diffusion-like smooth and advection-dominated
- Tests whether law type changes continuously with spectral decay or is mainly model-determined

---

## Execution order & dependencies

```
Tier 1 (text fixes) ──────────────────────────────────────────► can submit to workshop/preprint
         │
         ▼
Tier 2.1 (held-out prediction) ─┐
Tier 2.2 (within-slice bootstrap) ──► update figures/tables ──► stronger submission
         │
         ▼
Tier 3.1 (densify data axis)     ─┐
Tier 3.2 (MLP controlled exp)     ├──► selective-venue quality
Tier 3.3 (widen resolution grid)  │
Tier 3.4 (frequency filtering)   ─┘
```

Tier 1 uses existing text/data only.
Tier 2 uses existing data, new analysis code.
Tier 3 requires new training runs.
Tier 4 is documented as future work.

Distributed execution plan with current three RunPod workers is documented in
`ROADMAP_EXECUTION_PODS.md`.

---

## Success criteria

- [x] The BIC discussion explains the small-n penalty inversion; the paper does not claim BIC is "more conservative" at n = 5
- [ ] Every claim about a "winning law" is backed by Akaike weight > 0.5 *and* within-slice bootstrap stability, or explicitly flagged as ambiguous
- [x] The held-out prediction contest shows whether multi-law selection generalises better than a power-law baseline (either outcome is publishable) — code ready, data pending
- [x] Aggregation language is consistent throughout (mean Akaike weights, not raw votes)
- [x] Interpretive claims (log = high-dimensional, DeepONet = bottleneck) are framed as hypotheses, not diagnostics
- [x] Resolution axis conclusions are directional, with law identity presented as exploratory
- [x] MLP confound is discussed and bounded (ideally with a control in Tier 3.2) — `mlp_controlled` model ready, runs pending
- [x] Related work positions the paper against classic scaling-law methodology with the hybrid framing
