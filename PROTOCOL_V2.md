# PROTOCOL_V2 — Response-surface diagnostics for operator learning

**Date:** 2026-04-09
**Status:** Phase 1 deliverable. Self-contained protocol document for the Paper 2 reframe. Replaces the implicit "law-fit + AICc selection" protocol of the previous draft. Reading order: AUDIT.md → VALIDATION_REPORT.md → this document → RESULTS_*.md (Phases 3–5).

This document defines what the new paper measures, how it measures it, and how it interprets it. It commits to operational definitions where the previous draft was vague (regime labels, primary outcomes, inclusion rules) so that the analysis and rewrite phases can proceed without backsliding into law-first thinking.

---

## §0 Decisions taken (locked in 2026-04-09)

| # | Question | Decision |
|---|---|---|
| 1 | Strictness of "no law fits in main text" | (b) Law fits live in an appendix only; main text may use descriptive shorthand like "log-like" or "power-like" when describing response-surface regions, but never names a "winning" law family or AICc winner. |
| 2 | Regime taxonomy basis | Sensitivity-based, with thresholds calibrated empirically from the data in Phase 3. See §6. |
| 3 | Darcy's role | (a) Secondary case. Darcy's R range is half of Burgers's, so cross-resolution conclusions about Darcy are not drawn. Darcy still appears in main-text response-surface plots within its own R range. |
| 4 | Diffusion data axis | (b) Keep the sparse 5-N grid for deeponet/fno/mlp_baseline (50, 100, 500, 1000, 5000) and the 4-N grid for mlp_controlled. Document the caveat in Limitations and avoid drawing slope-based claims on Diffusion. |
| 5 | Seed parity | (b) Work with 2 data seeds × 3 train seeds (= 6 seed pairs) on the diagonal for the main grid; 2 × 2 = 4 seed pairs on cross-res; 3 × 3 = 9 seed pairs on `mlp_controlled`. Report seed variability honestly with its actual sample size; do not rerun. |
| 6 | Optimizer ablations | Reposition the 657 SGD-cosine + 756 Adam-2x-patience runs as evidence for the optimization-limited regime in Phase 5, not as a "robustness check" for law fits. |
| 7 | Deliverable location | Repo root, alongside AUDIT.md, VALIDATION_REPORT.md, ROADMAP.md. |
| 8 | Paper draft target | (b) New file `paper/main_v4.tex`, leave `paper/main.tex` unchanged until v4 is ready, swap in a single commit. |
| R=48 | Burgers data-generator bug | (a) Document and exclude. R=48 is excluded from primary response-surface heatmaps and reported separately as a known data-generator artifact (see VALIDATION_REPORT.md §5b). The data generator is *not* patched and the 729 affected cells are *not* re-run. |

---

## §1 Scope and central thesis

The paper studies how operator-learning models behave as a function of three jointly-varied experimental factors — dataset size N, training resolution R_train, and capacity C — and how that behavior transfers when the model is evaluated at a different resolution R_eval ≠ R_train. The product (N × R_train × R_eval × C) defines the *response surface* over which all primary outcomes are reported.

**Central thesis** (one sentence, to be tested by the analysis): operator-learning systems should be characterized by *where on the (N, R, C, R_eval) response surface they are operating* — data-limited, capacity-limited, resolution-sensitive, saturated, or unstable — rather than by the parametric law that best fits a 1D slice of error vs one axis.

**This thesis is not the same as the rejected ones.** Forbidden framings (carried forward from the reframe directive):

- "We found universal scaling laws."
- "Power laws explain operator-learning scaling."
- "The main result is which functional form wins."
- "Operator-learning scaling should be treated as a model-selection problem over law families."

Permitted descriptive shorthand: "log-like trend", "power-like decay", "saturating behavior", "near-flat region", as adjectives for response-surface regions. Such phrases must never carry the weight of a primary claim.

---

## §2 Experimental axes

| Axis | Symbol | Values | Notes |
|---|---|---|---|
| Task | T | {`burgers_operator`, `darcy`, `diffusion`} | Burgers is the densest grid and is the primary case. |
| Model family | M | {`deeponet`, `fno`, `mlp_baseline`, `mlp_controlled`} | `mlp_controlled` is only present on Burgers and Diffusion; not on Darcy. |
| Capacity | C | {`tiny`, `small`, `small-med`, `medium`, `med-large`, `large`, `xlarge`} | **Capacity slots are named, not strictly numerical.** "medium FNO" and "medium MLP" do not have the same parameter count. The protocol reports parameter count as the canonical capacity coordinate; the slot name is a label only. |
| Dataset size | N | task-dependent (see VALIDATION_REPORT §3) | Burgers/Darcy: 11 values from 50 to 5000. Diffusion: 5 values per model (different sets per model). |
| Training resolution | R_train | task-dependent (see VALIDATION_REPORT §3) | Burgers: 10 values from 16 to 512. Darcy/Diffusion: 4 values from 32 to 256. |
| Evaluation resolution | R_eval | {32, 64, 128, 256, 512} | The cross-resolution superset. Records exist for both the diagonal (R_eval = R_train) and the off-diagonal cells. |
| Data seed | s_d | {11, 22} (and {11, 22, 33} on `mlp_controlled`) | Controls the random task instances (initial conditions / coefficient fields). |
| Train seed | s_t | {101, 202, 303} (and {101, 202} on cross-res) | Controls model initialization and minibatch order. |

**Excluded from the main grid (Decision R=48):**

- All R_train = 48 cells on Burgers are documented in a separate appendix subsection but do not appear in primary response-surface heatmaps. Reason: VALIDATION_REPORT.md §5b shows the Burgers data generator produces NaN/Inf in 1–4% of samples *only* at R=48, so the 582 "diverged" runs at R=48 are not optimization failures but corrupt input data. Including them would inject spurious 80% divergence noise into otherwise-clean cells.

**Sub-grid stratification (Decision: per VALIDATION_REPORT §6):**

For each (T, M, C) slice, two strata are recognized:

- **Core rectangle:** the maximal (N, R) sub-rectangle where every cell has the modal seed-pair coverage for the slice. On Burgers/{deeponet,fno,mlp_baseline}, the core is N ∈ {50, 100, 200, 500, 1000, 2000, 5000} × R ∈ {16, 32, 64, 96, 128, 192, 256, 384, 512} (excluding R=48). On Darcy and Diffusion, the core is the entire stored grid.
- **Fill-in points:** cells that exist outside the core, typically at intermediate N values (150, 300, 750, 3000) restricted to the densest R values. Fill-in points are overlaid on the same heatmaps as the core but rendered as supplementary markers, not as primary cells.

Statistical claims and regime labels are computed *on the core only*. Fill-in points are visualization-only.

---

## §3 Per-run metadata schema

Every diagonal training run produces a `metrics.json` with the fields below. These are the only fields the protocol reads — no inference from path encoding, no inference from filenames.

```
{
  "model_name":      str,           # 'deeponet' | 'fno' | 'mlp_baseline' | 'mlp_controlled'
  "capacity_name":   str,           # capacity slot
  "dataset_size":    int,           # N
  "resolution":      int,           # R_train
  "parameter_count": int,           # canonical capacity coordinate
  "train_seed":      int,
  "best_epoch":      int,           # -1 if the run never completed an epoch
  "runtime_seconds": float,
  "test_rel_l2":     float,         # primary error
  "test_mse":        float,         # secondary error
  "val_rel_l2":      float,         # for early-stopping inspection
  "status":          'success' | 'failed',
  "eligible_for_fit": bool,         # legacy field; ignored by this protocol
  "diverged":        bool,
  "nan_detected":    bool,
  "failure_reason":  str | null,    # 'nan_or_inf' is the only observed value as of 2026-04-09
  "device":          str
}
```

Cross-resolution evaluation rows have the same schema *minus* `diverged`, `nan_detected`, `val_rel_l2`, `runtime_seconds`, and `best_epoch`. Convergence and instability information about a cross-res row must be inherited from its parent diagonal training row by joining on `(task, model, capacity, N, R_train, data_seed, train_seed)`. This join is implemented in `scripts/aggregate_runs.py` (existing) and must be respected by Phase 4.

**Path-encoded metadata** (used only as a join key, never as a source of values):

```
runs/task=<T>/model=<M>/capacity=<C>/N=<N>/R=<R_train>/data_seed=<s_d>/seed=<s_t>/[eval_resolution=<R_eval>/]metrics.json
```

---

## §4 Cell aggregation: from per-run records to per-cell statistics

A *cell* is a tuple (T, M, C, N, R_train) for diagonal cells, and (T, M, C, N, R_train, R_eval) for cross-res cells. For each cell, the protocol computes the following statistics over the seed dimension (data seed × train seed). Per-run records are preserved upstream by `scripts/aggregate_runs.py`.

### 4.1 Primary outcomes (every cell, both diagonal and cross-res)

| Statistic | Definition | Why |
|---|---|---|
| `n_runs` | Number of records in the cell. | Sample size. |
| `n_success` | Number of records with `status == 'success'`. | Convergence numerator. |
| `convergence_rate` | `n_success / n_runs`. | First-class outcome. Replaces "average over successes" for cells with mixed success. |
| `divergence_rate` | `n_diverged / n_runs`. | First-class outcome. **Excludes R=48 cells from any cross-cell aggregation** (see §2). |
| `instability_rate` | Fraction of records with `nan_detected OR diverged OR best_epoch == -1`. | Superset of divergence; catches edge cases. |
| `median_test_rel_l2` | Median `test_rel_l2` over successful records in the cell. | Replaces the previous protocol's mean. Robust to the few-sample-size regime where one bad seed dominates a mean. |
| `iqr_test_rel_l2` | Interquartile range of `test_rel_l2` over successful records. | Variability without distributional assumptions. |
| `seed_cv_test_rel_l2` | Coefficient of variation (std / |mean|) of `test_rel_l2` over successful records. | Cross-cell comparable variability. |
| `worst_test_rel_l2` | Maximum `test_rel_l2` over successful records. | Worst-seed bound; relevant to fragility. |
| `best_test_rel_l2` | Minimum `test_rel_l2` over successful records. | Best-seed bound; pairs with `worst` to bound the seed envelope. |

### 4.2 Secondary outcomes (diagonal only)

| Statistic | Definition |
|---|---|
| `median_best_epoch` | Median of `best_epoch` over successful records. |
| `median_runtime_seconds` | Median of `runtime_seconds` over successful records. |
| `parameter_count` | Constant within a cell; copied through. |
| `data_seeds_observed`, `train_seeds_observed` | Sets of seed values present. |

### 4.3 Cross-resolution outcomes (cross-res cells only, joined to parent diagonal)

| Statistic | Definition |
|---|---|
| `parent_convergence_rate` | The diagonal parent cell's convergence rate, inherited via join. Cross-res rows from a non-converged training run are excluded. |
| `degradation_vs_diagonal` | `median_test_rel_l2(cell) / median_test_rel_l2(diagonal cell at same N, R_train, R_eval=R_train)`. Ratio ≥ 1 means worse than diagonal. |
| `transfer_asymmetry` | For each (R_train, R_eval) pair, compare `degradation_vs_diagonal` against the swapped pair (R_eval, R_train) at the same (T, M, C, N). Asymmetry > 1 means train-low-test-high is worse than train-high-test-low (or vice versa). |

### 4.4 Aggregation script

`scripts/aggregate_runs.py` already preserves per-run records. A new script `scripts/aggregate_cells.py` (to be created in Phase 2 follow-up) will compute the §4.1–4.3 statistics and emit:

- `results/cell_stats_diagonal.csv` — one row per diagonal cell with all primary and secondary outcomes.
- `results/cell_stats_crossres.csv` — one row per cross-res cell with primary outcomes plus joined parent fields.

These two CSVs are the canonical input to Phases 3, 4, 5. No phase reads `runs_aggregate.csv` directly.

---

## §5 Inclusion / exclusion rules

Stated explicitly so the rewrite can cite them and the appendix can list them.

1. **R=48 excluded from primary plots and primary aggregates.** Reported separately in an appendix subsection with the per-R divergence table from VALIDATION_REPORT §5b. The 582 affected runs are reattributed from "training divergence" to "data-generator failure".
2. **Failed runs excluded from `median_*` statistics.** Failed runs still count toward `n_runs`, `convergence_rate`, `divergence_rate`, `instability_rate`. They do not contribute to medians, IQRs, or worst/best bounds.
3. **Fill-in cells (outside the core rectangle) excluded from regime classification.** They are visualized as supplementary markers but not assigned a regime label and not counted in regime tallies.
4. **Darcy excluded from any cross-resolution claim that requires R > 256.** Darcy's R range is {32, 64, 128, 256}; the new paper reports cross-res transfer for Darcy only within that range and does not extrapolate.
5. **Diffusion excluded from local-slope claims that require ≥ 6 N values.** Diffusion has 5 N values per model; any claim requiring a robust local sensitivity in the N direction is computed on Burgers and Darcy only and reported as Diffusion-specific separately when meaningful.
6. **Cross-res cells with `parent_convergence_rate < 1.0` excluded from primary cross-res aggregates.** A cross-res evaluation of a parent training run that itself failed inherits the failure.
7. **The `eligible_for_fit` field from the legacy schema is ignored.** It was a law-fit-specific filter and has no meaning in the new protocol. The exclusion rules above replace it.

---

## §6 Regime taxonomy

The five regime labels and their operational definitions. Each label is a function of *local sensitivities* of `median_test_rel_l2` to N, R_train, and parameter_count, computed on the core rectangle of each (T, M, C) slice. Thresholds are placeholders to be calibrated empirically in Phase 3.

### 6.1 Local sensitivities

For a cell (T, M, C, N_i, R_j) inside the core rectangle, define the discrete log-log derivatives:

```
S_N(i, j)  = ( log E(i+1, j) - log E(i-1, j) ) / ( log N_{i+1} - log N_{i-1} )
S_R(i, j)  = ( log E(i, j+1) - log E(i, j-1) ) / ( log R_{j+1} - log R_{j-1} )
S_C(i, j)  = ( log E(i, j) at C+1 - log E(i, j) at C-1 ) / ( log P_{C+1} - log P_{C-1} )
```

where `E = median_test_rel_l2`, `P = parameter_count`. Boundary cells use one-sided finite differences. Cells where any of the three neighbors is missing or has `convergence_rate < 1` are flagged as `regime = ambiguous` and not labeled.

The sign convention: scaling laws of the form E ∝ x^{-α} give negative log-log slopes (S < 0) when error decreases with the axis. So |S_N|, |S_R|, |S_C| measure how *steeply* error responds to that axis. Regime labels operate on absolute values.

### 6.2 Thresholds (placeholders, calibrated in Phase 3)

```
THRESH_HIGH = 0.20   # |S| above this counts as "responsive" on that axis
THRESH_LOW  = 0.05   # |S| below this counts as "saturated" on that axis
THRESH_DIV  = 0.10   # divergence_rate above this counts as "unstable"
THRESH_CV   = 0.50   # seed_cv above this counts as "unstable"
```

These are first-pass values. Phase 3 will plot the empirical distribution of |S_N|, |S_R|, |S_C|, divergence_rate, and seed_cv across all cells and tune the thresholds so that:

(a) the modal cell of the Burgers/FNO core falls in `saturated` if its three sensitivities are all in their bottom quartile;
(b) the cells the user / reviewer would label "obviously data-limited" by visual inspection are caught by the rule;
(c) the unstable threshold is just barely tight enough to flag the optimizer-ablation R cells in Phase 5 without flagging any R_train ≠ 48 main-grid cells.

The calibration process and final thresholds are recorded in RESULTS_RESPONSE_SURFACES.md.

### 6.3 Labels

Applied per-cell on the core rectangle. Labels are *not* mutually exclusive in the trivial sense — a cell can be both "data-limited" and "saturated on R" — so the protocol assigns a *primary label* (the regime that dominates) plus an optional secondary label.

| Primary label | Rule |
|---|---|
| `unstable` | `divergence_rate > THRESH_DIV` OR `seed_cv > THRESH_CV` OR `instability_rate > THRESH_DIV`. Takes precedence over all others; an unstable cell does not get a sensitivity-based label. |
| `data-limited` | `|S_N| > THRESH_HIGH` AND `|S_R| < THRESH_HIGH` AND `|S_C| < THRESH_HIGH`. |
| `resolution-sensitive` | `|S_R| > THRESH_HIGH` AND `|S_N| < THRESH_HIGH` AND `|S_C| < THRESH_HIGH`. |
| `capacity-limited` | `|S_C| > THRESH_HIGH` AND `|S_N| < THRESH_HIGH` AND `|S_R| < THRESH_HIGH`. |
| `saturated` | `|S_N| < THRESH_LOW` AND `|S_R| < THRESH_LOW` AND `|S_C| < THRESH_LOW`. |
| `mixed` | None of the above (multiple axes simultaneously responsive but none dominant). |
| `ambiguous` | Not enough neighbors to compute one of the three sensitivities, or boundary cell, or excluded by §5. |

### 6.4 Why sensitivity-based, not law-fit-based

The previous protocol identified regimes by which parametric law (power, log, exponential, …) best fit a 1D slice. That approach has three problems the new protocol fixes:

1. **It collapses 2D structure into 1D slices.** The decision of which axis to slice along is itself a modeling choice that biases the result. Sensitivity computed on the local response-surface neighborhood does not.
2. **It depends on which law family is in the candidate set.** Adding or removing a candidate can flip the "winner" without any change in the data.
3. **At small N (4–7 points per slice), the bootstrap-stability of any IC-selected winner is poor.** The reviewer's own held-out contest showed only 15/33 wins (45%) for the multi-law fitter against a power-law-only baseline, with mean improvement ~1.2%. Sensitivity-based labels do not require model selection.

Sensitivity-based labels also map directly to the user-facing question the paper is trying to answer ("when does adding more data / more resolution / more capacity help?") in a way law identity does not.

---

## §7 Cross-resolution transfer

This is the centerpiece of the new framing. The cross-res records on disk (13,440 of them, see VALIDATION_REPORT §1) make this analysis possible without any reruns.

### 7.1 Transfer matrix

For each (T, M, C, N) slice, the *transfer matrix* `K[i, j]` is defined as:

```
K[i, j] = median_test_rel_l2 over seeds at (R_train = R_i, R_eval = R_j)
```

where R_i ranges over the slice's R_train values and R_j ranges over the cross-res superset {32, 64, 128, 256, 512} intersected with the slice's R_eval values. R_train = 48 columns are excluded.

The diagonal entries `K[i, i]` (when R_i is in both the train and eval grids) are the standard "train and test at the same resolution" baseline. Off-diagonal entries are the transferred error.

### 7.2 Robustness and fragility scores

For each (T, M, C, N) slice:

| Score | Definition | Reading |
|---|---|---|
| `mean_degradation` | Mean over off-diagonal cells of `K[i, j] / K[i, i]`. | Average factor by which moving away from the training resolution degrades error. ≥ 1 by construction in expectation. |
| `worst_degradation` | Maximum over off-diagonal cells of `K[i, j] / K[i, i]`. | Worst-case fragility. The cell achieving the maximum is reported alongside. |
| `train_low_eval_high_ratio` | Mean of `K[i, j] / K[i, i]` over (i, j) with R_j > R_i. | "Did training at low resolution and testing high hurt?" |
| `train_high_eval_low_ratio` | Mean of `K[i, j] / K[i, i]` over (i, j) with R_j < R_i. | "Did training at high resolution and testing low hurt?" |
| `transfer_asymmetry` | Ratio of the previous two. | > 1: training low and testing high is the worse direction. |

These five scores are reported per-slice in RESULTS_TRANSFER.md as the primary cross-res evidence.

### 7.3 Existing primitives

`src/scaling_operator_learning/analysis/cross_resolution.py` already implements `build_transfer_matrix(df, ...)` and `resolution_transfer_gain(df, ...)`. Phase 4 needs to add a driver script `scripts/build_transfer_matrices.py` that iterates over all (T, M, C, N) slices, builds the matrix using the existing primitive, computes the §7.2 scores, and emits:

- `results/transfer_matrices.json` — full per-slice matrices and scores.
- `figures/transfer_matrix_<task>_<model>.{pdf,png}` — one centerpiece figure per (task, model) pair, faceted by capacity and N.

---

## §8 Convergence and instability (Phase 5)

The narrative here changes the most relative to the previous draft.

### 8.1 What "instability" means in the new protocol

A cell is *unstable* if its `divergence_rate > THRESH_DIV` or `seed_cv > THRESH_CV` (per §6.2). Unstable cells receive the `unstable` regime label and do not get a sensitivity-based label.

Two distinct sources of instability are recognized and reported separately:

- **Optimization-limited instability:** runs that completed at least one epoch but failed to converge (e.g. NaN appearing mid-training, or seed CV blowing up). The optimizer ablations (657 SGD-cosine + 756 Adam-2x-patience) are repositioned (Decision 6) as evidence here.
- **Data-generator failure:** the 582 R=48 Burgers runs (VALIDATION_REPORT §5b). These are *not* counted as optimization instability; they are reported in their own appendix subsection.

The paper's "582 diverged" macro stays numerically the same but is reattributed.

### 8.2 Per-cell visualizations

Phase 5 produces:

- A divergence-rate heatmap per (T, M, C) slice over (N, R) — `figures/divergence_heatmap_<task>_<model>.{pdf,png}`.
- A seed-CV heatmap per (T, M, C) slice — `figures/seed_cv_heatmap_<task>_<model>.{pdf,png}`.
- A scatter of `seed_cv` vs `divergence_rate` colored by regime label, one point per cell, all (T, M, C) overlaid — to show whether the unstable threshold is doing real work.

---

## §9 Law fits as appendix supplement

Decisions 1, 6 confirm: law fits stay only in an appendix.

- The existing artifacts in `results/multilaw_fits.json`, `results/multilaw_fits_restricted.json`, `results/within_slice_bootstrap.json`, `results/holdout_prediction_contest.json` are reused without rerunning.
- The appendix subsection (sketched in `APPENDIX_LAW_FITS.md`, deferred to Phase 6) presents the held-out contest's 15/33 win fraction and ~1.2% mean improvement as the *honest fragility evidence*, not as a "robustness check". The reframe message: "we tried law-family selection; the evidence that it adds little is the strongest justification for the response-surface framing."
- Bootstrap-instability of the law identity (22/27 slices "exploratory") is reported in the same appendix subsection as a corroborating signal.
- The main text does not name law families or AICc winners. It may use descriptive shorthand ("log-like trend in the high-N region") with a forward reference to the appendix.

---

## §10 What the protocol does NOT do

Stated explicitly to head off scope creep and to make the rewrite reviewable:

1. **No new training runs.** Phases 1–6 are pure analysis and rewriting on existing data. The Decision-5 default (work with what we have) means no rerun cost. If Phase 3/4/5 surface specific cells where reruns are clearly necessary, they go into a new "Phase 5b reruns" section, not into the main critical path.
2. **No new task implementations.** Burgers, Darcy, Diffusion as currently implemented. The R=48 data-bug is documented, not patched.
3. **No claims that require ≥ 3 data seeds at the cell level.** The protocol works with the 2-data-seed reality.
4. **No held-out exponent prediction.** The held-out contest is part of the appendix narrative as fragility evidence, not as a separate primary result.
5. **No "we found a universal scaling law".** Categorically forbidden by Decision 1 and the reframe directive.
6. **No claims about Diffusion local slopes** that aren't explicitly caveated by the 5-N-value sample size.
7. **No claims about Darcy cross-resolution behavior outside R ∈ {32, 64, 128, 256}.**

---

## §11 Phase 1 → Phase 2 → … → Phase 8 deliverables

| Phase | Deliverable | Status | Owner |
|---|---|---|---|
| 0 | `AUDIT.md` | done (committed e6adb3b → 6cb873b) | Claude |
| 1 | `PROTOCOL_V2.md` (this document) | **done with this commit** | Claude |
| 2 | `VALIDATION_REPORT.md` + `results/cell_coverage_*.csv` | done (committed 2b370ac, 5e83152, b8716b1) | Claude |
| 2b | `scripts/aggregate_cells.py` + `results/cell_stats_*.csv` | not started (extends Phase 2 with the §4 statistics) | Claude |
| 3 | `RESULTS_RESPONSE_SURFACES.md` + `figures/response_surface_*` + threshold calibration | not started | Claude |
| 4 | `RESULTS_TRANSFER.md` + `scripts/build_transfer_matrices.py` + `figures/transfer_matrix_*` | not started | Claude |
| 5 | `RESULTS_CONVERGENCE.md` + `figures/divergence_heatmap_*`, `figures/seed_cv_heatmap_*` | not started | Claude |
| 6 | `APPENDIX_LAW_FITS.md` (optional) | not started | Claude |
| 7 | `paper/main_v4.tex` rewrite | not started | Claude |
| 8 | `CLAIM_AUDIT.md` (per-claim evidence audit) | not started | Claude |

Phase ordering: 2b → 3 → 4 → 5 → 6 → 7 → 8. Phases 3, 4, 5 can run in parallel after 2b is done because they read the same `results/cell_stats_*.csv` files but produce independent figures and result documents.

---

## §12 What I'm doing immediately after this commit

Per the phase ordering, the next concrete action is **Phase 2b**: write `scripts/aggregate_cells.py` to compute the §4 statistics from `runs/runs_aggregate.csv` (and the existing `cell_coverage_*.csv` files), and emit `results/cell_stats_diagonal.csv` and `results/cell_stats_crossres.csv`. This is the canonical input for Phases 3, 4, 5 and is purely descriptive — no thresholds, no labels, no figures.

After Phase 2b, Phase 3 (response surfaces + threshold calibration) starts.

I will not begin the paper rewrite (Phase 7) until Phases 3–5 are finished and their result documents are committed.
