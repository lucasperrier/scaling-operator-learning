# Phase 0 Audit — `scaling-operator-learning`

**Date:** 2026-04-09
**Purpose:** Map the existing repository as preparation for repositioning Paper 2 from a law-fit / AICc-winners frame to a response-surface diagnostic frame. This document describes what is already on disk; it does not propose changes (those belong in PROTOCOL_V2 and the phase docs that follow).

**Verdict in one paragraph.** The repo is well-populated with experimental data — 17,930 diagonal training runs and 13,440 cross-resolution evaluation records, plus 1,413 ablation runs — across 3 tasks, 4 model families, 7 capacity levels, 4–11 dataset sizes, and 4–10 spatial resolutions. **Cross-resolution evaluation data already exists on disk in the form needed for transfer-matrix analysis**, which is the single most important enabler of the new framing: the centerpiece pivot does not require new experiments. The analysis and plotting code, by contrast, is built almost entirely around 1D-slice law-fitting (six-candidate AICc selection, bootstrap stability, held-out exponent prediction). The current paper draft `paper/main.tex` has changed its title toward "Beyond Power Laws: Scaling-Law Diagnostics" but its body, contributions list, and Methods section still center on multi-law model selection and AICc-winner-as-diagnostic — i.e. it is in the framing the user is rejecting. The reframe is therefore primarily an *analysis-and-rewrite* job, not a re-run job.

---

## 1. Top-level layout

| Path | Purpose | Notes |
|---|---|---|
| `src/scaling_operator_learning/` | Library: tasks, models, training, analysis modules | Hydra-free YAML+dataclass config |
| `configs/` | YAML experiment configs | One per task, plus pilot variants |
| `scripts/` | Sweep launchers, aggregation, plotting, analysis | See §5 |
| `runs/` | Main grid output (gitignored) | 31,370 records — see §3, §4 |
| `runs_ablation_sgd_cosine/` | Optimizer ablation: SGD+cosine on Burgers (tracked) | 657 records |
| `runs_ablation_adam_2x_patience/` | Optimizer ablation: Adam with 2× early-stopping patience (gitignored) | 756 records |
| `results/` | Aggregated analysis JSONs (law fits, bootstrap, held-out contest) | All law-centric — see §5 |
| `figures/` | Generated PNGs/PDFs | Mostly law-fit visualisations — see §6 |
| `paper/` | LaTeX drafts: `main.tex`, `main_v1.tex`, `main_v2.tex`, `main_v3.tex` | See §7 |
| `paper-reviews/` | External review (`review-2026-04-07-001.tex`) | See §8 |
| `INDEPENDENT_ANALYSIS_REPORT.md`, `ROADMAP.md`, `ROADMAP_EXECUTION_PODS.md`, `AGENT_PROMPT.md` | Project planning / hand-off notes (already in repo) | Pre-existing context |

---

## 2. Experimental axes that actually exist

**Tasks** (`src/scaling_operator_learning/tasks/`):

- `burgers_operator` — 1D viscous Burgers `u_t + u·u_x = ν·u_xx`, periodic, pseudo-spectral + RK4
- `darcy` — 1D Darcy flow `−d/dx(a(x)·du/dx) = f`, Dirichlet BCs, log-normal permeability, finite differences
- `diffusion` — 1D heat / smooth diffusion problem
- (`diffusion_filtered.py` exists in the tasks directory but does not appear in the main run tree)

**Model families** (`src/scaling_operator_learning/models/`):

- `mlp_baseline` — discretization-tied fully-connected R → hidden → R (parameter count grows with R)
- `deeponet` — branch + trunk decomposition
- `fno` — 1D Fourier Neural Operator
- **`mlp_controlled`** — capacity-controlled MLP variant designed to hold parameter count fixed across resolutions, isolating resolution effect from capacity confound. **This is important and under-recognized in the existing draft.** Present in `runs/task=burgers_operator/model=mlp_controlled/...` and `runs/task=diffusion/model=mlp_controlled/...` (not in Darcy).

**Capacity ladder** (in `src/scaling_operator_learning/models/__init__.py`): named slots `tiny`, `small`, `small-medium`, `medium`, `medium-large`, `large`, `xlarge`. Hidden widths are encoded as a per-name list (e.g. `medium=[128,128]`); FNO has its own `FNO_CAPACITY_GRID`. Parameter count is computed via `parameter_count()`. Note: capacity slots are *named*, not strictly numerical, so cross-architecture comparison at the same name is not a clean comparison at the same parameter count — the existing draft already concedes this in §1.

**Axes per run:**

- N (dataset size): task-dependent grid; Burgers/Darcy use ~11 values from 50–5000; Diffusion uses a sparser 5-value grid.
- R_train (training resolution): Burgers uses {16, 32, 48, 64, 96, 128, 192, 256, 384, 512}; Darcy and Diffusion use the narrower {32, 64, 128, 256}.
- R_eval (evaluation resolution): a 5-value superset {32, 64, 128, 256, 512} used for cross-resolution post-hoc evaluation.
- data_seed: actual values used in the main grid are `{11, 22}` (NOT `{11, 22, 33}` as the paper text previously claimed — this was the reviewer's finding).
- train_seed: `{101, 202, 303}` for most tasks, with some Diffusion cells limited to `{101, 202}`.

---

## 3. Run output format

**Directory pattern (diagonal training run):**

```
runs/task=<task>/model=<model>/capacity=<cap>/N=<N>/R=<R>/data_seed=<ds>/seed=<ts>/metrics.json
```

**Diagonal `metrics.json` fields** (verified by reading `runs/task=diffusion/model=fno/capacity=xlarge/N=50/R=32/data_seed=22/seed=202/metrics.json`):

```
model_name, capacity_name, dataset_size, resolution (= R_train),
parameter_count, train_seed, best_epoch, runtime_seconds,
test_rel_l2, test_mse, val_rel_l2,
status, eligible_for_fit, diverged, nan_detected, failure_reason,
device
```

This is enough for the new framing's diagnostic columns: convergence flag, divergence flag, NaN flag, best epoch, val/test error, runtime, parameter count.

**Directory pattern (cross-resolution evaluation):**

```
runs/.../seed=<ts>/eval_resolution=<R_eval>/metrics.json
```

**Cross-resolution `metrics.json` fields** (verified):

```
model_name, capacity_name, dataset_size, resolution (= R_train),
eval_resolution (= R_eval), parameter_count, train_seed,
test_rel_l2, test_mse,
status, eligible_for_fit
```

**Important asymmetry.** Cross-res records have *no* `diverged`, `nan_detected`, `val_rel_l2`, `runtime_seconds`, or `best_epoch` fields. Convergence/instability information about a cross-res evaluation lives only in its parent diagonal training record (same path, one level up). Phase 5 (convergence analysis) needs to join cross-res rows back to their parent's convergence flags — this is a small but real wrinkle.

**`scripts/aggregate_runs.py`** parses both directory patterns correctly. It already groups by `(model_name, capacity_name, parameter_count, dataset_size, resolution, task, eval_resolution)` and emits two CSVs: a per-run `runs_aggregate.csv` and a per-cell `grouped_metrics.csv` containing means and stderr. **Per-run records are preserved**, so re-aggregating with new statistics (median, IQR, divergence rate, instability rate, convergence rate) is purely a rewrite of `group_records()` — no data loss to recover from.

---

## 4. Coverage of the experimental grid

**Verified counts (from disk, not from text):**

| Tree | Records | What it is |
|---|---:|---|
| `runs/` diagonal (no `eval_resolution=` suffix) | 17,930 | Training runs, R_train = R_eval implicit |
| `runs/` cross-resolution (`eval_resolution=` suffix) | 13,440 | Post-hoc evaluations, R_train ≠ R_eval (or = R_eval and stored explicitly) |
| `runs/` total | **31,370** | Matches `\mainRuns` macro in paper |
| `runs_ablation_sgd_cosine/` | 657 | SGD+cosine optimizer ablation, Burgers only |
| `runs_ablation_adam_2x_patience/` | 756 | Adam, doubled patience |
| **Grand total on disk** | **32,783** | Matches `\totalRuns` macro |

**Tasks present in the main grid:** burgers_operator, darcy, diffusion (3).
**Models present per task:**
- burgers_operator: deeponet, fno, mlp_baseline, mlp_controlled
- darcy: deeponet, fno, mlp_baseline (no mlp_controlled)
- diffusion: deeponet, fno, mlp_baseline, mlp_controlled

**Density assessment** (qualitative — needs systematic verification in Phase 2):

- Burgers is the densest cell: 11 N-values × 10 R-values × 7 capacities × 2 data_seeds × 3 train_seeds. This is the cell that supports primary claims.
- Darcy is moderately dense: 11 N-values × 4 R-values. Narrower R range than Burgers — limits cross-resolution conclusions about Darcy.
- Diffusion has only 5 N-values (sparser data axis). The paper already concedes Diffusion MLP_baseline is on the sparser n=5 grid. This grid asymmetry is real and recurring in the reviewer's complaints.
- Holes likely exist at extreme capacities (tiny / xlarge) for some (task, N, R) cells — needs Phase 2 enumeration.

**Reviewer-verified failure data:**

- 582 Burgers training runs diverged (~3.3% of Burgers diagonal runs). Geographic distribution across (N, R, capacity, model) is unknown but recoverable from `runs_aggregate.csv`. The paper draft previously claimed "zero divergence"; the reviewer caught this and the fix is in `main.tex` (`582 diverged ($<$3.3\%)` in Methods).
- SGD+cosine ablation: 657/756 successful (the manuscript previously said 471/756 — also reviewer-corrected).

---

## 5. Existing analysis infrastructure

**Aggregation:** `scripts/aggregate_runs.py` (130 lines). Walks the runs tree, parses path-encoded metadata, joins with `metrics.json` payload, emits `runs_aggregate.csv` and `grouped_metrics.csv`. Preserves per-run records. **Adequate as a base layer for the new analyses; the grouping function will need to be replaced or augmented to add median, IQR, convergence rate, instability rate, and seed-level variance.**

**Law-fitting (the bulk of existing analysis effort):**

- `scripts/run_multilaw_analysis.py` (~550 lines) — primary engine. Fits 6 candidate functional forms (power, exponential, logarithmic, stretched exponential, saturation, linear) per (task, model, capacity) slice on each axis (N, R, D), AICc-selects a per-slice winner, computes bootstrap stability and BIC concordance.
- `scripts/run_within_slice_bootstrap.py` — within-slice bootstrap for law-fit stability. Outputs `results/within_slice_bootstrap.json`. **Reviewer-derived headline: 22/27 slices are "exploratory" (bootstrap-unstable), only 4–5 are "robust" — this is itself ammunition for the user's pivot, since it shows law identity is mostly not statistically supported.**
- `scripts/run_holdout_contest.py` — held-out exponent-prediction contest. Outputs `results/holdout_prediction_contest.json`. **The contest shows multi-law fitting beats power-law-only baseline by ~1.2% mean improvement (15/33 wins). This is roughly noise, and is the strongest internal evidence that the law-fit framing is not doing scientific work.**
- `scripts/run_analysis.py`, `scripts/run_fast_analysis.py` — older / faster fitting prototypes.

**Cross-resolution analysis:** `src/scaling_operator_learning/analysis/cross_resolution.py` (~58 lines). Two functions: `build_transfer_matrix()` pivots a long-form DataFrame into an (R_train × R_eval) matrix for a single (task, model, capacity, N) slice; `resolution_transfer_gain()` computes improvement vs the diagonal baseline. **The primitives exist but they're tiny — there is no script that systematically applies them across all (task, model) pairs, no robustness/fragility scores, no transfer-aware aggregation.** This is the largest piece of new analysis code needed.

**Convergence / instability analysis:** Effectively absent. `scripts/independent_analysis.py` (the recently added independent audit script) computes per-cell coefficients of variation and surfaces "noisiest regimes," but there is no code that builds a per-cell convergence-rate or instability-rate heatmap, and no narrative about optimization-limited regions.

**Other:**

- `scripts/run_optimizer_ablation.py` — drives the SGD/Adam ablations
- `scripts/run_deeponet_ablation.py` — interpolation ablation for DeepONet (results in `results/deeponet_interpolation_ablation.json`)
- `scripts/independent_analysis.py` — recently added; produces `INDEPENDENT_ANALYSIS_REPORT.md`. Worth re-reading in Phase 2.

---

## 6. Plotting

**Existing figures (in `figures/`):**

- `fig1_data_scaling.png`, `fig2_resolution_scaling.png`, `fig3_capacity_scaling.png` — 1D log-log slices, one axis at a time
- `fig4_exponent_comparison.png` — bar chart of fitted exponents with CIs
- `fig5_transfer_heatmap.png` — **a transfer matrix exists**, but it's one figure among many, not a centerpiece
- `multilaw_*.png` — full suite of multi-law fitting visualisations (per-axis slice fits, AICc weight bars, bootstrap stability, restricted-vs-full law comparison, error-floor heatmaps, per-slice fits across all configurations)
- `fig_*_per_task.png` — per-task faceted versions of the above

**What is missing for the new framing:**

- 2D response-surface heatmaps over (N, R) at fixed task / model / capacity. **None exist.** Code currently emits 1D slices only.
- 2D heatmaps over (R, D) at fixed N. None.
- Marginal value-of-resolution plots (improvement from R_k → R_{k+1} as a function of N, conditioned on model family). None.
- Regime maps (per-cell labels: data-limited / capacity-limited / resolution-sensitive / saturated / unstable). None.
- Per-cell divergence-rate heatmaps. None.
- Per-cell seed-variance heatmaps. None.
- Cross-resolution robustness summaries (mean degradation away from diagonal, fragility = worst-case degradation). None.
- Asymmetry plots (train-low / test-high vs train-high / test-low). None.

The transfer matrix figure that does exist (`fig5_transfer_heatmap`) is generated by `scripts/plot_figures.py` and treats transfer as a single supplementary figure. In the new framing it becomes one of the centerpieces.

---

## 7. Paper draft state

**Files:**

| File | Size | Title | State |
|---|---:|---|---|
| `paper/main_v1.tex` | 38 KB | early law-fitting draft | snapshot |
| `paper/main_v2.tex` | 56 KB | "Empirical Scaling Laws for Neural Operator Learning…" | snapshot |
| `paper/main_v3.tex` | 75 KB | same title as v2 | snapshot |
| `paper/main.tex` | 92 KB | **"Beyond Power Laws: Scaling-Law Diagnostics for Operator Learning"** | **current working draft** |

**Diagnosis of `paper/main.tex` (this is the file that matters):**

The title has moved partway toward the diagnostic framing. The body has not. The abstract still describes the methodological contribution as "fit six candidate functional forms per scaling axis and use the corrected Akaike Information Criterion (AICc) to let the data select the governing law" and concludes that "operator-learning scaling should be treated as a model-selection problem over law families rather than an exponent-estimation exercise." The contributions list (lines 125–160) reads:

1. "We introduce **multi-law model selection for scaling analysis**… The *identity* of the winning law becomes an interpretable diagnostic"
2. Run accounting
3. "Data scaling on the densified grid consistently selects laws with a finite error floor"
4. "Resolution scaling is non-universal" (this one survives the reframe — it is exactly a regime claim)
5. Estimated error floors $E_\infty$ from best per-axis fits

**Items 1, 3, and 5 are exactly the framings the user is rejecting.** Item 4 ("resolution scaling is non-universal") is on-frame and is in fact the seed of the new paper's central message. The Methods section (`§Multi-Law Scaling Framework`, line 215+) is structured around the six candidate forms and AICc — it will need to be rewritten as a §Experimental Framework section centered on response surfaces, seed protocol, and convergence accounting.

The good news: most of the run-accounting, task descriptions, model descriptions, and reviewer-fix corrections in `main.tex` are reusable infrastructure for the rewrite. The bad news: the entire §Multi-Law Scaling Framework / §Results-by-axis structure is in the wrong shape and the contributions list needs to be re-grounded.

The reviewer fixes (seed lists, divergence count, held-out tally, table consistency, optimizer description) are already incorporated into `main.tex` and should be carried forward into the rewrite — no need to re-litigate them.

---

## 8. External review summary (`paper-reviews/review-2026-04-07-001.tex`)

The review is dated 2026-04-07 and is **methodologically friendly** — the reviewer found no math, sign-convention, or notation-drift errors. All complaints are manuscript-internal:

1. **Experimental accounting was internally inconsistent** across at least 5 places: seed lists conflated 2 vs 3 of each kind; table totals didn't match prose; "zero divergence" contradicted `runs_aggregate.csv` (582 actual divergences); held-out contest tally said 27 in prose but 33 in table; SGD ablation success count was off (471 vs 657). The reviewer audited from data files directly. **All fixes are in `main.tex`.**
2. **The fitting-method description was technically incompatible with the actual SciPy call.** The manuscript said Levenberg–Marquardt; the code passes finite bounds to `scipy.optimize.curve_fit`, which switches to trust-region reflective. **Fixed in `main.tex`.**
3. **Resolution-axis winner for Diffusion–FNO was inconsistent across three tables** (linear / logarithmic / power). The reviewer audited against `results/multilaw_fits.json` and found the correct answer is logarithmic in the full fit, power in the restricted fit. **Fixed in `main.tex`.**
4. **The "all 9 task–model combinations on the densified grid" claim was overstated** — Diffusion MLP is on the sparser n=5 grid. **Fixed in `main.tex`.**
5. **BIC was framed as a robustness check** but the appendix admits BIC's penalty is weaker than AICc's at small n. **Reframed in `main.tex` as a "supplementary sensitivity check".**

The reviewer's overall posture matters for the reframe: they explicitly endorsed the *diagnostic* interpretation over the *predictive* one. They wrote: *"the value lies in identifying when resolution investment helps or hurts"* — this is essentially the user's pivot, and the reviewer would (presumably) be sympathetic to it.

**One reviewer observation that the reframe should especially lean on:** the held-out contest's headline number is `\holdoutWinFraction = 15/33 (45\%)` and `\holdoutMeanImprovement ≈ 1.2\%`. This is the paper's own admission, in its own macros, that AICc-based law selection beats a power-law-only baseline by approximately noise. The reframe should foreground this as evidence that *law identity is not the right level of abstraction*, and pivot the contribution to response-surface mapping.

---

## 9. Readiness for the new framing — yes / partial / no

| Capability the new framing requires | Status | Notes |
|---|---|---|
| Per-run convergence / divergence / NaN flags | ✅ **Yes** | In every diagonal `metrics.json`. Already aggregated by `aggregate_runs.py` (`divergence_rate`). |
| Per-run val/test errors and seed labels | ✅ **Yes** | Standard fields. Per-run records preserved by `aggregate_runs.py`. |
| Cross-resolution evaluation data on disk | ✅ **Yes** | 13,440 records with `eval_resolution=K` directories. **No reruns needed for transfer-first analysis.** |
| Per-cell variability (median, IQR, seed CV) | ⚠️ **Partial** | Per-run data exists; current grouping computes mean/std/stderr only. New aggregation needed. |
| 2D response-surface heatmaps over (N, R) | ❌ **No** | All current plots are 1D slices. Plotting code needs to be added. |
| Marginal value-of-resolution plots | ❌ **No** | None. |
| Regime maps (data/capacity/resolution/saturated/unstable) | ❌ **No** | No classifier exists. Definitions need to be operationalized in Phase 3. |
| Transfer-matrix construction code | ⚠️ **Partial** | `analysis/cross_resolution.py` has the primitive (~58 lines). No driver script applies it across the grid. |
| Transfer robustness / fragility scores | ❌ **No** | Not computed anywhere. |
| Per-cell convergence-rate heatmaps | ❌ **No** | The data is there; the visualisation isn't. |
| Optimizer-ablation reuse for instability narrative | ⚠️ **Partial** | Ablation runs exist (657 SGD + 756 Adam); their failure structure is already known to be non-trivial. Needs framing as instability evidence, not as "robustness check." |
| Law-fit code (kept as appendix-only) | ✅ **Yes** | Already exists in `run_multilaw_analysis.py`. Reposition, do not delete. |
| Bootstrap & held-out contest as fragility evidence | ✅ **Yes** | Already exists. Repositioned: these become *evidence that law-identity claims are fragile*, not standalone results. |
| Reviewer fixes incorporated into draft | ✅ **Yes** | All applied in `main.tex`. |

---

## 10. Coverage gaps that would block the new framing

**Gaps that block analysis but not experiments** (i.e. fixable in code, not in compute):

- **G1: No 2D response-surface plotting code.** Phase 3 will need new functions in (probably) a new `scripts/plot_response_surfaces.py`. The data is there.
- **G2: No transfer-matrix driver.** Phase 4 will need a new `scripts/build_transfer_matrices.py` (or extension of `cross_resolution.py`) that iterates across all (task, model, capacity, N) slices, builds the matrix, and computes the new robustness/fragility metrics. The data is there.
- **G3: No regime classifier.** Phase 3 needs an explicit, defensible definition of regime labels (data-limited, capacity-limited, resolution-sensitive, saturated, unstable) and a function that assigns them per cell from local sensitivities. *This is the most subjective new piece of code* and the place where I most need user input on definitions.
- **G4: No per-cell convergence visualizations.** Phase 5 needs a divergence-rate heatmap and a seed-variance heatmap per cell. The data is there; plotting code is not.
- **G5: New aggregation statistics.** `aggregate_runs.py` needs a sibling or replacement that emits median, IQR, convergence_rate, instability_rate, divergence_rate per cell. Per-run preservation already works.

**Gaps that may require new experiments (to be confirmed in Phase 2):**

- **G6: Darcy resolution range.** Darcy was only swept over R ∈ {32, 64, 128, 256}, half of Burgers's R range. This limits any cross-resolution claim about Darcy and probably means Darcy will end up as a secondary case in the rewrite, not a co-equal third task. Re-running Darcy at higher R is feasible but expensive — needs cost/benefit decision.
- **G7: Diffusion data axis density.** Diffusion uses only 5 N values; Burgers/Darcy use 11. This grid asymmetry is reviewer-flagged and was already a source of internal-inconsistency complaints. Densifying Diffusion would be straightforward but adds runtime.
- **G8: Per-task seed parity.** The paper currently uses 2 data seeds globally and 2–3 train seeds depending on task. The new framing's emphasis on seed variability would benefit from at least 3 data seeds everywhere. This is a real rerun cost — to be costed in Phase 2 if pursued.
- **G9: Cross-res seed coverage.** I have not yet checked whether every (R_train, R_eval) cell has all 3 train seeds; some may have only 1–2. Phase 2 should enumerate this.

**Not a gap, but a watch-out:**

- **W1: Cross-res records lack convergence flags.** When a cross-res evaluation row gets joined into a regime map, its parent diagonal training record's `diverged` / `nan_detected` status must be carried over via path matching. This is a small join logic issue, not a data issue.
- **W2: Capacity slots are named, not strictly numerical.** "medium FNO" and "medium MLP" do not have the same parameter count. Cross-architecture comparisons at the same capacity name are confounded. The `mlp_controlled` family was added to mitigate this for MLP specifically; it deserves more prominence in the rewrite.

---

## 11. Risks, ambiguities, and decisions needed from user before Phase 1

These are the questions I would like answered before I draft PROTOCOL_V2.md, because the answers shape its scope.

1. **How aggressive is the "no law fits in main text" rule?** Two readings of the user's directive:
   - *(a)* Law fits stay in an appendix as a compact descriptive supplement, but the main text is silent about candidate-law families and never names a "winner."
   - *(b)* Law fits stay in an appendix, but the main text *may* still use phrases like "log-like behavior" or "power-like trend" as descriptive shorthand for response-surface regions (the user's prompt explicitly allows this).
   I plan to assume (b) unless told otherwise. Confirm?

2. **Regime taxonomy.** I will need to commit, in PROTOCOL_V2, to operational definitions of the regime labels. My current sketch is roughly:
   - *data-limited:* local ∂E/∂N is large in magnitude relative to ∂E/∂N_max in this slice, AND local ∂E/∂R and ∂E/∂C are small.
   - *capacity-limited:* analogous on the C axis.
   - *resolution-sensitive:* analogous on the R axis.
   - *saturated:* all three local sensitivities are small in absolute terms.
   - *unstable:* divergence_rate or seed CV exceeds a threshold (TBD: maybe 10% divergence, or seed CV > 0.5).
   These are first-pass and will get refined. Are you comfortable with sensitivity-based definitions, or do you want a different basis (e.g. local fitted-slope, model-selection IC differences, or something explicitly tied to fitted exponents)?

3. **Darcy's role.** Given that Darcy's R range is half of Burgers's, do we (a) accept Darcy as a "secondary case" in the rewrite and not draw cross-resolution conclusions about it, (b) re-run Darcy at higher R (cost TBD in Phase 2), or (c) drop Darcy from the main text and move it to an appendix? Default if unspecified: (a).

4. **Diffusion's data axis.** Densify (5 → ~11 N values) or keep sparse and write the appropriate caveats? Default if unspecified: keep sparse, write the caveats, and note in Limitations.

5. **Seed parity (G8).** Do we re-run to bring data_seeds up to 3 across the board, or do we work with 2 data seeds and report the seed variability honestly with its actual sample size? Default: work with what we have; reruns are optional.

6. **Optimizer ablations.** The 657 SGD-cosine + 756 Adam-2x ablation runs were originally a "robustness check" for the law fits. In the new framing they fit naturally as evidence of optimization-limited regimes (Phase 5). I plan to reposition them. Confirm?

7. **Where to put the deliverables.** I plan to put `AUDIT.md`, `PROTOCOL_V2.md`, and the `RESULTS_*.md` notes at the repo root next to `ROADMAP.md` and `INDEPENDENT_ANALYSIS_REPORT.md`. Alternatively they could live in a `paper2/` or `notes/` directory. Default: repo root. Confirm?

8. **Paper draft target.** Do you want the rewrite delivered as a wholesale replacement of `paper/main.tex`, or as a new file `paper/main_v4.tex` (matching the v1/v2/v3 snapshot pattern) that becomes the new working draft? Default: new file `paper/main_v4.tex`, leave `paper/main.tex` unchanged until the v4 is ready, then swap them in a single commit.

---

## 12. What I am ready to do next, contingent on your answers

Once questions 1–8 are answered (or defaults accepted), I would proceed with:

- **Phase 1 — PROTOCOL_V2.md.** A self-contained protocol document defining axes, metadata schema, primary outcomes (median, IQR, convergence rate, instability rate, cross-res degradation), secondary outcomes, and the regime taxonomy. After this is in place, the paper can be written without needing to revert to law-first thinking.

- **Phase 2 — VALIDATION_REPORT.md.** A cell-by-cell coverage map for the main grid: which (task, model, capacity, N, R) cells have how many seeds, how many converged, how many diverged. Identify cells that should be primary, cells that are appendix-only, and cells that are pilot-only. New aggregation script if needed.

- **Phase 3 — Response-surface analysis.** New plotting code for (N, R) and (R, D) heatmaps; marginal value-of-resolution plots; regime maps. RESULTS_RESPONSE_SURFACES.md interprets each figure.

- **Phase 4 — Cross-resolution transfer.** New driver around `analysis/cross_resolution.py`; transfer-matrix figures per (task, model); robustness/fragility tables; RESULTS_TRANSFER.md.

- **Phase 5 — Convergence and instability.** Per-cell convergence-rate and instability-rate heatmaps from the existing flags; integration with optimizer ablations as an optimization-limited-regime story; RESULTS_CONVERGENCE.md.

- **Phase 6 (optional) — APPENDIX_LAW_FITS.md.** Reposition existing multi-law fitting and bootstrap as an appendix supplement, foregrounding the held-out-contest's "+1.2%" result as the honest fragility evidence.

- **Phase 7 — Paper rewrite.** New `paper/main_v4.tex` with the §Experimental Framework / §Response-Surface Results / §Cross-Resolution Transfer / §Convergence and Instability / §Practical Design Rules structure.

- **Phase 8 — CLAIM_AUDIT.md.** Per-claim evidence audit, removing universal-law language, confirming each surviving claim is grounded in a specific figure, table, or test.

I will pause here and not start Phase 1 until you have weighed in on the questions in §11.
