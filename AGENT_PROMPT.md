# Agent Prompt: Finish the Scaling-Operator-Learning Roadmap

You are an expert ML research engineer. Your job is to **finish the entire roadmap** for the paper "Scaling Laws for Operator Learning" by executing all remaining experiments on 3 GPU pods, analyzing results, generating figures, populating paper tables, and updating the paper to submission quality.

The codebase is at `/root/scaling-operator-learning` on each pod and `~/Documents/projects/scaling-operator-learning` on the local workstation. A Python venv is at `.venv/`. All code, configs, models, tasks, and scripts are already implemented and tested (16/16 pytest pass). Your job is **execution, analysis, and paper writing** — not code development.

---

## Phase 0: Pod Setup & Smoke Test

**Do this first on ALL THREE pods in parallel.**

### 0.1 Connect to each pod

```bash
ssh -p 26317 root@213.173.108.102   # p1 (classic_peach_piranha)
ssh -p 30412 root@213.173.108.102   # p2 (mid_emerald_goldfish)
ssh -p 21099 root@157.157.221.29    # p3 (musical_red_coral)
```

If ports have changed, check RunPod dashboard and update.

### 0.2 Bootstrap each pod

```bash
cd /root
if [ -d scaling-operator-learning ]; then
  cd scaling-operator-learning && git pull
else
  git clone <YOUR_REPO_URL> && cd scaling-operator-learning
fi
python -m venv .venv
. .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests/ -v  # must pass 16/16
```

### 0.3 Smoke test (ONE short run per pod to verify GPU training works)

```bash
python scripts/run_sweep.py \
  --config configs/burgers_operator.yaml \
  --models mlp_baseline --capacities tiny \
  --dataset-sizes 50 --resolutions 32 \
  --data-seeds 11 --train-seeds 101 \
  --device cuda -j 1
```

Confirm it writes `runs/task=burgers_operator/model=mlp_baseline/capacity=tiny/N=50/R=32/dseed=11_tseed=101/metrics.json` and completes without error. Delete the smoke-test output if it conflicts with real runs.

---

## Phase 1: Run ALL Tier 3 Experiments (Parallel Across 3 Pods)

Launch all experiments using the master orchestration script inside tmux sessions so they survive SSH disconnects.

### Pod 1 — Tier 3.1: Densified Data Axis

```bash
tmux new -s tier3
. .venv/bin/activate
bash scripts/run_tier3_all.sh --pod p1
```

This runs `run_sweep.py` for **Burgers, Darcy, Diffusion** with the expanded N grid:
`N = [50, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000, 5000]` (11 values, up from 7).
Uses configs: `configs/burgers_operator_dense_N.yaml`, `configs/darcy_dense_N.yaml`, `configs/diffusion_dense_N.yaml`.
Models: `mlp_baseline, deeponet, fno`. Seeds: `data=[11,22,33], train=[101,202,303]`.

**Expected runs:** 3 tasks × 3 models × 7 capacities × 11 N × 4 R × 9 seeds = **~83,160 runs** (but many already exist from original grid — the sweep skips existing `metrics.json` unless `--overwrite`). Net new runs are for the 4 added N values: `[150, 300, 750, 3000]`, so ~3 × 3 × 7 × 4 × 4 × 9 = **9,072 new runs**.

### Pod 2 — Tier 3.3: Widened Resolution + Tier 3.4: Filtered Diffusion

```bash
tmux new -s tier3
. .venv/bin/activate
bash scripts/run_tier3_all.sh --pod p2
```

This runs:
1. **Wide-R sweeps** for Burgers, Darcy, Diffusion with `R = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]` (10 values, up from 4). Configs: `configs/*_wide_R.yaml`.
2. **Frequency-filtered Diffusion** with `diffusion_filtered` task, `R = [32, 64, 128, 256, 512]`, `N = [500, 1000, 2000, 5000]`. Config: `configs/diffusion_filtered.yaml`.

### Pod 3 — Tier 3.2: MLP Controlled + Tier 3.5: DeepONet Ablation + Tier 3.6: Optimizer

```bash
tmux new -s tier3
. .venv/bin/activate
bash scripts/run_tier3_all.sh --pod p3
```

This runs:
1. **MLP controlled-parameter** model on Diffusion and Burgers (`mlp_controlled` model, all 7 capacities, N=[500..5000], R=[32..256]).
2. **DeepONet interpolation ablation** comparing interpolation vs subsampling on saved DeepONet checkpoints.
3. **Optimizer sensitivity** ablation (Adam-2×-patience + SGD-cosine) on Burgers and Diffusion at medium capacity.

### Monitoring

Check progress from local workstation:

```bash
bash scripts/monitor_pods.sh
# Or continuous:
watch -n 60 bash scripts/monitor_pods.sh
```

Check pod tmux sessions:
```bash
ssh -p 26317 root@213.173.108.102 "tmux attach -t tier3"
```

Count completed runs on a pod:
```bash
find runs -name metrics.json | wc -l
```

**Wait for ALL pods to complete before proceeding to Phase 2.**

---

## Phase 2: Pull Results & Merge

From the **local workstation**:

```bash
cd ~/Documents/projects/scaling-operator-learning

# Pull from each pod into staging directories
rsync -avz -e "ssh -p 26317" root@213.173.108.102:/root/scaling-operator-learning/runs/ runs_p1/
rsync -avz -e "ssh -p 30412" root@213.173.108.102:/root/scaling-operator-learning/runs/ runs_p2/
rsync -avz -e "ssh -p 21099" root@157.157.221.29:/root/scaling-operator-learning/runs/ runs_p3/

# Also pull ablation outputs from p3
rsync -avz -e "ssh -p 21099" root@157.157.221.29:/root/scaling-operator-learning/runs_ablation_adam_2x_patience/ runs_ablation_adam_2x_patience/
rsync -avz -e "ssh -p 21099" root@157.157.221.29:/root/scaling-operator-learning/runs_ablation_sgd_cosine/ runs_ablation_sgd_cosine/

# Pull DeepONet ablation results
rsync -avz -e "ssh -p 21099" root@157.157.221.29:/root/scaling-operator-learning/results/deeponet_interpolation_ablation.json results/

# Merge main runs into the canonical runs/ directory
rsync -av runs_p1/ runs/
rsync -av runs_p2/ runs/
rsync -av runs_p3/ runs/

# Verify merged run count (should be significantly more than the original ~5,040)
find runs -name metrics.json | wc -l
```

---

## Phase 3: Aggregate & Analyze

### 3.1 Aggregate all runs into CSVs

```bash
. .venv/bin/activate
python scripts/aggregate_runs.py --runs-root runs
```

**Outputs:**
- `runs/runs_aggregate.csv` — one row per run
- `runs/grouped_metrics.csv` — mean/std/stderr per (task, model, capacity, N, R)

Verify the CSV includes the new N values (150, 300, 750, 3000), new R values (16, 48, 96, 192, 384, 512), `mlp_controlled` model, and `diffusion_filtered` task.

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/grouped_metrics.csv')
print('Tasks:', sorted(df['task'].unique()))
print('Models:', sorted(df['model'].unique()))
print('N values:', sorted(df['dataset_size'].unique()))
print('R values:', sorted(df['resolution'].unique()))
print('Total rows:', len(df))
"
```

**Expected:** Tasks should include `diffusion_filtered`. Models should include `mlp_controlled`. N should include 150, 300, 750, 3000. R should include 16, 48, 96, etc.

**Note:** `grouped_metrics_train.csv` may need to exist for analysis scripts. If `aggregate_runs.py` doesn't produce it, check whether it's a filtered version (train metrics only). If it doesn't exist, the analysis scripts read `grouped_metrics.csv` and filter internally — verify by checking what `run_multilaw_analysis.py` reads. If needed:
```bash
cp runs/grouped_metrics.csv runs/grouped_metrics_train.csv
```

### 3.2 Also aggregate ablation runs

```bash
python scripts/aggregate_runs.py --runs-root runs_ablation_adam_2x_patience --out-dir runs_ablation_adam_2x_patience
python scripts/aggregate_runs.py --runs-root runs_ablation_sgd_cosine --out-dir runs_ablation_sgd_cosine
```

### 3.3 Run multi-law analysis (main analysis)

```bash
python scripts/run_multilaw_analysis.py
```

**Reads:** `runs/grouped_metrics_train.csv`
**Writes:** `results/multilaw_fits.json`, `results/multilaw_fits_restricted.json`

This fits all 6 candidate laws (power, exponential, logarithmic, stretched_exp, saturation, linear) to every slice of every (task, model, axis) combination, performs AICc model selection, computes Akaike weights, BIC concordance, and bootstrap stability (B=500).

**Key things to check in the output:**
1. With the denser N grid (11 points per data-axis slice instead of 7), does "logarithmic wins" still hold? Or does power-law become competitive?
2. What is BIC concordance now? At n=11, the small-n penalty inversion is much less severe.
3. Are bootstrap stability fractions higher with more data points?

### 3.4 Run scaling fits (power-law-only analysis)

```bash
python scripts/run_analysis.py --grouped-csv runs/grouped_metrics.csv --out-dir results --n-boot 1000
# OR use the fast version:
python scripts/run_fast_analysis.py
```

**Writes:** `results/scaling_fits.json`

### 3.5 Run held-out prediction contest (Tier 2.1)

```bash
python scripts/run_holdout_contest.py
```

**Writes:** `results/holdout_prediction_contest.json`

Check: Does the multi-law framework predict held-out slices better than power-law-only? Report win rates and mean RMSE ratios.

### 3.6 Run within-slice bootstrap (Tier 2.2)

```bash
python scripts/run_within_slice_bootstrap.py
```

**Writes:** `results/within_slice_bootstrap.json`

Check: What fraction of bootstrap resamples agree with the original AICc winner? Low values (<0.5) indicate the law selection is genuinely uncertain at that slice.

### 3.7 Analyze optimizer ablation results

Compare law-selection outcomes between:
1. Original runs (Adam, standard patience)
2. Adam with 2× patience (`runs_ablation_adam_2x_patience/`)
3. SGD + cosine (`runs_ablation_sgd_cosine/`)

Run multi-law analysis on each ablation set:
```bash
# You may need to temporarily point the analysis script at the ablation CSVs.
# One approach: copy the grouped CSV and run analysis separately.
python -c "
import pandas as pd
for variant in ['adam_2x_patience', 'sgd_cosine']:
    df = pd.read_csv(f'runs_ablation_{variant}/grouped_metrics.csv')
    print(f'\n=== {variant} ===')
    print('Tasks:', sorted(df['task'].unique()))
    print('Models:', sorted(df['model'].unique()))
    print('Rows:', len(df))
"
```

For a proper comparison, modify `run_multilaw_analysis.py` (or write a small wrapper) to accept `--input-csv` and `--output-prefix` arguments, then run:
```bash
python scripts/run_multilaw_analysis.py  # original
# Then manually compare law winners across the three conditions
```

The key question: **Does the AICc-winning law change when you switch optimizers?** If yes → law selection is sensitive to training details (a real concern). If no → law identity is robust to optimizer choice (strong result).

### 3.8 Analyze DeepONet interpolation ablation

The results are in `results/deeponet_interpolation_ablation.json`. Check:
- Does interpolation vs subsampling change cross-resolution transfer conclusions?
- If subsampling gets similar or better results → interpolation is not a confounder.
- If results diverge → the cross-resolution story for DeepONet needs caveats.

### 3.9 Analyze MLP controlled-parameter results

Compare `mlp_controlled` vs `mlp_baseline` on Diffusion (where "resolution hurts" is strongest):
```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/grouped_metrics.csv')
diff = df[df['task'] == 'diffusion']
for model in ['mlp_baseline', 'mlp_controlled']:
    sub = diff[diff['model'] == model]
    if len(sub) > 0:
        print(f'\n{model}:')
        for R in sorted(sub['resolution'].unique()):
            r_sub = sub[sub['resolution'] == R]
            print(f'  R={R}: mean_rel_l2 = {r_sub[\"test_rel_l2_mean\"].mean():.4f}')
"
```

The key question: **Does fixing parameter count eliminate the "resolution hurts" effect?**
- If mlp_controlled at R=256 performs similarly to R=32 → the confound (D ∝ R) was real.
- If mlp_controlled still degrades at high R → genuine resolution effect.

### 3.10 Analyze frequency-filtered diffusion

Compare `diffusion_filtered` vs `diffusion` at high R:
```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/grouped_metrics.csv')
for task in ['diffusion', 'diffusion_filtered']:
    sub = df[df['task'] == task]
    if len(sub) > 0:
        print(f'\n{task}:')
        for R in sorted(sub['resolution'].unique()):
            r_sub = sub[sub['resolution'] == R]
            print(f'  R={R}: mean_rel_l2 = {r_sub[\"test_rel_l2_mean\"].mean():.4f}')
"
```

Key question: **Does low-pass filtering high-R inputs fix the degradation?**
- If yes → extraneous high-frequency content is the mechanism behind "resolution hurts" for diffusion.
- If no → the effect is more fundamental (e.g., curse of dimensionality).

---

## Phase 4: Generate All Figures

```bash
. .venv/bin/activate

# Core figures (power-law fits)
python scripts/plot_figures.py --grouped-csv runs/grouped_metrics.csv --fits-json results/scaling_fits.json --out-dir figures

# Per-task figures
python scripts/plot_figures_pertask.py

# Multi-law figures (the main contribution)
python scripts/plot_multilaw_figures.py
```

**Expected outputs in `figures/`:**
- `fig1_data_scaling.pdf` — E vs N
- `fig2_resolution_scaling.pdf` — E vs R
- `fig3_capacity_scaling.pdf` — E vs D
- `fig4_exponent_comparison.pdf` — α, β, γ bars
- `fig5_transfer_heatmap.pdf` — cross-res transfer
- `fig_data_scaling_per_task.pdf`, `fig_resolution_scaling_per_task.pdf`, `fig_capacity_scaling_per_task.pdf`
- `fig_3d_exponents.pdf`
- `fig_cross_res_per_task.pdf`
- `multilaw_data_scaling.pdf`, `multilaw_resolution_scaling.pdf`, `multilaw_capacity_scaling.pdf`
- `multilaw_heatmap.pdf`, `multilaw_e_inf.pdf`, `multilaw_akaike_weights.pdf`
- `multilaw_bootstrap_stability.pdf`, `multilaw_restricted_comparison.pdf`

Visually inspect all figures. Check that multi-law overlay curves match the reported AICc winners. Verify the denser grids produce smoother scaling curves. Flag any anomalies (e.g., divergent runs, strange non-monotonic behavior at extreme R or N values).

---

## Phase 5: Update the Paper

Now populate all placeholder sections in `paper/main.tex` with real results. The paper structure is:

### 5.1 Update Section 5.8 — Held-Out Prediction Contest

Read `results/holdout_prediction_contest.json`. Replace the `[Placeholder]` table in Section 5.8 (`\label{tab:holdout}`) with actual results:

| Task | Model | Axis | Multi-law RMSE | Power-law RMSE | Winner |
|------|-------|------|---------------|----------------|--------|
| ... actual data ... |

Write interpretation: if multi-law wins on most axes → "the multi-law framework provides genuine predictive advantage, not just descriptive flexibility." If power-law wins or ties → "power-law extrapolation is competitive; the multi-law framework's added complexity is justified primarily by interpretive rather than predictive value."

### 5.2 Update Section 5.9 — Within-Slice Bootstrap

Read `results/within_slice_bootstrap.json`. Replace the `[Placeholder]` table with:

| Task | Model | Axis | Original Winner | Bootstrap Stability | Top-3 Laws (win%) |
|------|-------|------|----------------|--------------------|--------------------|
| ... actual data ... |

Key number: mean bootstrap stability across all slices. If >0.7 → "law selection is reasonably stable at the slice level." If <0.5 → "substantial within-slice uncertainty; the reported AICc winners should be interpreted cautiously."

### 5.3 Update Appendix B.1 — Densified Data Axis

Report how law selection changes with 11 N-values vs the original 7. Key comparisons:
- Did the logarithmic-dominant finding survive on the data axis?
- Is BIC concordance higher at n=11?
- Are Akaike weights more concentrated (less ambiguity)?

Include a comparison table: `(task, model) → winner@n=7 vs winner@n=11`. Highlight any flips.

### 5.4 Update Appendix B.2 — MLP Fixed-Parameter-Count

Report mlp_controlled results on Diffusion. Key table:

| R | mlp_baseline (D∝R) | mlp_controlled (D≈33K) | Conclusion |
|---|--------------------|-----------------------|------------|
| 32 | ... | ... | |
| 64 | ... | ... | |
| 128 | ... | ... | |
| 256 | ... | ... | |

Interpretation depends on whether the "resolution hurts" effect persists under controlled parameters.

### 5.5 Update Appendix B.3 — Widened Resolution Grid

Report whether the Burgers/Darcy/Diffusion trichotomy (helps/neutral/hurts) survives with 10 R-values. Look for:
- Regime changes (helps at low R, plateaus at high R)
- Whether the directional finding is stable or was an artifact of sparse sampling
- Updated law-selection results on the resolution axis

### 5.6 Update Appendix B.4 — Frequency-Filtered Diffusion

Report filtered vs unfiltered comparison at high R. Key finding: does low-pass filtering restore the benefit of higher resolution?

### 5.7 Update Appendix B.5 — DeepONet Interpolation Ablation

Report `results/deeponet_interpolation_ablation.json`. Compare interpolation vs subsampling cross-resolution errors.

### 5.8 Update Appendix B.6 — Optimizer Sensitivity

Report whether law winners change across Adam/Adam-2×/SGD conditions. This is a crucial robustness check.

### 5.9 Update Discussion Section (Section 6)

Revise discussion points in light of new results:
- **Section 6.1** (Law as Diagnostic): Strengthen or weaken the claim based on denser data axis results and bootstrap stability.
- **Section 6.2** (Resolution): Update with wide-R and filtered diffusion findings.
- **Section 6.3** (Architecture Recommendations): Update with DeepONet ablation results.
- **Section 6.4** (Three-Axis Power Law): Update with held-out prediction contest results.

### 5.10 Update Success Criteria

The one remaining unchecked criterion in `ROADMAP.md`:
> Every claim about a "winning law" is backed by Akaike weight > 0.5 *and* within-slice bootstrap stability, or explicitly flagged as ambiguous

Go through Section 5 and audit every law-selection claim. For each:
- If Akaike weight > 0.5 AND bootstrap stability > 0.7 → state confidently.
- If either is marginal → add "(moderately supported; see Appendix A.7 and Section 5.9)".
- If both are low → flag as "ambiguous under current sample sizes" and discuss what would resolve it.

---

## Phase 6: Final Compilation & Verification

### 6.1 Compile the paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Fix any LaTeX errors or missing references.

### 6.2 Final checklist

- [ ] All `[Placeholder]` tags removed from `main.tex`
- [ ] All tables populated with real numbers
- [ ] All figures regenerated with new data and referenced correctly
- [ ] Section 5.8 (holdout contest) has actual results and interpretation
- [ ] Section 5.9 (within-slice bootstrap) has actual results and interpretation
- [ ] All 6 Appendix B subsections populated
- [ ] Discussion (Section 6) updated to reflect new findings
- [ ] Success criteria in ROADMAP.md all checked
- [ ] No claims of "logarithmic wins" without Akaike weight + bootstrap backing
- [ ] BIC concordance discussion updated for denser grid
- [ ] `results/` directory contains all analysis JSONs
- [ ] `figures/` directory contains all regenerated PDFs

### 6.3 Update ROADMAP.md

Mark all remaining `[ ]` items as `[x]`:
- Tier 3.1: re-run law selection, re-run BIC concordance, report criterion robustness
- Tier 3.2: run on Diffusion, report confound resolution
- Tier 3.3: check trichotomy survival, look for regime changes
- Tier 3.4: compare filtered vs raw, report finding
- Tier 3.5: report cross-resolution conclusions
- Tier 3.6: check if law winners shift
- Final success criterion checkbox

---

## Key Decision Points (Requires Judgment)

At several points, results may surprise. Here's how to handle each scenario:

1. **Logarithmic no longer dominates with denser N grid:** This is actually a good result — it means law selection was unstable at n=7 and the paper should report this honestly. Frame as: "With a denser sampling grid, the data-scaling landscape becomes more nuanced, with [winning law] emerging as the primary functional form."

2. **Held-out contest shows power-law wins:** Frame honestly. The multi-law framework still has value for interpretation and diagnostics, even if OOS prediction isn't improved. The paper's contribution shifts from "multi-law predicts better" to "multi-law reveals structure that power-law fitting obscures."

3. **Within-slice bootstrap shows high instability (<0.3 stability):** Add strong caveats to all law-selection claims. This is the most methodologically honest outcome — it means with n=5-11 points, law identity is genuinely uncertain. Frame as a contribution: "We quantify the inherent uncertainty in functional-form selection at realistic sample sizes."

4. **MLP confound is confirmed (controlled model fixes resolution-hurts):** This changes the narrative for Section 6.2 significantly. The "resolution hurts" finding becomes "parameter-count confound" and the paper should report the control experiment prominently.

5. **Optimizer sensitivity flips law winners:** This would be a serious concern. Add a subsection discussing convergence as a confounder and suggest that law-selection studies should always report optimizer sensitivity. Frame constructively.

---

## File Reference

### Scripts (in execution order)
| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `scripts/run_tier3_all.sh --pod p1\|p2\|p3` | GPU training runs | Configs, source code | `runs/**/metrics.json` |
| `scripts/aggregate_runs.py` | Merge metrics into CSVs | `runs/**/metrics.json` | `runs/runs_aggregate.csv`, `runs/grouped_metrics.csv` |
| `scripts/run_multilaw_analysis.py` | Multi-law AICc fitting | `runs/grouped_metrics_train.csv` | `results/multilaw_fits.json`, `results/multilaw_fits_restricted.json` |
| `scripts/run_analysis.py` | Power-law-only fitting | `runs/grouped_metrics.csv` | `results/scaling_fits.json` |
| `scripts/run_holdout_contest.py` | Tier 2.1 OOS prediction | `runs/grouped_metrics_train.csv` | `results/holdout_prediction_contest.json` |
| `scripts/run_within_slice_bootstrap.py` | Tier 2.2 data-level bootstrap | `runs/grouped_metrics_train.csv` | `results/within_slice_bootstrap.json` |
| `scripts/plot_figures.py` | Core figures | CSVs + JSONs | `figures/*.pdf` |
| `scripts/plot_figures_pertask.py` | Per-task figures | CSVs + JSONs | `figures/*.pdf` |
| `scripts/plot_multilaw_figures.py` | Multi-law figures | CSVs + JSONs | `figures/multilaw_*.pdf` |

### Configs
| Config | N values | R values | Task |
|--------|----------|----------|------|
| `configs/burgers_operator.yaml` | [50,100,200,500,1000,2000,5000] | [32,64,128,256] | burgers_operator |
| `configs/darcy.yaml` | [50,100,200,500,1000,2000,5000] | [32,64,128,256] | darcy |
| `configs/diffusion.yaml` | [50,100,200,500,1000,2000,5000] | [32,64,128,256] | diffusion |
| `configs/burgers_operator_dense_N.yaml` | [50,100,150,200,300,500,750,1000,2000,3000,5000] | [32,64,128,256] | burgers_operator |
| `configs/darcy_dense_N.yaml` | [50,100,150,200,300,500,750,1000,2000,3000,5000] | [32,64,128,256] | darcy |
| `configs/diffusion_dense_N.yaml` | [50,100,150,200,300,500,750,1000,2000,3000,5000] | [32,64,128,256] | diffusion |
| `configs/burgers_operator_wide_R.yaml` | [50,100,200,500,1000,2000,5000] | [16,32,48,64,96,128,192,256,384,512] | burgers_operator |
| `configs/darcy_wide_R.yaml` | [50,100,200,500,1000,2000,5000] | [16,32,48,64,96,128,192,256,384,512] | darcy |
| `configs/diffusion_wide_R.yaml` | [50,100,200,500,1000,2000,5000] | [16,32,48,64,96,128,192,256,384,512] | diffusion |
| `configs/diffusion_filtered.yaml` | [500,1000,2000,5000] | [32,64,128,256,512] | diffusion_filtered |

### Models
| Model | Key Property |
|-------|-------------|
| `mlp_baseline` | R→hidden→R, param count scales with R |
| `deeponet` | Unstacked branch-trunk, cross-resolution via interpolation |
| `fno` | Spectral convolution, modes/width/layers grid |
| `mlp_controlled` | Like mlp_baseline but hidden width adjusted to hold ~33K params constant across R |

### Tasks
| Task | PDE | Solver |
|------|-----|--------|
| `burgers_operator` | Burgers equation | Pseudo-spectral RK4 |
| `darcy` | Darcy flow | Finite difference (Thomas algorithm) |
| `diffusion` | Diffusion equation | Exact Fourier |
| `diffusion_filtered` | Diffusion + low-pass FFT filter on ICs | Exact Fourier + spectral filter |

### Paper sections needing updates
| Section | Label | Content needed |
|---------|-------|---------------|
| 5.8 | `sec:holdout` | Holdout prediction contest results |
| 5.9 | `sec:within-slice-bootstrap` | Within-slice bootstrap results |
| B.1 | `app:dense-data` | Dense N law-selection comparison |
| B.2 | `app:mlp-controlled` | MLP fixed-param results |
| B.3 | `app:wide-res` | Wide R trichotomy check |
| B.4 | `app:filtered` | Filtered diffusion results |
| B.5 | `app:deeponet-ablation` | DeepONet interpolation comparison |
| B.6 | `app:optimizer` | Optimizer sensitivity results |
| 6.1–6.4 | `sec:discussion` | Updated discussion |

---

## Estimated Timeline

- Phase 0 (setup): ~15 minutes
- Phase 1 (training): This is the bottleneck. Dense-N has ~9K new runs, wide-R has ~13K new runs, other ablations are smaller. Total wall-clock depends on GPU speed. On A100: ~6-12 hours. On A6000/3090: ~12-24 hours.
- Phase 2 (pull/merge): ~10 minutes
- Phase 3 (analysis): ~30 minutes
- Phase 4 (figures): ~10 minutes
- Phase 5 (paper writing): ~2-3 hours of careful writing
- Phase 6 (compilation/verification): ~30 minutes

**Critical path:** Phase 1 (GPU training) dominates. Everything else can start as soon as results are merged.
