#!/usr/bin/env bash
# Tier 3 Master Orchestration Script
#
# Runs ALL Tier 3 experiments on a single machine (e.g., a GPU pod).
# For distributed execution across 3 pods, see ROADMAP_EXECUTION_PODS.md.
#
# Usage:
#   bash scripts/run_tier3_all.sh [--pod p1|p2|p3|all] [--dry-run]
#
# Pod assignment:
#   p1: Tier 3.1 (data-axis densification, all 3 tasks)
#   p2: Tier 3.3 (widened resolution grid) + Tier 3.4 (frequency-filtered diffusion)
#   p3: Tier 3.2 (MLP controlled) + Tier 3.5 (DeepONet ablation) + Tier 3.6 (optimizer sensitivity)
#   all: run everything sequentially (for a single powerful machine)
set -euo pipefail

POD="${1:---pod}"
POD="${2:-all}"
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pod) POD="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) shift ;;
  esac
done

run_cmd() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "▶ $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute: $2"
  else
    eval "$2" 2>&1 | tee "logs/tier3_$(echo "$1" | tr ' ' '_').log"
  fi
  echo ""
}

mkdir -p logs

# ═══════════════════════════════════════════════════════════════════════
# POD 1: Tier 3.1 — Data-axis densification (all 3 tasks)
# Adds N ∈ {150, 300, 750, 3000} → 11 data points per slice (up from 7)
# ═══════════════════════════════════════════════════════════════════════
if [[ "$POD" == "p1" || "$POD" == "all" ]]; then

  run_cmd "Tier 3.1 — Dense N: Burgers" \
    "python scripts/run_sweep.py \
      --config configs/burgers_operator_dense_N.yaml \
      --models mlp_baseline,deeponet,fno \
      --device cuda -j 1"

  run_cmd "Tier 3.1 — Dense N: Darcy" \
    "python scripts/run_sweep.py \
      --config configs/darcy_dense_N.yaml \
      --models mlp_baseline,deeponet,fno \
      --device cuda -j 1"

  run_cmd "Tier 3.1 — Dense N: Diffusion" \
    "python scripts/run_sweep.py \
      --config configs/diffusion_dense_N.yaml \
      --models mlp_baseline,deeponet,fno \
      --device cuda -j 1"

fi

# ═══════════════════════════════════════════════════════════════════════
# POD 2: Tier 3.3 — Widened resolution grid + Tier 3.4 — Filtered diffusion
# ═══════════════════════════════════════════════════════════════════════
if [[ "$POD" == "p2" || "$POD" == "all" ]]; then

  run_cmd "Tier 3.3 — Wide R: Burgers" \
    "python scripts/run_sweep.py \
      --config configs/burgers_operator_wide_R.yaml \
      --models mlp_baseline,deeponet,fno \
      --device cuda -j 1"

  run_cmd "Tier 3.3 — Wide R: Darcy" \
    "python scripts/run_sweep.py \
      --config configs/darcy_wide_R.yaml \
      --models mlp_baseline,deeponet,fno \
      --device cuda -j 1"

  run_cmd "Tier 3.3 — Wide R: Diffusion" \
    "python scripts/run_sweep.py \
      --config configs/diffusion_wide_R.yaml \
      --models mlp_baseline,deeponet,fno \
      --device cuda -j 1"

  run_cmd "Tier 3.4 — Frequency-filtered Diffusion" \
    "python scripts/run_sweep.py \
      --config configs/diffusion_filtered.yaml \
      --task diffusion_filtered \
      --models mlp_baseline,deeponet,fno \
      --device cuda -j 1"

fi

# ═══════════════════════════════════════════════════════════════════════
# POD 3: Tier 3.2 + 3.5 + 3.6 — MLP control + DeepONet ablation + optimizer
# ═══════════════════════════════════════════════════════════════════════
if [[ "$POD" == "p3" || "$POD" == "all" ]]; then

  # Tier 3.2: MLP controlled-parameter experiment on Diffusion
  run_cmd "Tier 3.2 — MLP controlled (Diffusion)" \
    "python scripts/run_sweep.py \
      --config configs/diffusion.yaml \
      --models mlp_controlled \
      --capacities tiny,small,small-med,medium,med-large,large,xlarge \
      --dataset-sizes 500,1000,2000,5000 \
      --resolutions 32,64,128,256 \
      --device cuda -j 1"

  # Also run on Burgers and Darcy for complete comparison
  run_cmd "Tier 3.2 — MLP controlled (Burgers)" \
    "python scripts/run_sweep.py \
      --config configs/burgers_operator.yaml \
      --models mlp_controlled \
      --capacities tiny,small,small-med,medium,med-large,large,xlarge \
      --dataset-sizes 500,1000,2000,5000 \
      --resolutions 32,64,128,256 \
      --device cuda -j 1"

  # Tier 3.5: DeepONet interpolation ablation
  run_cmd "Tier 3.5 — DeepONet interpolation ablation" \
    "python scripts/run_deeponet_ablation.py \
      --runs-root runs \
      --tasks burgers_operator,darcy,diffusion \
      --capacities medium,large,xlarge \
      --device cuda"

  # Tier 3.6: Optimizer sensitivity (Burgers, all models, medium cap)
  run_cmd "Tier 3.6 — Optimizer sensitivity (Burgers)" \
    "python scripts/run_optimizer_ablation.py \
      --config configs/burgers_operator.yaml \
      --models mlp_baseline,deeponet,fno \
      --capacities medium \
      --device cuda -j 1"

  # Tier 3.6: Also run on Diffusion for the most interesting case
  run_cmd "Tier 3.6 — Optimizer sensitivity (Diffusion)" \
    "python scripts/run_optimizer_ablation.py \
      --config configs/diffusion.yaml \
      --models mlp_baseline,deeponet,fno \
      --capacities medium \
      --device cuda -j 1"

fi

# ═══════════════════════════════════════════════════════════════════════
# Post-processing (run after all pods complete and results are merged)
# ═══════════════════════════════════════════════════════════════════════
if [[ "$POD" == "post" || "$POD" == "all" ]]; then

  run_cmd "Aggregate all runs" \
    "python scripts/aggregate_runs.py --runs-root runs"

  run_cmd "Re-run multilaw analysis with extended data" \
    "python scripts/run_multilaw_analysis.py"

  run_cmd "Re-run Tier 2 analyses" \
    "python scripts/run_holdout_contest.py && \
     python scripts/run_within_slice_bootstrap.py"

  run_cmd "Regenerate figures" \
    "python scripts/plot_figures.py \
      --grouped-csv runs/grouped_metrics.csv \
      --fits-json results/scaling_fits.json \
      --out-dir figures && \
     python scripts/plot_multilaw_figures.py"

fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Tier 3 execution complete for pod: $POD"
echo "  Logs saved to: logs/"
echo "════════════════════════════════════════════════════════════════"
