#!/usr/bin/env bash
# Run the full post-experiment pipeline: aggregate → analysis → figures
# Usage: bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 1: Aggregate runs ==="
python scripts/aggregate_runs.py --runs-root runs

echo ""
echo "=== Step 2: Scaling analysis ==="
python scripts/run_analysis.py \
  --grouped-csv runs/grouped_metrics.csv \
  --out-dir results

echo ""
echo "=== Step 3: Generate figures ==="
python scripts/plot_figures.py \
  --grouped-csv runs/grouped_metrics.csv \
  --fits-json results/scaling_fits.json \
  --out-dir figures

echo ""
echo "=== Done! ==="
echo "Results in: results/"
echo "Figures in: figures/"
