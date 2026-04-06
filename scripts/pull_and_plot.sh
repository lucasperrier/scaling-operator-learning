#!/usr/bin/env bash
# Pull results from remote and run the local pipeline
# Usage: bash scripts/pull_and_plot.sh <remote_alias>
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${1:-runpod}"
REMOTE_DIR="/workspace/projects/scaling-operator-learning"

echo "=== Pulling runs/ from ${REMOTE} ==="
rsync -avz --progress "${REMOTE}:${REMOTE_DIR}/runs/" runs/

echo ""
echo "=== Running pipeline ==="
bash scripts/run_pipeline.sh
