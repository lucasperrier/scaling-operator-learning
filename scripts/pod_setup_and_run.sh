#!/usr/bin/env bash
# Pod setup and experiment runner
# Usage: bash pod_setup_and_run.sh <task_config> <task_name>
# Example: bash pod_setup_and_run.sh configs/burgers_operator.yaml burgers_operator
set -euo pipefail

TASK_CONFIG="${1:?Usage: $0 <config_path> <task_name>}"
TASK_NAME="${2:?Usage: $0 <config_path> <task_name>}"

echo "=== Pod Setup for task: ${TASK_NAME} ==="
echo "Started at: $(date)"

# Install system deps if needed
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 || true

# Clone repo
cd /root
if [ -d "scaling-operator-learning" ]; then
    echo "Repo already exists, pulling latest..."
    cd scaling-operator-learning && git pull
else
    echo "Cloning repo..."
    git clone https://github.com/lucasperrier/scaling-operator-learning.git
    cd scaling-operator-learning
fi

# Install package
pip install -e . > /dev/null 2>&1
echo "Package installed."

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Launching sweep: ${TASK_NAME} ==="
echo "Config: ${TASK_CONFIG}"
echo "Essential grid: N=[50,100,500,1000,5000], seeds: data=[11,22], train=[101,202]"
echo ""

# Run the essential sweep
python scripts/run_sweep.py \
    --config "${TASK_CONFIG}" \
    --task "${TASK_NAME}" \
    --models mlp_baseline,deeponet,fno \
    --dataset-sizes 50,100,500,1000,5000 \
    --data-seeds 11,22 \
    --train-seeds 101,202 \
    --device cuda \
    -j 1

echo ""
echo "=== ${TASK_NAME} sweep complete at $(date) ==="
