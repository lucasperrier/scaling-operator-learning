#!/usr/bin/env bash
# Chain script for p3: wait for ablation to finish, then run wide R + dense N
# to offload work from p1 and p2.
set -euo pipefail
cd /root/scaling-operator-learning
source .venv/bin/activate

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Waiting for optimizer ablation process to finish..."
while pgrep -f "run_optimizer_ablation" > /dev/null 2>&1; do
    sleep 30
done
log "Ablation process done."

# Check if diffusion ablation still needs to run
ABL_DIRS=$(find . -maxdepth 1 -type d -name 'runs_ablation_*' 2>/dev/null)
DIFFUSION_DONE=0
for d in $ABL_DIRS; do
    COUNT=$(find "$d" -path '*diffusion*' -name metrics.json 2>/dev/null | wc -l)
    DIFFUSION_DONE=$((DIFFUSION_DONE + COUNT))
done

if [ "$DIFFUSION_DONE" -lt 100 ]; then
    log "Running diffusion optimizer ablation (Tier 3.6)..."
    python scripts/run_optimizer_ablation.py \
        --config configs/diffusion.yaml \
        --models mlp_baseline,deeponet,fno \
        --capacities medium \
        --device cuda -j 1
    log "Diffusion ablation done."
fi

log "Starting wide R sweeps (new resolutions only) to offload p2..."

log "Darcy wide R (resolutions: 16,48,96,192,384,512)..."
python scripts/run_sweep.py \
    --config configs/darcy_wide_R.yaml \
    --models mlp_baseline,deeponet,fno \
    --resolutions 16,48,96,192,384,512 \
    --device cuda -j 1

log "Diffusion wide R (resolutions: 16,48,96,192,384,512)..."
python scripts/run_sweep.py \
    --config configs/diffusion_wide_R.yaml \
    --models mlp_baseline,deeponet,fno \
    --resolutions 16,48,96,192,384,512 \
    --device cuda -j 1

log "Darcy dense N (new sizes: 150,300,750,3000)..."
python scripts/run_sweep.py \
    --config configs/darcy_dense_N.yaml \
    --models mlp_baseline,deeponet,fno \
    --dataset-sizes 150,300,750,3000 \
    --device cuda -j 1

log "Diffusion dense N (new sizes: 150,300,750,3000)..."
python scripts/run_sweep.py \
    --config configs/diffusion_dense_N.yaml \
    --models mlp_baseline,deeponet,fno \
    --dataset-sizes 150,300,750,3000 \
    --device cuda -j 1

log "All chain work complete!"
