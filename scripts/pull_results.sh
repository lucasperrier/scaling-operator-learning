#!/usr/bin/env bash
# Pull results from all 3 pods and merge into local workspace.
# Safe to run multiple times (merges non-destructively).
# Usage: bash scripts/pull_results.sh
set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

P1_ADDR="root@213.173.108.102"  P1_PORT=26317
P2_ADDR="root@213.173.108.102"  P2_PORT=30412
P3_ADDR="root@157.157.221.29"   P3_PORT=21099

REMOTE_DIR="/root/scaling-operator-learning"
LOCAL_RUNS="runs"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

pull_metrics_from_pod() {
    local label="$1" addr="$2" port="$3"
    log "Pulling from $label ($addr:$port)..."

    # Create tarball of metrics.json files on remote (very fast, ~2MB total)
    ssh $SSH_OPTS -i "$SSH_KEY" -p "$port" "$addr" \
        "cd $REMOTE_DIR && find runs runs_ablation_* -name metrics.json 2>/dev/null | tar czf /tmp/metrics_${label}.tar.gz -T -" \
        2>/dev/null || { log "  WARNING: $label failed (pod may be offline)"; return 0; }

    # Download tarball
    scp $SSH_OPTS -i "$SSH_KEY" -P "$port" "$addr:/tmp/metrics_${label}.tar.gz" "/tmp/metrics_${label}.tar.gz" \
        2>/dev/null || { log "  WARNING: scp from $label failed"; return 0; }

    # Extract (preserves directory structure, won't overwrite newer local files)
    tar xzf "/tmp/metrics_${label}.tar.gz" --keep-newer-files 2>/dev/null || true
    local count
    count=$(tar tzf "/tmp/metrics_${label}.tar.gz" 2>/dev/null | wc -l)
    log "  $label: $count metrics files pulled"
    rm -f "/tmp/metrics_${label}.tar.gz"
}

# Also pull special results files
pull_results_from_pod() {
    local label="$1" addr="$2" port="$3"
    for fname in deeponet_interpolation_ablation.json; do
        scp $SSH_OPTS -i "$SSH_KEY" -P "$port" "$addr:$REMOTE_DIR/results/$fname" "results/${fname}" \
            2>/dev/null || true
    done
}

log "Starting pull from all pods..."

pull_metrics_from_pod "p1" "$P1_ADDR" "$P1_PORT"
pull_metrics_from_pod "p2" "$P2_ADDR" "$P2_PORT"
pull_metrics_from_pod "p3" "$P3_ADDR" "$P3_PORT"
pull_results_from_pod "p3" "$P3_ADDR" "$P3_PORT"

log "Merge complete. Local run counts:"
for t in burgers_operator darcy diffusion diffusion_filtered; do
    count=$(find runs -path "*task=$t*" -name metrics.json 2>/dev/null | wc -l)
    [ "$count" -gt 0 ] && echo "  task=$t: $count"
done

for d in runs_ablation_*; do
    [ -d "$d" ] && echo "  $d: $(find "$d" -name metrics.json | wc -l)"
done

log "Done. Next: run aggregate_runs.py and re-run analysis."
