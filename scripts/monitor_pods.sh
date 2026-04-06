#!/usr/bin/env bash
# Monitor all RunPod workers from local machine.
# Usage: bash scripts/monitor_pods.sh
set -uo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# Use direct TCP endpoints from RunPod exposed ports (fallbacks from latest snapshot).
POD1_ADDR="${POD1_ADDR:-root@213.173.108.102}"
POD1_PORT="${POD1_PORT:-26317}"
POD1_LABEL="${POD1_LABEL:-classic_peach_piranha}"

POD2_ADDR="${POD2_ADDR:-root@213.173.108.102}"
POD2_PORT="${POD2_PORT:-30412}"
POD2_LABEL="${POD2_LABEL:-mid_emerald_goldfish}"

POD3_ADDR="${POD3_ADDR:-root@157.157.221.29}"
POD3_PORT="${POD3_PORT:-21099}"
POD3_LABEL="${POD3_LABEL:-musical_red_coral}"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=8"

TARGET_TOTAL="${TARGET_TOTAL:-5040}"

check_pod() {
    local label="$1"
    local addr="$2"
    local port="$3"

    echo ""
    echo "--- ${label} (${addr}:${port}) ---"
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} -p "${port}" "${addr}" -i "${SSH_KEY}" '
        set +e

        # GPU status
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed "s/^/GPU: /"
        else
            echo "GPU: nvidia-smi unavailable"
        fi

        # Progress snapshots
        ROOT="/root/scaling-operator-learning"
        if [ -d "$ROOT/runs" ]; then
            DONE=$(find "$ROOT/runs" -name metrics.json 2>/dev/null | wc -l)
            echo "Completed metrics: $DONE"
        else
            echo "Completed metrics: 0 (runs directory missing)"
        fi

        # Common active jobs
        PROCS=$(ps aux | grep -E "run_sweep.py|run_multilaw_analysis.py|run_analysis.py" | grep -v grep | wc -l)
        echo "Active analysis/train processes: $PROCS"

        # Latest logs
        if [ -d "$ROOT" ]; then
            LATEST=$(ls -1t "$ROOT"/*.log 2>/dev/null | head -n 3)
            if [ -n "$LATEST" ]; then
                while IFS= read -r f; do
                    [ -f "$f" ] || continue
                    echo "  log: $(basename "$f")"
                    tail -n 1 "$f" 2>/dev/null | sed "s/^/    /"
                done <<< "$LATEST"
            else
                echo "  log: none"
            fi
        fi
    ' 2>/dev/null || echo "  CONNECTION FAILED"
}

echo "========================================"
echo "  EXPERIMENT MONITOR — $(date '+%H:%M:%S')"
echo "========================================"

check_pod "$POD1_LABEL" "$POD1_ADDR" "$POD1_PORT"
check_pod "$POD2_LABEL" "$POD2_ADDR" "$POD2_PORT"
check_pod "$POD3_LABEL" "$POD3_ADDR" "$POD3_PORT"

echo ""
echo "========================================"
