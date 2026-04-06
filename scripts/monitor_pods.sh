#!/usr/bin/env bash
# Monitor all pods - run from local machine
# Usage: bash scripts/monitor_pods.sh
set -uo pipefail

SSH_KEY=~/.ssh/id_ed25519
POD1="root@157.157.221.29 -p 20661"
POD2="root@157.157.221.29 -p 29630"
POD3="root@213.192.2.122 -p 40003"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "========================================"
echo "  EXPERIMENT MONITOR — $(date '+%H:%M:%S')"
echo "========================================"

for POD_LABEL POD_ADDR LOG_PREFIX TASK in \
    "Pod1:Burgers(A4000)" "$POD1" "sweep_burgers" "burgers_operator" \
    "Pod2:Diffusion(L4)"  "$POD2" "sweep_diffusion" "diffusion" \
    "Pod3:Darcy(3090)"    "$POD3" "sweep_darcy" "darcy"; do

    echo ""
    echo "--- $POD_LABEL ---"
    # shellcheck disable=SC2086
    ssh $SSH_OPTS $POD_ADDR -i "$SSH_KEY" "
        # GPU utilization
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'

        # Completed runs
        N=\$(find /root/scaling-operator-learning/runs -name metrics.json 2>/dev/null | wc -l)
        echo \"Completed: \$N / 1680\"

        # Active processes
        PROCS=\$(ps aux | grep run_sweep | grep -v grep | wc -l)
        echo \"Active processes: \$PROCS\"

        # Latest log lines per resolution
        for f in /root/${LOG_PREFIX}*.log; do
            [ -f \"\$f\" ] || continue
            R=\$(basename \$f .log | sed 's/.*_R/R=/' | sed 's/${LOG_PREFIX}/main/')
            LAST=\$(tail -1 \"\$f\" 2>/dev/null)
            echo \"  \$R: \$LAST\"
        done
    " 2>/dev/null || echo "  CONNECTION FAILED"
done

echo ""
echo "========================================"
