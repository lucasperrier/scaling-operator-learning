#!/usr/bin/env bash
# Quick connector for the three RunPod workers.
# Usage:
#   bash scripts/connect_pods.sh p1
#   bash scripts/connect_pods.sh p2
#   bash scripts/connect_pods.sh p3
#   bash scripts/connect_pods.sh all
set -euo pipefail

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    echo "Usage: $0 {p1|p2|p3|all}"
    exit 1
fi

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-o StrictHostKeyChecking=no"

P1_ADDR="${POD1_ADDR:-root@213.173.108.102}"
P1_PORT="${POD1_PORT:-26317}"
P2_ADDR="${POD2_ADDR:-root@213.173.108.102}"
P2_PORT="${POD2_PORT:-30412}"
P3_ADDR="${POD3_ADDR:-root@157.157.221.29}"
P3_PORT="${POD3_PORT:-21099}"

connect_one() {
    local addr="$1"
    local port="$2"
    echo "Connecting to ${addr}:${port}"
    ssh ${SSH_OPTS} -i "${SSH_KEY}" -p "${port}" "${addr}"
}

case "${TARGET}" in
    p1)
        connect_one "${P1_ADDR}" "${P1_PORT}"
        ;;
    p2)
        connect_one "${P2_ADDR}" "${P2_PORT}"
        ;;
    p3)
        connect_one "${P3_ADDR}" "${P3_PORT}"
        ;;
    all)
        echo "p1: ${P1_ADDR}:${P1_PORT}"
        echo "p2: ${P2_ADDR}:${P2_PORT}"
        echo "p3: ${P3_ADDR}:${P3_PORT}"
        ;;
    *)
        echo "Invalid target: ${TARGET}"
        echo "Usage: $0 {p1|p2|p3|all}"
        exit 1
        ;;
esac
