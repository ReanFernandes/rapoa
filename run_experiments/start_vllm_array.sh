#!/usr/bin/env bash
# SLURM job array that starts one vLLM server per task across healthy GPU nodes.
# Each task claims one GPU, starts the server, waits for it to be ready, then
# registers itself in ~/vllm_nodes.txt for open_tunnels.sh to pick up.
#
# Usage (submit from project root):
#   sbatch --array=3-4 run_experiments/start_vllm_array.sh   # 2 nodes
#   sbatch --array=20-26 run_experiments/start_vllm_array.sh   # 4 nodes
#
#SBATCH --job-name=vllm-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu001.kisski,gpu006.kisski
#SBATCH --output=logs/vllm_%A_%a.log

set -e

REPO_ROOT="$HOME/rapoa"
PORT=$((7346 + SLURM_ARRAY_TASK_ID))   # task 1→7347, task 2→7348, task 3→7349 ...
NODE=$(hostname)
STATUS_FILE="$HOME/vllm_nodes.txt"
LOCK_FILE="$HOME/vllm_nodes.lock"

cd "$REPO_ROOT"

_update_status() {
    local new_status="$1"
    (
        flock -x 200
        # Remove any existing entry for this port, then append updated one
        tmp=$(mktemp)
        grep -v "^${PORT} " "$STATUS_FILE" 2>/dev/null > "$tmp" || true
        echo "${PORT} ${NODE} ${new_status}" >> "$tmp"
        mv "$tmp" "$STATUS_FILE"
    ) 200>"$LOCK_FILE"
}

_deregister() {
    (
        flock -x 200
        tmp=$(mktemp)
        grep -v "^${PORT} " "$STATUS_FILE" 2>/dev/null > "$tmp" || true
        mv "$tmp" "$STATUS_FILE"
    ) 200>"$LOCK_FILE"
    echo "[task ${SLURM_ARRAY_TASK_ID}] Deregistered port ${PORT}"
}

trap _deregister EXIT

echo "[task ${SLURM_ARRAY_TASK_ID}] Node: ${NODE}, tunnel port: ${PORT}"
_update_status "STARTING"

echo "[task ${SLURM_ARRAY_TASK_ID}] Starting vLLM server..."
bash run_experiments/start_server.sh &> ~/vllm_server_${SLURM_ARRAY_TASK_ID}.log &
SERVER_PID=$!

# Wait up to 10 minutes for the server to become ready
READY=0
for i in $(seq 1 60); do
    if curl -sf http://localhost:7347/v1/models > /dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[task ${SLURM_ARRAY_TASK_ID}] ERROR: vLLM process died. Check ~/vllm_server_${SLURM_ARRAY_TASK_ID}.log" >&2
        _update_status "FAILED"
        exit 1
    fi
    echo "[task ${SLURM_ARRAY_TASK_ID}] Waiting for server... (${i}/60)"
    sleep 10
done

if [ "$READY" -eq 0 ]; then
    echo "[task ${SLURM_ARRAY_TASK_ID}] ERROR: Server did not become ready within 10 minutes." >&2
    _update_status "FAILED"
    exit 1
fi

_update_status "READY"
echo "[task ${SLURM_ARRAY_TASK_ID}] Server READY. Run open_tunnels.sh on the login node."
echo "[task ${SLURM_ARRAY_TASK_ID}]   ssh -N -L ${PORT}:localhost:7347 ${NODE}"

wait "$SERVER_PID"
