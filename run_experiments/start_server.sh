#!/usr/bin/env bash
# Start a vLLM inference server for RAPOA experiments.
# Requires vllm to be installed: uv pip install vllm  (or: uv sync --extra server)
#
# Usage:
#   bash run_experiments/start_server.sh [--model MODEL] [--port PORT]
#
# Defaults:
#   --model  gpt-oss-20b   (must match conf/models/local.yaml)
#   --port   8000          (must match conf/models/local.yaml)

set -e

MODEL="gpt-oss-20b"
PORT=8000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --port)  PORT="$2";  shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if ! command -v vllm &>/dev/null; then
  echo "ERROR: vllm not found. Install it with:"
  echo "       uv pip install vllm"
  echo "  or:  uv sync --extra server"
  exit 1
fi

echo "Starting vLLM server ($MODEL) on port $PORT..."
echo "Server will be ready when you see: Uvicorn running on http://0.0.0.0:$PORT"
echo ""

vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --served-model-name "$MODEL"
