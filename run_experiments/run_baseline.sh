#!/usr/bin/env bash
# Run non-optimised baseline evaluation — progenitor prompts, no optimisation loop.
#
# Runs experiments/run.py directly for one or all BabyAI tasks and writes
# results to logs_baseline/. Use this to verify your setup, inspect agent
# behaviour, or reproduce the non-optimised numbers from the paper.
#
# Usage:
#   bash run_experiments/run_baseline.sh [OPTIONS]
#
# Examples:
#   # Quick check — SPA guided on GoTo, 10 episodes
#   bash run_experiments/run_baseline.sh
#
#   # Full paper baseline — all tasks, all inference seeds, matching eval protocol
#   bash run_experiments/run_baseline.sh --full-eval --all-tasks
#
#   # BALROG baseline, plain prompt, all tasks
#   bash run_experiments/run_baseline.sh --pipeline balrog --variant plain --all-tasks
#
#   # SPA on a single task with a specific endpoint
#   bash run_experiments/run_baseline.sh --task putnext --endpoint http://localhost:8000/v1
#
# Options:
#   --pipeline   spa | balrog              (default: spa)
#   --variant    guided | plain            (default: guided)
#   --task       goto | pickup | open | putnext | seq | all   (default: goto)
#   --all-tasks  Shorthand for --task all; runs all 5 tasks in parallel
#   --full-eval  Use paper eval protocol: env seeds 500-519, inference seeds 2-7
#                (120 episodes per task). Default: 10 episodes, inference seed 1.
#   --episodes   Number of episodes per inference seed (default: 10, or 20 with --full-eval)
#   --workers    Parallel episode workers per task (default: 10)
#   --history    History window for BALROG pipeline (default: 16)
#   --model      Model name passed to the inference server (default: HLP_MODEL_ID or gpt-oss-20b)
#   --endpoint   Inference server base URL (default: HLP_API_BASE or http://localhost:8000/v1)
#   --log-dir    Output directory (default: logs_baseline)
#   --no-gif     Skip GIF rendering — recommended for multi-episode runs (default: on)
#   --gif        Enable GIF rendering

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
RUNNER="$ROOT/experiments/run.py"

# ── Task registry ─────────────────────────────────────────────────────────────

declare -A TASK_ENV=(
  [goto]="BabyAI-MixedTrainLocal-v0/goto"
  [pickup]="BabyAI-MixedTrainLocal-v0/pickup"
  [open]="BabyAI-MixedTrainLocal-v0/open"
  [putnext]="BabyAI-MixedTrainLocal-v0/putnext"
  [seq]="BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to"
)
ALL_TASKS=(goto pickup open putnext seq)

# ── Defaults ──────────────────────────────────────────────────────────────────

PIPELINE="spa"
VARIANT="guided"
TASK="goto"
FULL_EVAL=false
EPISODES=10
WORKERS=10
HISTORY=16
MODEL="${HLP_MODEL_ID:-gpt-oss-20b}"
ENDPOINT="${HLP_API_BASE:-http://localhost:8000/v1}"
LOG_DIR="logs_baseline"
NO_GIF=true

# ── Argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)   PIPELINE="$2";  shift 2 ;;
    --variant)    VARIANT="$2";   shift 2 ;;
    --task)       TASK="$2";      shift 2 ;;
    --all-tasks)  TASK="all";     shift   ;;
    --full-eval)  FULL_EVAL=true; shift   ;;
    --episodes)   EPISODES="$2";  shift 2 ;;
    --workers)    WORKERS="$2";   shift 2 ;;
    --history)    HISTORY="$2";   shift 2 ;;
    --model)      MODEL="$2";     shift 2 ;;
    --endpoint)   ENDPOINT="$2";  shift 2 ;;
    --log-dir)    LOG_DIR="$2";   shift 2 ;;
    --no-gif)     NO_GIF=true;    shift   ;;
    --gif)        NO_GIF=false;   shift   ;;
    -h|--help)
      head -40 "$0" | grep "^#" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Validate inputs ───────────────────────────────────────────────────────────

case "$PIPELINE" in
  spa|balrog) ;;
  *) echo "ERROR: --pipeline must be 'spa' or 'balrog'"; exit 1 ;;
esac

case "$VARIANT" in
  guided|plain) ;;
  *) echo "ERROR: --variant must be 'guided' or 'plain'"; exit 1 ;;
esac

if [[ "$TASK" != "all" ]] && [[ -z "${TASK_ENV[$TASK]+x}" ]]; then
  echo "ERROR: --task must be one of: goto pickup open putnext seq all"
  exit 1
fi

# ── Resolve pipeline and variant to internal names ────────────────────────────

# Paper term → code term
[[ "$VARIANT" == "guided" ]] && PROMPT_VARIANT="rich" || PROMPT_VARIANT="minimal"
[[ "$PIPELINE" == "spa" ]]   && PIPELINE_FLAG="with_descriptor" || PIPELINE_FLAG="balrog_baseline"

# ── Resolve eval protocol ─────────────────────────────────────────────────────

if $FULL_EVAL; then
  ENV_SEEDS=($(seq 500 519))     # 20 env seeds matching paper hold-out set
  INFERENCE_SEEDS=(2 3 4 5 6 7)  # 6 inference seeds matching paper protocol
  [[ "$EPISODES" == "10" ]] && EPISODES=20  # bump default if not explicitly set
else
  ENV_SEEDS=($(seq 42 $((42 + EPISODES - 1))))
  INFERENCE_SEEDS=(1)
fi

# ── Resolve task list ─────────────────────────────────────────────────────────

if [[ "$TASK" == "all" ]]; then
  TASKS=("${ALL_TASKS[@]}")
else
  TASKS=("$TASK")
fi

# ── Export endpoint so run.py's HLP_API_BASE fallback picks it up ─────────────

export HLP_API_BASE="$ENDPOINT"
export HLP_MODEL_ID="$MODEL"

# ── Summary ───────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════"
echo "  RAPOA — Non-optimised baseline"
echo "  Pipeline : $PIPELINE ($PIPELINE_FLAG)"
echo "  Variant  : $VARIANT ($PROMPT_VARIANT)"
echo "  Tasks    : ${TASKS[*]}"
echo "  Episodes : $EPISODES env seeds × ${#INFERENCE_SEEDS[@]} inference seed(s) = $((EPISODES * ${#INFERENCE_SEEDS[@]})) per task"
echo "  Model    : $MODEL"
echo "  Endpoint : $ENDPOINT"
echo "  Log dir  : $LOG_DIR"
echo "═══════════════════════════════════════════════════════"

# ── Runner ────────────────────────────────────────────────────────────────────

run_task() {
  local task="$1"
  local env_id="${TASK_ENV[$task]}"

  for inf_seed in "${INFERENCE_SEEDS[@]}"; do
    local out_dir="$ROOT/$LOG_DIR/${task}/${PIPELINE}/${VARIANT}/seed_${inf_seed}"
    mkdir -p "$out_dir"

    local cmd=(
      "$PYTHON" "$RUNNER"
      --env            "$env_id"
      --pipeline       "$PIPELINE_FLAG"
      --prompt-variant "$PROMPT_VARIANT"
      --seed-list      "${ENV_SEEDS[@]}"
      --inference-seed "$inf_seed"
      --model          "$MODEL"
      --workers        "$WORKERS"
      --log-dir        "$out_dir"
    )

    [[ "$PIPELINE_FLAG" == "balrog_baseline" ]] && cmd+=(--history-window "$HISTORY")
    $NO_GIF && cmd+=(--no-gif)

    echo "[$(date +%H:%M:%S)] Starting: task=$task inf_seed=$inf_seed"
    "${cmd[@]}"
    echo "[$(date +%H:%M:%S)] Done:     task=$task inf_seed=$inf_seed → $out_dir"
  done
}

# ── Launch — parallel across tasks, sequential across inference seeds ─────────

if [[ "${#TASKS[@]}" -gt 1 ]]; then
  pids=()
  for task in "${TASKS[@]}"; do
    run_task "$task" &
    pids+=($!)
  done
  # Wait for all and collect exit codes
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "ERROR: task subprocess $pid failed"; failed=1; }
  done
  [[ "$failed" -eq 1 ]] && exit 1
else
  run_task "${TASKS[0]}"
fi

echo ""
echo "Results written to: $ROOT/$LOG_DIR/"
