#!/usr/bin/env bash
# Evaluate the final incumbent prompts from one or more mean-gated optimisation runs.
#
# Reads incumbent_agent_prompt.txt + incumbent_descriptor_prompt.txt from
# optimization_runs/{run_id}/{variant}/{task}/ and runs with_descriptor
# pipeline on all tasks / variants found in each run directory.
#
# Results are written to logs_fresh_eval_optimised/{run_id}/ so each run's
# results stay separated and load_fresh_eval() can distinguish them.
#
# Usage (from project root):
#   bash run_experiments/run_optimised_eval.sh \
#       --run-ids babyai/gpt-oss-20b/spa_guided_20260520_143022/spa_mean_valbag_t005_rich
#
# Options:
#   --run-ids         Space-separated list of run IDs inside optimization_runs/ (required)
#   --opt-runs-dir    Root of optimisation runs (default: optimization_runs)
#   --log-root        Root under which per-run-id subdirs are created
#                     (default: logs_fresh_eval_optimised)
#   --episodes        Episodes per seed per task (default: 20)
#   --env-seed        Environment seed (default: 500)
#   --inference-seeds Space-separated list (default: 2 3 4 5 6 7)
#   --workers         Parallel episode workers (default: 4)
#   --seed-batch-size Max seeds to run in parallel per run-id (default: 0 = all at once)
#   --tasks           Space-separated task names to evaluate (default: all found in run dir)
#                     e.g. --tasks goto pickup open
#
# Example:
#   bash run_experiments/run_optimised_eval.sh \
#       --run-ids babyai/gpt-oss-20b/spa_guided_20260520_143022/spa_mean_valbag_t005_rich \
#       --workers 4 --tasks goto pickup open pick_up_seq_go_to

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_IDS=()
TASKS_FILTER=()
USE_BEST_T=0
OPT_RUNS_DIR="optimization_runs"
LOG_ROOT="logs_fresh_eval_optimised"
EPISODES=20
ENV_SEED=500
INFERENCE_SEEDS=(2 3 4 5 6 7)
WORKERS=4
SEED_BATCH_SIZE=0

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-ids)         shift; while [[ $# -gt 0 && "$1" != --* ]]; do RUN_IDS+=("$1"); shift; done ;;
        --tasks)           shift; while [[ $# -gt 0 && "$1" != --* ]]; do TASKS_FILTER+=("$1"); shift; done ;;
        --use-best-t)      USE_BEST_T=1;           shift ;;
        --opt-runs-dir)    OPT_RUNS_DIR="$2";     shift 2 ;;
        --log-root)        LOG_ROOT="$2";          shift 2 ;;
        --episodes)        EPISODES="$2";          shift 2 ;;
        --env-seed)        ENV_SEED="$2";          shift 2 ;;
        --inference-seeds) shift; INFERENCE_SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do INFERENCE_SEEDS+=("$1"); shift; done ;;
        --workers)         WORKERS="$2";           shift 2 ;;
        --seed-batch-size) SEED_BATCH_SIZE="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "ERROR: --run-ids is required"
    echo "  e.g. --run-ids babyai/gpt-oss-20b/spa_guided_20260520_143022/spa_mean_valbag_t005_rich"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"

# ── Run helper ────────────────────────────────────────────────────────────────
_run_task() {
    # Evaluate one task directory. Reads pipeline/variant/history from run_config.json.
    local TASK_DIR="$1"
    local LABEL="$2"
    local INFERENCE_SEED="$3"
    local LOG_DIR="$4"
    local COND_LOG_DIR="$5"

    local AGENT_FILE DESC_FILE
    if [[ "$USE_BEST_T" -eq 1 ]]; then
        AGENT_FILE="$TASK_DIR/best_t_agent_prompt.txt"
        DESC_FILE="$TASK_DIR/best_t_descriptor_prompt.txt"
    else
        AGENT_FILE="$TASK_DIR/incumbent_agent_prompt.txt"
        DESC_FILE="$TASK_DIR/incumbent_descriptor_prompt.txt"
    fi
    local CFG="$TASK_DIR/run_config.json"

    [[ ! -f "$AGENT_FILE" ]] && { echo "  SKIP $LABEL — no $(basename $AGENT_FILE)" >&2; return; }
    [[ ! -f "$CFG" ]] && { echo "  SKIP $LABEL — no run_config.json" >&2; return; }

    # Read pipeline, prompt_variant from task-level run_config (optimise.py config)
    local PIPELINE PROMPT_VARIANT HISTORY_WINDOW AGENT_MULTI_TURN TASK_NAME
    PIPELINE=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('pipeline','with_descriptor'))" "$CFG")
    PROMPT_VARIANT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('prompt_variant','rich'))" "$CFG")
    TASK_NAME=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['env'].split('/')[-1])" "$CFG")

    # Skip if this task+seed already has a completed summary in the log dir
    local n_done
    n_done=$(find "$LOG_DIR" -path "*/$TASK_NAME/*" -name "run_summary.json" \
             -path "*/iseed_${INFERENCE_SEED}/*" 2>/dev/null | wc -l)
    if [[ "$n_done" -gt 0 ]]; then
        echo "  SKIP $LABEL seed $INFERENCE_SEED — already evaluated" >&2
        return
    fi

    # history_window and agent_multi_turn are not in the task-level config — read from the first episode run_config
    read -r HISTORY_WINDOW AGENT_MULTI_TURN < <(python3 -c "
import json, pathlib, sys
ep_cfg = next(iter(sorted(pathlib.Path(sys.argv[1]).glob('env_round_0/**/run_config.json'))), None)
if ep_cfg:
    d = json.load(open(ep_cfg))
    hw  = d.get('history_window') or ''
    amt = '1' if d.get('agent_multi_turn') else '0'
    print(hw, amt)
else:
    print('', '0')
" "$TASK_DIR")

    local SAFE="${LABEL//[ \/]/_}"
    local CLOG="$COND_LOG_DIR/${SAFE}__iseed_${INFERENCE_SEED}.log"

    local CMD=(python "$REPO_ROOT/experiments/run.py"
        --env "BabyAI-MixedTrainLocal-v0/$TASK_NAME"
        --episodes "$EPISODES"
        --env-seed "$ENV_SEED"
        --workers "$WORKERS"
        --log-dir "$LOG_DIR"
        --pipeline "$PIPELINE"
        --prompt-variant "$PROMPT_VARIANT"
        --agent-prompt-file "$AGENT_FILE"
        --inference-seed "$INFERENCE_SEED"
        --no-gif
    )
    # Add descriptor only for with_descriptor pipeline
    if [[ "$PIPELINE" == "with_descriptor" ]]; then
        [[ ! -f "$DESC_FILE" ]] && { echo "  SKIP $LABEL — no incumbent_descriptor_prompt.txt" >&2; return; }
        CMD+=(--descriptor-prompt-file "$DESC_FILE")
    fi
    # Add history window (all pipelines) and agent-multi-turn for h16 spa conditions
    [[ -n "$HISTORY_WINDOW" ]] && CMD+=(--history-window "$HISTORY_WINDOW")
    [[ "$AGENT_MULTI_TURN" == "1" ]] && CMD+=(--agent-multi-turn)

    echo "  + [$LABEL | $PIPELINE | seed $INFERENCE_SEED]" >&2
    "${CMD[@]}" > "$CLOG" 2>&1 &
    # Store PID in shared variable — avoids $() subshell which would reparent
    # the background process to init, making wait() fail with "not a child"
    _TASK_PID=$!
}

_run_one_seed() {
    local INFERENCE_SEED="$1"
    local RUN_ID="$2"
    local OPT_DIR="$3"
    local LOG_DIR="$4"
    local COND_LOG_DIR="$5"

    local _PIDS=()
    _TASK_PID=""

    mapfile -t TOP_DIRS < <(find "$OPT_DIR" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort)

    for TOP in "${TOP_DIRS[@]}"; do
        local TOP_DIR="$OPT_DIR/$TOP"

        # Detect structure: flat ({task}/) vs nested ({variant}/{task}/)
        # If incumbent_agent_prompt.txt exists directly here, this IS the task dir
        if [[ -f "$TOP_DIR/incumbent_agent_prompt.txt" ]]; then
            if [[ ${#TASKS_FILTER[@]} -gt 0 ]] && ! printf '%s\n' "${TASKS_FILTER[@]}" | grep -qx "$TOP"; then
                continue
            fi
            _run_task "$TOP_DIR" "opt/$RUN_ID/$TOP" \
                      "$INFERENCE_SEED" "$LOG_DIR" "$COND_LOG_DIR"
            [[ -n "$_TASK_PID" ]] && _PIDS+=("$_TASK_PID")
        else
            # Nested: TOP is a variant, look one level deeper for task dirs
            mapfile -t TASK_DIRS < <(find "$TOP_DIR" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort)
            for TASK in "${TASK_DIRS[@]}"; do
                local TASK_DIR="$TOP_DIR/$TASK"
                [[ ! -f "$TASK_DIR/incumbent_agent_prompt.txt" ]] && continue
                if [[ ${#TASKS_FILTER[@]} -gt 0 ]] && ! printf '%s\n' "${TASKS_FILTER[@]}" | grep -qx "$TASK"; then
                    continue
                fi
                _run_task "$TASK_DIR" "opt/$RUN_ID/$TOP/$TASK" \
                          "$INFERENCE_SEED" "$LOG_DIR" "$COND_LOG_DIR"
                [[ -n "$_TASK_PID" ]] && _PIDS+=("$_TASK_PID")
            done
        fi
    done

    local failed=0
    for pid in "${_PIDS[@]}"; do
        wait "$pid" || { echo "  WARNING pid $pid exited non-zero — check $COND_LOG_DIR" >&2; failed=$((failed + 1)); }
    done
    echo "  seed $INFERENCE_SEED batch done ($failed failure(s))"
}

# ── Main loop: one sub-process per run-id ─────────────────────────────────────
_TOP_PIDS=()

for RUN_ID in "${RUN_IDS[@]}"; do
    OPT_DIR="$REPO_ROOT/$OPT_RUNS_DIR/$RUN_ID"
    if [[ ! -d "$OPT_DIR" ]]; then
        echo "WARNING: $OPT_DIR not found — skipping $RUN_ID"
        continue
    fi

    if [[ "$USE_BEST_T" -eq 1 ]]; then
        LOG_DIR="$LOG_ROOT/${RUN_ID}_best_t"
    else
        LOG_DIR="$LOG_ROOT/$RUN_ID"
    fi
    COND_LOG_DIR="$REPO_ROOT/$LOG_DIR/condition_logs"
    mkdir -p "$COND_LOG_DIR"

    mapfile -t VARIANTS_PREVIEW < <(find "$OPT_DIR" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort)

    # Warn about any tasks where no mutations were accepted —
    # eval still runs to provide independent reproducibility evidence.
    ZERO_MUT=$(python3 -c "
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
zero = []
for log in sorted(base.rglob('optimisation_log.jsonl')):
    if 'env_round' in str(log) or 'opt_cycle' in str(log): continue
    try:
        records = [json.loads(l) for l in open(log)]
        n = sum(1 for r in records if r.get('record_type')=='opt_cycle'
                and r.get('opt_cycle_outcome')=='accepted')
        total = sum(1 for r in records if r.get('record_type')=='opt_cycle')
        if total > 0 and n == 0:
            zero.append(f'{log.parent.parent.name}/{log.parent.name}')
    except: pass
print(' '.join(zero))
" "$OPT_DIR" 2>/dev/null)

    echo ""
    echo "============================================================"
    echo "  Run ID:   $RUN_ID"
    echo "  Variants: ${VARIANTS_PREVIEW[*]}"
    echo "  Log dir:  $LOG_DIR"
    if [[ -n "$ZERO_MUT" ]]; then
        echo "  NOTE — zero mutations accepted in: $ZERO_MUT"
        echo "         Eval runs anyway (reproducibility evidence)."
    fi
    echo "============================================================"

    (
        MAX_PARALLEL_SEEDS=${SEED_BATCH_SIZE}
        [[ "$MAX_PARALLEL_SEEDS" -eq 0 ]] && MAX_PARALLEL_SEEDS=${#INFERENCE_SEEDS[@]}

        _SEED_PIDS=()

        for INFERENCE_SEED in "${INFERENCE_SEEDS[@]}"; do
            while [[ ${#_SEED_PIDS[@]} -ge $MAX_PARALLEL_SEEDS ]]; do
                wait -n 2>/dev/null || true
                _running=()
                for pid in "${_SEED_PIDS[@]}"; do
                    kill -0 "$pid" 2>/dev/null && _running+=("$pid")
                done
                _SEED_PIDS=("${_running[@]}")
            done

            ( _run_one_seed "$INFERENCE_SEED" "$RUN_ID" "$OPT_DIR" "$LOG_DIR" "$COND_LOG_DIR" ) &
            _SEED_PIDS+=($!)
        done

        wait "${_SEED_PIDS[@]}"
        echo "  $RUN_ID — all seeds done."
    ) &
    _TOP_PIDS+=($!)
done

echo ""
echo "  Waiting for all run-ids to finish ..."
wait "${_TOP_PIDS[@]}"

echo ""
echo "============================================================"
echo "  Optimised eval complete."
echo "  Results under: $LOG_ROOT/"
for RUN_ID in "${RUN_IDS[@]}"; do
    echo "    $LOG_ROOT/$RUN_ID/"
done
echo "============================================================"
