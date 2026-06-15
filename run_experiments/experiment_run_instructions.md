# Experiment Run Instructions

All experiments are launched through `run_pipeline.py` with a config from `conf/configs/`.

---

## Setup: Local (single server)

For local development or single-GPU use, start one vLLM server and point the configs at it.

**1. Start the server:**
```bash
bash run_experiments/start_server.sh
```

**2. Set `models: local` in your config** (all provided configs default to this). Edit `conf/models/local.yaml` to match your model name and endpoint if needed:
```yaml
strong:
  name: your-model-name
  endpoint: http://localhost:8000/v1
```

**3. Verify connectivity:**
```bash
python -c "from src.llm.server_utils import find_all_gpu_servers; print(find_all_gpu_servers())"
```
If this returns an empty list the pipeline falls back to `HLP_API_BASE`. Set it if needed:
```bash
export HLP_API_BASE=http://localhost:8000/v1
```

---

## Setup: Cluster (SLURM, multi-GPU)

The cluster workflow uses a three-component setup that must run concurrently during a campaign:

### Step 1 — Submit the vLLM job array

From the login node, submit `start_vllm_array.sh` as a SLURM job array. Each array task starts one vLLM server on one GPU node and registers itself in `~/vllm_nodes.txt` once ready.

```bash
sbatch --array=1-4 run_experiments/start_vllm_array.sh   # 4 GPU nodes
```

Wait until `~/vllm_nodes.txt` shows `READY` entries:
```bash
watch cat ~/vllm_nodes.txt
```

### Step 2 — Open SSH tunnels

In a separate terminal or tmux pane on the login node, run `open_tunnels.sh`. This reads `~/vllm_nodes.txt` and opens an SSH tunnel for each READY node, writing the active port list to `~/hlp_ports.txt` which the pipeline reads for endpoint discovery.

```bash
# One-shot (re-run if new nodes come online)
bash run_experiments/open_tunnels.sh

# Watch mode — auto-opens tunnels as new nodes become ready (recommended)
bash run_experiments/open_tunnels.sh --watch
```

Keep this running in a tmux pane for the duration of the campaign. The pipeline's endpoint watcher picks up changes to `~/hlp_ports.txt` within 30 seconds.

### Step 3 — Launch the campaign

Once tunnels are open, set `models: cluster` in your config and run the pipeline from the login node (or in another tmux pane):

```bash
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml
```

The pipeline auto-discovers all reachable GPU endpoints via `~/hlp_ports.txt` and scales workers accordingly.

---

## Running experiments

```bash
# Smoke test — 1 task, 2 cycles, verify the pipeline works end-to-end
python run_experiments/run_pipeline.py conf/configs/smoke_test.yaml

# SPA guided — HSP, LSP, always-accept, module ablations
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml

# SPA plain — same conditions, plain prompt initialisation
python run_experiments/run_pipeline.py conf/configs/main/spa_plain.yaml

# BALROG baseline — all history window and prompt conditions
python run_experiments/run_pipeline.py conf/configs/main/balrog.yaml

# Threshold sensitivity sweep
python run_experiments/run_pipeline.py conf/configs/ablations/threshold_sweep.yaml

# Dry run — print all commands without executing
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml --dry-run
```

---

## CLI overrides

Any config parameter can be overridden from the command line using dotpath syntax:

```bash
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml \
  optimisation.opt_cycles=5 \
  rollout.workers=10
```

---

## Resuming interrupted runs

Each campaign uses a seconds-precise timestamp in its path, so resumption requires `--campaign-override` pointing to the original directory:

```bash
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml \
  --campaign-override babyai/gpt-oss-20b/spa_guided_20260520_143022
```

The pipeline skips any `(task × experiment)` pair that already has a complete `optimisation_log.jsonl`. Ensure `resume: true` in your config (all provided configs default to this).

---

## Log directory structure

```
optimization_runs/
  {env}/                              e.g. babyai
    {model}/                          e.g. gpt-oss-20b
      {run_name}_{YYYYMMDD_HHMMSS}/   e.g. spa_guided_20260520_143022
        {slug}/                       e.g. spa_mean_valbag_t005_rich
          {task}/                     e.g. mixed_train_goto
            run_config.json
            optimisation_log.jsonl
            incumbent_agent_prompt.txt
            incumbent_descriptor_prompt.txt
            opt_cycle_NNN/

logs_fresh_eval_optimised/            (separate root, same structure below)
  {env}/{model}/{run_name}_{timestamp}/{slug}/{task}/
    run_summary.json
    trajectory.jsonl
    episode_NNN.done
```

Optimisation logs are small and irreplaceable — they contain the full prompt evolution history. Fresh eval logs are large and regeneratable from the incumbent prompts.

---

## Checking progress

```bash
# Optimisation progress across all runs
python tools/opt_progress.py

# Fresh eval progress
python tools/progress.py

# Live monitoring
watch -n 60 'python tools/opt_progress.py'
```

---

## Running fresh evals manually

Fresh eval runs automatically after each experiment via `run_pipeline.py`. To run manually on a completed optimisation:

```bash
bash run_experiments/run_optimised_eval.sh \
  --run-ids babyai/gpt-oss-20b/spa_guided_20260520_143022/spa_mean_valbag_t005_rich \
  --opt-runs-dir optimization_runs \
  --log-root logs_fresh_eval_optimised \
  --episodes 20 \
  --env-seed 500 \
  --inference-seeds 2 3 4 5 6 7 \
  --workers 20
```

---

## Creating a new config

Copy an existing config from `conf/configs/` as a starting point, or refer to `conf/config.yaml` for the full parameter reference. At minimum you need `run_name` and `experiments`:

```yaml
env: babyai
models: local
agent: descriptor_actor
run_name: my_experiment
resume: true

experiments:
  - type: spa
    prompt_variant: rich
    module_constraint: both
    acceptance:
      rule: mean
      reward_threshold: 0.05
      min_discordant_pairs: 4
      p_threshold: 0.05
      validation_strategy: validation_bag
```

See `conf/README.md` for the full config system documentation.
