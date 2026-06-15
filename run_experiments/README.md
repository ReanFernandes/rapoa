# run_experiments/ — Campaign Orchestration

This directory contains the master orchestrator and supporting scripts for running full experiment campaigns.

---

## run_pipeline.py — Main Entry Point

Loads a config, spawns optimisation subprocesses across all tasks and experimental conditions in parallel, then triggers fresh evaluation on the final optimised prompts.

```bash
# Run a full campaign
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml

# Dry run — print all commands without executing
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml --dry-run

# CLI overrides using dotpath syntax
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml \
  optimisation.opt_cycles=5 \
  rollout.workers=4

# Resume an interrupted campaign
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml \
  --campaign-override babyai/gpt-oss-20b/spa_guided_20260520_143022
```

Resume granularity is one full optimisation cycle — if a run is interrupted mid-cycle, that cycle reruns from the Behaviour Analyser call.

---

## Supporting Scripts

| Script | Purpose | Setup |
|--------|---------|-------|
| `run_baseline.sh` | Non-optimised baseline eval — progenitor prompts, no optimisation. Start here to verify your setup or reproduce paper baseline numbers. | Local or cluster |
| `run_optimised_eval.sh` | Fresh evaluation on optimised incumbent prompts. Called automatically by `run_pipeline.py`. | Local or cluster |
| `start_server.sh` | Start a single vLLM server on one GPU. | Local |
| `start_vllm_array.sh` | SLURM job array — starts one vLLM server per GPU node. | **Cluster only** |
| `open_tunnels.sh` | Opens SSH tunnels from the login node to each GPU node running vLLM. | **Cluster only** |

For full cluster setup instructions (SLURM job submission, tunnel management, running campaigns across multiple GPUs) see [`experiment_run_instructions.md`](experiment_run_instructions.md).

### run_baseline.sh examples

```bash
# Quick sanity check — SPA guided on GoTo, 10 episodes
bash run_experiments/run_baseline.sh

# Full paper baseline protocol — all tasks, all inference seeds (120 episodes/task)
bash run_experiments/run_baseline.sh --full-eval --all-tasks

# BALROG baseline, plain prompt, all tasks
bash run_experiments/run_baseline.sh --pipeline balrog --variant plain --all-tasks

# Single task with explicit endpoint
bash run_experiments/run_baseline.sh --task putnext --endpoint http://localhost:8000/v1
```

---

## Log Structure

Optimisation runs write to:

```
optimization_runs/{env}/{model}/{run_name}_{YYYYMMDD_HHMMSS}/{slug}/{task}/
  run_config.json
  optimisation_log.jsonl
  incumbent_agent_prompt.txt       ← optimised actor prompt
  incumbent_descriptor_prompt.txt  ← optimised descriptor prompt
  opt_cycle_NNN/
```

Fresh evaluation writes to:

```
logs_fresh_eval_optimised/{env}/{model}/{run_name}_{YYYYMMDD_HHMMSS}/{slug}/{task}/
  run_summary.json
  trajectory.jsonl
  episode_NNN.done
```

Optimisation logs are small and irreplaceable (they contain the full prompt evolution history). Fresh eval logs are large and regeneratable from the incumbent prompts.

---

## experiment_run_instructions.md

Full step-by-step guide for running campaigns on the cluster, including server startup, tunnel setup, and progress monitoring.
