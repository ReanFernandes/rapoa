# experiments/ — Core Runners

This directory contains the two core scripts that do the actual work. They are typically invoked by `run_experiments/run_pipeline.py` as subprocesses, but can be called directly for development and debugging.

---

## run.py — Episode Runner

Runs a fixed number of episodes for a single experimental condition and writes a trajectory log.

```bash
python experiments/run.py \
  --env BabyAI-GoToRedBallNoDist-v0 \
  --episodes 10 \
  --pipeline with_descriptor \
  --prompt-variant rich \
  --model gpt-oss-20b \
  --endpoint http://127.0.0.1:1234/v1 \
  --log-dir logs/
```

Key flags:

| Flag | Description |
|------|-------------|
| `--pipeline` | `with_descriptor` (SPA) or `balrog_baseline` |
| `--prompt-variant` | `rich` (guided) or `minimal` (plain) |
| `--episodes` | Number of episodes to run |
| `--workers` | Parallel episode workers |
| `--history-window` | Rolling conversation history length (BALROG baseline only) |
| `--actor-prompt-file` | Path to an incumbent prompt file (used by the optimiser) |
| `--no-gif` | Skip GIF rendering (recommended for optimisation runs) |

Run `python experiments/run.py --help` for the full flag list.

---

## optimise.py — Optimisation Loop

Runs the full SPA optimisation loop for a single task: repeatedly running the Behaviour Analyser → Mutator → Evaluator cycle to refine the actor and descriptor prompts.

```bash
python experiments/optimise.py \
  --env BabyAI-GoToRedBallNoDist-v0 \
  --opt-cycles 20 \
  --prompt-variant rich \
  --model gpt-oss-20b \
  --endpoint http://127.0.0.1:1234/v1 \
  --log-dir optimization_runs/
```

Key flags:

| Flag | Description |
|------|-------------|
| `--opt-cycles` | Number of optimisation cycles |
| `--ba-episodes` | Episodes per Behaviour Analyser call |
| `--t-size` | T pool size for candidate selection |
| `--reward-threshold` | δ acceptance threshold; use `--reward-threshold=-inf` for always-accept |
| `--rule` | `mean` or `wilcoxon` — evaluation acceptance rule |
| `--module-constraint` | `actor`, `descriptor`, or `random` — restrict which module BA can target |
| `--actor-history-window` | Rolling history for the actor (SPA h16 variant) |

The loop writes `optimisation_log.jsonl` and `incumbent_{actor,descriptor}_prompt.txt` to the log directory. Each entry in the JSONL file is one cycle record with full timing, token counts, and acceptance outcome.

---

## How They Relate

`optimise.py` calls `run.py` as a subprocess at three points per cycle:

1. **env_round** — baseline episodes to establish the incumbent's current performance
2. **V eval** — fast validation of a candidate prompt mutation
3. **T eval** — final tournament selection between incumbent and accepted candidates

`run_experiments/run_pipeline.py` orchestrates `optimise.py` across multiple tasks and experimental conditions in parallel, then fires `run_optimised_eval.sh` to run fresh evaluation on the final incumbents.
