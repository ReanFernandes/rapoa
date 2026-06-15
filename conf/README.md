# conf/ — Configuration System

Experiments are fully driven by YAML configs composed from this directory. There are no hardcoded experiment parameters in the pipeline code.

---

## Structure

```
conf/
  config.yaml               Full parameter reference — read this to understand every field
  env/
    babyai.yaml             Environment structure: action names, direction names, obs fields, default task list
  task/
    babyai/                 One file per task — gym_id and task_family
  models/
    cluster.yaml            Model registry for cluster use (dynamic endpoint discovery)
    local.yaml              Model registry for local/single-server use
  agent/
    descriptor_actor.yaml   Two-LLM SPA pipeline
    balrog.yaml             BALROG baseline (rolling history, h=16)
    balrog_short_history.yaml  BALROG baseline (h=1)
  rollout/
    default.yaml            Episode parameters: max_steps, workers, actor/descriptor generation settings
  optimisation/
    default.yaml            Optimisation loop: cycles, BA episodes, T pool size, seeds, BA/Mutator settings
  evaluation/
    default.yaml            Fresh eval settings: seeds, episodes, workers
  configs/
    smoke_test.yaml         End-to-end verification — run this first on a new install
    main/
      spa_guided.yaml       SPA guided prompt: HSP, LSP, AA, module ablations
      spa_plain.yaml        SPA plain prompt: same conditions
      balrog.yaml           BALROG baseline: h16 + h1, guided + plain, all acceptance conditions
    ablations/
      threshold_sweep.yaml       δ ∈ {0.00, 0.02, 0.05, 0.10} × {HSP, LSP}, SPA guided
      threshold_sweep_h16.yaml   Same with 16-step actor history window
      random_module.yaml         Random module pre-selection (BA attribution ablation)
```

`conf/config.yaml` is the primary reference. Every parameter is documented inline with valid values and a one-line description.

---

## How Configs Compose

A runnable config in `conf/configs/` declares which group files to use and then overrides specific values. The loader in `src/config/loader.py` merges them in order:

1. Group defaults (`env/`, `models/`, `agent/`, `rollout/`, `optimisation/`, `evaluation/`)
2. User config overrides from `conf/configs/main/ or conf/configs/ablations/`

For example, `conf/configs/smoke_test.yaml` sets `models: local`, picks two tasks, and sets `opt_cycles: 2` — everything else inherits from the group defaults.

---

## Running a Config

```bash
python run_experiments/run_pipeline.py conf/configs/smoke_test.yaml
```

### CLI Overrides

Any config field can be overridden on the command line using dotpath syntax:

```bash
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml \
  optimisation.opt_cycles=5 \
  rollout.workers=4
```

### Resuming a Run

```bash
python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml \
  --campaign-override babyai/gpt-oss-20b/spa_guided_20260520_143022
```

---

## Writing Your Own Config

Create a new file in `conf/configs/`. At minimum you need:

```yaml
env: babyai
models: local          # or cluster
agent: descriptor_actor

experiments:
  - type: spa
    prompt_variant: rich       # rich (guided) or minimal (plain)
    validation_strategy: validation_bag   # validation_bag (HSP) or train_signal (LSP)
    reward_threshold: 0.05     # δ acceptance threshold; use -.inf for always-accept
```

Everything not specified inherits from the group defaults. See `conf/config.yaml` for the full list of overridable fields.

---

## Models Config

`conf/models/local.yaml` is for single-server local use. Edit it to match your inference server:

```yaml
strong:
  name: your-model-name      # Model ID your server advertises
  endpoint: http://127.0.0.1:1234/v1
```

`conf/models/cluster.yaml` uses `endpoint: null` which triggers dynamic endpoint discovery via `~/hlp_ports.txt` — this is specific to our cluster setup and not needed for single-server use.
