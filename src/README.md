# src/ — Library Code

The core library. All pipeline code, optimisation logic, environment abstractions, and LLM client wrappers live here.

---

## Subpackages

### `config/`
OmegaConf-based config loading and validation. `load_config(path)` is the single entry point — it composes the group files from `conf/`, merges user overrides, and runs `validate_config()` before returning. Schema is defined as Python dataclasses in `schema.py`.

### `environment/`
Environment adapter abstraction. `BaseEnvironmentAdapter` (in `base.py`) defines the interface between the pipeline and the environment: `action_names`, `direction_names`, `get_mission()`, `get_scene_text()`, `get_inventory_text()`. `MiniGridAdapter` (in `minigrid.py`) implements this for BabyAI/MiniGrid.

To add a new environment, subclass `BaseEnvironmentAdapter` and implement all abstract methods. The pipeline auto-loads the adapter from `conf/env/{env_family}.yaml` at startup — no pipeline code changes required.

### `llm/`
LLM client wrappers.
- `client.py` — `OpenAIClient`: connection-reusing OpenAI-compatible client with retry logic.
- `server_utils.py` — `find_all_gpu_servers()` for dynamic endpoint discovery; `start_endpoint_watcher()` for hot-swapping endpoints mid-run.
- `balrog_adapter.py` — Thin adapter that makes our client interface compatible with BALROG's agent interface.

### `optimization/`
The BA → Mutator → Evaluator optimisation loop components.
- `behaviour_analyser.py` — `BehaviourAnalyser`: sends episode trajectories to the LLM and extracts a structured diagnosis (`BAOutput`) identifying which module to modify and what to change.
- `mutator.py` — `Mutator`: applies a BA diagnosis to the current incumbent prompt and produces a revised candidate.
- `evaluator.py` — `Evaluator`: runs V and T evaluation episodes and decides whether to accept a candidate (mean rule or Wilcoxon test, gated by `reward_threshold`).
- `hereditary.py` — Builds the hereditary context: the chain of accepted mutations shown to the BA on subsequent cycles.
- `preprocessing.py` — Compresses episode trajectories for BA input.

### `pipeline/`
Per-step orchestration of the two-LLM pipeline.
- `descriptor_agent.py` — `DescriptorAgent`: calls Descriptor then Actor on each env step, manages prompt loading and multi-turn history.
- `descriptor.py` — `Descriptor`: wraps the descriptor LLM call.
- `parsing.py` — Extracts `PLAN` and `ACTION` fields from actor responses; falls back to `"go forward"` on parse failure.

### `logging/`
Run logging infrastructure.
- `episode_logger.py` — `EpisodeLogger`: writes `trajectory.jsonl` and episode summary files.
- `run_directory.py` — Path construction and `run_config.json` serialisation.
- `rendering.py` — ASCII grid rendering for trajectory inspection.

### `utils/`
Shared helpers.
- `config.py` — `load_prompts()`: loads and assembles the prompt blocks for a given environment family, task, and variant.
- `minigrid_maps.py` — Object, colour, and state string mappings for MiniGrid.
- `bitmask.py` — Bitmask stringification utility.
