"""Shared environment and LLM configuration."""

import os
from pathlib import Path


DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_ADAPTER_PATH = "./llc-lora-adapter"
DEFAULT_ENV_NAME = "BabyAI-GoToRedBall-v0"
DEFAULT_RECORDS_PATH = "records/babyai/BabyAI-MixedTrainLocal-v0"
TRAIN_DATA_FILE = "llm_train_data.jsonl"
SUCCESS_TRAJECTORIES_FILE = "success_trajectories.json"

ACTION_MAP = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

# OpenAI-compatible local endpoint (e.g. LM Studio)
HLP_API_BASE = os.environ.get("HLP_API_BASE", "http://127.0.0.1:1234/v1")
HLP_API_KEY = os.environ.get("HLP_API_KEY", "lm-studio")
HLP_MODEL_ID = os.environ.get("HLP_MODEL_ID", "local-model")
HLP_TEMPERATURE = float(os.environ.get("HLP_TEMPERATURE", "0.6"))
HLP_MAX_TOKENS = int(os.environ.get("HLP_MAX_TOKENS", "8092"))

PROMPTS_DIR = Path("prompts")


def load_prompts(
    env_family: str = "babyai",
    task: str | None = None,
    prompt_variant: str = "rich",
    multi_turn: bool = False,
) -> dict[str, str]:
    """Load prompt blocks for a given environment family and variant.

    Args:
        env_family:     Environment family directory under prompts/.
        task:           Optional task name for per-task task_layer.txt.
        prompt_variant: "rich" or "minimal" — selects the _rich or _minimal
                        suffixed variants of agent/descriptor/balrog instruction files.
        multi_turn:     When True, loads agent_instructions_{variant}_mt.txt for the
                        agent_instructions key if it exists, falling back to the
                        standard variant file. Has no effect on other prompt keys.

    Returns a dict with keys:
        "environment_layer"        — shared, no variant suffix
        "agent_instructions"       — variant-suffixed (MT variant if multi_turn=True)
        "descriptor_instructions"  — variant-suffixed
        "balrog_instructions"      — variant-suffixed (for balrog_baseline pipeline)
        "task_layer"               — per-task (only if task given and file exists)

    Prompt files are read from:
        prompts/{env_family}/environment_layer.txt
        prompts/{env_family}/agent_instructions_{variant}.txt
        prompts/{env_family}/agent_instructions_{variant}_mt.txt  (multi-turn variant)
        prompts/{env_family}/descriptor_instructions_{variant}.txt
        prompts/{env_family}/balrog_instructions_{variant}.txt
        prompts/{env_family}/tasks/{task}/task_layer.txt  (optional)
    """
    family_dir = PROMPTS_DIR / env_family
    prompts: dict[str, str] = {}

    # environment_layer has no variant — it is shared infrastructure
    env_path = family_dir / "environment_layer.txt"
    if env_path.exists():
        prompts["environment_layer"] = env_path.read_text()

    for key, stem in (
        ("agent_instructions",      "agent_instructions"),
        ("descriptor_instructions", "descriptor_instructions"),
        ("balrog_instructions",     "balrog_instructions"),
    ):
        if key == "agent_instructions" and multi_turn:
            mt_path = family_dir / f"{stem}_{prompt_variant}_mt.txt"
            if mt_path.exists():
                prompts[key] = mt_path.read_text()
                continue
        path = family_dir / f"{stem}_{prompt_variant}.txt"
        if path.exists():
            prompts[key] = path.read_text()

    if task is not None:
        task_path = family_dir / "tasks" / task / "task_layer.txt"
        if task_path.exists():
            prompts["task_layer"] = task_path.read_text()

    return prompts
