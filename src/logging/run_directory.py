"""Run directory creation and configuration persistence."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def parse_env_id(env_id: str) -> tuple[str, str]:
    """Best-effort split of an environment ID into (family, task).

    Strips the version suffix then splits on the first hyphen:
        'BabyAI-GoToRedBall-v0'          -> ('BabyAI', 'GoToRedBall')
        'MiniGrid-Empty-5x5-v0'          -> ('MiniGrid', 'Empty-5x5')
        'BabyAI-MixedTrainLocal-v0/goto' -> ('BabyAI', 'goto')

    Falls back to (env_id, 'unknown') if no hyphen is found.
    """
    if "/" in env_id:
        base, subtype = env_id.split("/", 1)
        family, _ = parse_env_id(base)
        return family, subtype.replace(" ", "_")
    name = re.sub(r"-v\d+$", "", env_id)
    parts = name.split("-", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (env_id, "unknown")


def create_run_directory(
    base_log_directory: Path,
    env_family: str,
    task: str,
    model: str,
    pipeline: str,
    prompt_variant: str,
    conversation_mode: str,
    reasoning: bool,
    inference_seed: int | None = None,
) -> Path:
    """Create a timestamped run directory under a structured hierarchy.

    Structure:
        base / env_family / task / model / pipeline / prompt_variant / conversation_mode / (reasoning|no_reasoning) / inference_seed / timestamp
    """
    model_safe = model.replace("/", "--")
    reasoning_dir = "reasoning" if reasoning else "no_reasoning"
    seed_dir = f"iseed_{inference_seed}" if inference_seed is not None else "iseed_none"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = (
        base_log_directory
        / env_family
        / task
        / model_safe
        / pipeline
        / prompt_variant
        / conversation_mode
        / reasoning_dir
        / seed_dir
        / timestamp
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    log.info("Created run log directory: %s", run_directory)
    return run_directory


def save_run_config(run_directory: Path, config: dict) -> None:
    """Persist the run configuration as JSON in the run directory."""
    config_path = run_directory / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, default=str)
    log.info("Saved run config to %s", config_path)
