"""Per-episode logger — writes grid renders and LLM I/O logs per step."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.logging.rendering import render_full_grid, render_partial_observation

log = logging.getLogger(__name__)


def _format_status(terminated: bool, truncated: bool) -> str:
    if terminated:
        return "Status: TERMINATED"
    if truncated:
        return "Status: TRUNCATED"
    return "Status: ongoing"


class EpisodeLogger:
    """Writes two separate log files per episode — one for grid renders,
    one for LLM input/output.

    Usage as a context manager::

        with EpisodeLogger(...) as logger:
            logger.log_step(...)
            ...
            logger.log_episode_summary(...)
    """

    def __init__(
        self,
        run_directory: Path,
        episode_number: int,
        environment_name: str,
        seed: int,
    ):
        self._episode_number = episode_number
        self._environment_name = environment_name
        self._seed = seed

        grid_directory = run_directory / "grid_renders"
        llm_directory = run_directory / "llm_output"
        grid_directory.mkdir(parents=True, exist_ok=True)
        llm_directory.mkdir(parents=True, exist_ok=True)

        grid_file_path = grid_directory / f"episode_{episode_number:03d}.log"
        llm_file_path = llm_directory / f"episode_{episode_number:03d}.log"

        self._grid_file = open(grid_file_path, "w", encoding="utf-8")
        self._llm_file = open(llm_file_path, "w", encoding="utf-8")

        self._write_headers()

    def _write_headers(self) -> None:
        header = (
            f"{'=' * 70}\n"
            f"Episode {self._episode_number}\n"
            f"Environment: {self._environment_name}\n"
            f"Seed: {self._seed}\n"
            f"{'=' * 70}\n\n"
        )
        self._grid_file.write(header)
        self._llm_file.write(header)
        self._grid_file.flush()
        self._llm_file.flush()

    def log_step(
        self,
        step_number: int,
        raw_observation: dict,
        full_grid_encoded: np.ndarray,
        agent_position: tuple[int, int],
        agent_direction: int,
        agent_representation: str | None,
        llm_response: str | None,
        action_name: str,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Log a single environment step to both log files."""
        self._write_grid_step(
            step_number, raw_observation, full_grid_encoded,
            agent_position, agent_direction,
            action_name, reward, terminated, truncated,
        )
        self._write_llm_step(
            step_number, agent_representation, llm_response, action_name, reward,
            terminated, truncated,
        )

    def _write_grid_step(
        self,
        step_number: int,
        raw_observation: dict,
        full_grid_encoded: np.ndarray,
        agent_position: tuple[int, int],
        agent_direction: int,
        action_name: str,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        separator = f"{'─' * 70}\n"
        self._grid_file.write(f"\n{separator}Step {step_number}\n{separator}\n")

        mission = raw_observation.get("mission", "")

        self._grid_file.write("[ Full Environment Grid ]\n\n")
        full_grid_text = render_full_grid(
            full_grid_encoded, agent_position, agent_direction, mission,
        )
        self._grid_file.write(full_grid_text)
        self._grid_file.write("\n\n")

        image_grid = raw_observation.get("image")
        direction = raw_observation.get("direction", 0)

        if isinstance(image_grid, np.ndarray) and image_grid.shape == (7, 7, 3):
            self._grid_file.write("[ Agent's Partial Observation (7x7) ]\n\n")
            partial_obs_text = render_partial_observation(image_grid, direction, mission)
            self._grid_file.write(partial_obs_text)
            self._grid_file.write("\n")

        status = _format_status(terminated, truncated)
        self._grid_file.write(
            f"\nAction: {action_name}  |  Reward: {reward:.4f}  |  {status}\n"
        )
        self._grid_file.flush()

    def _write_llm_step(
        self,
        step_number: int,
        agent_representation: str | None,
        llm_response: str | None,
        action_name: str,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        separator = f"{'─' * 70}\n"
        self._llm_file.write(f"\n{separator}Step {step_number}\n{separator}\n")

        self._llm_file.write(">>> PROMPT SENT TO LLM >>>\n\n")
        if agent_representation is not None:
            self._llm_file.write(agent_representation)
            self._llm_file.write("\n")
        else:
            self._llm_file.write("(no agent — random action mode)\n")

        self._llm_file.write("\n<<< LLM RESPONSE <<<\n\n")
        if llm_response is not None:
            self._llm_file.write(llm_response)
            self._llm_file.write("\n")
        else:
            self._llm_file.write("(no LLM response)\n")

        status = _format_status(terminated, truncated)
        self._llm_file.write(
            f"\nAction: {action_name}  |  Reward: {reward:.4f}  |  {status}\n"
        )
        self._llm_file.flush()

    def log_episode_summary(self, total_reward: float, total_steps: int) -> None:
        """Write a summary block at the end of both log files."""
        success = total_reward > 0
        status_label = "SUCCESS" if success else "FAILED"
        summary = (
            f"\n{'=' * 70}\n"
            f"Episode Summary\n"
            f"  Result:       {status_label}\n"
            f"  Total reward: {total_reward:.4f}\n"
            f"  Total steps:  {total_steps}\n"
            f"{'=' * 70}\n"
        )
        self._grid_file.write(summary)
        self._llm_file.write(summary)
        self._grid_file.flush()
        self._llm_file.flush()

    def close(self) -> None:
        """Flush and close both log files."""
        for file_handle in (self._grid_file, self._llm_file):
            if file_handle and not file_handle.closed:
                file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
