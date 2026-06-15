"""MiniGrid/BabyAI environment adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environment.base import BaseEnvironmentAdapter
from src.utils.minigrid_maps import OBJECT_TO_STR, COLOR_TO_STR, STATE_TO_STR

# Canonical action and direction name lists for MiniGrid/BabyAI environments.
# Index order matches MiniGrid's integer action enum.
# Imported directly by run.py for action parsing and prompt construction.
ACTION_NAMES    = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle", "done"]
DIRECTION_NAMES = ["right", "down", "left", "up"]

# Agent position in the 7×7 agent-relative obs frame
_AGENT_X = 3
_AGENT_Y = 6


class MiniGridAdapter(BaseEnvironmentAdapter):
    """Adapter for MiniGrid / BabyAI environments.

    Observation format after BabyAITextCleanLangWrapper:
      obs["text"]["long_term_context"]  — rule-based scene text from gen_obs_desc()
      obs["mission"]                    — goal string
      obs["direction"]                  — integer index into DIRECTION_NAMES
      obs["image"]                      — PIL Image (POV render)

    The raw 7×7×3 numpy image is also available pre-wrapper for parse_observation.
    """

    @property
    def action_names(self) -> list[str]:
        return ACTION_NAMES

    @property
    def direction_names(self) -> list[str]:
        return DIRECTION_NAMES

    def get_mission(self, obs: Any) -> str:
        return obs.get("mission", "")

    def get_scene_text(self, obs: Any) -> str:
        return obs.get("text", {}).get("long_term_context", "")

    def get_inventory_text(self, env: Any) -> str | None:
        carrying = env.unwrapped.carrying
        if carrying is None:
            return None
        return f"{carrying.color} {carrying.type}"

    def parse_observation(self, raw_obs: Any) -> list[str]:
        """Extract object lines from the 7×7×3 MiniGrid image array.

        Returns a list of human-readable strings describing visible objects
        and their relative positions. Called by format_observation_input.
        """
        img: np.ndarray = raw_obs["image"]
        lines: list[str] = []

        for x in range(7):
            for y in range(7):
                obj_idx = img[x, y, 0]
                if obj_idx <= 1:        # skip unseen (0) and empty (1)
                    continue

                dx = x - _AGENT_X
                dy = _AGENT_Y - y

                if dx == 0 and dy == 0:
                    continue

                obj   = OBJECT_TO_STR.get(obj_idx, f"object_{obj_idx}")
                color = COLOR_TO_STR.get(img[x, y, 1], f"color_{img[x, y, 1]}")

                parts: list[str] = []
                if dy > 0:
                    parts.append(f"{dy} step{'s' if dy > 1 else ''} forward")
                if dx != 0:
                    side = "right" if dx > 0 else "left"
                    parts.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} {side}")

                pos = " and ".join(parts) if parts else "at your location"

                if obj_idx == 4:        # door — include open/closed/locked state
                    state = STATE_TO_STR.get(img[x, y, 2], "unknown state")
                    lines.append(f"- {color} {obj} ({state}), {pos}")
                else:
                    lines.append(f"- {color} {obj}, {pos}")

        return lines

    def format_observation_input(self, obs: Any) -> str:
        """Build the user message sent to the Descriptor LLM.

        Combines mission, facing direction, and visible objects into the
        structured input the Descriptor expects for BabyAI environments.
        """
        direction = (
            DIRECTION_NAMES[obs["direction"]]
            if obs["direction"] < len(DIRECTION_NAMES)
            else str(obs["direction"])
        )
        object_lines = self.parse_observation(obs)
        objects_block = (
            "\n".join(object_lines) if object_lines else "- nothing notable visible"
        )
        return (
            f"Mission: {obs['mission']}\n"
            f"Facing: {direction}\n\n"
            f"Visible objects:\n{objects_block}"
        )
