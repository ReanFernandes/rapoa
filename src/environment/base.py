"""Abstract base class for environment adapters.

An adapter is the bridge between a BALROG-wrapped environment and the SPA
pipeline. BALROG normalises all environments to a common observation shape
(obs["text"]["long_term_context"], obs["image"], env.language_action_space).
The adapter adds the SPA-specific layer on top:

  - action_names              : ordered list of valid action strings, used for
                                action parsing and prompt construction
  - direction_names           : ordered list of direction name strings indexed
                                by obs["direction"]
  - format_observation_input  : builds the exact user message sent to the
                                Descriptor LLM from the normalised BALROG obs
  - get_mission(obs)          : extracts the goal/mission string from obs
  - get_scene_text(obs)       : extracts the primary scene description text
  - get_inventory_text(env)   : extracts carrying/inventory as a display string

To add a new environment:
  1. Ensure a BALROG wrapper exists for it (produces obs["text"][...])
  2. Subclass BaseEnvironmentAdapter
  3. Implement the abstract members below (and override optional ones as needed)
  4. Write env and task configs for the new environment
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironmentAdapter(ABC):

    @property
    @abstractmethod
    def action_names(self) -> list[str]:
        """Ordered list of valid action name strings for this environment.

        Index order must match the environment's integer action enum so that
        ``env.step(action_names[i])`` and ``env.step(i)`` are equivalent.
        Used by the agent prompt and by the action parser to map LLM output
        to an integer action index.
        """
        pass

    @property
    @abstractmethod
    def direction_names(self) -> list[str]:
        """Ordered list of direction name strings indexed by obs["direction"].

        For BabyAI/MiniGrid: ["right", "down", "left", "up"] (4 directions).
        For NetHack-style envs: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"].
        """
        pass

    @abstractmethod
    def format_observation_input(self, obs: Any) -> str:
        """Build the user message sent to the Descriptor LLM.

        Receives the full observation dict after BALROG's wrapper has
        normalised it. Should combine whichever fields are relevant for
        this environment into a single string the Descriptor can reason over.

        Args:
            obs: Normalised observation dict. Always contains at minimum:
                 obs["text"]["long_term_context"] (str) — scene description
                 obs["mission"] (str)              — goal string (BabyAI)

        Returns:
            A single string forming the user message for the Descriptor LLM.
        """
        pass

    def get_mission(self, obs: Any) -> str:
        """Extract the mission/goal string from an observation.

        Default: reads obs["mission"]. Override for environments that store
        the goal elsewhere or use a different key.
        """
        return obs.get("mission", "")

    def get_scene_text(self, obs: Any) -> str:
        """Extract the primary scene description text from an observation.

        Default: reads obs["text"]["long_term_context"] (BALROG standard).
        Override if the environment surfaces scene text at a different path.
        """
        return obs.get("text", {}).get("long_term_context", "")

    def get_inventory_text(self, env: Any) -> str | None:
        """Extract the current carrying/inventory state as a display string.

        Default: returns None (no inventory concept). Override for environments
        with carrying or inventory mechanics.

        Args:
            env: The live environment object (post-step or post-reset).

        Returns:
            Human-readable inventory string, or None if not applicable.
        """
        return None
