"""Two-module pipeline orchestrator.

Coordinates the Descriptor LLM and Agent LLM calls for each environment step.
Descriptor produces a mission-focused scene summary; Agent receives that
summary and outputs PLAN + ACTION.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.client import OpenAIClient
from src.environment.minigrid import ACTION_NAMES, DIRECTION_NAMES
from src.pipeline.descriptor import Descriptor
from src.pipeline.parsing import extract_plan_and_action, action_to_index
from src.utils.config import load_prompts

log = logging.getLogger(__name__)

_PROMPTS = load_prompts()
AGENT_SYSTEM_PROMPT = _PROMPTS.get("agent_instructions", "")


class DescriptorAgent:
    """Orchestrates the two-LLM pipeline per environment step.

    Two separate LLM calls happen per step:
      1. Descriptor: raw obs  →  natural language description
      2. Agent:      description  →  PLAN + ACTION

    Both use the same OpenAIClient configuration but maintain completely
    separate conversation histories.
    """

    def __init__(
        self,
        agent_client_factory,
        descriptor_client_factory=None,
        agent_multi_turn: bool = False,
        descriptor_multi_turn: bool = False,
        history_window: int = 16,
        prompt_variant: str = "rich",
        agent_prompt_override: str | None = None,
        descriptor_prompt_override: str | None = None,
    ):
        self.agent_client = agent_client_factory()
        self.descriptor = Descriptor(
            client=(descriptor_client_factory or agent_client_factory)(),
            multi_turn=descriptor_multi_turn,
            history_window=history_window,
            prompt_variant=prompt_variant,
        )
        self.agent_multi_turn = agent_multi_turn
        self.history_window = history_window
        self.prompt_variant = prompt_variant
        self.agent_prompt_override = agent_prompt_override
        self.descriptor_prompt_override = descriptor_prompt_override
        self.plan: Optional[str] = None
        self._conversation: list[dict[str, str]] = []
        self.last_descriptor_output: Optional[str] = None
        self.last_user_prompt: Optional[str] = None
        self.last_llm_response: Optional[str] = None
        self.last_agent_reasoning: Optional[str] = None
        self.last_parse_failed: bool = False
        self.last_descriptor_usage: Optional[dict] = None
        self.last_agent_usage: Optional[dict] = None
        self._last_agent_pos: Optional[tuple[int, int]] = None
        self._last_direction: Optional[str] = None
        self.reset()

    def reset(self):
        """Clear conversation history and plan for a new episode."""
        agent_prompt = self.agent_prompt_override or load_prompts(
            prompt_variant=self.prompt_variant, multi_turn=self.agent_multi_turn
        ).get("agent_instructions", "")
        self._conversation = [{"role": "system", "content": agent_prompt}]
        self.descriptor.reset()
        self.plan = None
        self.last_descriptor_output = None
        self.last_user_prompt = None
        self.last_llm_response = None
        self.last_agent_reasoning = None
        self.last_parse_failed = False
        self.last_descriptor_usage = None
        self.last_agent_usage = None
        self._last_agent_pos = None
        self._last_direction = None

    def act(
        self,
        obs: dict,
        info: dict | None = None,
        prev_action: str | None = None,
        agent_pos: tuple[int, int] | None = None,
    ) -> int:
        """Pick the next action given the current observation.

        Returns:
            int: action index into the environment's action list.
        """
        self._last_agent_pos = agent_pos
        self._last_direction = (
            DIRECTION_NAMES[obs["direction"]]
            if obs["direction"] < len(DIRECTION_NAMES)
            else str(obs["direction"])
        )

        description = self.descriptor.describe(obs, prompt=self.descriptor_prompt_override or None)
        self.last_descriptor_output = description
        self.last_descriptor_usage = self.descriptor.last_usage

        carrying = (info or {}).get("carrying")
        user_msg = self._build_prompt(obs, description, prev_action, carrying)
        self.last_user_prompt = user_msg

        if self.agent_multi_turn:
            self._conversation.append({"role": "user", "content": user_msg})
            messages = self._conversation
        else:
            messages = [self._conversation[0], {"role": "user", "content": user_msg}]

        raw, self.last_agent_reasoning, self.last_agent_usage = self.agent_client.generate_with_reasoning(messages)

        if self.agent_multi_turn:
            self._conversation.append({"role": "assistant", "content": raw})
            max_msgs = 1 + 2 * self.history_window
            if len(self._conversation) > max_msgs:
                self._conversation = [self._conversation[0]] + self._conversation[-2 * self.history_window:]

        self.last_llm_response = raw

        plan, action_str = extract_plan_and_action(raw)
        if plan and plan.lower() != "no changes.":
            self.plan = plan

        action_idx = action_to_index(action_str)
        if action_idx is None:
            log.warning("Could not parse action from LLM response, defaulting to 'go forward': %s", raw)
            action_idx = ACTION_NAMES.index("go forward")
            self.last_parse_failed = True
        else:
            self.last_parse_failed = False

        return action_idx

    def _build_prompt(
        self,
        obs: dict,
        description: str,
        prev_action: str | None,
        carrying: str | None = None,
    ) -> str:
        parts: list[str] = []
        if prev_action is not None and not self.agent_multi_turn:
            parts.append(f"Previous action: {prev_action}")
        parts.append(f"Mission: {obs['mission']}")
        parts.append(f"Carrying: {carrying if carrying else 'nothing'}")
        parts.append(f"\nObservation:\n{description}")
        if self.plan and not self.agent_multi_turn:
            parts.append(f"\nCurrent plan:\n{self.plan}")
        parts.append("\nProvide your PLAN and ACTION.")
        return "\n".join(parts)

    def combined_log_entry(self) -> str:
        """Format descriptor output + agent prompt + reasoning traces for the llm_output log."""
        pos_str = (
            f"({self._last_agent_pos[0]}, {self._last_agent_pos[1]})"
            if self._last_agent_pos is not None else "unknown"
        )
        debug_header = f"[ DEBUG INFO ]\nFacing: {self._last_direction or 'unknown'}  |  Position: {pos_str}"

        descriptor_reasoning = (
            f"\n\n[ DESCRIPTOR REASONING ]\n\n{self.descriptor.last_reasoning}"
            if self.descriptor.last_reasoning else ""
        )
        agent_reasoning = (
            f"\n\n[ AGENT REASONING ]\n\n{self.last_agent_reasoning}"
            if self.last_agent_reasoning else ""
        )

        return (
            f"{debug_header}\n\n"
            "[ DESCRIPTOR OUTPUT ]\n\n"
            f"{self.last_descriptor_output or '(none)'}"
            f"{descriptor_reasoning}\n\n"
            "[ AGENT PROMPT ]\n\n"
            f"{self.last_user_prompt or '(none)'}"
            f"{agent_reasoning}"
        )
