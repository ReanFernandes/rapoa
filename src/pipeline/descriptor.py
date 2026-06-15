"""Descriptor LLM module.

Converts the rule-based scene text (from BALROG's BabyAI wrapper) into a
mission-focused natural language summary for the Agent LLM.
"""

from __future__ import annotations

import logging

from src.llm.client import OpenAIClient
from src.utils.config import load_prompts

log = logging.getLogger(__name__)

_PROMPTS = load_prompts()
DESCRIPTOR_SYSTEM_PROMPT = _PROMPTS.get("descriptor_instructions", "")


class Descriptor:
    """Calls a local LLM to produce a contextualised scene description.

    Receives the rule-based scene text from the environment wrapper
    (``obs["text"]["long_term_context"]``) and asks the LLM to interpret
    it in terms of the mission.
    """

    def __init__(self, client: OpenAIClient | None = None, multi_turn: bool = False, history_window: int = 16, prompt_variant: str = "rich"):
        self.client = client or OpenAIClient()
        self.multi_turn = multi_turn
        self.history_window = history_window
        self.prompt_variant = prompt_variant
        self._conversation: list[dict[str, str]] = []
        self.last_reasoning: str | None = None
        self.last_usage: dict | None = None

    def reset(self):
        """Clear conversation history and reasoning trace between episodes."""
        self._conversation = []
        self.last_reasoning = None
        self.last_usage = None

    def describe(self, obs: dict, prompt: str | None = None) -> str:
        """Return a natural language description of the observation.

        Args:
            obs:    Observation dict containing at minimum:
                    ``text["long_term_context"]`` (str) — rule-based scene text,
                    and ``mission`` (str).
            prompt: System prompt override. If None, reloads from config each
                    call so prompt updates are picked up without a restart.

        Returns:
            Natural language description produced by the LLM.
        """
        if prompt is None:
            prompt = load_prompts(prompt_variant=self.prompt_variant).get("descriptor_instructions", "")
        user_msg = self._format_input(obs)
        system_msg = {"role": "system", "content": prompt}
        if self.multi_turn:
            if not self._conversation:
                self._conversation.append(system_msg)
            self._conversation.append({"role": "user", "content": user_msg})
            messages = self._conversation
        else:
            messages = [system_msg, {"role": "user", "content": user_msg}]
        response, self.last_reasoning, self.last_usage = self.client.generate_with_reasoning(messages)
        if self.multi_turn:
            self._conversation.append({"role": "assistant", "content": response})
            max_msgs = 1 + 2 * self.history_window
            if len(self._conversation) > max_msgs:
                self._conversation = [self._conversation[0]] + self._conversation[-2 * self.history_window:]
        return response

    @staticmethod
    def _format_input(obs: dict) -> str:
        """Build the user message sent to the descriptor LLM."""
        scene_text = obs["text"]["long_term_context"]
        return (
            f"Mission: {obs['mission']}\n\n"
            f"Scene:\n{scene_text}"
        )
