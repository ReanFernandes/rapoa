"""Adapter that wraps OpenAIClient to satisfy BALROG's client interface.

BALROG agents call ``client.generate(messages)`` where ``messages`` is a list
of ``balrog.prompt_builder.history.Message`` objects, and expect an
``LLMResponse`` namedtuple back.  Our ``OpenAIClient`` speaks plain dicts and
returns ``(content, reasoning, usage)``.  This adapter bridges the two without
modifying either side.

Image attachments on Message objects are silently dropped — our inference
server is text-only.
"""

from __future__ import annotations

from balrog.client import LLMResponse

from src.llm.client import OpenAIClient


class BALROGClientAdapter:
    """Wraps OpenAIClient to satisfy BALROG's ``generate(messages)`` interface."""

    def __init__(self, client: OpenAIClient):
        self._client = client
        self.last_messages: list[dict] = []
        self.last_reasoning: str | None = None  # LM Studio reasoning, saved before BALROG overwrites it
        self.last_usage: dict = {}

    def generate(self, messages) -> LLMResponse:
        dict_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        self.last_messages = dict_messages
        content, reasoning, usage = self._client.generate_with_reasoning(dict_messages)
        self.last_reasoning = reasoning  # save before BALROG's _extract_final_answer overwrites LLMResponse.reasoning
        self.last_usage = usage or {}
        usage = self.last_usage
        return LLMResponse(
            model_id=self._client.model,
            completion=content or "",
            stop_reason=usage.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
            reasoning=reasoning,
        )
