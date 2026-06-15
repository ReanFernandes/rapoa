"""Thin OpenAI-compatible client that targets a local inference server."""

from __future__ import annotations

import logging
import random
import threading
import time
import openai
from openai import OpenAI

from src.utils.config import (
    HLP_API_BASE,
    HLP_API_KEY,
    HLP_MODEL_ID,
    HLP_TEMPERATURE,
    HLP_MAX_TOKENS,
)

log = logging.getLogger(__name__)


class OpenAIClient:
    """Wraps the OpenAI Python SDK, pre-configured for a local endpoint."""

    # When set, all instances use this list instead of their per-instance _endpoints.
    _shared_endpoints: list[str] | None = None
    _shared_endpoints_lock: threading.Lock = threading.Lock()

    # One persistent OpenAI client per endpoint URL — reuses TCP connections.
    _client_cache: dict[str, OpenAI] = {}
    _client_cache_lock: threading.Lock = threading.Lock()

    @classmethod
    def update_endpoints(cls, endpoints: list[str]) -> None:
        """Replace the active endpoint list across all instances."""
        with cls._shared_endpoints_lock:
            cls._shared_endpoints = list(endpoints)
        log.info("Endpoints updated (%d node(s)): %s", len(endpoints), endpoints)
        print(f"[endpoint-watcher] Nodes updated ({len(endpoints)}): {endpoints}")

    @classmethod
    def _get_client(cls, endpoint: str, api_key: str) -> OpenAI:
        """Return a cached OpenAI client for this endpoint, creating one if needed."""
        with cls._client_cache_lock:
            if endpoint not in cls._client_cache:
                cls._client_cache[endpoint] = OpenAI(base_url=endpoint, api_key=api_key)
            return cls._client_cache[endpoint]

    _RETRYABLE = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
        openai.RateLimitError,
    )

    def __init__(
        self,
        base_url: str = HLP_API_BASE,
        api_key: str = HLP_API_KEY,
        model: str = HLP_MODEL_ID,
        temperature: float = HLP_TEMPERATURE,
        max_tokens: int = HLP_MAX_TOKENS,
        max_retries: int = 500,
        endpoints: list[str] | None = None,
        inference_seed: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.inference_seed = inference_seed
        self._endpoints = endpoints or [base_url]
        self._api_key = api_key

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Send a chat-completion request and return the assistant content."""
        content, _, _ = self.generate_with_reasoning(messages)
        return content

    def generate_with_reasoning(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, str | None, dict]:
        """Send a chat-completion request and return (content, reasoning, usage).

        ``reasoning`` is populated from ``message.reasoning`` (LM Studio) or
        ``message.reasoning_content`` (vLLM) if either is present and non-empty,
        otherwise ``None``.

        ``usage`` is a dict with keys:
            prompt_tokens, completion_tokens, finish_reason, latency_s
        Values are None if the server does not return them.
        """
        wait = 0.0

        for attempt in range(self.max_retries):
            with self.__class__._shared_endpoints_lock:
                active = self.__class__._shared_endpoints or self._endpoints
            endpoint = random.choice(active)
            client = self.__class__._get_client(endpoint, self._api_key)

            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.inference_seed,
                )
                break
            except self._RETRYABLE as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = random.uniform(4, 6)
                log.warning(
                    "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.max_retries, wait, e,
                )
            time.sleep(wait)

        latency_s = round(time.perf_counter() - t0, 3)

        msg = response.choices[0].message
        content = msg.content or ""
        raw_reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
        )
        reasoning = raw_reasoning.strip() if raw_reasoning and raw_reasoning.strip() else None

        u = response.usage
        usage = {
            "prompt_tokens":     u.prompt_tokens     if u else None,
            "completion_tokens": u.completion_tokens if u else None,
            "finish_reason":     response.choices[0].finish_reason,
            "latency_s":         latency_s,
        }

        log.debug("LLM response: %s", content)
        return content, reasoning, usage
