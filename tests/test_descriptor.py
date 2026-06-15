"""Tests for src/perception/descriptor.py — Descriptor."""

import pytest

from src.pipeline.descriptor import Descriptor


class _FakeClient:
    def __init__(self, response: str):
        self._response = response
        self.last_messages = None

    def generate(self, messages):
        self.last_messages = messages
        return self._response

    def generate_with_reasoning(self, messages):
        self.last_messages = messages
        return self._response, None, {"prompt_tokens": None, "completion_tokens": None, "finish_reason": "stop", "latency_s": 0.0}


def _make_obs(
    mission: str = "go to the red ball",
    scene_text: str = "a red ball 1 step forward",
    direction: int = 0,
) -> dict:
    return {
        "text": {"long_term_context": scene_text, "short_term_context": ""},
        "mission": mission,
        "direction": direction,
    }


def test_describe_sends_system_prompt():
    client = _FakeClient("Red ball ahead.")
    d = Descriptor(client=client)
    d.describe(_make_obs(), prompt="TEST_PROMPT")
    assert client.last_messages[0]["role"] == "system"
    assert client.last_messages[0]["content"] == "TEST_PROMPT"


def test_describe_user_message_contains_mission():
    client = _FakeClient("ok")
    d = Descriptor(client=client)
    d.describe(_make_obs(mission="go to the blue key"))
    user_content = client.last_messages[1]["content"]
    assert "go to the blue key" in user_content


def test_describe_user_message_contains_scene_text():
    client = _FakeClient("ok")
    d = Descriptor(client=client)
    d.describe(_make_obs(scene_text="a red ball 3 steps forward and 1 step right"))
    user_content = client.last_messages[1]["content"]
    assert "red ball" in user_content


def test_describe_returns_llm_output():
    client = _FakeClient("Scene: red ball 1 step forward.")
    d = Descriptor(client=client)
    result = d.describe(_make_obs())
    assert result == "Scene: red ball 1 step forward."
