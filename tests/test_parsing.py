"""Tests for response parsing helpers.

These functions are critical to the pipeline — a broken parser means every
action silently defaults to 'turn left'. They live in src/pipeline/parsing.py.
"""

import pytest

from src.pipeline.parsing import extract_plan_and_action as _extract_plan_and_action, action_to_index as _action_to_index


# ---------------------------------------------------------------------------
# _extract_plan_and_action
# ---------------------------------------------------------------------------

def test_extracts_plan_and_action():
    response = "PLAN: move toward the red ball\nACTION: go forward"
    plan, action = _extract_plan_and_action(response)
    assert plan == "move toward the red ball"
    assert action == "go forward"


def test_extracts_multiline_plan():
    response = "PLAN: step 1: turn right\nstep 2: go forward\nACTION: turn right"
    plan, action = _extract_plan_and_action(response)
    assert "step 1" in plan
    assert "step 2" in plan
    assert action == "turn right"


def test_action_only_no_plan():
    response = "ACTION: pick up"
    plan, action = _extract_plan_and_action(response)
    assert plan is None
    assert action == "pick up"


def test_plan_only_no_action():
    response = "PLAN: explore the room"
    plan, action = _extract_plan_and_action(response)
    assert plan == "explore the room"
    assert action is None


def test_neither_present():
    response = "I think I should go forward."
    plan, action = _extract_plan_and_action(response)
    assert plan is None
    assert action is None


def test_case_insensitive_action():
    response = "PLAN: done\naction: GO FORWARD"
    plan, action = _extract_plan_and_action(response)
    assert action == "go forward"


def test_action_with_surrounding_text():
    # LLM wraps the action in a sentence — should still extract it
    response = "PLAN: navigate\nACTION: go forward (to reach the target)"
    plan, action = _extract_plan_and_action(response)
    assert action == "go forward"


def test_plan_no_changes_preserved():
    # "No changes." is handled upstream — the parser should still return it
    response = "PLAN: No changes.\nACTION: turn left"
    plan, action = _extract_plan_and_action(response)
    assert plan == "No changes."
    assert action == "turn left"


# ---------------------------------------------------------------------------
# _action_to_index
# ---------------------------------------------------------------------------

def test_exact_match_all_actions():
    from src.environment.minigrid import ACTION_NAMES
    for idx, name in enumerate(ACTION_NAMES):
        assert _action_to_index(name) == idx


def test_none_input_returns_none():
    assert _action_to_index(None) is None


def test_unknown_action_returns_none():
    assert _action_to_index("fly") is None


def test_partial_match():
    # If the LLM returns e.g. "go forward please", partial match should work
    assert _action_to_index("go forward please") == 2  # index of "go forward"


def test_case_insensitive_exact():
    assert _action_to_index("TURN LEFT") == 0
    assert _action_to_index("Turn Right") == 1
