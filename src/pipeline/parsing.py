"""LLM response parsing helpers for the two-module pipeline.

Extracts PLAN and ACTION fields from agent LLM responses and maps
action strings to their indices. Kept separate so the Behaviour Analyser
and other optimisation components can import parsing logic without pulling
in the full pipeline.
"""

from __future__ import annotations

import re

from src.environment.minigrid import ACTION_NAMES

_ACTION_RE = re.compile(
    r"ACTION:\s*(" + "|".join(ACTION_NAMES) + r")",
    re.IGNORECASE,
)


def extract_plan_and_action(response_text: str) -> tuple[str | None, str | None]:
    """Extract PLAN and ACTION fields from a raw LLM response.

    Returns:
        (plan, action) where either may be None if not found.
        action is lowercased if found.
    """
    plan_match = re.search(
        r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL,
    )
    action_match = _ACTION_RE.search(response_text)
    plan = plan_match.group(1).strip() if plan_match else None
    action = action_match.group(1).strip().lower() if action_match else None
    return plan, action


def action_to_index(action: str | None) -> int | None:
    """Map an action string to its index in ACTION_NAMES.

    Tries exact match first, then substring match.
    Returns None if no match found.
    """
    if action is None:
        return None
    action = action.strip().lower()
    for idx, name in enumerate(ACTION_NAMES):
        if action == name:
            return idx
    for idx, name in enumerate(ACTION_NAMES):
        if name in action:
            return idx
    return None
