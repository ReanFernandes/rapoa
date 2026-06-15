"""Level 1 trajectory preprocessing for the Behaviour Analyser.

Strips structurally redundant fields and formats episode steps into a
compact, LLM-readable text representation. No semantic compression is
applied — the BA receives the full observational content of each step.

Fields stripped per step
------------------------
- raw_direction         : redundant with direction
- agent_finish_reason   : always "stop" in normal operation
- token counts, latency : irrelevant for behaviour attribution
- parse_failed          : folded into an inline flag when True
- blocked_fwd           : folded into an inline flag when True
- mission               : moved to episode header, not repeated
- episode number        : in header

Fields always kept per step
---------------------------
- step number, facing direction, agent position
- scene_text (raw environment observation)
- descriptor_output (descriptor's goal-conditioned summary)
- action taken
- plan
- target_in_grid and target_visible (only when at least one is True
  or they disagree — the gap between them is the Descriptor signal)
- reward on the final step only (always 0 mid-episode)
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Step formatting
# ---------------------------------------------------------------------------

def _format_step(step: dict) -> str:
    lines = []

    # Header line — episode number prefix keeps the BA anchored across multi-episode calls
    ep = step.get("episode")
    step_id = f"E{ep}.S{step['step']}" if ep is not None else f"Step {step['step']}"
    header = (
        f"{step_id}"
        f" | facing {step['direction']}"
        f" | pos: {step['agent_pos']}"
    )

    # target_in_grid is ground truth from the environment (target in raw scene_text).
    # target_visible is an unreliable substring match against descriptor_output —
    # it fires True whenever the target name appears in the text, including when
    # the descriptor correctly reports the target is absent ("blue key not visible").
    # We annotate only on target_in_grid; the BA reads the descriptor text directly
    # to judge whether it handled the target correctly.
    if step.get("target_in_grid"):
        header += " | target in scene"

    lines.append(header)

    # Raw scene text — newlines replaced with " / " for compactness
    scene = step.get("scene_text", "").replace("\n", " / ")
    lines.append(f"  Raw scene: {scene}")

    # Descriptor output — absent for without_descriptor pipeline
    descriptor = step.get("descriptor_output")
    if descriptor is not None:
        descriptor_clean = " ".join(descriptor.split())
        lines.append(f"  Descriptor: {descriptor_clean}")

    # Action line with inline flags
    action = step.get("action", "unknown")
    flags = []
    if step.get("parse_failed"):
        flags.append("PARSE FAILED")
    if step.get("blocked_fwd"):
        flags.append("BLOCKED")
    action_line = f"  Action: {action}"
    if flags:
        action_line += f"  [{', '.join(flags)}]"
    lines.append(action_line)

    # Plan
    plan = step.get("plan")
    if plan:
        lines.append(f"  Plan: {plan}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

def _extract_outcome(steps: list[dict], summary: dict | None) -> tuple[bool, float, int]:
    """Return (success, reward, total_steps) for an episode."""
    if summary:
        return (
            bool(summary.get("success", False)),
            float(summary.get("total_reward", 0.0)),
            int(summary.get("total_steps", len(steps))),
        )
    if not steps:
        return False, 0.0, 0
    last = steps[-1]
    success = bool(last.get("terminated") and float(last.get("reward", 0)) > 0)
    return success, float(last.get("reward", 0.0)), len(steps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_episode(
    steps: list[dict],
    summary: dict | None = None,
    episode_num: int | None = None,
) -> str:
    """Apply Level 1 compression to one episode.

    Args:
        steps:       Step records for the episode from trajectory.jsonl.
        summary:     Optional summary record (type='summary').
        episode_num: Episode number used in the boundary marker.

    Returns:
        Compressed text representation ready for inclusion in a BA prompt.
    """
    if not steps:
        return ""

    mission = steps[0].get("mission", "unknown")
    success, reward, total_steps = _extract_outcome(steps, summary)

    ep_label = f"Episode {episode_num}" if episode_num is not None else "Episode"
    outcome_tag = (
        f"SUCCESS ({total_steps} steps)"
        if success
        else f"FAILED (truncated at {total_steps} steps)"
    )

    sep = "=" * 20
    lines = [
        f"{sep} {ep_label} — {outcome_tag} {sep}",
        f"Mission: {mission}",
        "Steps below are causally independent from all prior episodes.",
        "",
    ]

    for step in steps:
        lines.append(_format_step(step))
        lines.append("")

    # Reward appears on the terminal marker only — always 0 mid-episode
    terminal = "TERMINATED" if success else "TRUNCATED"
    lines.append(f"{terminal} | reward: {reward:.3f}")

    return "\n".join(lines)


def load_episode_groups(
    trajectory_path: Path,
) -> dict[int, tuple[list[dict], dict | None]]:
    """Read trajectory.jsonl and group records by episode number.

    Returns:
        Mapping {episode_num: (step_records, summary_or_None)},
        keyed in ascending episode order.
    """
    steps_by_ep: dict[int, list[dict]] = {}
    summary_by_ep: dict[int, dict] = {}

    with open(trajectory_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ep = record.get("episode", 0)
            if record.get("type") == "summary":
                summary_by_ep[ep] = record
            else:
                steps_by_ep.setdefault(ep, []).append(record)

    all_eps = sorted(set(steps_by_ep) | set(summary_by_ep))
    return {
        ep: (steps_by_ep.get(ep, []), summary_by_ep.get(ep))
        for ep in all_eps
    }


def compress_for_ba(
    trajectory_path: Path,
    episode_indices: list[int] | None = None,
) -> str:
    """Compress selected episodes from a trajectory file into BA input.

    Args:
        trajectory_path: Path to trajectory.jsonl.
        episode_indices: 1-indexed episode numbers to include.
                         None includes all episodes.

    Returns:
        Concatenated compressed text of all selected episodes.
    """
    groups = load_episode_groups(trajectory_path)

    selected = (
        [ep for ep in sorted(episode_indices) if ep in groups]
        if episode_indices is not None
        else sorted(groups)
    )

    parts = [
        compress_episode(groups[ep][0], groups[ep][1], episode_num=ep)
        for ep in selected
    ]
    return "\n\n".join(p for p in parts if p)
