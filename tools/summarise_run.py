"""Print a compact trajectory summary for a descriptor pipeline run.

Reads trajectory.jsonl from a run directory and prints one line per step,
making it easy to see whether the agent was exploring, looping, or making
progress — without reading through verbose LLM logs.

Usage:
    # Most recent run:
    python scripts/summarise_run.py

    # Specific run:
    python scripts/summarise_run.py --run logs/with_descriptor/2026-03-17_14-28-59_BabyAI-GoToRedBall-v0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DIRECTION_ARROWS = {"right": "→", "down": "↓", "left": "←", "up": "↑"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compact trajectory summary")
    parser.add_argument(
        "--run", default=None,
        help="Path to run directory. Defaults to the most recent run in logs/with_descriptor/.",
    )
    parser.add_argument(
        "--log-dir", default="logs/with_descriptor",
        help="Base log directory to search for the most recent run.",
    )
    return parser.parse_args()


def most_recent_run(base: Path) -> Path:
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [r for r in runs if r.is_dir() and (r / "trajectory.jsonl").exists()]
    if not runs:
        raise FileNotFoundError(f"No runs with trajectory.jsonl found in {base}")
    return runs[0]


def load_run_config(run_dir: Path) -> dict:
    cfg = run_dir / "run_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text())
    return {}


def load_trajectory(run_dir: Path) -> list[dict]:
    path = run_dir / "trajectory.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarise(run_dir: Path):
    cfg    = load_run_config(run_dir)
    events = load_trajectory(run_dir)

    steps    = [e for e in events if "step" in e and e.get("type") != "summary"]
    summaries = {e["episode"]: e for e in events if e.get("type") == "summary"}

    model   = cfg.get("model") or "unknown model"
    env     = cfg.get("env") or run_dir.name
    n_ep    = cfg.get("episodes", max((s["episode"] for s in steps), default=0))

    print(f"\nRun:    {run_dir}")
    print(f"Model:  {model}   Env: {env}   Episodes: {n_ep}")

    by_episode: dict[int, list[dict]] = {}
    for s in steps:
        by_episode.setdefault(s["episode"], []).append(s)

    total_successes = 0
    total_reward    = 0.0

    for ep_num in sorted(by_episode.keys()):
        ep_steps = by_episode[ep_num]
        ep_summary = summaries.get(ep_num, {})
        success = ep_summary.get("success", False)
        n_steps = ep_summary.get("total_steps", len(ep_steps))
        reward  = ep_summary.get("total_reward", 0.0)

        total_successes += int(success)
        total_reward    += reward

        result = "SUCCESS" if success else "FAILED "
        print(f"\n  Episode {ep_num} — {result}  ({n_steps} steps)")

        action_counts = Counter(s["action"] for s in ep_steps)
        prev_pos  = None
        loops     = 0

        for s in ep_steps:
            pos    = tuple(s["agent_pos"])
            arrow  = DIRECTION_ARROWS.get(s["direction"], "?")
            action = s["action"]
            vis    = "●" if s["target_visible"] else "·"

            # Detect no-movement (same position as previous step)
            stuck = ""
            if prev_pos is not None and pos == prev_pos and action == "forward":
                loops += 1
                stuck = " ← blocked"

            print(
                f"    step {s['step']:3d}  ({pos[0]:2d},{pos[1]:2d}){arrow}"
                f"  {action:<8s}  {vis}{stuck}"
            )
            prev_pos = pos

        # Per-episode stats
        dominant = action_counts.most_common(1)[0]
        target_steps = sum(1 for s in ep_steps if s["target_visible"])
        print(
            f"    ─── dominant action: {dominant[0]} ({dominant[1]}×)"
            f"  |  target visible: {target_steps}/{n_steps} steps"
            f"  |  blocked fwd: {loops}"
        )

    print(f"\n  Overall: {total_successes}/{n_ep} success"
          f"  avg reward: {total_reward / max(n_ep, 1):.4f}\n")


def main():
    args = parse_args()
    run_dir = Path(args.run) if args.run else most_recent_run(Path(args.log_dir))
    summarise(run_dir)


if __name__ == "__main__":
    main()
