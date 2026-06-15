"""Replay a recorded episode and save it as an animated GIF.

Reads actions from trajectory.jsonl, re-runs the environment with the
original seed, and captures RGB frames using MiniGrid's built-in renderer.

Usage:
    # Most recent run, episode 1:
    python scripts/animate_episode.py

    # Specific run and episode:
    python scripts/animate_episode.py --run logs/with_descriptor/2026-03-17_15-00-00_BabyAI-GoToRedBall-v0 --episode 2

    # Control speed (ms per frame):
    python scripts/animate_episode.py --episode 1 --frame-ms 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Animate a recorded episode as a GIF")
    parser.add_argument("--run",      default=None, help="Path to run directory. Defaults to most recent.")
    parser.add_argument("--log-dir",  default="logs/with_descriptor")
    parser.add_argument("--episode",  type=int, default=1, help="Episode number to animate")
    parser.add_argument("--frame-ms", type=int, default=400, help="Milliseconds per frame")
    parser.add_argument("--out",      default=None, help="Output GIF path. Defaults to <run_dir>/episode_<N>.gif")
    return parser.parse_args()


def most_recent_run(base: Path) -> Path:
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [r for r in runs if r.is_dir() and (r / "trajectory.jsonl").exists()]
    if not runs:
        raise FileNotFoundError(f"No runs with trajectory.jsonl found in {base}")
    return runs[0]


def load_episode(run_dir: Path, episode: int) -> tuple[dict, list[str]]:
    """Return (run_config, list_of_action_names) for the given episode."""
    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    trajectory = [
        json.loads(line)
        for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
        if line.strip()
    ]

    actions = [
        e["action"]
        for e in trajectory
        if e.get("episode") == episode and e.get("type") != "summary"
    ]

    if not actions:
        raise ValueError(f"No steps found for episode {episode} in {run_dir / 'trajectory.jsonl'}")

    return cfg, actions


ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def replay(env_id: str, seed: int, actions: list[str], frame_ms: int) -> list[Image.Image]:
    """Replay actions in the environment and return a list of PIL frames."""
    minigrid.register_minigrid_envs()
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)

    frames: list[Image.Image] = []

    def _stamp_step(frame: Image.Image, step: int) -> Image.Image:
        frame = frame.copy()
        draw = ImageDraw.Draw(frame)
        try:
            font = ImageFont.load_default(size=14)
        except TypeError:
            font = ImageFont.load_default()
        label = f"Step {step}"
        x, y = 4, 4
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + dx, y + dy), label, fill=(0, 0, 0), font=font)
        draw.text((x, y), label, fill=(255, 255, 255), font=font)
        return frame

    # Capture initial frame
    frames.append(_stamp_step(Image.fromarray(env.render()), 0))

    for step, action_name in enumerate(actions, start=1):
        action_idx = ACTION_NAMES.index(action_name)
        _, _, terminated, truncated, _ = env.step(action_idx)
        frames.append(_stamp_step(Image.fromarray(env.render()), step))
        if terminated or truncated:
            break

    env.close()
    return frames


def save_gif(frames: list[Image.Image], out_path: Path, frame_ms: int):
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
    )
    print(f"Saved {len(frames)}-frame GIF → {out_path}")


def main():
    args = parse_args()

    run_dir = Path(args.run) if args.run else most_recent_run(Path(args.log_dir))
    cfg, actions = load_episode(run_dir, args.episode)

    env_id = cfg.get("env", "BabyAI-GoToRedBall-v0")
    seed   = cfg.get("seed", 42) + (args.episode - 1)

    print(f"Run:     {run_dir}")
    print(f"Episode: {args.episode}  |  Env: {env_id}  |  Seed: {seed}  |  Steps: {len(actions)}")

    frames = replay(env_id, seed, actions, args.frame_ms)

    out_path = Path(args.out) if args.out else run_dir / f"episode_{args.episode:03d}.gif"
    save_gif(frames, out_path, args.frame_ms)


if __name__ == "__main__":
    main()
