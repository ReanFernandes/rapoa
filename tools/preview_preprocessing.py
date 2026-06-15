"""Preview the Level 1 preprocessing output for a trajectory file.

Shows the compressed text that the Behaviour Analyser will receive,
along with a token estimate and episode summary.

Usage
-----
# All episodes in a run directory
python tools/preview_preprocessing.py --run-dir logs/BabyAI/goto/.../

# Specific episodes
python tools/preview_preprocessing.py --run-dir logs/BabyAI/goto/.../ --episodes 1 3 5

# Pipe to less for long output
python tools/preview_preprocessing.py --run-dir logs/BabyAI/goto/.../ | less
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.preprocessing import compress_for_ba, load_episode_groups


def main():
    parser = argparse.ArgumentParser(description="Preview BA preprocessing output")
    parser.add_argument("--run-dir", required=True, help="Run directory containing trajectory.jsonl")
    parser.add_argument("--episodes", nargs="*", type=int, default=None,
                        help="Episode numbers to include (default: all)")
    parser.add_argument("--no-text", action="store_true",
                        help="Print summary only, skip the compressed text")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    trajectory_path = run_dir / "trajectory.jsonl"
    if not trajectory_path.exists():
        # search one level deeper (run_dir might be the session root)
        candidates = list(run_dir.rglob("trajectory.jsonl"))
        if not candidates:
            print(f"No trajectory.jsonl found under {run_dir}", file=sys.stderr)
            sys.exit(1)
        if len(candidates) > 1:
            print(f"Multiple trajectory.jsonl found — specify a single run directory:", file=sys.stderr)
            for c in candidates:
                print(f"  {c}", file=sys.stderr)
            sys.exit(1)
        trajectory_path = candidates[0]

    groups = load_episode_groups(trajectory_path)
    all_eps = sorted(groups)
    selected = args.episodes if args.episodes else all_eps

    # Summary table
    print(f"Trajectory: {trajectory_path}")
    print(f"Episodes in file: {len(all_eps)}  |  Selected: {len(selected)}")
    print()
    print(f"{'Ep':>4}  {'Outcome':10}  {'Steps':>6}  {'Reward':>8}")
    print("-" * 36)
    for ep in all_eps:
        _, summary = groups[ep]
        if summary:
            marker = ">>>" if ep in selected else "   "
            outcome = "SUCCESS" if summary.get("success") else "failed"
            print(f"{marker}{ep:>3}  {outcome:10}  {summary.get('total_steps', '?'):>6}  {summary.get('total_reward', 0):>8.3f}")
    print()

    if args.no_text:
        return

    text = compress_for_ba(trajectory_path, episode_indices=selected if args.episodes else None)
    approx_tokens = len(text) // 4

    print(f"Compressed output — {len(text):,} chars  (~{approx_tokens:,} tokens)")
    print("=" * 70)
    print(text)


if __name__ == "__main__":
    main()
