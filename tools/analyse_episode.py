"""Run the Behaviour Analyser on a completed run directory.

Loads the trajectory, applies Level 1 preprocessing, calls the BA,
and prints the structured output. Use this to manually verify that
the BA's attributions match your own reading of the trajectory before
running the full optimisation loop.

The LLM server must be reachable (same setup as running experiments).

Usage
-----
# Analyse all episodes in a run (random sample up to 5)
python tools/analyse_episode.py --run-dir logs/BabyAI/goto/.../

# Analyse specific episodes
python tools/analyse_episode.py --run-dir logs/BabyAI/goto/.../ --episodes 3 7 12

# Print the full prompt sent to the BA (without calling the LLM)
python tools/analyse_episode.py --run-dir logs/BabyAI/goto/.../ --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import OpenAIClient
from src.llm.server_utils import find_all_gpu_servers
from src.optimization.behaviour_analyser import BehaviourAnalyser
from src.optimization.preprocessing import compress_for_ba, load_episode_groups
from src.utils.config import load_prompts, HLP_MODEL_ID

_MAX_EPISODES = 5
_SEED = 42


def _find_trajectory(run_dir: Path) -> Path:
    direct = run_dir / "trajectory.jsonl"
    if direct.exists():
        return direct
    candidates = list(run_dir.rglob("trajectory.jsonl"))
    if not candidates:
        print(f"No trajectory.jsonl found under {run_dir}", file=sys.stderr)
        sys.exit(1)
    if len(candidates) > 1:
        print("Multiple trajectory.jsonl found — specify a single run directory:", file=sys.stderr)
        for c in candidates:
            print(f"  {c}", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def _load_run_config(run_dir: Path) -> dict:
    for directory in [run_dir, *run_dir.parents]:
        cfg = directory / "run_config.json"
        if cfg.exists():
            return json.loads(cfg.read_text())
    return {}


def _select_episodes(groups: dict, requested: list[int] | None) -> list[int]:
    """Return episode indices to pass to the BA."""
    if requested:
        missing = [ep for ep in requested if ep not in groups]
        if missing:
            print(f"Warning: episodes not found in trajectory: {missing}", file=sys.stderr)
        return [ep for ep in requested if ep in groups]

    # Random sample up to _MAX_EPISODES (successes first to fill cheaply, then failures)
    successes = [ep for ep, (_, s) in groups.items() if s and s.get("success")]
    failures  = [ep for ep, (_, s) in groups.items() if s and not s.get("success")]

    rng = random.Random(_SEED)
    selected = successes[:]
    rng.shuffle(failures)
    for ep in failures:
        if len(selected) >= _MAX_EPISODES:
            break
        selected.append(ep)
    return sorted(selected[:_MAX_EPISODES])


def main():
    parser = argparse.ArgumentParser(description="Run the Behaviour Analyser on a run directory")
    parser.add_argument("--run-dir",  required=True, help="Run directory (must contain trajectory.jsonl)")
    parser.add_argument("--episodes", nargs="*", type=int, default=None,
                        help="Episode numbers to analyse (default: random sample up to 5)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print the prompt sent to the BA without calling the LLM")
    parser.add_argument("--model",    default=None, help="Override model ID")
    parser.add_argument("--no-save",  action="store_true",
                        help="Do not save output to ~/BA-analyses-traces")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    trajectory_path = _find_trajectory(run_dir)

    # Infer prompt variant and task from run_config.json
    cfg = _load_run_config(run_dir)
    prompt_variant = cfg.get("prompt_variant", "rich")
    env_id = cfg.get("env", "")
    task = None
    if "/" in env_id:
        task = env_id.split("/", 1)[1]

    prompts = load_prompts(env_family="babyai", task=task, prompt_variant=prompt_variant)

    groups = load_episode_groups(trajectory_path)
    selected = _select_episodes(groups, args.episodes)

    if not selected:
        print("No episodes to analyse.", file=sys.stderr)
        sys.exit(1)

    # Episode summary
    print(f"Run:      {trajectory_path}")
    print(f"Variant:  {prompt_variant}  |  Task: {task or 'unknown'}")
    print(f"Episodes: {selected}")
    print()
    for ep in selected:
        _, summary = groups[ep]
        if summary:
            outcome = "SUCCESS" if summary.get("success") else "failed"
            print(f"  Episode {ep:>3}: {outcome} ({summary.get('total_steps')} steps, reward {summary.get('total_reward', 0):.3f})")
    print()

    # Build the BA
    if args.dry_run:
        ba = BehaviourAnalyser(client=None, pipeline_mode="with_descriptor")  # type: ignore
        compressed = compress_for_ba(trajectory_path, selected)
        messages = ba._build_messages(
            compressed_trajectory=compressed,
            agent_prompt=prompts.get("agent_instructions", ""),
            env_prompt=prompts.get("environment_layer", ""),
            descriptor_prompt=prompts.get("descriptor_instructions"),
            task_prompt=prompts.get("task_layer"),
            hereditary_context=None,
        )
        for msg in messages:
            print(f"=== [{msg['role'].upper()}] ===")
            print(msg["content"])
            print()
        approx_tokens = sum(len(m["content"]) // 4 for m in messages)
        print(f"--- Approx total prompt tokens: ~{approx_tokens:,} ---")
        return

    # Live LLM call
    cluster_endpoints = find_all_gpu_servers()
    endpoints = cluster_endpoints if cluster_endpoints else None
    model = args.model or ("gpt-oss-20b" if cluster_endpoints else HLP_MODEL_ID)
    # BA calls need more completion budget than agent calls — staged reasoning
    # produces substantial visible chain-of-thought before the output block.
    client = OpenAIClient(model=model, endpoints=endpoints, max_tokens=32768)
    ba = BehaviourAnalyser(client=client, pipeline_mode="with_descriptor")

    print("Calling BA...")
    output = ba.analyse_from_trajectory(
        trajectory_path=trajectory_path,
        episode_indices=selected,
        agent_prompt=prompts.get("agent_instructions", ""),
        env_prompt=prompts.get("environment_layer", ""),
        descriptor_prompt=prompts.get("descriptor_instructions"),
        task_prompt=prompts.get("task_layer"),
    )

    lines = _format_output(output)
    result_text = "\n".join(lines)
    print(result_text)

    if not args.dry_run and not args.no_save:
        _save_output(result_text, task or "unknown", selected, output)


def _format_output(output) -> list[str]:
    lines = ["=" * 60]
    if output.prompt_tokens is not None:
        total = (output.prompt_tokens or 0) + (output.completion_tokens or 0)
        reason = output.finish_reason or "?"
        trunc_warn = "  ⚠ HIT MAX_TOKENS" if reason == "length" else ""
        lines.append(
            f"Tokens:  {output.prompt_tokens:,} prompt + "
            f"{output.completion_tokens:,} completion = {total:,} total"
            f"  |  {output.latency_s:.1f}s  |  finish={reason}{trunc_warn}"
        )
        lines.append("")
    lines.append(f"TYPE:    {output.output_type.upper()}")
    if output.implicated_module:
        lines.append(f"MODULE:  {output.implicated_module}")
    if output.failure_step is not None:
        lines.append(f"STEP:    {output.failure_step}")
    if output.change_type:
        lines.append(f"CHANGE:  {output.change_type.upper()}")
    if output.skip_reason:
        lines.append(f"SKIP_REASON: {output.skip_reason}")
    if output.parse_error:
        lines.append(f"PARSE ERROR: {output.parse_error}")
    if output.location:
        lines += ["", "LOCATION:", output.location]
    lines += ["", "CHARACTERISATION:", output.characterisation]
    if output.suggested_change:
        lines += ["", "SUGGESTED CHANGE:", output.suggested_change]
    if output.candidate_queue:
        lines += ["", "ADDITIONAL CANDIDATES:"]
        for i, c in enumerate(output.candidate_queue, 1):
            step_str = str(c.step) if c.step is not None else "none"
            lines.append(f"  {i}. [{c.module}] step {step_str}"
                         + (f"  [{c.change_type}]" if c.change_type else ""))
            if c.location:
                lines.append(f"     Location: {c.location}")
            lines.append(f"     Characterisation: {c.description}")
            if c.suggested_change:
                lines.append(f"     Suggested change: {c.suggested_change}")
    if output.raw_reasoning:
        lines += ["", "--- REASONING TRACE ---", output.raw_reasoning]
    raw_label = "--- RAW RESPONSE (parse failed) ---" if output.parse_error else "--- RAW RESPONSE ---"
    lines += ["", raw_label, output.raw_response]
    lines.append("=" * 60)
    return lines


def _save_output(text: str, task: str, episodes: list, output) -> None:
    save_dir = Path.home() / "BA-analyses-traces"
    save_dir.mkdir(exist_ok=True)
    ep_str = "-".join(str(e) for e in episodes)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"{task}_ep{ep_str}_{output.output_type}_{ts}.txt"
    path = save_dir / fname
    path.write_text(text)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
