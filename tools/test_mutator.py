"""Run the BA then the Mutator on a completed run directory.

Produces a full trace saved to ~/ba_mutator_traces/ containing:
  - Run metadata
  - Compressed trajectory sent to the BA
  - Full BA prompt (system + user messages)
  - BA parsed output, reasoning trace, raw response, token metrics
  - Full Mutator prompt (system + user messages)
  - Mutator parsed output, diff, revised prompt, reasoning trace, raw response, token metrics

Use this to validate the BA → Mutator pipeline before running the full
optimisation loop, and as the authoritative log whenever the agent runs
experiments autonomously.

Both the BA and Mutator LLM servers must be reachable.

Usage
-----
# Run BA + Mutator on a run directory (random sample up to 5 episodes)
python tools/test_mutator.py --run-dir logs/BabyAI/open/.../

# Specific episodes
python tools/test_mutator.py --run-dir logs/BabyAI/open/.../ --episodes 3 7 14

# Print the full prompt sent to the Mutator without calling the LLM
python tools/test_mutator.py --run-dir logs/BabyAI/open/.../ --dry-run

# Do not save output (print only)
python tools/test_mutator.py --run-dir logs/BabyAI/open/.../ --no-save
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import OpenAIClient
from src.llm.server_utils import find_all_gpu_servers
from src.optimization.behaviour_analyser import BehaviourAnalyser, BAOutput
from src.optimization.mutator import Mutator, MutatorOutput
from src.optimization.preprocessing import compress_for_ba, load_episode_groups
from src.utils.config import load_prompts, HLP_MODEL_ID

_MAX_EPISODES = 5
_SEED = 42
_MAX_RETRIES = 3
_SAVE_DIR = Path.home() / "ba_mutator_traces"


def _find_trajectory(run_dir: Path) -> Path:
    direct = run_dir / "trajectory.jsonl"
    if direct.exists():
        return direct
    candidates = list(run_dir.rglob("trajectory.jsonl"))
    if not candidates:
        print(f"No trajectory.jsonl found under {run_dir}", file=sys.stderr)
        sys.exit(1)
    if len(candidates) > 1:
        print("Multiple trajectory.jsonl found — specify a single run directory:",
              file=sys.stderr)
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
    if requested:
        missing = [ep for ep in requested if ep not in groups]
        if missing:
            print(f"Warning: episodes not found: {missing}", file=sys.stderr)
        return [ep for ep in requested if ep in groups]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BA then Mutator on a run directory — full trace saved to "
                    "~/ba_mutator_traces/"
    )
    parser.add_argument("--run-dir",  required=True)
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print Mutator prompt without calling the LLM")
    parser.add_argument("--model",    default=None, help="Override model ID")
    parser.add_argument("--no-save",  action="store_true",
                        help="Print only — do not save to ~/ba_mutator_traces/")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    trajectory_path = _find_trajectory(run_dir)

    cfg = _load_run_config(run_dir)
    prompt_variant = cfg.get("prompt_variant", "rich")
    env_id = cfg.get("env", "")
    task = env_id.split("/", 1)[1] if "/" in env_id else None

    prompts = load_prompts(
        env_family="babyai", task=task, prompt_variant=prompt_variant
    )

    groups   = load_episode_groups(trajectory_path)
    selected = _select_episodes(groups, args.episodes)

    if not selected:
        print("No episodes to analyse.", file=sys.stderr)
        sys.exit(1)

    print(f"Run:      {trajectory_path}")
    print(f"Variant:  {prompt_variant}  |  Task: {task or 'unknown'}")
    print(f"Episodes: {selected}")
    for ep in selected:
        _, summary = groups[ep]
        if summary:
            outcome = "SUCCESS" if summary.get("success") else "failed"
            print(f"  Episode {ep:>3}: {outcome} "
                  f"({summary.get('total_steps')} steps, "
                  f"reward {summary.get('total_reward', 0):.3f})")
    print()

    cluster_endpoints = find_all_gpu_servers()
    endpoints = cluster_endpoints if cluster_endpoints else None
    model = args.model or ("gpt-oss-20b" if cluster_endpoints else HLP_MODEL_ID)

    # -----------------------------------------------------------------------
    # Compress trajectory
    # -----------------------------------------------------------------------
    compressed = compress_for_ba(trajectory_path, selected)

    # -----------------------------------------------------------------------
    # Step 1: BA
    # -----------------------------------------------------------------------
    ba_client = OpenAIClient(model=model, endpoints=endpoints, max_tokens=32768)
    ba = BehaviourAnalyser(client=ba_client, pipeline_mode="with_descriptor")

    print("Running BA...")
    ba_output = ba.analyse(
        compressed_trajectory=compressed,
        agent_prompt=prompts.get("agent_instructions", ""),
        env_prompt=prompts.get("environment_layer", ""),
        descriptor_prompt=prompts.get("descriptor_instructions"),
        task_prompt=prompts.get("task_layer"),
    )
    ba_messages = ba.last_messages or []

    if ba_output.output_type == "skip":
        lines = _build_trace_header(run_dir, trajectory_path, prompt_variant,
                                    task, selected, groups, model)
        lines += _format_ba_section(ba_messages, ba_output, compressed)
        lines += [
            "", "=" * 70,
            f"BA returned SKIP ({ba_output.skip_reason}) — nothing to mutate.",
            "=" * 70,
        ]
        _print_and_save("\n".join(lines), task, selected, "skip", args.no_save)
        return

    if ba_output.parse_error:
        lines = _build_trace_header(run_dir, trajectory_path, prompt_variant,
                                    task, selected, groups, model)
        lines += _format_ba_section(ba_messages, ba_output, compressed)
        lines += ["", f"BA parse error — cannot proceed to Mutator."]
        _print_and_save("\n".join(lines), task, selected, "ba_parse_error", args.no_save)
        return

    if not ba_output.implicated_module:
        print("BA output has no implicated module — cannot proceed.", file=sys.stderr)
        return

    module = ba_output.implicated_module
    current_prompt = (
        prompts.get("agent_instructions", "")
        if module == "agent"
        else prompts.get("descriptor_instructions", "")
    )

    # -----------------------------------------------------------------------
    # Step 2: dry-run prints Mutator prompt and exits
    # -----------------------------------------------------------------------
    mutator_client = OpenAIClient(model=model, endpoints=endpoints, max_tokens=16384)
    mutator = Mutator(client=mutator_client)

    if args.dry_run:
        mut_messages = mutator._build_messages(
            module=module,
            change_type=ba_output.change_type or "add",
            location=ba_output.location or "",
            characterisation=ba_output.characterisation,
            suggested_change=ba_output.suggested_change or "",
            current_prompt=current_prompt,
            env_prompt=prompts.get("environment_layer", ""),
            task_layer=prompts.get("task_layer"),
            hereditary_context=None,
        )
        print("\n=== MUTATOR PROMPT (dry-run) ===")
        for msg in mut_messages:
            print(f"\n--- [{msg['role'].upper()}] ---")
            print(msg["content"])
        approx = sum(len(m["content"]) // 4 for m in mut_messages)
        print(f"\n--- Approx prompt tokens: ~{approx:,} ---")
        return

    # -----------------------------------------------------------------------
    # Step 3: Mutator (with retries)
    # -----------------------------------------------------------------------
    print("Running Mutator...")
    mutator_output = None
    for attempt in range(1, _MAX_RETRIES + 1):
        result = mutator.mutate(
            module=module,
            change_type=ba_output.change_type or "add",
            location=ba_output.location or "",
            characterisation=ba_output.characterisation,
            suggested_change=ba_output.suggested_change or "",
            current_prompt=current_prompt,
            env_prompt=prompts.get("environment_layer", ""),
            task_layer=prompts.get("task_layer"),
            hereditary_context=None,
        )
        if not result.parse_error:
            mutator_output = result
            break
        print(f"  Attempt {attempt}/{_MAX_RETRIES} — parse error: {result.parse_error}")
        if attempt == _MAX_RETRIES:
            mutator_output = result
    mut_messages = mutator.last_messages or []

    # -----------------------------------------------------------------------
    # Assemble and output full trace
    # -----------------------------------------------------------------------
    lines = _build_trace_header(run_dir, trajectory_path, prompt_variant,
                                task, selected, groups, model)
    lines += _format_ba_section(ba_messages, ba_output, compressed)
    lines += _format_mutator_section(mut_messages, mutator_output, current_prompt)

    trace = "\n".join(lines)
    print(trace)

    outcome_tag = "parse_error" if mutator_output.parse_error else \
                  f"{module}_{ba_output.change_type or 'unknown'}"
    _print_and_save(trace, task or "unknown", selected, outcome_tag, args.no_save)


# ---------------------------------------------------------------------------
# Trace assembly helpers
# ---------------------------------------------------------------------------

def _build_trace_header(
    run_dir: Path,
    trajectory_path: Path,
    prompt_variant: str,
    task: str | None,
    selected: list[int],
    groups: dict,
    model: str,
) -> list[str]:
    lines = [
        "=" * 70,
        "FULL BA → MUTATOR TRACE",
        f"Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run dir:  {trajectory_path}",
        f"Task:     {task or 'unknown'}  |  Variant: {prompt_variant}",
        f"Model:    {model}",
        f"Episodes: {selected}",
    ]
    for ep in selected:
        _, summary = groups[ep]
        if summary:
            outcome = "SUCCESS" if summary.get("success") else "failed"
            lines.append(f"  Episode {ep:>3}: {outcome} "
                         f"({summary.get('total_steps')} steps, "
                         f"reward {summary.get('total_reward', 0):.3f})")
    lines.append("=" * 70)
    return lines


def _format_ba_section(
    messages: list[dict],
    output: BAOutput,
    compressed: str,
) -> list[str]:
    lines = ["", "=" * 70, "BEHAVIOUR ANALYSER", "=" * 70]

    # Token / latency header
    if output.prompt_tokens is not None:
        total = (output.prompt_tokens or 0) + (output.completion_tokens or 0)
        trunc = "  ⚠ HIT MAX_TOKENS" if output.finish_reason == "length" else ""
        lines.append(
            f"Tokens: {output.prompt_tokens:,} prompt + "
            f"{output.completion_tokens:,} completion = {total:,} total  |  "
            f"{output.latency_s:.1f}s  |  finish={output.finish_reason}{trunc}"
        )

    # Parsed fields
    lines += [
        "",
        f"TYPE:   {output.output_type.upper()}",
    ]
    if output.implicated_module:
        lines.append(f"MODULE: {output.implicated_module}")
    if output.failure_step is not None:
        lines.append(f"STEP:   {output.failure_step}")
    if output.change_type:
        lines.append(f"CHANGE: {output.change_type.upper()}")
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
            lines.append(f"  {i}. [{c.module}] step {c.step}  [{c.change_type}]")
            if c.location:
                lines.append(f"     Location: {c.location}")
            lines.append(f"     Characterisation: {c.description}")
            if c.suggested_change:
                lines.append(f"     Suggested change: {c.suggested_change}")

    # Compressed trajectory
    lines += ["", "─" * 70, "COMPRESSED TRAJECTORY (sent to BA)", "─" * 70, compressed]

    # Full BA prompt
    lines += ["", "─" * 70, "BA INPUT PROMPT", "─" * 70]
    for msg in messages:
        lines += [f"=== [{msg['role'].upper()}] ===", msg["content"], ""]

    # Reasoning trace
    if output.raw_reasoning:
        lines += ["─" * 70, "BA REASONING TRACE", "─" * 70, output.raw_reasoning]

    # Raw response
    raw_label = "BA RAW RESPONSE (parse failed)" if output.parse_error \
        else "BA RAW RESPONSE"
    lines += ["", "─" * 70, raw_label, "─" * 70, output.raw_response]

    return lines


def _format_mutator_section(
    messages: list[dict],
    output: MutatorOutput,
    original_prompt: str,
) -> list[str]:
    lines = ["", "=" * 70, "MUTATOR", "=" * 70]

    if output.prompt_tokens is not None:
        total = (output.prompt_tokens or 0) + (output.completion_tokens or 0)
        trunc = "  ⚠ HIT MAX_TOKENS" if output.finish_reason == "length" else ""
        lines.append(
            f"Tokens: {output.prompt_tokens:,} prompt + "
            f"{output.completion_tokens:,} completion = {total:,} total  |  "
            f"{output.latency_s:.1f}s  |  finish={output.finish_reason}{trunc}"
        )

    if output.parse_error:
        lines += ["", f"PARSE ERROR: {output.parse_error}"]
    else:
        lines += [
            "",
            f"SECTION:   {output.section}",
            f"CHANGE:    {output.change}",
            "",
            "PRINCIPLE:",
            output.principle,
        ]
        if output.conflict_note:
            lines += ["", f"CONFLICT NOTE: {output.conflict_note}"]

        # Diff
        orig_lines    = original_prompt.splitlines(keepends=True)
        revised_lines = output.revised_prompt.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            orig_lines, revised_lines,
            fromfile="original", tofile="revised", lineterm=""
        ))
        if diff:
            lines += ["", "─" * 70, "DIFF", "─" * 70]
            lines.extend(line.rstrip("\n") for line in diff)
        else:
            lines += ["", "─" * 70, "DIFF: no changes detected", "─" * 70]

        lines += ["", "─" * 70, "REVISED PROMPT", "─" * 70, output.revised_prompt]

    # Full Mutator prompt
    lines += ["", "─" * 70, "MUTATOR INPUT PROMPT", "─" * 70]
    for msg in messages:
        lines += [f"=== [{msg['role'].upper()}] ===", msg["content"], ""]

    # Reasoning trace
    if output.raw_reasoning:
        lines += ["─" * 70, "MUTATOR REASONING TRACE", "─" * 70, output.raw_reasoning]

    # Raw response
    raw_label = "MUTATOR RAW RESPONSE (parse failed)" if output.parse_error \
        else "MUTATOR RAW RESPONSE"
    lines += ["", "─" * 70, raw_label, "─" * 70, output.raw_response]

    return lines


def _print_and_save(
    trace: str,
    task: str,
    episodes: list,
    tag: str,
    no_save: bool,
) -> None:
    if no_save:
        return
    _SAVE_DIR.mkdir(exist_ok=True)
    ep_str = "-".join(str(e) for e in episodes)
    ts     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname  = f"{task}_ep{ep_str}_{tag}_{ts}.txt"
    path   = _SAVE_DIR / fname
    path.write_text(trace, encoding="utf-8")
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
