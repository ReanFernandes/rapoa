"""Optimisation loop — RAPOA.

Terminology
-----------
env_round           One call to run.py producing episodes (trajectories + rewards).
                    The agent prompt does not change within an env_round.
opt_cycle           One BA → Mutator → Evaluator sequence. Consumes the V bag
                    from the preceding env_round. May update the incumbent prompt.
V bag               Pool of episodes run each env_round with the current incumbent.
                    Size: ba_episodes × max_skip_resample (derived, not a free parameter).
BA train signal     Random sample of --ba-episodes (fixed 6) from the V bag,
                    sent to the BA. Manually verified to fit BA context window.
T pool              Fixed held-out episodes established in env_round 0. Never re-run.
                    Size: --t-size (default 20).
Paired evaluation   Running the challenger on the same seeds as the incumbent and
                    comparing rewards directly. Controls for episode difficulty.
Validation strategy How many V bag episodes the paired evaluation uses:
                    train_signal  — same 6 episodes the BA saw (in-sample)
                    validation_bag — all 20 V bag episodes (out-of-sample for 14)

Loop structure
--------------
env_round 0  :  run (v_bag_size + t_size) episodes → first v_bag_size → V bag,
                last t_size → T pool (fixed forever). No coin flip.
opt_cycle 1  :  BA on sample from V bag → Mutator → paired V eval → T eval → accept
env_round 1  :  run v_bag_size V episodes with current incumbent
opt_cycle 2  :  BA on sample from V bag → Mutator → paired V eval → T eval → accept
env_round 2  :  run v_bag_size V episodes with current incumbent
...

Seed collision guarantee: T seeds are [env_seed + v_bag_size, ..., env_seed + v_bag_size + t_size - 1].
env_round k (k≥1) uses seeds starting at env_seed + v_bag_size + t_size + (k-1)*v_bag_size.
No V seed ever appears in T.

Usage
-----
    python experiments/optimise.py \\
        --env BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to \\
        --prompt-variant minimal \\
        --opt-cycles 20 --ba-episodes 6 --max-skip-resample 3 --t-size 20 \\
        --validation-strategy validation_bag \\
        --rule mean --reward-threshold 0.05 \\
        --model gpt-oss-20b --inference-seed 1 --workers 20
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

_HERE   = Path(__file__).resolve().parent
_ROOT   = _HERE.parent
_PYTHON = _ROOT / ".venv" / "bin" / "python"

import sys
sys.path.insert(0, str(_ROOT))

from src.llm.client import OpenAIClient
from src.llm.server_utils import find_all_gpu_servers, start_endpoint_watcher
from src.optimization.behaviour_analyser import BehaviourAnalyser
from src.optimization.evaluator import Evaluator
from src.optimization.hereditary import (
    append_entry, update_outcome, render_hereditary_context, prompt_hash,
)
from src.optimization.mutator import Mutator
from src.optimization.preprocessing import compress_for_ba
from src.utils.config import load_prompts


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RAPOA optimisation loop")
    p.add_argument("--env",                  default="BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to")
    p.add_argument("--pipeline",             default="with_descriptor",
                   choices=["with_descriptor", "balrog_baseline"])
    p.add_argument("--prompt-variant",       default="rich", choices=["rich", "minimal"])
    p.add_argument("--model",                default=None)
    p.add_argument("--inference-seed",       type=int, default=None,
                   help="LLM sampling seed for episode runs and BA/Mutator calls")
    p.add_argument("--env-seed",             type=int, default=42,
                   help="Base env seed for all episode generation")
    p.add_argument("--opt-cycles",           type=int, default=10,
                   help="Number of opt_cycles to run")
    p.add_argument("--t-size",               type=int, default=20,
                   help="T pool size. Collected once in env_round 0, fixed for entire run.")
    p.add_argument("--ba-episodes",          type=int, default=6,
                   help="BA train signal: episodes sampled from V bag per BA call. "
                        "Fixed at 6 (manually verified to fit BA context window).")
    p.add_argument("--max-skip-resample",    type=int, default=3,
                   help="Max times to resample V bag and retry BA on SKIP before moving on.")
    p.add_argument("--validation-strategy",  default="validation_bag",
                   choices=["train_signal", "validation_bag"],
                   help="train_signal: paired V eval on the 6 BA episodes only. "
                        "validation_bag: paired V eval on all v-bag-size episodes.")
    p.add_argument("--workers",              type=int, default=10,
                   help="Parallel episode workers passed to run.py")
    p.add_argument("--max-steps",            type=int, default=64,
                   help="Max steps per episode")
    p.add_argument("--log-dir",              default="optimization_runs")
    p.add_argument("--run-id",               default=None,
                   help="Run identifier (default: timestamp)")
    p.add_argument("--rule",                 default="mean",
                   choices=["mean", "wilcoxon"],
                   help="Acceptance rule. Use --reward-threshold -inf for always-accept behaviour.")
    p.add_argument("--p-threshold",          type=float, default=0.05)
    p.add_argument("--reward-threshold",     type=float, default=0.05,
                   help="Mean reward improvement required for acceptance. "
                        "Pass -inf to accept every challenger (always-accept).")
    p.add_argument("--min-discordant-pairs", type=int,   default=4,
                   help="Minimum discordant episode pairs required for a non-insufficient_signal "
                        "verdict. Set to 0 to disable the check (useful with --reward-threshold 0 "
                        "where you want to accept neutral mutations at floor performance).")
    p.add_argument("--history-window",        type=int, default=16,
                   help="Rolling history window for BALROG pipeline (default: 16).")
    p.add_argument("--actor-history-window",  type=int, default=None,
                   help="Rolling history window for the actor in with_descriptor pipeline. "
                        "None (default) = single-turn. Integer = multi-turn with that window size.")
    p.add_argument("--module-constraint",    default="both",
                   choices=["both", "actor", "descriptor", "random"],
                   help="Restrict which module the BA may implicate. "
                        "'both' (default) = free choice. "
                        "'agent'/'descriptor' = forced attribution to that module only.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# env_round helper
# ---------------------------------------------------------------------------

def _dynamic_workers(base_workers: int, n_seeds: int) -> int:
    """Return worker count scaled to live endpoint count, capped at n_seeds."""
    n_endpoints = len(find_all_gpu_servers())
    scaled = max(base_workers, n_endpoints * 2) if n_endpoints else base_workers
    return min(n_seeds, scaled)


def _run_env_round(
    round_num: int,
    seeds: list[int],
    incumbent_agent_path: Path,
    incumbent_descriptor_path: Path,
    log_dir: Path,
    args,
) -> tuple[Path, Path, list[float]]:
    """Execute one env_round. Returns (run_directory, trajectory_path, rewards)."""
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(_PYTHON),
        str(_ROOT / "experiments" / "run.py"),
        "--env",                    args.env,
        "--pipeline",               args.pipeline,
        "--prompt-variant",         args.prompt_variant,
        "--seed-list",              *[str(s) for s in seeds],
        "--workers",                str(_dynamic_workers(args.workers, len(seeds))),
        "--no-gif",
        "--max-steps-per-episode",  str(args.max_steps),
        "--agent-prompt-file",      str(incumbent_agent_path),
        "--descriptor-prompt-file", str(incumbent_descriptor_path),
        "--log-dir",                str(log_dir),
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.inference_seed is not None:
        cmd += ["--inference-seed", str(args.inference_seed)]
    if args.pipeline == "balrog_baseline":
        cmd += ["--history-window", str(args.history_window)]
    elif args.pipeline == "with_descriptor" and getattr(args, "actor_history_window", None) is not None:
        cmd += ["--agent-multi-turn", "--history-window", str(args.actor_history_window)]

    print(f"\n  [env_round {round_num}] Running {len(seeds)} episodes...")
    subprocess.run(cmd, check=True, cwd=str(_ROOT))

    summaries = list(log_dir.rglob("run_summary.json"))
    if not summaries:
        raise RuntimeError(f"env_round {round_num}: no run_summary.json under {log_dir}")

    summary_path  = max(summaries, key=lambda p: p.stat().st_mtime)
    run_directory = summary_path.parent
    trajectory    = run_directory / "trajectory.jsonl"

    with open(summary_path) as f:
        data = json.load(f)

    rewards = [ep["total_reward"] for ep in data["episodes"]]
    print(f"  [env_round {round_num}] Done — mean reward: {sum(rewards)/len(rewards):.3f}  "
          f"successes: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")
    return run_directory, trajectory, rewards


# ---------------------------------------------------------------------------
# Optimisation log
# ---------------------------------------------------------------------------

def _append_log(log_path: Path, record: dict) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    with log_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_resume_state(
    run_dir: Path,
    opt_log: Path,
    args,
    rng: random.Random,
) -> dict | None:
    """Return resume state if a partial run exists, otherwise None (fresh start).

    Reads optimisation_log.jsonl to reconstruct:
      - cycles_done       : number of fully completed opt_cycles
      - t_seeds           : T pool seeds (constant)
      - t_incumbent_rewards : T pool rewards tracking through accepted mutations
      - v_seeds           : V bag seeds going into the next cycle
      - v_rewards         : V bag rewards going into the next cycle
      - v_trajectory      : Path to the trajectory for the next cycle's V bag

    Also advances rng past all consumed samples so sampling continues correctly.
    """
    records = _read_log(opt_log)
    if not records:
        return None

    setup = next(
        (r for r in records if r.get("record_type") == "env_round_setup"
         and r.get("env_round") == 0),
        None,
    )
    if setup is None:
        return None

    t_seeds             = setup["t_seeds"]
    t_incumbent_rewards = setup["t_rewards"]

    cycle_records = sorted(
        [r for r in records if r.get("record_type") == "opt_cycle"],
        key=lambda r: r["opt_cycle"],
    )
    cycles_done = len(cycle_records)

    # Advance rng past all BA sample draws in completed cycles.
    fake_ep_nums = list(range(1, args.v_bag_size + 1))
    for cr in cycle_records:
        for _ in range(cr.get("ba_attempts", 1)):
            rng.sample(fake_ep_nums, min(args.ba_episodes, len(fake_ep_nums)))

    # Track t_incumbent_rewards through accepted mutations.
    for cr in cycle_records:
        if cr.get("opt_cycle_outcome") == "accepted":
            for cand in cr.get("candidates_tried", []):
                t_res = cand.get("t_result") or {}
                if t_res.get("verdict") == "accepted":
                    challenger = t_res.get("challenger_rewards", [])
                    if challenger:
                        t_incumbent_rewards = challenger
                    break

    # Restore V bag from the env_round that followed the last completed cycle.
    if cycles_done == 0:
        # env_round_0 done, no cycles yet — V bag comes from the setup record.
        v_seeds   = setup["v_seeds"]
        v_rewards = setup["v_rewards"]
        env_round_dir = run_dir / "env_round_0"
    else:
        last = cycle_records[-1]
        env_round_dir = Path(last["env_round_out_dir"]).parent \
            if last.get("env_round_out_dir") else run_dir / f"env_round_{cycles_done}"
        # Seeds are fully deterministic from args — recompute rather than trust the log.
        offset   = args.v_bag_size + args.t_size + (cycles_done - 1) * args.v_bag_size
        v_seeds  = [args.env_seed + offset + i for i in range(args.v_bag_size)]
        summaries = list(env_round_dir.rglob("run_summary.json"))
        if not summaries:
            print(f"  [resume] WARNING: env_round_{cycles_done} has no run_summary.json — "
                  f"cannot resume past cycle {cycles_done}.")
            return None
        with open(max(summaries, key=lambda p: p.stat().st_mtime)) as f:
            data = json.load(f)
        v_rewards = [ep["total_reward"] for ep in data["episodes"]]

    trajectories = list(env_round_dir.rglob("trajectory.jsonl"))
    if not trajectories:
        print(f"  [resume] WARNING: no trajectory.jsonl in {env_round_dir} — "
              "cannot resume.")
        return None
    trajectories = [max(trajectories, key=lambda p: p.stat().st_mtime)]

    return {
        "cycles_done":          cycles_done,
        "t_seeds":              t_seeds,
        "t_incumbent_rewards":  t_incumbent_rewards,
        "v_seeds":              v_seeds,
        "v_rewards":            v_rewards,
        "v_trajectory":         trajectories[0],
    }


# ---------------------------------------------------------------------------
# Prompt cache
# ---------------------------------------------------------------------------

def _cache_prompt(cache_dir: Path, module: str, prompt_text: str) -> str:
    """Write prompt to cache_dir/{module}_{hash}.txt. Returns the hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = prompt_hash(prompt_text)
    (cache_dir / f"{module}_{h}.txt").write_text(prompt_text, encoding="utf-8")
    return h


# ---------------------------------------------------------------------------
# Output saving helpers
# ---------------------------------------------------------------------------

def _save_ba_output(out_path: Path, ba_output, messages: list | None) -> None:
    lines = [
        f"TYPE:   {ba_output.output_type}",
        f"MODULE: {ba_output.implicated_module}",
        f"STEP:   {ba_output.failure_step}",
        f"TOKENS: prompt={ba_output.prompt_tokens}  "
        f"completion={ba_output.completion_tokens}  latency={ba_output.latency_s:.1f}s",
    ]
    if ba_output.parse_error:
        lines.append(f"PARSE ERROR: {ba_output.parse_error}")
    lines += ["\n--- CHARACTERISATION ---", ba_output.characterisation or ""]
    if ba_output.raw_reasoning:
        lines += ["\n--- REASONING TRACE ---", ba_output.raw_reasoning]
    lines += ["\n--- RAW RESPONSE ---", ba_output.raw_response]
    if messages:
        lines.append("\n--- FULL INPUT PROMPT ---")
        for msg in messages:
            lines.append(f"[{msg['role'].upper()}]\n{msg['content']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _save_mutator_output(out_path: Path, mut_out, messages: list | None) -> None:
    lines = [
        f"SECTION: {mut_out.section}",
        f"CHANGE:  {mut_out.change}",
        f"TOKENS:  prompt={mut_out.prompt_tokens}  "
        f"completion={mut_out.completion_tokens}  latency={mut_out.latency_s:.1f}s",
    ]
    if mut_out.conflict_note:
        lines.append(f"CONFLICT NOTE: {mut_out.conflict_note}")
    if mut_out.parse_error:
        lines.append(f"PARSE ERROR: {mut_out.parse_error}")
    lines += ["\n--- PRINCIPLE ---", mut_out.principle or "",
              "\n--- REVISED PROMPT ---", mut_out.revised_prompt or "",
              "\n--- RAW RESPONSE ---", mut_out.raw_response]
    if messages:
        lines.append("\n--- FULL INPUT PROMPT ---")
        for msg in messages:
            lines.append(f"[{msg['role'].upper()}]\n{msg['content']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    # v_bag_size is derived — not a free parameter
    args.v_bag_size = args.ba_episodes * args.max_skip_resample

    rng = random.Random(args.env_seed)

    run_id  = args.run_id or time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_start_time = time.perf_counter()

    print(f"\nOptimisation run   : {run_dir}")
    print(f"Env                : {args.env}")
    _rule_display = "always_accept" if math.isinf(args.reward_threshold) and args.reward_threshold < 0 else f"{args.rule} (δ={args.reward_threshold})"
    print(f"Acceptance rule    : {_rule_display}")
    print(f"Validation strategy: {args.validation_strategy}")
    print(f"V bag size         : {args.v_bag_size}  |  BA train signal: {args.ba_episodes}")
    print(f"T pool size        : {args.t_size}  |  opt_cycles: {args.opt_cycles}")

    hereditary_log = run_dir / "hereditary.jsonl"
    opt_log        = run_dir / "optimisation_log.jsonl"

    from src.logging.run_directory import parse_env_id as _parse_env_id
    _env_family, _ = _parse_env_id(args.env)
    _actor_multi_turn = (
        args.pipeline == "with_descriptor"
        and getattr(args, "actor_history_window", None) is not None
    )
    prompts    = load_prompts(env_family=_env_family.lower(), prompt_variant=args.prompt_variant,
                              multi_turn=_actor_multi_turn)
    env_prompt = prompts["environment_layer"]

    incumbent_agent_path      = run_dir / "incumbent_agent_prompt.txt"
    incumbent_descriptor_path = run_dir / "incumbent_descriptor_prompt.txt"
    prompt_cache_dir          = run_dir / "prompt_cache"

    resume = _load_resume_state(run_dir, opt_log, args, rng)

    if resume is not None:
        cycles_done         = resume["cycles_done"]
        t_seeds             = resume["t_seeds"]
        t_incumbent_rewards = resume["t_incumbent_rewards"]
        v_seeds             = resume["v_seeds"]
        v_rewards           = resume["v_rewards"]
        v_trajectory        = resume["v_trajectory"]
        v_ep_nums           = list(range(1, args.v_bag_size + 1))
        # Incumbent prompts are already on disk (updated in place on every accept).
        incumbent_agent_prompt      = incumbent_agent_path.read_text()
        incumbent_descriptor_prompt = incumbent_descriptor_path.read_text()
        if args.pipeline == "balrog_baseline":
            args.module_constraint = "actor"
        print(f"\nResuming from cycle {cycles_done + 1} / {args.opt_cycles}  "
              f"({cycles_done} cycle(s) already complete)")
    else:
        cycles_done = 0
        if args.pipeline == "balrog_baseline":
            incumbent_agent_prompt      = prompts["balrog_instructions"]
            incumbent_descriptor_prompt = ""
            args.module_constraint      = "actor"
        else:
            incumbent_agent_prompt      = prompts["agent_instructions"]
            incumbent_descriptor_prompt = prompts["descriptor_instructions"]

        incumbent_agent_path.write_text(incumbent_agent_prompt)
        incumbent_descriptor_path.write_text(incumbent_descriptor_prompt)
        _cache_prompt(prompt_cache_dir, "actor",      incumbent_agent_prompt)
        _cache_prompt(prompt_cache_dir, "descriptor", incumbent_descriptor_prompt)
        (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    if cycles_done >= args.opt_cycles:
        print(f"\nAll {args.opt_cycles} cycle(s) already complete — nothing to do.")
        return

    cluster_endpoints = find_all_gpu_servers()
    client_kwargs: dict = {}
    if cluster_endpoints:
        client_kwargs["endpoints"] = cluster_endpoints
        client_kwargs["model"]     = args.model or "gpt-oss-20b"
        print(f"Cluster nodes      : {cluster_endpoints}")

        def _on_endpoints_change(new_endpoints: list[str]) -> None:
            if not new_endpoints:
                return
            OpenAIClient.update_endpoints(new_endpoints)

        start_endpoint_watcher(_on_endpoints_change)
        print("Endpoint watcher started — edit ~/hlp_ports.txt to add/remove nodes mid-run")
    if args.model:
        client_kwargs["model"] = args.model
    if args.inference_seed is not None:
        client_kwargs["inference_seed"] = args.inference_seed

    _ba_pipeline_mode = "monolithic" if args.pipeline == "balrog_baseline" else "with_descriptor"
    ba        = BehaviourAnalyser(client=OpenAIClient(**client_kwargs, max_tokens=32768), pipeline_mode=_ba_pipeline_mode)
    mutator   = Mutator(client=OpenAIClient(**client_kwargs))
    evaluator = Evaluator(
        rule=args.rule,
        p_threshold=args.p_threshold,
        reward_threshold=args.reward_threshold,
        min_discordant_pairs=args.min_discordant_pairs,
    )

    # -----------------------------------------------------------------------
    # env_round 0 — V bag + T pool (skipped on resume)
    # Seeds: [env_seed .. env_seed+v_bag_size-1]          → V bag
    #        [env_seed+v_bag_size .. +v_bag_size+t_size-1] → T pool
    # -----------------------------------------------------------------------
    if resume is None:
        print(f"\n{'=' * 60}")
        print(f"env_round 0 — {args.v_bag_size} V + {args.t_size} T episodes")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()

        round0_seeds = [args.env_seed + i for i in range(args.v_bag_size + args.t_size)]
        round0_dir, v_trajectory, round0_rewards = _run_env_round(
            round_num=0,
            seeds=round0_seeds,
            incumbent_agent_path=incumbent_agent_path,
            incumbent_descriptor_path=incumbent_descriptor_path,
            log_dir=run_dir / "env_round_0",
            args=args,
        )

        v_seeds   = round0_seeds[:args.v_bag_size]
        v_rewards = round0_rewards[:args.v_bag_size]
        v_ep_nums = list(range(1, args.v_bag_size + 1))

        t_seeds             = round0_seeds[args.v_bag_size:]
        t_incumbent_rewards = round0_rewards[args.v_bag_size:]

        print(f"  V bag: {len(v_seeds)} episodes  |  T pool: {len(t_seeds)} episodes (fixed)")

        _append_log(opt_log, {
            "record_type":              "env_round_setup",
            "env_round":                0,
            "incumbent_prompt_version": prompt_hash(incumbent_agent_prompt),
            "v_seeds":                  v_seeds,
            "t_seeds":                  t_seeds,
            "v_rewards":                v_rewards,
            "t_rewards":                t_incumbent_rewards,
            "env_round_dir":            str(round0_dir),
            "wall_time_s":              round(time.perf_counter() - t0, 2),
        })

    # -----------------------------------------------------------------------
    # opt_cycles — start from cycles_done+1 on resume
    # -----------------------------------------------------------------------
    for opt_cycle in range(cycles_done + 1, args.opt_cycles + 1):
        t0 = time.perf_counter()
        _eff_module = (
            random.choice(["actor", "descriptor"])
            if args.module_constraint == "random"
            else args.module_constraint
        )
        print(f"\n{'=' * 60}")
        print(f"opt_cycle {opt_cycle} / {args.opt_cycles}  "
              f"(env_round {opt_cycle - 1} V bag)")
        print(f"{'=' * 60}")

        opt_cycle_dir = run_dir / f"opt_cycle_{opt_cycle}"
        opt_cycle_dir.mkdir(parents=True, exist_ok=True)

        # Build episode → (seed, reward) lookup for this V bag
        ep_lookup = {
            ep_num: (seed, reward)
            for ep_num, seed, reward in zip(v_ep_nums, v_seeds, v_rewards)
        }

        hereditary_ctx = (
            render_hereditary_context(hereditary_log)
            if hereditary_log.exists() else None
        )

        # -- BA with SKIP resampling --
        ba_output       = None
        sampled_ep_nums = None
        max_attempts    = args.max_skip_resample + 1
        ba_attempts_detail: list[dict] = []

        for attempt in range(max_attempts):
            sampled_ep_nums = sorted(
                rng.sample(v_ep_nums, min(args.ba_episodes, len(v_ep_nums)))
            )
            compressed = compress_for_ba(v_trajectory, sampled_ep_nums)

            attempt_label = f"attempt_{attempt + 1}" if attempt > 0 else "attempt_1"
            print(f"  Running BA on episodes {sampled_ep_nums} ({attempt_label})...")

            _t_ba_attempt = time.perf_counter()
            ba_output = ba.analyse(
                compressed_trajectory=compressed,
                agent_prompt=incumbent_agent_prompt,
                env_prompt=env_prompt,
                descriptor_prompt=incumbent_descriptor_prompt,
                hereditary_context=hereditary_ctx,
                module_constraint=_eff_module,
            )
            ba_attempts_detail.append({
                "attempt":           attempt + 1,
                "wall_time_s":       round(time.perf_counter() - _t_ba_attempt, 2),
                "latency_s":         ba_output.latency_s,
                "prompt_tokens":     ba_output.prompt_tokens,
                "completion_tokens": ba_output.completion_tokens,
                "outcome":           ba_output.output_type,
            })
            _save_ba_output(
                opt_cycle_dir / f"ba_output_{attempt_label}.txt",
                ba_output, ba.last_messages,
            )
            print(f"  BA: type={ba_output.output_type}  "
                  f"module={ba_output.implicated_module}  "
                  f"tokens={ba_output.prompt_tokens}/{ba_output.completion_tokens}")

            if ba_output.output_type != "skip":
                break
            if attempt < max_attempts - 1:
                print(f"  BA SKIP — resampling V bag "
                      f"({attempt + 1}/{args.max_skip_resample} resample(s) used)...")

        # Derive train signal seeds/rewards from the final BA sample
        train_seeds   = [ep_lookup[ep][0] for ep in sampled_ep_nums]
        train_rewards = [ep_lookup[ep][1] for ep in sampled_ep_nums]

        # Validation seeds: determined by --validation-strategy
        if args.validation_strategy == "train_signal":
            eval_v_seeds   = train_seeds
            eval_v_rewards = train_rewards
        else:  # validation_bag
            eval_v_seeds   = v_seeds
            eval_v_rewards = v_rewards

        cycle_record: dict = {
            "record_type":              "opt_cycle",
            "opt_cycle":                opt_cycle,
            "env_round_in":             opt_cycle - 1,
            "env_round_out":            opt_cycle,
            "validation_strategy":      args.validation_strategy,
            "random_module_draw":       _eff_module if args.module_constraint == "random" else None,
            "incumbent_prompt_version": prompt_hash(incumbent_agent_prompt),
            "v_seeds":                  v_seeds,
            "t_seeds":                  t_seeds,
            "v_incumbent_rewards":      v_rewards,
            "t_incumbent_rewards":      t_incumbent_rewards,
            "ba_sample_ep_nums":        sampled_ep_nums,
            "ba_sample_seeds":          train_seeds,
            "eval_v_seeds":             eval_v_seeds,
            "ba_attempts":              attempt + 1,
            "ba_attempts_detail":       ba_attempts_detail,
            "ba_wall_time_s":           round(sum(a["wall_time_s"] for a in ba_attempts_detail), 2),
            "env_round_in_dir":         str(v_trajectory.parent),
            "ba_output": {
                "type":             ba_output.output_type,
                "module":           ba_output.implicated_module,
                "step":             ba_output.failure_step,
                "characterisation": ba_output.characterisation,
                "skip_reason":      ba_output.skip_reason,
                "prompt_tokens":    ba_output.prompt_tokens,
                "completion_tokens":ba_output.completion_tokens,
                "latency_s":        ba_output.latency_s,
            },
            "candidates_tried":   [],
            "opt_cycle_outcome":  None,
            "new_prompt_version": None,
            "env_round_out_dir":  None,
            "wall_time_s":        None,
        }

        if ba_output.output_type == "skip" or ba_output.implicated_module is None:
            print(f"  BA returned SKIP after {attempt + 1} attempt(s) — no mutation.")
            cycle_record["opt_cycle_outcome"] = "ba_skip"
        else:
            primary = {
                "module":          ba_output.implicated_module,
                "change_type":     ba_output.change_type or "add",
                "location":        ba_output.location or "",
                "characterisation":ba_output.characterisation,
                "suggested_change":ba_output.suggested_change or "",
            }
            additional = [
                {
                    "module":          c.module,
                    "change_type":     c.change_type or "add",
                    "location":        c.location or "",
                    "characterisation":c.description,
                    "suggested_change":c.suggested_change or "",
                }
                for c in ba_output.candidate_queue
                if c.module and c.suggested_change
            ]
            candidates     = [primary] + additional
            opt_outcome    = "rejected_all"
            new_prompt_ver = None

            # Safeguard: filter candidates to respect --module-constraint.
            # BA may still output the wrong module despite prompt instruction.
            if _eff_module != "both":
                allowed = [c for c in candidates if c["module"] == _eff_module]
                if not allowed:
                    print(f"  All {len(candidates)} BA candidate(s) violated "
                          f"module_constraint='{_eff_module}' — constraint_skip.")
                    opt_outcome = "constraint_skip"
                    candidates  = []
                else:
                    n_filtered = len(candidates) - len(allowed)
                    if n_filtered:
                        print(f"  Filtered {n_filtered} candidate(s) violating "
                              f"module_constraint='{_eff_module}' — "
                              f"{len(allowed)} remaining.")
                    candidates = allowed

            for rank, cand in enumerate(candidates, start=1):
                print(f"\n  Candidate {rank}/{len(candidates)}: "
                      f"module={cand['module']}  change_type={cand['change_type']}")

                current_prompt = (
                    incumbent_agent_prompt if cand["module"] == "actor"
                    else incumbent_descriptor_prompt
                )

                _t_mut = time.perf_counter()
                mut_out = mutator.mutate(
                    module=cand["module"],
                    change_type=cand["change_type"],
                    location=cand["location"],
                    characterisation=cand["characterisation"],
                    suggested_change=cand["suggested_change"],
                    current_prompt=current_prompt,
                    env_prompt=env_prompt,
                    hereditary_context=hereditary_ctx,
                )
                _mut_wall_s = round(time.perf_counter() - _t_mut, 2)
                _save_mutator_output(
                    opt_cycle_dir / f"mutator_output_{rank}.txt",
                    mut_out, mutator.last_messages,
                )
                if mut_out.revised_prompt and not mut_out.parse_error:
                    _cache_prompt(prompt_cache_dir, cand["module"], mut_out.revised_prompt)

                if mut_out.parse_error or not mut_out.revised_prompt:
                    print(f"    Mutator error: {mut_out.parse_error}")
                    cycle_record["candidates_tried"].append({
                        "candidate_rank":         rank,
                        "module":                 cand["module"],
                        "mutator_wall_time_s":    _mut_wall_s,
                        "mutator_latency_s":      mut_out.latency_s,
                        "mutator_prompt_tokens":  mut_out.prompt_tokens,
                        "mutator_completion_tokens": mut_out.completion_tokens,
                        "mutator_error":          mut_out.parse_error,
                        "v_result": None, "t_result": None,
                    })
                    continue

                if len(mut_out.revised_prompt.strip()) < 50:
                    print(f"    Mutator output too short — skipping")
                    cycle_record["candidates_tried"].append({
                        "candidate_rank":         rank,
                        "module":                 cand["module"],
                        "mutator_wall_time_s":    _mut_wall_s,
                        "mutator_latency_s":      mut_out.latency_s,
                        "mutator_prompt_tokens":  mut_out.prompt_tokens,
                        "mutator_completion_tokens": mut_out.completion_tokens,
                        "mutator_error":          "revised_prompt too short",
                        "v_result": None, "t_result": None,
                    })
                    continue

                hkey = opt_cycle * 100 + rank
                append_entry(hereditary_log, {
                    "cycle":                  hkey,
                    "prompt_version":         prompt_hash(current_prompt),
                    "revised_prompt_version": prompt_hash(mut_out.revised_prompt),
                    "module":                 cand["module"],
                    "change_type":            cand["change_type"],
                    "ba_characterisation":    cand["characterisation"],
                    "section":                mut_out.section,
                    "change":                 mut_out.change,
                    "principle":              mut_out.principle,
                    "conflict_note":          mut_out.conflict_note,
                    "outcome":                "pending",
                })

                cand_record: dict = {
                    "candidate_rank":            rank,
                    "module":                    cand["module"],
                    "mutator_section":           mut_out.section,
                    "mutator_change":            mut_out.change,
                    "mutator_principle":         mut_out.principle,
                    "mutator_conflict_note":     mut_out.conflict_note,
                    "mutator_wall_time_s":       _mut_wall_s,
                    "mutator_latency_s":         mut_out.latency_s,
                    "mutator_prompt_tokens":     mut_out.prompt_tokens,
                    "mutator_completion_tokens": mut_out.completion_tokens,
                    "v_result": None,
                    "t_result": None,
                }

                # Paired V evaluation (seeds depend on validation_strategy)
                _t_v = time.perf_counter()
                v_eval = evaluator.evaluate(
                    stage="V",
                    seeds=eval_v_seeds,
                    incumbent_rewards=eval_v_rewards,
                    candidate_prompt=mut_out.revised_prompt,
                    module=cand["module"],
                    env=args.env, pipeline=args.pipeline,
                    prompt_variant=args.prompt_variant, model=args.model,
                    inference_seed=args.inference_seed,
                    workers=_dynamic_workers(args.workers, len(eval_v_seeds)),
                    history_window=args.history_window,
                    actor_history_window=getattr(args, "actor_history_window", None),
                    log_dir=opt_cycle_dir,
                )
                cand_record["v_result"] = {
                    "verdict":           v_eval.verdict,
                    "p_value":           v_eval.p_value,
                    "net_mean_reward":   v_eval.net_mean_reward,
                    "n_positive":        v_eval.n_positive,
                    "n_negative":        v_eval.n_negative,
                    "n_tied":            v_eval.n_tied,
                    "challenger_rewards":v_eval.challenger_rewards,
                    "wall_time_s":       round(time.perf_counter() - _t_v, 2),
                    "n_episodes":        len(eval_v_seeds),
                }
                print(f"    V verdict: {v_eval.verdict}  "
                      f"net_reward={v_eval.net_mean_reward}  p={v_eval.p_value}")

                if v_eval.verdict == "rejected":
                    update_outcome(hereditary_log, hkey, "rejected")
                    cycle_record["candidates_tried"].append(cand_record)
                    continue

                # T acceptance check — skipped if no T pool configured (--t-size 0)
                if t_seeds:
                    _t_t = time.perf_counter()
                    t_eval = evaluator.evaluate(
                        stage="T",
                        seeds=t_seeds,
                        incumbent_rewards=t_incumbent_rewards,
                        candidate_prompt=mut_out.revised_prompt,
                        module=cand["module"],
                        env=args.env, pipeline=args.pipeline,
                        prompt_variant=args.prompt_variant, model=args.model,
                        inference_seed=args.inference_seed,
                        workers=_dynamic_workers(args.workers, len(t_seeds)),
                        history_window=args.history_window,
                        actor_history_window=getattr(args, "actor_history_window", None),
                        log_dir=opt_cycle_dir,
                    )
                    cand_record["t_result"] = {
                        "verdict":            t_eval.verdict,
                        "p_value":            t_eval.p_value,
                        "net_mean_reward":    t_eval.net_mean_reward,
                        "n_positive":         t_eval.n_positive,
                        "n_negative":         t_eval.n_negative,
                        "n_tied":             t_eval.n_tied,
                        "challenger_rewards": t_eval.challenger_rewards,
                        "wall_time_s":        round(time.perf_counter() - _t_t, 2),
                        "n_episodes":         len(t_seeds),
                    }
                    t_verdict = t_eval.verdict
                    t_challenger_rewards = t_eval.challenger_rewards
                    print(f"    T verdict: {t_eval.verdict}  "
                          f"net_reward={t_eval.net_mean_reward}  p={t_eval.p_value}")
                else:
                    cand_record["t_result"] = {"verdict": "accepted", "note": "no_t_pool"}
                    t_verdict = "accepted"
                    t_challenger_rewards = []
                    print(f"    T verdict: accepted (no T pool — t_size=0)")

                cycle_record["candidates_tried"].append(cand_record)

                if t_verdict == "accepted":
                    if cand["module"] == "actor":
                        incumbent_agent_prompt = mut_out.revised_prompt
                        incumbent_agent_path.write_text(incumbent_agent_prompt)
                    else:
                        incumbent_descriptor_prompt = mut_out.revised_prompt
                        incumbent_descriptor_path.write_text(incumbent_descriptor_prompt)

                    if t_challenger_rewards:
                        t_incumbent_rewards = t_challenger_rewards

                    update_outcome(hereditary_log, hkey, "accepted")
                    new_prompt_ver = prompt_hash(mut_out.revised_prompt)
                    opt_outcome    = "accepted"
                    print(f"\n  *** ACCEPTED candidate {rank} ***")
                    break
                else:
                    update_outcome(hereditary_log, hkey, "rejected")

            cycle_record["opt_cycle_outcome"] = opt_outcome
            cycle_record["new_prompt_version"] = new_prompt_ver

        # -- env_round after opt_cycle: fresh V bag with current incumbent --
        # Seeds start at env_seed + v_bag_size + t_size + (opt_cycle-1)*v_bag_size
        # ensuring no collision with T seeds or previous V bags.
        round_offset = args.v_bag_size + args.t_size + (opt_cycle - 1) * args.v_bag_size
        round_seeds  = [args.env_seed + round_offset + i for i in range(args.v_bag_size)]

        _t_round_post = time.perf_counter()
        round_dir, v_trajectory, v_rewards = _run_env_round(
            round_num=opt_cycle,
            seeds=round_seeds,
            incumbent_agent_path=incumbent_agent_path,
            incumbent_descriptor_path=incumbent_descriptor_path,
            log_dir=run_dir / f"env_round_{opt_cycle}",
            args=args,
        )
        v_seeds   = round_seeds
        v_ep_nums = list(range(1, args.v_bag_size + 1))

        _candidates = cycle_record["candidates_tried"]
        cycle_record["env_round_out_dir"]          = str(round_dir)
        cycle_record["env_round_post_wall_time_s"] = round(time.perf_counter() - _t_round_post, 2)
        cycle_record["env_round_post_n_episodes"]  = len(round_seeds)
        cycle_record["n_candidates_tried"]         = len(_candidates)
        cycle_record["n_v_evals"]                  = sum(1 for c in _candidates if c.get("v_result") is not None)
        cycle_record["n_t_evals"]                  = sum(1 for c in _candidates if c.get("t_result") is not None and c["t_result"].get("note") != "no_t_pool")
        cycle_record["wall_time_s"]                = round(time.perf_counter() - t0, 2)
        _append_log(opt_log, cycle_record)

        print(f"\n  opt_cycle {opt_cycle} complete: "
              f"{cycle_record.get('opt_cycle_outcome')}  "
              f"({cycle_record['wall_time_s']:.1f}s)")

    _append_log(opt_log, {
        "record_type":       "run_summary",
        "opt_cycles":        args.opt_cycles,
        "total_wall_time_s": round(time.perf_counter() - run_start_time, 2),
        "env":               args.env,
        "workers":           args.workers,
        "ba_episodes":       args.ba_episodes,
        "max_skip_resample": args.max_skip_resample,
        "t_size":            args.t_size,
    })

    print(f"\n{'=' * 60}")
    print("Optimisation complete.")
    print(f"  Run directory        : {run_dir}")
    print(f"  Final agent prompt   : {incumbent_agent_path}")
    print(f"  Final desc. prompt   : {incumbent_descriptor_path}")
    print(f"  Optimisation log     : {opt_log}")
    print(f"  Hereditary log       : {hereditary_log}")


if __name__ == "__main__":
    run(parse_args())
