"""Optimisation progress tracker.

Reads optimisation_log.jsonl and run_config.json from each task run and
prints a compact live summary.

Usage:
    python tools/opt_progress.py
    python tools/opt_progress.py --opt-dir optimization_runs
    python tools/opt_progress.py --run-id babyai/openai--gpt-oss-20b/primary_20_20260520_143022/spa_mean_valbag_t005_rich
    watch -n 60 'python tools/opt_progress.py'

The tool auto-detects all slug directories under --opt-dir by searching
for optimisation_log.jsonl files. Pass --run-id to inspect a specific slug.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Legacy constants — kept for backward-compat display only, not used for path scanning
VARIANTS = ["minimal", "rich"]
TASKS    = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to",
            "mixed_train_goto", "mixed_train_pickup", "mixed_train_open",
            "mixed_train_putnext", "mixed_train_pick_up_seq_go_to"]


# ── Log reading ──────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return records


def _load_run(run_dir: Path) -> dict:
    """Return a summary dict for one optimisation run directory."""
    cfg_path = run_dir / "run_config.json"
    log_path = run_dir / "optimisation_log.jsonl"

    if not cfg_path.exists():
        return {"status": "waiting", "cycles_done": 0, "opt_cycles": 20}

    with open(cfg_path) as f:
        cfg = json.load(f)

    opt_cycles  = cfg.get("opt_cycles", 20)
    records     = _read_jsonl(log_path)
    cycle_recs  = [r for r in records if r.get("record_type") == "opt_cycle"]
    cycles_done = len(cycle_recs)

    # Outcome distribution
    outcomes = {"accepted": 0, "ba_skip": 0, "rejected": 0, "other": 0}
    for r in cycle_recs:
        o = r.get("opt_cycle_outcome", "")
        if o == "accepted":
            outcomes["accepted"] += 1
        elif o == "ba_skip":
            outcomes["ba_skip"] += 1
        elif o in ("rejected", "rejected_all", "all_candidates_failed"):
            outcomes["rejected"] += 1
        else:
            outcomes["other"] += 1

    # Latest V incumbent mean reward
    v_mean = None
    if cycle_recs:
        last = cycle_recs[-1]
        rewards = last.get("v_incumbent_rewards") or []
        if rewards:
            v_mean = sum(rewards) / len(rewards)

    # Latest BA output type
    last_ba = None
    if cycle_recs:
        last_ba = cycle_recs[-1].get("ba_output", {}).get("type")

    # Latest V net mean reward (challenger vs incumbent)
    last_v_net = None
    if cycle_recs:
        tried = cycle_recs[-1].get("candidates_tried") or []
        for cand in tried:
            v_res = cand.get("v_result") or {}
            if v_res.get("net_mean_reward") is not None:
                last_v_net = v_res["net_mean_reward"]
                break

    v_bag_size = cfg.get("v_bag_size", 20)
    t_size     = cfg.get("t_size", 0)

    # Active env_round: round 0 while no cycles done, else round = cycles_done
    active_round    = cycles_done  # env_round_0 before cycle 1, env_round_N during cycle N+1
    round_eps_done  = 0
    round_eps_total = v_bag_size + (t_size if active_round == 0 else 0)

    round_dir = run_dir / f"env_round_{active_round}"
    if round_dir.exists():
        round_eps_done = len(list(round_dir.rglob("episode_*.done"))) or len(list(round_dir.rglob("episode_*.gif")))

    # Status
    if cycles_done >= opt_cycles:
        status = "done"
    elif cycles_done == 0 and not log_path.exists():
        status = "env_round_0"
    elif cycles_done == 0:
        status = "cycle_1_running"
    else:
        status = f"cycle_{cycles_done + 1}_running"

    # Wall time: last cycle and total
    last_wall  = None
    total_wall = None
    if cycle_recs:
        last_wall  = cycle_recs[-1].get("wall_time_s")
        walls      = [r.get("wall_time_s") for r in cycle_recs if r.get("wall_time_s") is not None]
        total_wall = round(sum(walls), 1) if walls else None

    return {
        "status":           status,
        "cycles_done":      cycles_done,
        "opt_cycles":       opt_cycles,
        "outcomes":         outcomes,
        "v_mean":           v_mean,
        "last_v_net":       last_v_net,
        "last_ba":          last_ba,
        "last_wall":        last_wall,
        "total_wall":       total_wall,
        "active_round":     active_round,
        "round_eps_done":   round_eps_done,
        "round_eps_total":  round_eps_total,
    }


# ── Display helpers ──────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 10) -> str:
    if not total:
        return "[" + "?" * width + "]"
    filled = int(width * done / total)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def _fmt_status(info: dict) -> str:
    s    = info["status"]
    done = info.get("round_eps_done", 0)
    total= info.get("round_eps_total", 0)
    eps  = f" ({done}/{total} eps)" if total else ""
    if s == "done":        return "done"
    if s == "waiting":     return "waiting"
    if s == "env_round_0": return f"round_0 running{eps}"
    n = s.replace("_running", "").replace("cycle_", "cycle ")
    return f"{n} running{eps}"


def _fmt_v_mean(info: dict) -> str:
    v = info.get("v_mean")
    return f"{v:.2f}" if v is not None else "  —  "


def _fmt_net(info: dict) -> str:
    v = info.get("last_v_net")
    if v is None:
        return "   —  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"


def _fmt_outcomes(info: dict) -> str:
    o = info.get("outcomes", {})
    return f"A:{o.get('accepted',0)} S:{o.get('ba_skip',0)} R:{o.get('rejected',0)}"


def _fmt_wall(seconds: float | None) -> str:
    if seconds is None:
        return "  —  "
    if seconds < 3600:
        return f"{seconds/60:.0f}m"
    return f"{seconds/3600:.1f}h"


# ── Structure detection ───────────────────────────────────────────────────────

def _build_run_rows(base_dir: Path) -> list[tuple[str, Path]]:
    """Return [(label, run_dir), ...] by scanning base_dir for task subdirectories.

    Supports both the new structure ({slug}/{task}/opt_log) and legacy structures
    ({exp}/{variant}/{task}/opt_log and {exp}/{constraint}/{variant}/{task}/opt_log)
    by scanning for directories that contain optimisation_log.jsonl or run_config.json
    at increasing depth levels.
    """
    def _has_run_files(d: Path) -> bool:
        return (d / "optimisation_log.jsonl").exists() or (d / "run_config.json").exists()

    # 1-level deep: new structure — {slug}/{task}/
    rows = sorted(
        [(d.name, d) for d in base_dir.iterdir() if d.is_dir() and _has_run_files(d)],
        key=lambda x: x[0],
    )
    if rows:
        return rows

    # 2-level deep: legacy standard — {exp}/{variant}/{task}/
    rows = sorted(
        [(f"{v.name}/{t.name}", t)
         for v in base_dir.iterdir() if v.is_dir()
         for t in v.iterdir() if t.is_dir() and _has_run_files(t)],
        key=lambda x: x[0],
    )
    if rows:
        return rows

    # 3-level deep: legacy module ablation — {exp}/{constraint}/{variant}/{task}/
    rows = sorted(
        [(f"{c.name}/{v.name}/{t.name}", t)
         for c in base_dir.iterdir() if c.is_dir()
         for v in c.iterdir() if v.is_dir()
         for t in v.iterdir() if t.is_dir() and _has_run_files(t)],
        key=lambda x: x[0],
    )
    return rows


# ── Per-experiment display ────────────────────────────────────────────────────

def _print_experiment(base_dir: Path, show_history: bool = False) -> None:
    rows = _build_run_rows(base_dir)

    runs = {label: _load_run(run_dir) for label, run_dir in rows}

    n_total   = len(runs)
    n_done    = sum(1 for r in runs.values() if r["status"] == "done")
    n_active  = sum(1 for r in runs.values() if r["status"] not in ("done", "waiting"))
    n_waiting = sum(1 for r in runs.values() if r["status"] == "waiting")
    opt_cycles = next(iter(runs.values()))["opt_cycles"]
    total_cyc  = sum(r["cycles_done"] for r in runs.values())
    total_exp  = n_total * opt_cycles

    print(f"\n{'='*72}")
    print(f"  {base_dir.name}")
    print(f"{'='*72}")
    print(f"  Runs    : {n_done}/{n_total} done   {n_active} active   {n_waiting} waiting")
    print(f"  Cycles  : {total_cyc}/{total_exp} total across all runs")

    print(f"\n{'─'*80}")
    print(f"  {'run':<30}  {'progress':<14}  {'V mean':>6}  {'Δ last':>6}  {'outcomes':<14}  {'elapsed':>7}  status")
    print(f"{'─'*80}")

    prev_group = None
    for label, run_dir in rows:
        info      = runs[label]
        group     = label.split("/")[0]
        if group != prev_group and prev_group is not None:
            print()
        prev_group = group
        bar       = _bar(info["cycles_done"], info["opt_cycles"])
        cycle_str = f"{bar} {info['cycles_done']:>2}/{info['opt_cycles']}"
        print(
            f"  {label:<30}  {cycle_str:<14}  "
            f"{_fmt_v_mean(info):>6}  "
            f"{_fmt_net(info):>6}  "
            f"{_fmt_outcomes(info):<14}  "
            f"{_fmt_wall(info.get('total_wall')):>7}  "
            f"{_fmt_status(info)}"
        )

    active_runs = [
        (label, run_dir) for label, run_dir in rows
        if runs[label]["status"] not in ("done", "waiting")
        and runs[label]["cycles_done"] > 0
    ]
    if active_runs and show_history:
        print(f"\n{'─'*72}")
        print("  Recent cycle history (active runs)")
        print(f"{'─'*72}")
        for label, run_dir in active_runs:
            log_path = run_dir / "optimisation_log.jsonl"
            records  = [r for r in _read_jsonl(log_path) if r.get("record_type") == "opt_cycle"]
            print(f"  {label}")
            for r in records[-3:]:
                cyc     = r.get("opt_cycle", "?")
                outcome = r.get("opt_cycle_outcome", "?")
                rewards = r.get("v_incumbent_rewards") or []
                vmean   = f"{sum(rewards)/len(rewards):.2f}" if rewards else " — "
                wall    = r.get("wall_time_s")
                wall_s  = f"{wall:.0f}s" if wall else "?"
                ba_type = r.get("ba_output", {}).get("type", "?")
                n_cand  = r.get("n_candidates_tried", len(r.get("candidates_tried") or []))
                n_v     = r.get("n_v_evals", "?")
                n_t     = r.get("n_t_evals", "?")
                net_str = " — "
                for c in (r.get("candidates_tried") or []):
                    vr = c.get("v_result") or {}
                    if vr.get("net_mean_reward") is not None:
                        net_str = f"{vr['net_mean_reward']:+.2f}"
                        break
                print(f"    cycle {cyc:>2}  outcome={outcome:<22}  V={vmean}  Δ={net_str}  BA={ba_type}  cand={n_cand} V-evals={n_v} T-evals={n_t}  ({wall_s})")
            print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimisation progress tracker")
    parser.add_argument("--opt-dir", default="optimization_runs",
                        help="Root optimisation runs directory (default: optimization_runs)")
    parser.add_argument("--run-id", nargs="+", default=None,
                        help="One or more run IDs to inspect (default: all runs auto-detected)")
    parser.add_argument("--history", action="store_true",
                        help="Show recent cycle history for active runs (default: off)")
    args = parser.parse_args()

    opt_dir = Path(args.opt_dir)
    now = time.strftime("%H:%M:%S")
    print(f"\n  Optimisation Progress   [{now}]")

    if args.run_id:
        base_dirs = [opt_dir / rid for rid in args.run_id]
    else:
        # Auto-detect slug directories by finding all optimisation_log.jsonl files
        # and taking their parent's parent (task_dir.parent = slug_dir)
        logs = list(opt_dir.rglob("optimisation_log.jsonl"))
        slug_dirs = sorted(
            set(p.parent.parent for p in logs),
            key=lambda d: d.stat().st_mtime,
        )
        if not slug_dirs:
            print(f"No optimisation run directories found under {opt_dir}")
            return
        base_dirs = slug_dirs

    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"\n  [not found] {base_dir}")
            continue
        _print_experiment(base_dir, show_history=args.history)

    print(f"{'='*72}\n")
    print("  Refresh:  watch -n 60 'python tools/opt_progress.py'")
    print("  Single:   python tools/opt_progress.py --run-id babyai/openai--gpt-oss-20b/primary_20_20260520_143022/spa_mean_valbag_t005_rich")
    print()


if __name__ == "__main__":
    main()
