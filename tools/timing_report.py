"""Compute cost estimation for optimisation runs.

Reads optimisation_log.jsonl files and prints a table of wall-clock times,
LLM token counts, and per-phase breakdowns — suitable for reporting in a
paper's compute cost section.

Usage:
    python tools/timing_report.py
    python tools/timing_report.py --opt-dir optimization_runs/babyai/gpt-oss-20b/primary_20_xxx
    python tools/timing_report.py --run-id babyai/gpt-oss-20b/primary_20_xxx/spa_mean_valbag_t005_rich/mixed_train_goto
    python tools/timing_report.py --csv   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_OPT_DIR = Path("optimization_runs")


# ── Log reading ───────────────────────────────────────────────────────────────

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


# ── Per-run analysis ──────────────────────────────────────────────────────────

def _analyse_run(run_dir: Path) -> dict | None:
    log_path = run_dir / "optimisation_log.jsonl"
    cfg_path = run_dir / "run_config.json"
    if not log_path.exists():
        return None

    records     = _read_jsonl(log_path)
    cycle_recs  = [r for r in records if r.get("record_type") == "opt_cycle"]
    setup_rec   = next((r for r in records if r.get("record_type") == "env_round_setup"), None)
    summary_rec = next((r for r in records if r.get("record_type") == "run_summary"), None)

    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)

    n_cycles = len(cycle_recs)
    if n_cycles == 0:
        return None

    # ── Wall times ────────────────────────────────────────────────────────────
    cycle_walls   = [r.get("wall_time_s") for r in cycle_recs if r.get("wall_time_s") is not None]
    setup_wall    = setup_rec.get("wall_time_s") if setup_rec else None
    total_wall    = summary_rec.get("total_wall_time_s") if summary_rec else None
    if total_wall is None and cycle_walls:
        total_wall = sum(cycle_walls) + (setup_wall or 0)

    ba_walls      = [r.get("ba_wall_time_s") for r in cycle_recs if r.get("ba_wall_time_s") is not None]
    round_walls   = [r.get("env_round_post_wall_time_s") for r in cycle_recs if r.get("env_round_post_wall_time_s") is not None]

    # Mutator+eval wall = cycle_wall - ba_wall - env_round_post_wall
    mut_eval_walls: list[float] = []
    for r in cycle_recs:
        cw = r.get("wall_time_s")
        bw = r.get("ba_wall_time_s")
        rw = r.get("env_round_post_wall_time_s")
        if cw is not None and bw is not None and rw is not None:
            mut_eval_walls.append(max(0.0, cw - bw - rw))

    # ── Token counts ──────────────────────────────────────────────────────────
    ba_pt = ba_ct = 0
    for r in cycle_recs:
        bo = r.get("ba_output") or {}
        ba_pt += bo.get("prompt_tokens") or 0
        ba_ct += bo.get("completion_tokens") or 0
        # Also accumulate per-attempt detail if present
        for att in r.get("ba_attempts_detail") or []:
            # ba_output already covers the final attempt; detail covers all
            pass  # use ba_output totals only (final attempt = what matters)

    mut_pt = mut_ct = 0
    v_eps_total = t_eps_total = 0
    for r in cycle_recs:
        for cand in r.get("candidates_tried") or []:
            mut_pt += cand.get("mutator_prompt_tokens") or 0
            mut_ct += cand.get("mutator_completion_tokens") or 0
            vr = cand.get("v_result") or {}
            tr = cand.get("t_result") or {}
            v_eps_total += vr.get("n_episodes") or 0
            if tr.get("note") != "no_t_pool":
                t_eps_total += tr.get("n_episodes") or 0

    # Round episodes (from env_round_setup + post-cycle rounds)
    env_eps_total = 0
    if setup_rec:
        v_seeds = setup_rec.get("v_seeds") or []
        t_seeds = setup_rec.get("t_seeds") or []
        env_eps_total += len(v_seeds) + len(t_seeds)
    for r in cycle_recs:
        env_eps_total += r.get("env_round_post_n_episodes") or 0

    # ── Eval counts ───────────────────────────────────────────────────────────
    n_v_evals = sum(r.get("n_v_evals") or 0 for r in cycle_recs)
    n_t_evals = sum(r.get("n_t_evals") or 0 for r in cycle_recs)
    n_cands   = sum(r.get("n_candidates_tried") or 0 for r in cycle_recs)
    n_ba_skips = sum(1 for r in cycle_recs if r.get("opt_cycle_outcome") == "ba_skip")

    return {
        "run_dir":          run_dir,
        "env":              cfg.get("env") or summary_rec.get("env") if summary_rec else "?",
        "n_cycles":         n_cycles,
        "opt_cycles":       cfg.get("opt_cycles", "?"),
        "workers":          cfg.get("workers") or (summary_rec.get("workers") if summary_rec else None),
        "ba_episodes":      cfg.get("ba_episodes"),
        "t_size":           cfg.get("t_size"),
        # Wall times
        "total_wall_s":     total_wall,
        "setup_wall_s":     setup_wall,
        "mean_cycle_wall_s":sum(cycle_walls) / len(cycle_walls) if cycle_walls else None,
        "mean_ba_wall_s":   sum(ba_walls) / len(ba_walls) if ba_walls else None,
        "mean_round_wall_s":sum(round_walls) / len(round_walls) if round_walls else None,
        "mean_mut_eval_wall_s": sum(mut_eval_walls) / len(mut_eval_walls) if mut_eval_walls else None,
        # Tokens
        "ba_prompt_tokens":       ba_pt,
        "ba_completion_tokens":   ba_ct,
        "mut_prompt_tokens":      mut_pt,
        "mut_completion_tokens":  mut_ct,
        # Eval counts
        "n_cands":          n_cands,
        "n_v_evals":        n_v_evals,
        "n_t_evals":        n_t_evals,
        "n_ba_skips":       n_ba_skips,
        "env_eps_total":    env_eps_total,
        "v_eps_total":      v_eps_total,
        "t_eps_total":      t_eps_total,
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float | None, width: int = 8) -> str:
    if seconds is None:
        return f"{'—':>{width}}"
    if seconds < 120:
        return f"{seconds:.0f}s".rjust(width)
    if seconds < 3600:
        return f"{seconds/60:.1f}m".rjust(width)
    return f"{seconds/3600:.2f}h".rjust(width)


def _fmt_pct(part: float | None, total: float | None) -> str:
    if part is None or not total:
        return "  — "
    return f"{100*part/total:4.0f}%"


def _fmt_k(n: int) -> str:
    if n == 0:
        return "0"
    if n < 1000:
        return str(n)
    return f"{n/1000:.1f}k"


def _print_run(info: dict) -> None:
    run_parts = info["run_dir"].parts
    # Show last 3 path components for readability
    label = "/".join(run_parts[-3:]) if len(run_parts) >= 3 else str(info["run_dir"])

    tw   = info["total_wall_s"]
    mcw  = info["mean_cycle_wall_s"]
    mbaw = info["mean_ba_wall_s"]
    mrw  = info["mean_round_wall_s"]
    mmew = info["mean_mut_eval_wall_s"]

    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"{'─'*72}")
    print(f"  Cycles completed   : {info['n_cycles']} / {info['opt_cycles']}")
    print(f"  Workers            : {info['workers'] or '?'}")
    print(f"  BA episodes        : {info['ba_episodes'] or '?'}   T size: {info['t_size'] or '?'}")
    print()
    print(f"  Total wall time    : {_fmt_time(tw).strip()}")
    if info["setup_wall_s"]:
        print(f"    setup (env_round_0)    : {_fmt_time(info['setup_wall_s']).strip()}")
    if mcw is not None:
        print(f"    mean per opt_cycle     : {_fmt_time(mcw).strip()}")
        if mbaw is not None:
            print(f"      ├─ BA phase          : {_fmt_time(mbaw).strip()}  {_fmt_pct(mbaw, mcw)}")
        if mmew is not None:
            print(f"      ├─ mutator+eval       : {_fmt_time(mmew).strip()}  {_fmt_pct(mmew, mcw)}")
        if mrw is not None:
            print(f"      └─ env_round (post)   : {_fmt_time(mrw).strip()}  {_fmt_pct(mrw, mcw)}")
    print()
    print(f"  Episode counts")
    print(f"    env_round (rollout) : {info['env_eps_total']}")
    print(f"    V evals             : {info['v_eps_total']}  ({info['n_v_evals']} eval runs)")
    print(f"    T evals             : {info['t_eps_total']}  ({info['n_t_evals']} eval runs)")
    print(f"    BA skips            : {info['n_ba_skips']} / {info['n_cycles']} cycles")
    print()
    ba_total_t = info["ba_prompt_tokens"] + info["ba_completion_tokens"]
    mut_total_t = info["mut_prompt_tokens"] + info["mut_completion_tokens"]
    total_t = ba_total_t + mut_total_t
    if total_t > 0:
        print(f"  Token counts (BA + Mutator LLM calls only)")
        print(f"    BA    : {_fmt_k(info['ba_prompt_tokens'])} in / {_fmt_k(info['ba_completion_tokens'])} out")
        print(f"    Mutator: {_fmt_k(info['mut_prompt_tokens'])} in / {_fmt_k(info['mut_completion_tokens'])} out")
        print(f"    Total : {_fmt_k(ba_total_t + mut_total_t)} tokens  "
              f"(excludes rollout actor/descriptor — see trajectory.jsonl)")
    else:
        print(f"  Token counts: not available (run older format, no token fields in log)")


def _save_csv(results: list[dict], output_path: str = "timing_results.csv") -> None:
    cols = [
        "run_dir", "n_cycles", "opt_cycles", "workers",
        "total_wall_s", "mean_cycle_wall_s", "mean_ba_wall_s",
        "mean_round_wall_s", "mean_mut_eval_wall_s",
        "env_eps_total", "v_eps_total", "t_eps_total",
        "n_v_evals", "n_t_evals", "n_ba_skips",
        "ba_prompt_tokens", "ba_completion_tokens",
        "mut_prompt_tokens", "mut_completion_tokens",
    ]
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for info in results:
                row = []
                for c in cols:
                    v = info.get(c)
                    row.append("" if v is None else str(v))
                f.write(",".join(row) + "\n")
        print(f"CSV output successfully saved to {output_path}")
    except OSError as e:
        print(f"Error saving CSV file: {e}", file=sys.stderr)


# ── Run discovery ─────────────────────────────────────────────────────────────

def _find_run_dirs(base: Path) -> list[Path]:
    """Return all directories under base that contain an optimisation_log.jsonl."""
    return sorted(p.parent for p in base.rglob("optimisation_log.jsonl"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute cost report for optimisation runs.")
    ap.add_argument("--opt-dir",  default=None,
                    help="Root directory to scan (e.g. optimization_runs/babyai/gpt-oss-20b/primary_20_xxx)")
    ap.add_argument("--run-id",   default=None,
                    help="Specific run ID path (relative to --opt-dir root, or absolute)")
    ap.add_argument("--csv",      action="store_true",
                    help="Output machine-readable CSV instead of human-readable table")
    args = ap.parse_args()

    if args.run_id:
        run_dir = Path(args.run_id)
        if not run_dir.is_absolute():
            run_dir = _OPT_DIR / run_dir
        run_dirs = [run_dir]
    elif args.opt_dir:
        base = Path(args.opt_dir)
        run_dirs = _find_run_dirs(base)
        if not run_dirs:
            # Maybe it's a run dir itself
            run_dirs = [base]
    else:
        run_dirs = _find_run_dirs(_OPT_DIR)

    if not run_dirs:
        print("No optimisation_log.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for rd in run_dirs:
        info = _analyse_run(rd)
        if info:
            results.append(info)

    if not results:
        print("No completed runs found (0 opt_cycle records).", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        _save_csv(results, "timing_results.csv")
        return

    print(f"\nTiming Report — {len(results)} run(s)")
    for info in results:
        _print_run(info)

    if len(results) > 1:
        total_t = sum(r["total_wall_s"] or 0 for r in results if r["total_wall_s"])
        print(f"\n{'═'*72}")
        print(f"  AGGREGATE  ({len(results)} runs)   total elapsed: {_fmt_time(total_t).strip()}")


if __name__ == "__main__":
    main()