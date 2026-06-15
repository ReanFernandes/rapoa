"""Fresh eval progress tracker for optimised prompts.

Scans logs_fresh_eval_optimised/ and shows per-campaign, per-slug, per-task
completion status and success rates. Cross-references optimization_runs/ to
show slugs whose optimisation is still running or pending fresh eval.

Usage:
    python tools/progress.py
    python tools/progress.py --campaign primary_20_20260522_110326
    python tools/progress.py --smoke        # include smoke test campaigns
    watch -n 60 'python tools/progress.py'
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

_CAMPAIGN_RE = re.compile(r".+_\d{8}_\d{6}$")

TASK_SHORT = {
    "goto":              "goto",
    "pickup":            "pickup",
    "open":              "open",
    "putnext":           "putnext",
    "pick_up_seq_go_to": "seq",
}

SKIP_PREFIXES = ("smoke_test",)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 10) -> str:
    if not total:
        return "[" + "?" * width + "]"
    filled = int(width * done / total)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def _fmt_sr(sr: float | None) -> str:
    return f"{sr:.0%}" if sr is not None else "  —  "


# ── Optimisation cross-reference ──────────────────────────────────────────────

def _opt_slug_complete(slug_dir: Path) -> bool:
    """True if every task under this optimisation slug has a run_summary record."""
    task_dirs = [d for d in slug_dir.iterdir() if d.is_dir()]
    if not task_dirs:
        return False
    for task_dir in task_dirs:
        log = task_dir / "optimisation_log.jsonl"
        if not log.exists():
            return False
        lines = [l.strip() for l in log.read_text().splitlines() if l.strip()]
        if not lines:
            return False
        try:
            if json.loads(lines[-1]).get("record_type") != "run_summary":
                return False
        except json.JSONDecodeError:
            return False
    return True


def get_opt_slug_status(opt_campaign_dir: Path) -> dict[str, str]:
    """
    Return {slug_name: status} for all slugs in an optimisation campaign dir.
    Status: "opt_running" | "eval_pending"
    """
    if not opt_campaign_dir or not opt_campaign_dir.exists():
        return {}
    result = {}
    for slug_dir in opt_campaign_dir.iterdir():
        if not slug_dir.is_dir():
            continue
        result[slug_dir.name] = (
            "eval_pending" if _opt_slug_complete(slug_dir) else "opt_running"
        )
    return result


# ── Fresh eval data loading ───────────────────────────────────────────────────

def _load_run_dir(run_dir: Path) -> dict:
    """Load one (iseed, timestamp) run directory."""
    summary = run_dir / "run_summary.json"
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)
        return {
            "status":       "complete",
            "n_episodes":   data.get("n_episodes", 0),
            "success_rate": data.get("summary", {}).get("success_rate"),
        }
    ep_done = len(list(run_dir.glob("episode_*.done")))
    if ep_done:
        return {"status": "in_progress", "n_episodes": ep_done, "success_rate": None}
    return {"status": "waiting", "n_episodes": 0, "success_rate": None}


def _find_run_dirs(slug_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Return [(task, iseed, run_dir), ...] for all runs under a slug directory.
    Path layout: {env_family}/{task}/{model}/{pipeline}/{variant}/{conv}/{reasoning}/{iseed}/{ts}/
    """
    seen: set[tuple[str, str]] = set()
    results = []
    for sentinel in [*slug_dir.rglob("run_summary.json"), *slug_dir.rglob("episode_001.done")]:
        run_dir = sentinel.parent
        if sentinel.name == "episode_001.done" and (run_dir / "run_summary.json").exists():
            continue
        parts = sentinel.relative_to(slug_dir).parts
        if len(parts) < 9:
            continue
        task, iseed = parts[1], parts[7]
        key = (task, iseed)
        if key in seen:
            continue
        seen.add(key)
        results.append((task, iseed, run_dir))
    return results


def load_slug_tasks(slug_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {task: {iseed: run_info}} for all runs under slug_dir."""
    tasks: dict[str, dict[str, dict]] = defaultdict(dict)
    for task, iseed, run_dir in _find_run_dirs(slug_dir):
        tasks[task][iseed] = _load_run_dir(run_dir)
    return dict(tasks)


def _agg(seeds: dict[str, dict]) -> dict:
    """Aggregate per-task seed runs into summary stats."""
    runs = list(seeds.values())
    nc = sum(1 for r in runs if r["status"] == "complete")
    ni = sum(1 for r in runs if r["status"] == "in_progress")
    ep_done = sum(r["n_episodes"] for r in runs)
    srs = [r["success_rate"] for r in runs if r["success_rate"] is not None]
    ep_per_seed = next(
        (r["n_episodes"] for r in runs if r["status"] == "complete" and r["n_episodes"]), 20
    )
    return {
        "n_complete":   nc,
        "n_inprog":     ni,
        "n_total":      len(runs),
        "ep_done":      ep_done,
        "ep_expected":  len(runs) * ep_per_seed,
        "ep_per_seed":  ep_per_seed,
        "success_rate": sum(srs) / len(srs) if srs else None,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def _print_campaign(campaign_dir: Path, opt_campaign_dir: Path | None) -> None:
    eval_slug_dirs = {d.name: d for d in campaign_dir.iterdir() if d.is_dir()}
    opt_slug_status = get_opt_slug_status(opt_campaign_dir)

    # All expected slugs = union of what's in fresh eval + what's in opt_runs
    all_slugs = sorted(set(eval_slug_dirs) | set(opt_slug_status))
    if not all_slugs:
        return

    # Load fresh eval data for slugs that have it
    slug_data: dict[str, dict | None] = {}
    for slug_name in all_slugs:
        slug_data[slug_name] = load_slug_tasks(eval_slug_dirs[slug_name]) if slug_name in eval_slug_dirs else None

    # Campaign-level totals — include estimates for missing slugs
    present_aggs = [
        _agg(seeds)
        for td in slug_data.values() if td is not None
        for seeds in td.values()
    ]
    total_nc  = sum(a["n_complete"] for a in present_aggs)
    total_nt  = sum(a["n_total"]    for a in present_aggs)
    total_epd = sum(a["ep_done"]    for a in present_aggs)
    total_epe = sum(a["ep_expected"]for a in present_aggs)

    # Estimate missing slug contributions from completed slugs
    n_missing = sum(1 for td in slug_data.values() if td is None)
    if n_missing and present_aggs:
        n_present_slugs = sum(1 for td in slug_data.values() if td is not None)
        avg_nt  = total_nt  / n_present_slugs
        avg_epe = total_epe / n_present_slugs
        total_nt  += round(avg_nt  * n_missing)
        total_epe += round(avg_epe * n_missing)

    n_slugs_expected = len(all_slugs)
    n_slugs_done = sum(
        1 for slug, td in slug_data.items()
        if td is not None
        and all(_agg(seeds)["n_complete"] == _agg(seeds)["n_total"] > 0
                for seeds in td.values())
    )

    n_tasks_done_total = sum(
        1 for td in slug_data.values() if td is not None
        for seeds in td.values()
        if _agg(seeds)["n_complete"] == _agg(seeds)["n_total"] > 0
    )
    n_tasks_expected_total = n_slugs_expected * 5  # 5 tasks per slug

    campaign_bar = _bar(n_slugs_done, n_slugs_expected)
    print(f"\n{'='*72}")
    print(f"  {campaign_dir.name}")
    print(f"{'='*72}")
    print(f"  {campaign_bar} {n_slugs_done}/{n_slugs_expected} slugs   "
          f"Tasks: {n_tasks_done_total}/{n_tasks_expected_total}   "
          f"Episodes: {total_epd}/{total_epe}")
    print()
    print(f"  {'slug':<42}  {'tasks':>6}  {'episodes':>12}  {'SR':>5}  status")
    print(f"  {'─'*42}  {'─'*6}  {'─'*12}  {'─'*5}  {'─'*16}")

    for slug_name in all_slugs:
        td = slug_data[slug_name]

        # Slug not yet in fresh eval — show its optimisation status
        if td is None:
            opt_st = opt_slug_status.get(slug_name, "unknown")
            label = "opt running" if opt_st == "opt_running" else "eval pending"
            print(f"  {slug_name:<42}  0/—    0/—       —    {label}")
            print()
            continue

        slug_aggs = [_agg(seeds) for seeds in td.values()]
        snc  = sum(a["n_complete"] for a in slug_aggs)
        sni  = sum(a["n_inprog"]   for a in slug_aggs)
        snt  = sum(a["n_total"]    for a in slug_aggs)
        sepd = sum(a["ep_done"]    for a in slug_aggs)
        sepe = sum(a["ep_expected"]for a in slug_aggs)
        srs  = [a["success_rate"] for a in slug_aggs if a["success_rate"] is not None]
        slug_sr = sum(srs) / len(srs) if srs else None

        n_tasks_done = sum(
            1 for seeds in td.values()
            if _agg(seeds)["n_complete"] == _agg(seeds)["n_total"] > 0
        )
        n_tasks_total = len(td)
        if n_tasks_done == n_tasks_total and n_tasks_total > 0:
            status = "done"
        elif sni > 0:
            status = f"{sni} running"
        elif n_tasks_done > 0:
            status = f"{n_tasks_done}/{n_tasks_total} tasks"
        else:
            status = "waiting"

        print(f"  {slug_name:<42}  {n_tasks_done}/{n_tasks_total}  {sepd:>5}/{sepe:<5}  {_fmt_sr(slug_sr):>5}  {status}")

        # Per-task detail for incomplete slugs
        if n_tasks_done < n_tasks_total and n_tasks_total > 0:
            for task in sorted(td.keys()):
                a = _agg(td[task])
                short = TASK_SHORT.get(task, task)
                nc, ni, nt = a["n_complete"], a["n_inprog"], a["n_total"]
                if nc == nt and nt > 0:
                    t_status = "✓"
                elif ni > 0:
                    t_status = f"●{a['ep_done']}/{a['ep_expected']}"
                else:
                    t_status = "·"
                print(
                    f"    {'└─ ' + short:<40}  "
                    f"{'':>3}{nc:>2}/{nt:<3}  "
                    f"{a['ep_done']:>5}/{a['ep_expected']:<5}  "
                    f"{_fmt_sr(a['success_rate']):>5}  {t_status}"
                )
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh eval progress tracker")
    parser.add_argument("--log-dir", default="logs_fresh_eval_optimised",
                        help="Root fresh eval directory (default: logs_fresh_eval_optimised)")
    parser.add_argument("--opt-dir", default="optimization_runs",
                        help="Root optimisation runs directory for cross-reference "
                             "(default: optimization_runs)")
    parser.add_argument("--campaign", nargs="+", default=None, metavar="ID",
                        help="Filter to specific campaign IDs")
    parser.add_argument("--smoke", action="store_true",
                        help="Include smoke test campaigns (excluded by default)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    opt_dir = Path(args.opt_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    now = time.strftime("%H:%M:%S")
    print(f"\n  Fresh Eval Progress   [{now}]")

    # Collect all campaign dirs from fresh eval log root
    campaign_dirs = []
    for env_dir in sorted(log_dir.iterdir()):
        if not env_dir.is_dir():
            continue
        for model_dir in sorted(env_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for campaign_dir in sorted(model_dir.iterdir()):
                if not campaign_dir.is_dir():
                    continue
                if not _CAMPAIGN_RE.match(campaign_dir.name):
                    continue
                if not args.smoke and any(campaign_dir.name.startswith(p) for p in SKIP_PREFIXES):
                    continue
                if args.campaign and campaign_dir.name not in args.campaign:
                    continue
                campaign_dirs.append((env_dir.name, model_dir.name, campaign_dir))

    if not campaign_dirs:
        print("No fresh eval campaigns found.")
        return

    for env_name, model_name, campaign_dir in campaign_dirs:
        opt_campaign_dir = opt_dir / env_name / model_name / campaign_dir.name
        _print_campaign(campaign_dir, opt_campaign_dir)

    print(f"{'='*72}\n")
    print("  Refresh:  watch -n 60 'python tools/progress.py'")
    print()


if __name__ == "__main__":
    main()
