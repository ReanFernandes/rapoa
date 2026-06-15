"""
gen_all_results_table.py

Comprehensive results table — every campaign × slug × task.

Two subgroups per condition:
  Opt    : mean(t_incumbent_rewards) from the final opt_cycle in
           optimisation_log.jsonl (= T-pool SR of the final incumbent)
  Eval   : fresh eval success rate mean ± std across inference seeds

Results shown per task (goto / pickup / open / putnext / pick_up_seq_go_to)
plus an aggregate row (macro-average over all tasks with data).

Smoke-test campaigns are excluded automatically.

Usage:
    cd final_paper_plotting
    python gen_all_results_table.py                  # terminal table
    python gen_all_results_table.py --csv            # terminal + save CSV
    python gen_all_results_table.py --csv --no-term  # CSV only
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_PROJ       = _HERE.parent
OPT_ROOT    = _PROJ / "optimization_runs"  / "babyai" / "gpt-oss-20b"
EVAL_ROOT   = _PROJ / "logs_fresh_eval_optimised" / "babyai" / "gpt-oss-20b"
TABLES_DIR  = _HERE / "figures" / "tables"

TASKS = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]
TASK_OPT_NAMES = {
    "goto":              "mixed_train_goto",
    "pickup":            "mixed_train_pickup",
    "open":              "mixed_train_open",
    "putnext":           "mixed_train_putnext",
    "pick_up_seq_go_to": "mixed_train_pick_up_seq_go_to",
}
TASK_LABELS = {
    "goto":              "GoTo",
    "pickup":            "PickUp",
    "open":              "Open",
    "putnext":           "PutNext",
    "pick_up_seq_go_to": "Seq",
}

SKIP_PREFIX = "smoke_test"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_opt_t_score(log_path: Path) -> float | None:
    """Mean T-pool reward of the final incumbent (last opt_cycle record)."""
    try:
        records = [json.loads(l) for l in open(log_path)]
    except Exception:
        return None
    cycles = [r for r in records if r.get("record_type") == "opt_cycle"]
    if not cycles:
        return None
    last = cycles[-1]
    rewards = last.get("t_incumbent_rewards")
    if not rewards:
        return None
    return float(np.mean(rewards))


def load_opt_cycles(log_path: Path) -> int:
    """Number of completed opt cycles."""
    try:
        records = [json.loads(l) for l in open(log_path)]
    except Exception:
        return 0
    return sum(1 for r in records if r.get("record_type") == "opt_cycle")


def load_eval_sr(eval_slug_dir: Path, task: str) -> tuple[float, float] | None:
    """
    Fresh eval success rate: (mean, std) across inference seeds for one task.
    std is computed over per-seed means.
    Returns None if no data found.
    """
    seed_srs: dict[str, list[bool]] = defaultdict(list)

    for summary in sorted(eval_slug_dir.rglob("run_summary.json")):
        parts = summary.parts
        try:
            # Path: .../{slug}/BabyAI/{task}/{model}/{pipeline}/{variant}/{conv_mode}/{reasoning}/{iseed}/{ts}/run_summary.json
            slug_idx = next(i for i, p in enumerate(parts) if p == eval_slug_dir.name)
            task_in_path = parts[slug_idx + 2]   # skip the env-family dir (BabyAI)
        except (StopIteration, IndexError):
            continue
        if task_in_path != task:
            continue
        try:
            iseed_idx = next(i for i, p in enumerate(parts) if p.startswith("iseed_"))
            iseed = parts[iseed_idx]
        except StopIteration:
            iseed = "iseed_unknown"
        try:
            d = json.load(open(summary))
            seed_srs[iseed].extend(d["episodes"])
        except Exception:
            continue

    if not seed_srs:
        return None

    per_seed_means = [
        sum(1 for e in eps if e["success"]) / len(eps)
        for eps in seed_srs.values() if eps
    ]
    if not per_seed_means:
        return None
    return float(np.mean(per_seed_means)), float(np.std(per_seed_means))


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_runs() -> dict:
    """
    Returns nested dict: campaign → slug → task → {opt_log, eval_dir}
    Sorted alphabetically. Smoke-test campaigns excluded.
    """
    result = {}
    for campaign_dir in sorted(OPT_ROOT.iterdir()):
        if not campaign_dir.is_dir():
            continue
        if campaign_dir.name.startswith(SKIP_PREFIX):
            continue
        slugs = {}
        for slug_dir in sorted(campaign_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            tasks = {}
            for task_dir in sorted(slug_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                log = task_dir / "optimisation_log.jsonl"
                if not log.exists():
                    continue
                tasks[task_dir.name] = {
                    "opt_log":  log,
                    "eval_dir": EVAL_ROOT / campaign_dir.name / slug_dir.name,
                }
            if tasks:
                slugs[slug_dir.name] = tasks
        if slugs:
            result[campaign_dir.name] = slugs
    return result


# ── Per-row data ──────────────────────────────────────────────────────────────

def row_data(campaign: str, slug: str, tasks_dict: dict) -> dict:
    """
    Collect opt T-score and eval SR for every canonical task.
    Returns dict: task → {"opt": float|None, "eval_mean": float|None, "eval_std": float|None, "cycles": int}
    plus "aggregate" key.
    """
    eval_slug_dir = EVAL_ROOT / campaign / slug
    out = {}

    for short_task in TASKS:
        opt_task = TASK_OPT_NAMES[short_task]
        entry = tasks_dict.get(opt_task)

        opt_score = None
        cycles    = 0
        if entry:
            opt_score = load_opt_t_score(entry["opt_log"])
            cycles    = load_opt_cycles(entry["opt_log"])

        eval_result = load_eval_sr(eval_slug_dir, short_task) if eval_slug_dir.exists() else None
        eval_mean   = eval_result[0] if eval_result else None
        eval_std    = eval_result[1] if eval_result else None

        out[short_task] = {
            "opt":       opt_score,
            "eval_mean": eval_mean,
            "eval_std":  eval_std,
            "cycles":    cycles,
        }

    # Aggregate across tasks that have data
    opt_vals  = [v["opt"]       for v in out.values() if v["opt"]       is not None]
    eval_vals = [v["eval_mean"] for v in out.values() if v["eval_mean"] is not None]

    out["aggregate"] = {
        "opt":       float(np.mean(opt_vals))  if opt_vals  else None,
        "opt_std":   float(np.std(opt_vals))   if len(opt_vals) > 1  else None,
        "eval_mean": float(np.mean(eval_vals)) if eval_vals else None,
        "eval_std":  float(np.std(eval_vals))  if len(eval_vals) > 1 else None,
        "n_tasks_opt":  len(opt_vals),
        "n_tasks_eval": len(eval_vals),
    }
    return out


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v: float | None, decimals: int = 1) -> str:
    return f"{100*v:.{decimals}f}" if v is not None else "—"


def _pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    if std is None or std == 0.0:
        return f"{100*mean:.1f}"
    return f"{100*mean:.1f}±{100*std:.1f}"


# ── Terminal output ───────────────────────────────────────────────────────────

def print_table(runs: dict, all_data: dict):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        _rich = True
    except ImportError:
        _rich = False

    task_labels = [TASK_LABELS[t] for t in TASKS]

    if _rich:
        console = Console(width=260)
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("Campaign",  style="dim", no_wrap=True, min_width=14)
        table.add_column("Slug",      no_wrap=True, min_width=20)
        table.add_column("Cyc",       justify="right", min_width=3)
        for tl in task_labels:
            table.add_column(f"{tl}\nOpt%",  justify="right", min_width=7)
            table.add_column(f"{tl}\nEval%", justify="right", min_width=9)
        table.add_column("Agg\nOpt%",  justify="right", min_width=8, style="bold")
        table.add_column("Agg\nEval%", justify="right", min_width=10, style="bold")

        prev_campaign = None
        for campaign, slugs in runs.items():
            for slug, tasks_dict in slugs.items():
                d = all_data[(campaign, slug)]
                # avg cycles across tasks
                cyc_vals = [d[t]["cycles"] for t in TASKS if d[t]["cycles"] > 0]
                cyc_str  = str(int(np.mean(cyc_vals))) if cyc_vals else "—"

                camp_str = campaign if campaign != prev_campaign else ""
                prev_campaign = campaign

                cells = [camp_str, slug, cyc_str]
                for t in TASKS:
                    cells.append(_pct(d[t]["opt"]))
                    cells.append(_pm(d[t]["eval_mean"], d[t]["eval_std"]))
                agg = d["aggregate"]
                cells.append(_pct(agg["opt"]))
                cells.append(_pm(agg["eval_mean"], agg["eval_std"]))
                table.add_row(*cells)

            table.add_section()

        console.print(table)
    else:
        # Fallback: plain text
        header = ["Campaign", "Slug", "Cyc"]
        for tl in task_labels:
            header += [f"{tl}_Opt", f"{tl}_Eval"]
        header += ["Agg_Opt", "Agg_Eval"]
        print("\t".join(header))
        for campaign, slugs in runs.items():
            for slug, tasks_dict in slugs.items():
                d = all_data[(campaign, slug)]
                cyc_vals = [d[t]["cycles"] for t in TASKS if d[t]["cycles"] > 0]
                row = [campaign, slug, str(int(np.mean(cyc_vals))) if cyc_vals else "—"]
                for t in TASKS:
                    row.append(_pct(d[t]["opt"]))
                    row.append(_pm(d[t]["eval_mean"], d[t]["eval_std"]))
                agg = d["aggregate"]
                row.append(_pct(agg["opt"]))
                row.append(_pm(agg["eval_mean"], agg["eval_std"]))
                print("\t".join(row))


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csv(runs: dict, all_data: dict):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "all_results.csv"

    fieldnames = ["campaign", "slug", "avg_cycles"]
    for t in TASKS:
        tl = TASK_LABELS[t]
        fieldnames += [
            f"{tl}_opt_t_score",
            f"{tl}_eval_mean",
            f"{tl}_eval_std",
            f"{tl}_eval_n_seeds",
        ]
    fieldnames += [
        "agg_opt_mean", "agg_opt_std", "agg_opt_n_tasks",
        "agg_eval_mean", "agg_eval_std", "agg_eval_n_tasks",
    ]

    def _v(x): return f"{x:.4f}" if x is not None else ""

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for campaign, slugs in runs.items():
            for slug, tasks_dict in slugs.items():
                d = all_data[(campaign, slug)]
                cyc_vals = [d[t]["cycles"] for t in TASKS if d[t]["cycles"] > 0]
                row = {
                    "campaign":   campaign,
                    "slug":       slug,
                    "avg_cycles": f"{np.mean(cyc_vals):.1f}" if cyc_vals else "",
                }
                for t in TASKS:
                    tl = TASK_LABELS[t]
                    cell = d[t]
                    # count eval seeds
                    eval_slug_dir = EVAL_ROOT / campaign / slug
                    seeds = set()
                    if eval_slug_dir.exists():
                        for s in eval_slug_dir.rglob("run_summary.json"):
                            parts = s.parts
                            try:
                                iseed_idx = next(
                                    i for i, p in enumerate(parts) if p.startswith("iseed_")
                                )
                                task_idx = next(
                                    i for i, p in enumerate(parts) if p == eval_slug_dir.name
                                ) + 2  # skip env-family dir (BabyAI)
                                if parts[task_idx] == t:
                                    seeds.add(parts[iseed_idx])
                            except StopIteration:
                                pass
                    row[f"{tl}_opt_t_score"] = _v(cell["opt"])
                    row[f"{tl}_eval_mean"]   = _v(cell["eval_mean"])
                    row[f"{tl}_eval_std"]    = _v(cell["eval_std"])
                    row[f"{tl}_eval_n_seeds"] = str(len(seeds))
                agg = d["aggregate"]
                row["agg_opt_mean"]    = _v(agg["opt"])
                row["agg_opt_std"]     = _v(agg.get("opt_std"))
                row["agg_opt_n_tasks"] = str(agg["n_tasks_opt"])
                row["agg_eval_mean"]   = _v(agg["eval_mean"])
                row["agg_eval_std"]    = _v(agg.get("eval_std"))
                row["agg_eval_n_tasks"] = str(agg["n_tasks_eval"])
                w.writerow(row)

    print(f"\nCSV saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     action="store_true", help="Save CSV to figures/tables/all_results.csv")
    parser.add_argument("--no-term", action="store_true", help="Skip terminal table output")
    args = parser.parse_args()

    print("Discovering runs...")
    runs = discover_runs()

    print("Loading data...")
    all_data = {}
    for campaign, slugs in runs.items():
        for slug, tasks_dict in slugs.items():
            all_data[(campaign, slug)] = row_data(campaign, slug, tasks_dict)

    n_slugs = sum(len(s) for s in runs.values())
    print(f"  {len(runs)} campaigns, {n_slugs} slug×campaign pairs\n")

    if not args.no_term:
        print_table(runs, all_data)

    if args.csv:
        save_csv(runs, all_data)
