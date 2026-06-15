"""
Appendix per-seed success rate table — task-grouped layout.

For each task, produces one LaTeX table:
  rows = conditions (non-optimised then optimised), columns = S2…S7 | Mean | ±Std

Non-optimised data: FRESH_EVAL_DIR  (logs_fresh_eval/)
Optimised data:     OPT_EVAL_DIR    (logs_fresh_eval_optimised/) — rows are silently
                    omitted if the directory does not exist or contains no data yet.

Output:
    final_paper_plotting/tables/appendix_seed_table.tex
    final_paper_plotting/tables/appendix_seed_table.txt

Usage:
    cd final_paper_plotting
    python gen_appendix_seed_table.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import (
    AGENT_NAME, TASKS, TASK_LABELS,
    FRESH_EVAL_DIR, PROJECT_ROOT,
)

OPT_EVAL_DIR = PROJECT_ROOT / "logs_fresh_eval_optimised"

# ── Condition definitions ─────────────────────────────────────────────────────
# Each entry: (cond_key, display_label, eval_dir_key)
# eval_dir_key: "nonopt" → FRESH_EVAL_DIR, "opt" → OPT_EVAL_DIR

COND_DEFS = [
    # ── Non-optimised ─────────────────────────────────────────────────────────
    (("balrog_baseline", "minimal", "history_1step"),  "BALROG / min / 1-step",  "nonopt"),
    (("balrog_baseline", "minimal", "history_16step"), "BALROG / min / 16-step", "nonopt"),
    (("balrog_baseline", "rich",    "history_1step"),  "BALROG / rich / 1-step", "nonopt"),
    (("balrog_baseline", "rich",    "history_16step"), "BALROG / rich / 16-step","nonopt"),
    (("with_descriptor", "minimal", "single_turn"),    f"{AGENT_NAME} / minimal","nonopt"),
    (("with_descriptor", "rich",    "single_turn"),    f"{AGENT_NAME} / rich",   "nonopt"),
    # ── Optimised (uncomment when logs_fresh_eval_optimised/ is populated) ────
    # (("with_descriptor", "minimal", "single_turn"),  f"{AGENT_NAME} / min (opt)",  "opt"),
    # (("with_descriptor", "rich",    "single_turn"),  f"{AGENT_NAME} / rich (opt)", "opt"),
    # (("balrog_baseline", "minimal", "history_1step"),"BALROG / min (opt)",         "opt"),
]

ROWS = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]
INFERENCE_SEEDS = ["iseed_2", "iseed_3", "iseed_4", "iseed_5", "iseed_6", "iseed_7"]
SEED_LABELS     = ["S2", "S3", "S4", "S5", "S6", "S7"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_seed(log_dir: Path) -> dict:
    """
    Returns data[(pipeline, variant, conv_mode)][task][iseed] = success_rate.
    Silently returns empty dict if log_dir does not exist.
    """
    if not log_dir.exists():
        return {}

    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for summary_path in sorted(log_dir.rglob("run_summary.json")):
        parts = summary_path.parts
        try:
            anchor = next(i for i, p in enumerate(parts) if p == log_dir.name)
            _, task, _, pipeline, variant, conv_mode, reasoning, iseed, ts = (
                parts[anchor+1:anchor+10]
            )
        except (StopIteration, ValueError):
            continue
        if task not in TASKS:
            continue
        with open(summary_path) as f:
            d = json.load(f)
        raw[(pipeline, variant, conv_mode)][task][iseed].extend(d["episodes"])

    result = {}
    for cond_key, task_seeds in raw.items():
        result[cond_key] = {}
        for task, seed_eps in task_seeds.items():
            result[cond_key][task] = {}
            for iseed, eps in seed_eps.items():
                if eps:
                    result[cond_key][task][iseed] = (
                        sum(1 for e in eps if e["success"]) / len(eps)
                    )
    return result


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{100*v:.0f}\\%"

def fmt_pct_plain(v, width=6) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'—':>{width}}"
    return f"{100*v:>{width-1}.1f}%"


# ── LaTeX generation ──────────────────────────────────────────────────────────

def make_latex_task_table(task: str, nonopt_data: dict, opt_data: dict) -> str:
    n_seed = len(SEED_LABELS)
    col_spec = "l" + "r" * n_seed + "|rr"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(
        f"  \\caption{{Per-inference-seed success rates for \\textbf{{{TASK_LABELS[task]}}}. "
        r"Each cell: success rate over 20 episodes. "
        r"Mean and $\pm$Std computed across the six inference seeds.}}"
    )
    lines.append(f"  \\label{{tab:seed_{task}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    seed_hdr = " & ".join(SEED_LABELS)
    lines.append(f"    Condition & {seed_hdr} & Mean & $\\pm$Std \\\\")
    lines.append(r"    \midrule")

    prev_group = None
    for cond_key, cond_label, dir_key in COND_DEFS:
        src = nonopt_data if dir_key == "nonopt" else opt_data
        task_data = src.get(cond_key, {}).get(task, {})

        # Check if this row has any data at all
        seed_vals = [task_data.get(s) for s in INFERENCE_SEEDS]
        if all(v is None for v in seed_vals) and dir_key == "opt":
            continue  # skip opt rows that haven't been run yet

        # Group separator: blank row between non-opt and opt blocks
        group = dir_key
        if prev_group is not None and group != prev_group:
            lines.append(r"    \midrule")
        prev_group = group

        valid = [v for v in seed_vals if v is not None]
        mean_v = float(np.mean(valid)) if valid else float("nan")
        std_v  = float(np.std(valid))  if len(valid) > 1 else 0.0

        cells = " & ".join(fmt_pct(v) for v in seed_vals)
        safe_label = cond_label.replace("/", r"\slash ").replace("_", r"\_")
        lines.append(
            f"    {safe_label} & {cells} & "
            f"{fmt_pct(mean_v)} & {fmt_pct(std_v)} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


# ── Plain-text generation ─────────────────────────────────────────────────────

def make_text_task_table(task: str, nonopt_data: dict, opt_data: dict) -> str:
    cond_w = 26
    col_w  = 7

    header = f"{'Condition':<{cond_w}}" + "".join(f"{s:>{col_w}}" for s in SEED_LABELS)
    header += f"{'Mean':>{col_w}}{'Std':>{col_w}}"
    sep = "-" * len(header)

    lines = []
    lines.append(f"\n{'='*len(header)}")
    lines.append(f"Task: {TASK_LABELS[task]}")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)

    prev_group = None
    for cond_key, cond_label, dir_key in COND_DEFS:
        src = nonopt_data if dir_key == "nonopt" else opt_data
        task_data = src.get(cond_key, {}).get(task, {})

        seed_vals = [task_data.get(s) for s in INFERENCE_SEEDS]
        if all(v is None for v in seed_vals) and dir_key == "opt":
            continue

        group = dir_key
        if prev_group is not None and group != prev_group:
            lines.append(sep)
        prev_group = group

        valid = [v for v in seed_vals if v is not None]
        mean_v = float(np.mean(valid)) if valid else float("nan")
        std_v  = float(np.std(valid))  if len(valid) > 1 else 0.0

        row = f"{cond_label:<{cond_w}}"
        row += "".join(fmt_pct_plain(v, col_w) for v in seed_vals)
        row += fmt_pct_plain(mean_v, col_w) + fmt_pct_plain(std_v, col_w)
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "tables"
    out_dir.mkdir(exist_ok=True)

    nonopt_data = load_per_seed(FRESH_EVAL_DIR)
    opt_data    = load_per_seed(OPT_EVAL_DIR)

    if not opt_data:
        print(f"  Note: {OPT_EVAL_DIR.name}/ not found or empty — optimised rows omitted.")

    latex_parts = [
        r"% Auto-generated by gen_appendix_seed_table.py — do not edit by hand",
        r"% Requires \usepackage{booktabs} in your preamble",
        "",
    ]
    text_parts = []

    for task in ROWS:
        latex_parts.append(make_latex_task_table(task, nonopt_data, opt_data))
        text_parts.append(make_text_task_table(task, nonopt_data, opt_data))

    latex_path = out_dir / "appendix_seed_table.tex"
    text_path  = out_dir / "appendix_seed_table.txt"

    latex_path.write_text("\n".join(latex_parts))
    text_path.write_text("\n".join(text_parts) + "\n")

    print(f"  Saved: {latex_path}")
    print(f"  Saved: {text_path}")
    print()

    for part in text_parts:
        print(part)
