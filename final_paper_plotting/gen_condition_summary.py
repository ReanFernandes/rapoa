"""
Condition summary table — mean SR per condition across all 5 tasks.

Rows    = every evaluated condition (non-optimised + all optimised)
Columns = GoTo | PickUp | Open | PickUp→GoTo | PutNext | Mean (avg over 5 tasks)

Output (both generated):
    tables/condition_summary.txt   — plain-text for reference / writing
    tables/condition_summary.tex   — LaTeX tabular for paper use

Usage:
    cd final_paper_plotting
    python gen_condition_summary.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import (
    AGENT_NAME, TASKS, TASK_LABELS,
    FRESH_EVAL_DIR,
    VARIANT_LABELS, AA_LABEL, T_STAR, FINAL_INC, SP_LABELS, DELTA,
    SLUGS, eval_dir, latest_campaign,
)

_V = VARIANT_LABELS

ROWS = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]

# ── Campaign selection ────────────────────────────────────────────────────────
# Override these to target a specific campaign by name, e.g.:
#   CAMPAIGN_PRIMARY = "primary_20_20260521_143022"
CAMPAIGN_PRIMARY = latest_campaign(SLUGS["hsp_rich"])
CAMPAIGN_THRESH  = latest_campaign(SLUGS["thresh002_hsp"])
CAMPAIGN_ABLATION = latest_campaign(SLUGS["actor_ablation_rich"])

# ── Condition definitions ─────────────────────────────────────────────────────
# (group_label, cond_label, cond_key, eval_dir_path)

_CK_OURS_MIN  = ("with_descriptor", "minimal", "single_turn")
_CK_OURS_RICH = ("with_descriptor", "rich",    "single_turn")
_CK_BALROG    = ("balrog_baseline", "minimal", "history_16step")

_HSP    = f"HSP, {DELTA}=0.05"
_LSP    = f"LSP, {DELTA}=0.05"
_AA_FIN = f"{AA_LABEL}, {FINAL_INC}"

def _e(campaign, slug_key): return eval_dir(campaign, SLUGS[slug_key]) if campaign else None

CONDITIONS = [
    # ── Non-optimised baselines ───────────────────────────────────────────────
    ("BALROG baseline", f"plain / 1-step",   ("balrog_baseline","minimal","history_1step"),  FRESH_EVAL_DIR),
    ("BALROG baseline", f"plain / 16-step",  ("balrog_baseline","minimal","history_16step"), FRESH_EVAL_DIR),
    ("BALROG baseline", f"guided / 1-step",  ("balrog_baseline","rich",   "history_1step"),  FRESH_EVAL_DIR),
    ("BALROG baseline", f"guided / 16-step", ("balrog_baseline","rich",   "history_16step"), FRESH_EVAL_DIR),
    (f"{AGENT_NAME} baseline", f"plain",  _CK_OURS_MIN,  FRESH_EVAL_DIR),
    (f"{AGENT_NAME} baseline", f"guided", _CK_OURS_RICH, FRESH_EVAL_DIR),

    # ── Gated — HSP δ=0.05 ───────────────────────────────────────────────────
    (_HSP, f"BALROG {_V['minimal']}/16-step", _CK_BALROG,    _e(CAMPAIGN_PRIMARY, "balrog_minimal")),
    (_HSP, "plain",        _CK_OURS_MIN,  _e(CAMPAIGN_PRIMARY, "hsp_minimal")),
    (_HSP, "guided",       _CK_OURS_RICH, _e(CAMPAIGN_PRIMARY, "hsp_rich")),

    # ── Gated — LSP δ=0.05 ───────────────────────────────────────────────────
    (_LSP, f"BALROG {_V['minimal']}/16-step", _CK_BALROG,    _e(CAMPAIGN_PRIMARY, "balrog_minimal")),
    (_LSP, "plain",        _CK_OURS_MIN,  _e(CAMPAIGN_PRIMARY, "lsp_minimal")),
    (_LSP, "guided",       _CK_OURS_RICH, _e(CAMPAIGN_PRIMARY, "lsp_rich")),

    # ── Always-accept ─────────────────────────────────────────────────────────
    (_AA_FIN, f"BALROG {_V['minimal']}/16-step", _CK_BALROG,    _e(CAMPAIGN_PRIMARY, "balrog_minimal")),
    (_AA_FIN, "plain",        _CK_OURS_MIN,  _e(CAMPAIGN_PRIMARY, "always_accept_minimal")),
    (_AA_FIN, "guided",       _CK_OURS_RICH, _e(CAMPAIGN_PRIMARY, "always_accept_rich")),

    # ── Threshold sweep — HSP ────────────────────────────────────────────────
    (f"HSP, {DELTA}=0.02", "plain",  _CK_OURS_MIN,  _e(CAMPAIGN_THRESH, "thresh002_hsp")),
    (f"HSP, {DELTA}=0.02", "guided", _CK_OURS_RICH, _e(CAMPAIGN_THRESH, "thresh002_hsp")),
    (f"HSP, {DELTA}=0.10", "plain",  _CK_OURS_MIN,  _e(CAMPAIGN_THRESH, "thresh010_hsp")),
    (f"HSP, {DELTA}=0.10", "guided", _CK_OURS_RICH, _e(CAMPAIGN_THRESH, "thresh010_hsp")),

    # ── Threshold sweep — LSP ────────────────────────────────────────────────
    (f"LSP, {DELTA}=0.02", "plain",  _CK_OURS_MIN,  _e(CAMPAIGN_THRESH, "thresh002_lsp")),
    (f"LSP, {DELTA}=0.02", "guided", _CK_OURS_RICH, _e(CAMPAIGN_THRESH, "thresh002_lsp")),
    (f"LSP, {DELTA}=0.10", "plain",  _CK_OURS_MIN,  _e(CAMPAIGN_THRESH, "thresh010_lsp")),
    (f"LSP, {DELTA}=0.10", "guided", _CK_OURS_RICH, _e(CAMPAIGN_THRESH, "thresh010_lsp")),

    # ── Module ablation ───────────────────────────────────────────────────────
    ("Ablation: actor-only",      "plain",  _CK_OURS_MIN,  _e(CAMPAIGN_ABLATION, "actor_ablation_minimal")),
    ("Ablation: actor-only",      "guided", _CK_OURS_RICH, _e(CAMPAIGN_ABLATION, "actor_ablation_rich")),
    ("Ablation: descriptor-only", "plain",  _CK_OURS_MIN,  _e(CAMPAIGN_ABLATION, "descriptor_ablation_minimal")),
    ("Ablation: descriptor-only", "guided", _CK_OURS_RICH, _e(CAMPAIGN_ABLATION, "descriptor_ablation_rich")),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sr_with_std(log_dir: Path) -> dict:
    """
    Returns {cond_key: {task: (mean_sr, std_across_seeds)}}.
    std is computed across 6 inference-seed means — same as heatmap ± values.
    """
    if not log_dir.exists():
        return {}
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for summary in sorted(log_dir.rglob("run_summary.json")):
        parts = summary.parts
        try:
            anchor = next(i for i, p in enumerate(parts) if p == log_dir.name)
            _, task, _, pipeline, variant, conv_mode, _, iseed, _ = parts[anchor+1:anchor+10]
        except (StopIteration, ValueError):
            continue
        if task not in TASKS:
            continue
        with open(summary) as f:
            d = json.load(f)
        raw[(pipeline, variant, conv_mode)][task][iseed].extend(d["episodes"])

    result = {}
    for ck, task_seeds in raw.items():
        result[ck] = {}
        for task, seed_eps in task_seeds.items():
            all_eps = [e for eps in seed_eps.values() for e in eps]
            if not all_eps:
                continue
            mean_sr = sum(1 for e in all_eps if e["success"]) / len(all_eps)
            seed_srs = [
                sum(1 for e in eps if e["success"]) / len(eps)
                for eps in seed_eps.values() if eps
            ]
            std_sr = float(np.std(seed_srs)) if len(seed_srs) > 1 else 0.0
            result[ck][task] = (mean_sr, std_sr)
    return result


def get_row(cond_key, eval_dir: Path, cache: dict) -> dict:
    """Returns {task: (mean_sr, std_sr)} for a condition."""
    key = str(eval_dir)
    if key not in cache:
        cache[key] = load_sr_with_std(eval_dir)
    return cache[key].get(cond_key, {})


# ── Build data matrix ─────────────────────────────────────────────────────────

def build_table() -> list:
    """Returns list of (group, variant, vals, mean_sr, mean_std) per condition.
    vals = [(mean_sr, std_sr) or None] per task.
    mean_sr = mean across tasks. mean_std = mean of per-task stds across tasks.
    """
    cache = {}
    rows = []
    for group, variant, cond_key, eval_dir in CONDITIONS:
        task_data = get_row(cond_key, eval_dir, cache)
        vals = [task_data.get(t) for t in ROWS]   # (mean, std) or None per task
        valid_means = [v[0] for v in vals if v is not None]
        valid_stds  = [v[1] for v in vals if v is not None]
        mean_sr  = float(np.mean(valid_means)) if valid_means else None
        mean_std = float(np.mean(valid_stds))  if valid_stds  else None
        rows.append((group, variant, vals, mean_sr, mean_std))
    return rows


# ── Plain-text output ─────────────────────────────────────────────────────────

def fmt(v, width=7) -> str:
    if v is None:
        return f"{'—':>{width}}"
    return f"{100*v:>{width-1}.1f}%"


def make_text_table(data: list) -> str:
    task_hdrs = [TASK_LABELS[t] for t in ROWS]
    col_w = 13
    grp_w = 28
    var_w = 14

    hdr = f"{'Group':<{grp_w}} {'Variant':<{var_w}}"
    for th in task_hdrs:
        hdr += f"{th:>{col_w}}"
    hdr += f"{'Mean±Std':>{col_w+4}}"
    sep = "─" * (len(hdr))

    lines = [sep, hdr, sep]
    prev_group = None
    for group, variant, vals, mean_sr, mean_std in data:
        if prev_group is not None and group != prev_group:
            lines.append(sep)
        prev_group = group
        row = f"{group:<{grp_w}} {variant:<{var_w}}"
        for v in vals:
            sr = v[0] if v is not None else None
            row += fmt(sr, col_w)
        if mean_sr is not None and mean_std is not None:
            row += f"  {100*mean_sr:>5.1f}±{100*mean_std:4.1f}%"
        else:
            row += f"{'—':>{col_w+4}}"
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


# ── LaTeX output ──────────────────────────────────────────────────────────────

def fmt_tex(v) -> str:
    if v is None:
        return r"\textemdash"
    return f"{100*v:.1f}\\%"


def make_latex_table(data: list) -> str:
    n_tasks = len(ROWS)
    col_spec = "ll" + "r" * n_tasks + "r"
    task_hdrs = " & ".join(f"\\textbf{{{TASK_LABELS[t]}}}" for t in ROWS)

    lines = [
        r"% Auto-generated by gen_condition_summary.py — do not edit by hand",
        r"% Requires \usepackage{booktabs} in your preamble",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Mean success rate per condition across all five BabyAI tasks. "
        r"Results from the fresh evaluation protocol (env seed 500, "
        r"inference seeds 2--7, 20 episodes per seed).}",
        r"  \label{tab:condition_summary}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    \\textbf{{Condition}} & \\textbf{{Variant}} & {task_hdrs} & \\textbf{{Mean}} \\\\",
        r"    \midrule",
    ]

    prev_group = None
    for group, variant, vals, mean_sr, mean_std in data:
        if prev_group is not None and group != prev_group:
            lines.append(r"    \midrule")
        prev_group = group
        safe_group   = group.replace("_", r"\_").replace("δ", r"$\delta$").replace("∞", r"$\infty$").replace("-∞", r"$-\infty$")
        safe_variant = variant.replace("_", r"\_")
        cells = " & ".join(fmt_tex(v[0] if v else None) for v in vals)
        mean_cell = f"{100*mean_sr:.1f}\\% $\\pm$ {100*mean_std:.1f}\\%" if mean_sr is not None else r"\textemdash"
        lines.append(f"    {safe_group} & {safe_variant} & {cells} & {mean_cell} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "tables"
    out_dir.mkdir(exist_ok=True)

    data = build_table()

    txt = make_text_table(data)
    tex = make_latex_table(data)

    txt_path = out_dir / "condition_summary.txt"
    tex_path = out_dir / "condition_summary.tex"
    txt_path.write_text(txt + "\n")
    tex_path.write_text(tex + "\n")

    print(f"  Saved: {txt_path}")
    print(f"  Saved: {tex_path}")
    print()
    print(txt)
