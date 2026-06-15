"""
Headliner heatmap — success rate by task × condition.

Shows every optimised condition alongside non-optimised baselines.
Columns with no eval data yet render as "—". Comment out any COLS entry to hide it.

Usage:
    cd final_paper_plotting
    python plot_heatmap.py [--cols <name>]

    --cols all        Full heatmap with every condition (default)
    --cols section1   baselines + gated HSP + gated LSP              (12 cols)
    --cols section2   baselines + gated HSP + gated LSP + AA         (15 cols)
    --cols section3   baselines + gated HSP + all module ablations   (15 cols)
    --cols section4   baselines + gated HSP + AA                     (12 cols)

Output: figures/heatmap_<name>.pdf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import (
    AGENT_NAME, TASKS, TASK_LABELS, TASK_OPT_NAMES,
    FRESH_EVAL_DIR, EVAL_RUNS_ROOT, CAMPAIGN_IDS,
    COL_LABEL_OVERRIDES, GROUP_LABEL_OVERRIDES,
    VARIANT_LABELS, BALROG_OPT_LABEL, OURS_OPT_LABEL_MIN, OURS_OPT_LABEL_RICH,
    DELTA, AA_LABEL, T_STAR, FINAL_INC, SP_LABELS,
    SLUGS, eval_dir, opt_dir,
    apply_neurips_style, save,
)

_V = VARIANT_LABELS   # shorthand: _V["minimal"] == "plain"

apply_neurips_style()

# ── Campaign selection ────────────────────────────────────────────────────────
# Experiments are split across campaigns: rich conditions + ablations in primary_20;
# minimal conditions in primary_minimal; BALROG variants in balrog_all.
CP_R   = CAMPAIGN_IDS["primary_20"]       # primary: rich conditions + ablations
CP_M   = CAMPAIGN_IDS["primary_minimal"]  # primary: minimal conditions + ablations
CP_B   = CAMPAIGN_IDS["balrog_all"]       # balrog h16 rich opt conditions
CP_T   = CAMPAIGN_IDS["thresh_sweep"]     # threshold sweep
CP_RND = CAMPAIGN_IDS["random_module"]    # random module ablation
CP_H1  = CAMPAIGN_IDS["balrog_h1"]        # balrog h1 re-run (correctly at h1)

# Backward-compat aliases (used in gen_paper_tables.py)
CAMPAIGN_PRIMARY  = CP_R
CAMPAIGN_THRESH   = CP_T
CAMPAIGN_ABLATION = CP_R

# Eval root alias for gen_paper_tables.py
OPT_EVAL_ROOT = EVAL_RUNS_ROOT

# ── Column definitions ────────────────────────────────────────────────────────
# Each entry: (cond_key, label, group, opt_run_id, eval_dir)
#
# cond_key   : (pipeline, variant, conv_mode)
# label      : column header text (use \n for two-line labels)
# group      : bracket group header; same string = same bracket
# opt_run_id : run dir name used to detect zero-mutation tasks (None = no marker)
# eval_dir   : where to load eval data from
#              FRESH_EVAL_DIR      → non-optimised baselines
#              OPT_EVAL_ROOT/{id} → optimised fresh-eval results

_CK_OURS_MIN    = ("with_descriptor", "minimal", "single_turn")
_CK_OURS_RICH   = ("with_descriptor", "rich",    "single_turn")
_CK_BALROG_MIN  = ("balrog_baseline", "minimal", "history_16step")
_CK_BALROG_RICH = ("balrog_baseline", "rich",    "history_16step")
_CK_B_MIN_1     = ("balrog_baseline", "minimal", "history_1step")
_CK_B_R_1       = ("balrog_baseline", "rich",    "history_1step")
_CK_BALROG      = _CK_BALROG_MIN   # backward-compat alias

_HSP    = f"HSP, {DELTA}=0.05"
_LSP    = f"LSP, {DELTA}=0.05"
_AA_BT  = f"{AA_LABEL}, {T_STAR}"
_AA_FIN = f"{AA_LABEL}, {FINAL_INC}"

COLS = [
    # ── Non-optimised baselines ───────────────────────────────────────────────
    (("balrog_baseline", "minimal", "history_1step"),  f"{_V['minimal']}\n1-step",  "BALROG\nbaseline", None, FRESH_EVAL_DIR),
    (("balrog_baseline", "minimal", "history_16step"), f"{_V['minimal']}\n16-step", "BALROG\nbaseline", None, FRESH_EVAL_DIR),
    (("balrog_baseline", "rich",    "history_1step"),  f"{_V['rich']}\n1-step",     "BALROG\nbaseline", None, FRESH_EVAL_DIR),
    (("balrog_baseline", "rich",    "history_16step"), f"{_V['rich']}\n16-step",    "BALROG\nbaseline", None, FRESH_EVAL_DIR),

    (_CK_OURS_MIN,  _V["minimal"], f"{AGENT_NAME}\nbaseline", None, FRESH_EVAL_DIR),
    (_CK_OURS_RICH, _V["rich"],    f"{AGENT_NAME}\nbaseline", None, FRESH_EVAL_DIR),

    # ── Gated — high selection pressure (HSP, δ=0.05) ────────────────────────
    (_CK_BALROG_MIN,  BALROG_OPT_LABEL,    _HSP, opt_dir(CP_B, SLUGS["balrog_minimal_hsp"]), eval_dir(CP_B, SLUGS["balrog_minimal_hsp"])),
    (_CK_BALROG_RICH, BALROG_OPT_LABEL,    _HSP, opt_dir(CP_B, SLUGS["balrog_rich_hsp"]),   eval_dir(CP_B, SLUGS["balrog_rich_hsp"])),
    (_CK_OURS_MIN,    OURS_OPT_LABEL_MIN,  _HSP, opt_dir(CP_M, SLUGS["hsp_minimal"]),       eval_dir(CP_M, SLUGS["hsp_minimal"])),
    (_CK_OURS_RICH,   OURS_OPT_LABEL_RICH, _HSP, opt_dir(CP_R, SLUGS["hsp_rich"]),          eval_dir(CP_R, SLUGS["hsp_rich"])),

    # ── Gated — low selection pressure (LSP, δ=0.05) ─────────────────────────
    (_CK_BALROG_MIN,  BALROG_OPT_LABEL,    _LSP, opt_dir(CP_B, SLUGS["balrog_minimal_lsp"]), eval_dir(CP_B, SLUGS["balrog_minimal_lsp"])),
    (_CK_BALROG_RICH, BALROG_OPT_LABEL,    _LSP, opt_dir(CP_B, SLUGS["balrog_rich_lsp"]),    eval_dir(CP_B, SLUGS["balrog_rich_lsp"])),
    (_CK_OURS_MIN,    OURS_OPT_LABEL_MIN,  _LSP, opt_dir(CP_M, SLUGS["lsp_minimal"]),      eval_dir(CP_M, SLUGS["lsp_minimal"])),
    (_CK_OURS_RICH,   OURS_OPT_LABEL_RICH, _LSP, opt_dir(CP_R, SLUGS["lsp_rich"]),         eval_dir(CP_R, SLUGS["lsp_rich"])),

    # ── δ=-∞  (always-accept) — final incumbent ───────────────────────────────
    (_CK_BALROG_MIN,  BALROG_OPT_LABEL,    _AA_FIN, opt_dir(CP_R, SLUGS["balrog_minimal"]),          eval_dir(CP_R, SLUGS["balrog_minimal"])),
    (_CK_BALROG_RICH, BALROG_OPT_LABEL,    _AA_FIN, opt_dir(CP_B, SLUGS["balrog_rich"]),             eval_dir(CP_B, SLUGS["balrog_rich"])),
    (_CK_OURS_MIN,    OURS_OPT_LABEL_MIN,  _AA_FIN, opt_dir(CP_M, SLUGS["always_accept_minimal"]),   eval_dir(CP_M, SLUGS["always_accept_minimal"])),
    (_CK_OURS_RICH,   OURS_OPT_LABEL_RICH, _AA_FIN, opt_dir(CP_R, SLUGS["always_accept_rich"]),      eval_dir(CP_R, SLUGS["always_accept_rich"])),

    # ── Threshold sweep — HSP ────────────────────────────────────────────────
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  f"HSP, {DELTA}=0.00", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh000_hsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh000_hsp"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, f"HSP, {DELTA}=0.00", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh000_hsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh000_hsp"])),
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  f"HSP, {DELTA}=0.02", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh002_hsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh002_hsp"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, f"HSP, {DELTA}=0.02", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh002_hsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh002_hsp"])),
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  f"HSP, {DELTA}=0.10", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh010_hsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh010_hsp"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, f"HSP, {DELTA}=0.10", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh010_hsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh010_hsp"])),

    # ── Threshold sweep — LSP ────────────────────────────────────────────────
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  f"LSP, {DELTA}=0.00", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh000_lsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh000_lsp"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, f"LSP, {DELTA}=0.00", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh000_lsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh000_lsp"])),
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  f"LSP, {DELTA}=0.02", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh002_lsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh002_lsp"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, f"LSP, {DELTA}=0.02", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh002_lsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh002_lsp"])),
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  f"LSP, {DELTA}=0.10", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh010_lsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh010_lsp"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, f"LSP, {DELTA}=0.10", opt_dir(CAMPAIGN_THRESH, SLUGS["thresh010_lsp"]), eval_dir(CAMPAIGN_THRESH, SLUGS["thresh010_lsp"])),

    # ── Module ablation ───────────────────────────────────────────────────────
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  "Ablation\nActor-only",      opt_dir(CP_M, SLUGS["actor_ablation_minimal"]),      eval_dir(CP_M, SLUGS["actor_ablation_minimal"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, "Ablation\nActor-only",      opt_dir(CP_R, SLUGS["actor_ablation_rich"]),         eval_dir(CP_R, SLUGS["actor_ablation_rich"])),
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  "Ablation\nDescriptor-only", opt_dir(CP_M, SLUGS["descriptor_ablation_minimal"]), eval_dir(CP_M, SLUGS["descriptor_ablation_minimal"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, "Ablation\nDescriptor-only", opt_dir(CP_R, SLUGS["descriptor_ablation_rich"]),    eval_dir(CP_R, SLUGS["descriptor_ablation_rich"])),

    # ── Random module ablation ────────────────────────────────────────────────
    (_CK_OURS_MIN,  OURS_OPT_LABEL_MIN,  "Ablation\nRandom-module",   opt_dir(CP_RND, SLUGS["random_hsp_minimal"]),        eval_dir(CP_RND, SLUGS["random_hsp_minimal"])),
    (_CK_OURS_RICH, OURS_OPT_LABEL_RICH, "Ablation\nRandom-module",   opt_dir(CP_RND, SLUGS["random_hsp_rich"]),           eval_dir(CP_RND, SLUGS["random_hsp_rich"])),

    # ── BALROG h1 optimised ───────────────────────────────────────────────────
    (_CK_B_MIN_1, f"BALROG\n{_V['minimal']}\nAA",  "BALROG\nh1 opt", opt_dir(CP_H1, SLUGS["balrog_h1_minimal"]),     eval_dir(CP_H1, SLUGS["balrog_h1_minimal"])),
    (_CK_B_R_1,  f"BALROG\n{_V['rich']}\nAA",      "BALROG\nh1 opt", opt_dir(CP_H1, SLUGS["balrog_h1_rich"]),        eval_dir(CP_H1, SLUGS["balrog_h1_rich"])),
    (_CK_B_MIN_1, f"BALROG\n{_V['minimal']}\nHSP", "BALROG\nh1 opt", opt_dir(CP_H1, SLUGS["balrog_h1_minimal_hsp"]), eval_dir(CP_H1, SLUGS["balrog_h1_minimal_hsp"])),
    (_CK_B_R_1,  f"BALROG\n{_V['rich']}\nHSP",     "BALROG\nh1 opt", opt_dir(CP_H1, SLUGS["balrog_h1_rich_hsp"]),    eval_dir(CP_H1, SLUGS["balrog_h1_rich_hsp"])),
    (_CK_B_MIN_1, f"BALROG\n{_V['minimal']}\nLSP", "BALROG\nh1 opt", opt_dir(CP_H1, SLUGS["balrog_h1_minimal_lsp"]), eval_dir(CP_H1, SLUGS["balrog_h1_minimal_lsp"])),
    (_CK_B_R_1,  f"BALROG\n{_V['rich']}\nLSP",     "BALROG\nh1 opt", opt_dir(CP_H1, SLUGS["balrog_h1_rich_lsp"]),    eval_dir(CP_H1, SLUGS["balrog_h1_rich_lsp"])),

    # ── No-win-condition ablation — commented out (sanity check only) ─────────
    # (_CK_OURS_RICH, f"{_V['rich']}\nnowincond", "Ours\n(no wincond)", None, FRESH_EVAL_DIR.parent / "logs_fresh_eval_nowincond"),
]

ROWS = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]

GROUP_COLORS = {
    "BALROG\nbaseline":               "#0072B2",   # Okabe blue
    f"{AGENT_NAME}\nbaseline":        "#D55E00",   # Okabe vermillion
    f"HSP, {DELTA}=0.05":            "#CC79A7",   # Okabe reddish purple
    f"LSP, {DELTA}=0.05":            "#882255",   # dark magenta
    f"{AA_LABEL}, {T_STAR}":         "#E69F00",   # Okabe orange
    f"{AA_LABEL}, {FINAL_INC}":      "#56B4E9",   # Okabe sky blue
    f"HSP, {DELTA}=0.00":            "#66C2A5",   # light teal
    f"HSP, {DELTA}=0.02":            "#009E73",   # Okabe bluish green
    f"HSP, {DELTA}=0.10":            "#005a3e",   # dark green
    f"LSP, {DELTA}=0.00":            "#88CCEE",   # light blue
    f"LSP, {DELTA}=0.02":            "#0072B2",   # Okabe blue
    f"LSP, {DELTA}=0.10":            "#003f63",   # dark blue
    "Ablation\nActor-only":       "#44BB99",   # teal
    "Ablation\nDescriptor-only": "#AA4488",   # magenta
    "Ablation\nRandom-module":   "#DDAA33",   # gold
    "BALROG\nh1 opt":            "#44BB99",   # teal
}

# ── Named column subsets ──────────────────────────────────────────────────────
# Select via: python plot_heatmap.py --cols <name>
# Each list is a subset of COLS indices (0-based).  Add new subsets here.
# Index map (matches COLS order above):
#   0–5  : Non-opt baselines (BALROG min/1s, min/16s, rich/1s, rich/16s; Ours min, rich)
#   6–9  : Gated HSP (BALROG-min, BALROG-rich, Ours/min, Ours/rich)
#   10–13: Gated LSP (BALROG-min, BALROG-rich, Ours/min, Ours/rich)
#   14–17: Always-accept end-of-run (BALROG-min, BALROG-rich, Ours/min, Ours/rich)
#   18–23: Threshold sweep HSP (t=0.00/0.02/0.10 × min/rich) — 6 entries
#   24–29: Threshold sweep LSP (t=0.00/0.02/0.10 × min/rich) — 6 entries
#   30–31: Ablation Actor-only (min, rich)
#   32–33: Ablation Descriptor-only (min, rich)
#   34–35: Ablation Random-module (min, rich)
#   36–41: BALROG h1 opt (AA×2, HSP×2, LSP×2 — min/rich pairs)

_NON_OPT   = list(range(0, 6))    # all 6 non-opt baselines
_VALBAG    = list(range(6, 10))   # 4 gated HSP
_TRAINSIG  = list(range(10, 14))  # 4 gated LSP
_AA_EOR    = list(range(14, 18))  # 4 AA end-of-run
_ABLATION  = list(range(30, 36))  # 6 ablation entries (actor, descriptor, random)
_BALROG_H1 = list(range(36, 42))  # 6 BALROG h1 opt entries

NAMED_COLS = {
    # Full kitchen-sink view
    "all":      list(range(len(COLS))),

    # Section 1: baselines + gated HSP + gated LSP
    "section1": _NON_OPT + _VALBAG + _TRAINSIG,

    # Section 2: baselines + all gated + always-accept
    "section2": _NON_OPT + _VALBAG + _TRAINSIG + _AA_EOR,

    # Section 3: baselines + gated HSP (reference) + module ablation
    "section3": _NON_OPT + _VALBAG + _ABLATION,

    # Section 4: baselines + gated HSP + always-accept
    "section4": _NON_OPT + _VALBAG + _AA_EOR,

    # BALROG h1 opt: baselines + all 6 h1 conditions
    "balrog_h1": _NON_OPT + _BALROG_H1,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_fresh_eval(log_dir: Path) -> dict:
    """
    Returns per-condition per-task stats including std of per-seed success rates.
    std is computed across inference-seed means — measures LLM stochasticity.
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

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
        data[(pipeline, variant, conv_mode)][task][iseed].extend(d["episodes"])

    result = {}
    for cond_key, task_seeds in data.items():
        result[cond_key] = {}
        for task, seed_eps in task_seeds.items():
            all_eps  = [e for eps in seed_eps.values() for e in eps]
            n        = len(all_eps)
            seed_srs = [
                sum(1 for e in eps if e["success"]) / len(eps)
                for eps in seed_eps.values() if eps
            ]
            result[cond_key][task] = dict(
                sr  = sum(1 for e in all_eps if e["success"]) / n,
                std = float(np.std(seed_srs)) if len(seed_srs) > 1 else 0.0,
                ms  = sum(e["total_steps"] for e in all_eps) / n,
                n   = n,
            )
    return result


# ── Zero-mutation detection ───────────────────────────────────────────────────

def load_mutation_counts(cols: list, rows: list) -> dict:
    """
    Returns zero_mut[(col_idx, row_idx)] = True when the opt run for that
    (condition, task) accepted zero mutations.

    The 4th element of each col tuple is now a Path to the slug-level opt dir
    (or None for non-optimised baselines).
    """
    # Reverse mapping: opt task name → eval short name
    _OPT_TO_EVAL = {v: k for k, v in TASK_OPT_NAMES.items()}
    _cache: dict[int, set] = {}  # keyed by col index

    zero_mut: dict[tuple, bool] = {}

    for j, (cond_key, _, _, run_dir, _) in enumerate(cols):
        if not run_dir or not isinstance(run_dir, Path) or not run_dir.exists():
            continue
        if j not in _cache:
            zero_tasks: set[str] = set()
            for log in run_dir.rglob("optimisation_log.jsonl"):
                if any(p.startswith(("opt_cycle_", "env_round_", "eval_")) for p in log.parts):
                    continue
                opt_task = log.parent.name
                eval_task = _OPT_TO_EVAL.get(opt_task, opt_task)
                try:
                    records = [json.loads(line) for line in open(log)]
                    n_accepted = sum(
                        1 for r in records
                        if r.get("record_type") == "opt_cycle"
                        and r.get("opt_cycle_outcome") == "accepted"
                    )
                    if n_accepted == 0:
                        zero_tasks.add(eval_task)
                except Exception:
                    pass
            _cache[j] = zero_tasks

        for i, task in enumerate(rows):
            if task in _cache[j]:
                zero_mut[(j, i)] = True

    return zero_mut


# ── Build matrix ──────────────────────────────────────────────────────────────

def build_matrix(rows: list, cols: list) -> tuple:
    _cache: dict[Path, dict] = {}

    def _get(eval_dir: Path) -> dict:
        if eval_dir not in _cache:
            _cache[eval_dir] = load_fresh_eval(eval_dir) if eval_dir.exists() else {}
        return _cache[eval_dir]

    sr_mat  = np.full((len(rows), len(cols)), np.nan)
    std_mat = np.full((len(rows), len(cols)), np.nan)
    ms_mat  = np.full((len(rows), len(cols)), np.nan)

    for i, task in enumerate(rows):
        for j, (cond_key, _, _, _, eval_dir) in enumerate(cols):
            cell = _get(eval_dir).get(cond_key, {}).get(task)
            if cell:
                sr_mat[i, j]  = cell["sr"]
                std_mat[i, j] = cell["std"]
                ms_mat[i, j]  = cell["ms"]

    return sr_mat, std_mat, ms_mat


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_heatmap(sr_mat, std_mat, rows, cols, zero_mut=None):
    n_rows = len(rows)
    n_cols = len(cols)

    HEADER_H = 0.65
    CELL_H   = 0.44
    CELL_W   = 0.58
    fig_h    = HEADER_H + n_rows * CELL_H + 0.45
    fig_w    = max(5.0, 1.35 + n_cols * CELL_W)

    # Dynamic margins: left leaves room for task-name y-tick labels (~1.5 in)
    left_frac  = max(0.08, 1.5 / fig_w)
    right_frac = min(0.96, 1 - 0.8 / fig_w)   # 0.8 in for colorbar

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(top=0.82, bottom=0.12,
                        left=left_frac, right=right_frac)

    cmap = plt.cm.viridis
    cmap.set_bad("0.93")

    im = ax.imshow(sr_mat, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")

    # Best value per row — overall and within baseline columns only
    row_best = np.nanmax(sr_mat, axis=1, keepdims=True)
    is_best  = np.isclose(sr_mat, row_best) & ~np.isnan(sr_mat)

    baseline_js = [j for j, (*_, eval_dir) in enumerate(cols)
                   if eval_dir == FRESH_EVAL_DIR]
    is_best_baseline = np.zeros((n_rows, n_cols), dtype=bool)
    if baseline_js:
        base_sub = sr_mat[:, baseline_js]
        base_max = np.nanmax(base_sub, axis=1, keepdims=True)
        for jl, jg in enumerate(baseline_js):
            is_best_baseline[:, jg] = (
                np.isclose(sr_mat[:, jg], base_max[:, 0]) & ~np.isnan(sr_mat[:, jg])
            )

    for i in range(n_rows):
        for j in range(n_cols):
            v   = sr_mat[i, j]
            std = std_mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7.5, color="0.5")
            else:
                text_col = "white" if v < 0.25 else "0.1"
                ax.text(j, i - 0.13, f"{100*v:.0f}%",
                        ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color=text_col)
                if not np.isnan(std):
                    ax.text(j, i + 0.22, f"±{100*std:.0f}%",
                            ha="center", va="center",
                            fontsize=6.0, color=text_col)
                if is_best[i, j]:
                    ax.plot([j - 0.28, j + 0.28], [i + 0.38, i + 0.38],
                            color=text_col, linewidth=1.2,
                            solid_capstyle="round", clip_on=False)
                if is_best_baseline[i, j]:
                    ax.plot([j - 0.28, j + 0.28], [i - 0.40, i - 0.40],
                            color=text_col, linewidth=1.0, linestyle="--",
                            solid_capstyle="round", clip_on=False)
                if zero_mut and zero_mut.get((j, i)):
                    ax.text(j + 0.38, i - 0.38, "◦",
                            ha="right", va="top", fontsize=6,
                            color=text_col, clip_on=True)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([TASK_LABELS[t] for t in rows], fontsize=8.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [COL_LABEL_OVERRIDES.get(lbl, lbl) for _, lbl, *_ in cols],
        fontsize=7.0,
    )
    ax.tick_params(length=0, pad=3)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    groups = [g for _, _, g, *_ in cols]
    for j in range(1, n_cols):
        if groups[j] != groups[j-1]:
            ax.axvline(j - 0.5, color="0.25", linewidth=1.5)

    ax.spines[:].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Success rate", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    # ── Group header brackets ─────────────────────────────────────────────────
    trans = ax.get_xaxis_transform()

    group_spans: dict[str, list] = {}
    for j, (_, _, g, *_) in enumerate(cols):
        group_spans.setdefault(g, []).append(j)

    Y_BRACKET = 1.05
    Y_LINE    = 1.09
    Y_TEXT    = 1.13

    for g, idxs in group_spans.items():
        x0  = idxs[0]  - 0.4
        x1  = idxs[-1] + 0.4
        xm  = (x0 + x1) / 2
        col = GROUP_COLORS.get(g, "0.4")

        ax.plot([x0, x0, x1, x1], [Y_BRACKET, Y_LINE, Y_LINE, Y_BRACKET],
                transform=trans, color=col, linewidth=1.2,
                clip_on=False, solid_capstyle="round")
        ax.text(xm, Y_TEXT, GROUP_LABEL_OVERRIDES.get(g, g),
                transform=trans, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=col, clip_on=False)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cols", default="all",
        choices=list(NAMED_COLS.keys()),
        help="Named column subset to plot (default: all)",
    )
    args = parser.parse_args()

    indices  = NAMED_COLS[args.cols]
    active   = [COLS[i] for i in indices]
    fname    = f"heatmap_{args.cols}"

    sr_mat, std_mat, _ = build_matrix(ROWS, active)
    zero_mut = load_mutation_counts(active, ROWS)
    fig = plot_heatmap(sr_mat, std_mat, ROWS, active, zero_mut=zero_mut)
    save(fig, fname)
    plt.show()