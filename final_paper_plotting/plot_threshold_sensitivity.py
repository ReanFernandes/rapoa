"""
Threshold sensitivity — per-task grouped bar chart.

4 subplots (2 selection pressures × 2 prompt variants).
x-axis : 5 tasks
Bars    : one per condition (non-opt, δ=-∞, δ=0.02, δ=0.05, δ=0.10), grouped per task.
◦ on bar top = zero accepted mutations (prompt unchanged from baseline).

Usage:
    cd final_paper_plotting
    python plot_threshold_sensitivity.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import (
    TASKS, TASK_LABELS, TASK_OPT_NAMES,
    FRESH_EVAL_DIR,
    VARIANT_LABELS, SP_LABELS, DELTA, AA_LABEL, T_STAR, FINAL_INC,
    FIG_WIDTH_FULL,
    CAMPAIGN_IDS,
    SLUGS, eval_dir as _eval_dir, opt_dir,
    apply_neurips_style, save,
)

_V = VARIANT_LABELS

apply_neurips_style()

VARIANTS   = ["minimal", "rich"]
VAL_STRATS = ["valbag", "trainsig"]

_CK_MIN  = ("with_descriptor", "minimal", "single_turn")
_CK_RICH = ("with_descriptor", "rich",    "single_turn")

ROWS = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]

# ── Campaign selection ────────────────────────────────────────────────────────
CP_PRIMARY = CAMPAIGN_IDS["primary_20"]
CP_MINIMAL = CAMPAIGN_IDS["primary_minimal"]
CP_THRESH  = CAMPAIGN_IDS["thresh_sweep"]

# ── Condition definitions ─────────────────────────────────────────────────────
COND_DEFS = [
    ("non-opt",                   "#999999", "baseline"),
    (f"{AA_LABEL}, {FINAL_INC}",  "#E69F00", "aa"),
    (f"{DELTA}=0.00",             "#88CCEE", "000"),
    (f"{DELTA}=0.02",             "#56B4E9", "002"),
    (f"{DELTA}=0.05",             "#0072B2", "default"),
    (f"{DELTA}=0.10",             "#003f63", "010"),
]

# Slug mapping for threshold conditions: (val_strat, variant, key) → (SLUGS key, campaign)
# thresh_sweep only has rich; default (t005) lives in the primary campaigns.
_THRESH_MAP = {
    ("valbag",   "rich",    "000"):     ("thresh000_hsp", CP_THRESH),
    ("valbag",   "rich",    "002"):     ("thresh002_hsp", CP_THRESH),
    ("valbag",   "rich",    "default"): ("hsp_rich",      CP_PRIMARY),
    ("valbag",   "rich",    "010"):     ("thresh010_hsp", CP_THRESH),
    ("valbag",   "minimal", "default"): ("hsp_minimal",   CP_MINIMAL),
    ("trainsig", "rich",    "000"):     ("thresh000_lsp", CP_THRESH),
    ("trainsig", "rich",    "002"):     ("thresh002_lsp", CP_THRESH),
    ("trainsig", "rich",    "default"): ("lsp_rich",      CP_PRIMARY),
    ("trainsig", "rich",    "010"):     ("thresh010_lsp", CP_THRESH),
    ("trainsig", "minimal", "default"): ("lsp_minimal",   CP_MINIMAL),
}

def _thresh_eval_dir(val_strat: str, variant: str, key: str) -> "Path | None":
    entry = _THRESH_MAP.get((val_strat, variant, key))
    if not entry:
        return None
    sk, campaign = entry
    return _eval_dir(campaign, SLUGS[sk])

def _thresh_opt_dir(val_strat: str, variant: str, key: str) -> "Path | None":
    entry = _THRESH_MAP.get((val_strat, variant, key))
    if not entry:
        return None
    sk, campaign = entry
    return opt_dir(campaign, SLUGS[sk])

def _aa_eval_dir(variant: str) -> Path:
    if variant == "minimal":
        return _eval_dir(CP_MINIMAL, SLUGS["always_accept_minimal"])
    return _eval_dir(CP_PRIMARY, SLUGS["always_accept_rich"])

# ── Layout ────────────────────────────────────────────────────────────────────
N_CONDS    = len(COND_DEFS)
BAR_W      = 0.13
INNER_GAP  = 0.025
GROUP_W    = N_CONDS * BAR_W + (N_CONDS - 1) * INNER_GAP
TASK_STEP  = GROUP_W + 0.28

_OFFSETS = [(i * (BAR_W + INNER_GAP) - (N_CONDS - 1) * (BAR_W + INNER_GAP) / 2)
            for i in range(N_CONDS)]
_TASK_X  = [t * TASK_STEP for t in range(len(ROWS))]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_zero_mutations(run_dir: "Path | None") -> set:
    """Return set of short task names where no mutations were accepted."""
    _OPT_TO_EVAL = {v: k for k, v in TASK_OPT_NAMES.items()}
    zero = set()
    if not run_dir or not run_dir.exists():
        return zero
    for log in run_dir.rglob("optimisation_log.jsonl"):
        if any(p.startswith(("opt_cycle_", "env_round_", "eval_")) for p in log.parts):
            continue
        opt_task  = log.parent.name
        eval_task = _OPT_TO_EVAL.get(opt_task, opt_task)
        try:
            records = [json.loads(l) for l in open(log)]
            n = sum(1 for r in records
                    if r.get("record_type") == "opt_cycle"
                    and r.get("opt_cycle_outcome") == "accepted")
            if n == 0:
                zero.add(eval_task)
        except Exception:
            pass
    return zero


def load_sr(log_dir: Path, cond_key: tuple) -> dict:
    data = defaultdict(list)
    for summary in sorted(log_dir.rglob("run_summary.json")):
        parts = summary.parts
        try:
            anchor = next(i for i, p in enumerate(parts) if p == log_dir.name)
            _, task, _, pipeline, variant, conv_mode, *_ = parts[anchor + 1:]
        except (StopIteration, ValueError):
            continue
        if task not in TASKS or (pipeline, variant, conv_mode) != cond_key:
            continue
        with open(summary) as f:
            d = json.load(f)
        data[task].extend(d["episodes"])
    return {task: sum(1 for e in eps if e["success"]) / len(eps)
            for task, eps in data.items() if eps}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_threshold_sensitivity():
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH_FULL, 5.6),
        sharey=True,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.20,
                        hspace=0.35, wspace=0.08)

    for row_idx, val_strat in enumerate(VAL_STRATS):
        for col_idx, variant in enumerate(VARIANTS):
            ax       = axes[row_idx][col_idx]
            cond_key = _CK_MIN if variant == "minimal" else _CK_RICH

            base_sr  = load_sr(FRESH_EVAL_DIR, cond_key)
            aa_dir   = _aa_eval_dir(variant)
            aa_sr    = load_sr(aa_dir, cond_key) if aa_dir.exists() else {}

            thresh_data  = {}
            thresh_zeros = {}
            for _, _, key in COND_DEFS:
                if key in ("baseline", "aa"):
                    continue
                ed = _thresh_eval_dir(val_strat, variant, key)
                od = _thresh_opt_dir(val_strat, variant, key)
                thresh_data[key]  = load_sr(ed, cond_key) if ed and ed.exists() else {}
                thresh_zeros[key] = load_zero_mutations(od)

            for ti, task in enumerate(ROWS):
                tx = _TASK_X[ti]
                for ci, (_, color, key) in enumerate(COND_DEFS):
                    bx = tx + _OFFSETS[ci]
                    if key == "baseline":
                        sr = base_sr.get(task, 0.0)
                        ax.bar(bx, sr, BAR_W, color=color, alpha=0.75,
                               hatch="///", edgecolor="white",
                               linewidth=0.3, zorder=2)
                    elif key == "aa":
                        sr = aa_sr.get(task, 0.0)
                        ax.bar(bx, sr, BAR_W, color=color, alpha=0.88,
                               edgecolor="white", linewidth=0.3, zorder=2)
                    elif key == "aa_bt":
                        sr = aa_bt_sr.get(task, 0.0)
                        ax.bar(bx, sr, BAR_W, color=color, alpha=0.88,
                               hatch="\\\\", edgecolor="white",
                               linewidth=0.3, zorder=2)
                    else:
                        sr = thresh_data[key].get(task, 0.0)
                        ax.bar(bx, sr, BAR_W, color=color, alpha=0.88,
                               edgecolor="white", linewidth=0.3, zorder=2)
                        if task in thresh_zeros.get(key, set()):
                            ax.text(bx, sr + 0.012, "◦",
                                    ha="center", va="bottom",
                                    fontsize=5.5, color="0.35", zorder=3)

            # Light vertical separators between task groups
            for ti in range(1, len(ROWS)):
                sep = (_TASK_X[ti - 1] + _TASK_X[ti]) / 2
                ax.axvline(sep, color="0.88", linewidth=0.6, zorder=0)

            ax.set_xlim(_TASK_X[0] - GROUP_W / 2 - 0.1,
                        _TASK_X[-1] + GROUP_W / 2 + 0.1)
            ax.set_xticks(_TASK_X)
            ax.set_xticklabels([TASK_LABELS[t] for t in ROWS],
                               fontsize=7, rotation=20, ha="right")
            ax.set_ylim(0, 1.10)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6.5)
            ax.tick_params(length=2, pad=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title((_V[variant]+' prompt').title(), fontsize=8, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"{SP_LABELS[val_strat]}\nSuccess rate", fontsize=7.5)

    # ── Legend ────────────────────────────────────────────────────────────────
    _HATCH = {"baseline": "///", "aa_bt": "\\\\"}
    cond_patches = []
    for label, color, key in COND_DEFS:
        cond_patches.append(
            mpatches.Patch(facecolor=color, alpha=0.88,
                           hatch=_HATCH.get(key), edgecolor="white",
                           label=label)
        )
    zero_handle = plt.Line2D([0], [0], marker="$◦$", color="0.35",
                              markersize=7, linestyle="none",
                              label="zero mutations")
    fig.legend(handles=cond_patches + [zero_handle],
               loc="lower center", bbox_to_anchor=(0.50, 0.06),
               ncol=len(COND_DEFS) + 1, fontsize=8.5,
               framealpha=0.9, edgecolor="0.8")

    return fig


if __name__ == "__main__":
    fig = plot_threshold_sensitivity()
    save(fig, "threshold_sensitivity")
    plt.show()
