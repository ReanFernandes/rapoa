"""
Efficiency figure — token cost and step efficiency for non-optimised conditions.

Layout: 2 rows × 5 task columns.
  Row 1 — Token cost per episode, split by outcome:
           Solid bar = mean tokens on successful episodes
           Hatched bar on top = mean tokens on failed episodes
           Annotated n_success / n_total across all seeds
  Row 2 — Steps on successful episodes:
           Bar = mean across 6 seed-means
           Dots = individual per-seed means (6 points per condition)

Aggregation: per-seed means are computed first (6 seeds × 20 episodes each),
then the bar shows the mean of those 6 seed-means, and dots show all 6.
This is consistent with the 6-seed-means treatment used everywhere else.

Usage:
    cd final_paper_plotting
    python plot_efficiency.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import (
    TASKS, TASK_LABELS, FRESH_EVAL_DIR,
    FIG_WIDTH_FULL,
    apply_neurips_style, save,
)

apply_neurips_style()

EFFICIENCY_FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "efficiency"

def _save(fig, name):
    EFFICIENCY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = EFFICIENCY_FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")

# ── Conditions ────────────────────────────────────────────────────────────────
# (pipeline, variant, conv_mode) → display label
CONDITIONS = [
    (("balrog_baseline", "minimal", "history_1step"),  "BALROG\nmin 1s"),
    (("balrog_baseline", "minimal", "history_16step"), "BALROG\nmin 16s"),
    (("balrog_baseline", "rich",    "history_1step"),  "BALROG\nrich 1s"),
    (("balrog_baseline", "rich",    "history_16step"), "BALROG\nrich 16s"),
    (("with_descriptor", "minimal", "single_turn"),    "Ours\nminimal"),
    (("with_descriptor", "rich",    "single_turn"),    "Ours\nrich"),
]

# Okabe-Ito — matches main heatmap GROUP_COLORS where applicable
COND_COLORS = {
    ("balrog_baseline", "minimal", "history_1step"):  "#56B4E9",
    ("balrog_baseline", "minimal", "history_16step"): "#0072B2",
    ("balrog_baseline", "rich",    "history_1step"):  "#56B4E9",
    ("balrog_baseline", "rich",    "history_16step"): "#009E73",
    ("with_descriptor", "minimal", "single_turn"):    "#E69F00",
    ("with_descriptor", "rich",    "single_turn"):    "#D55E00",
}

ROWS = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_efficiency_data(log_dir: Path) -> dict:
    """
    Returns data[(pipeline, variant, conv_mode)][task][iseed] = list of episode dicts.
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

    return data


def episode_tokens(ep: dict, pipeline: str) -> int:
    agent = ep.get("agent_prompt_tokens", 0) + ep.get("agent_completion_tokens", 0)
    desc  = (ep.get("descriptor_prompt_tokens", 0) + ep.get("descriptor_completion_tokens", 0)
             if pipeline == "with_descriptor" else 0)
    return agent + desc


def compute_stats(data: dict, cond_key: tuple, task: str) -> dict | None:
    """
    Pool all episodes across all inference seeds.
    Returns mean tokens for failed and successful episodes separately,
    plus pooled SR.
    """
    pipeline  = cond_key[0]
    seed_data = data.get(cond_key, {}).get(task, {})
    if not seed_data:
        return None

    success_tokens, fail_tokens = [], []
    for episodes in seed_data.values():
        for ep in episodes:
            tok = episode_tokens(ep, pipeline)
            if ep["success"]:
                success_tokens.append(tok)
            else:
                fail_tokens.append(tok)

    n_success = len(success_tokens)
    n_total   = n_success + len(fail_tokens)

    return dict(
        mean_fail_tok    = float(np.mean(fail_tokens))    if fail_tokens    else 0.0,
        mean_success_tok = float(np.mean(success_tokens)) if success_tokens else 0.0,
        n_success        = n_success,
        n_total          = n_total,
        sr               = n_success / n_total if n_total else 0.0,
    )


# Short x-axis labels — used inside each subplot
COND_SHORT = ["B\nmin\n1s", "B\nmin\n16s", "B\nrich\n1s", "B\nrich\n16s",
              "Ours\nmin", "Ours\nrich"]


def plot_efficiency(data: dict):
    n_tasks = len(ROWS)
    n_conds = len(CONDITIONS)

    fig, axes = plt.subplots(1, n_tasks, figsize=(FIG_WIDTH_FULL, 3.2),
                             sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22,
                        wspace=0.08)

    x     = np.arange(n_conds)
    bar_w = 0.62

    for col, task in enumerate(ROWS):
        ax = axes[col]

        for i, (cond_key, _) in enumerate(CONDITIONS):
            stats = compute_stats(data, cond_key, task)
            if stats is None:
                continue
            color     = COND_COLORS[cond_key]
            fail_tok  = stats["mean_fail_tok"]
            succ_tok  = stats["mean_success_tok"]
            sr        = stats["sr"]

            # Failed episodes — bottom, solid
            if fail_tok > 0:
                ax.bar(x[i], fail_tok, bar_w, color=color, alpha=0.85)

            # Successful episodes — on top, lighter with hatch
            if succ_tok > 0:
                ax.bar(x[i], succ_tok, bar_w, bottom=fail_tok,
                       color=color, alpha=0.40, hatch="///",
                       edgecolor=color, linewidth=0)

            # SR% annotated in white inside the fail bar (rotated)
            # Only if the bar is tall enough to fit text
            sr_label = f"{100*sr:.0f}%"
            if fail_tok > 0:
                ax.text(x[i], fail_tok * 0.5, sr_label,
                        ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold",
                        rotation=90)
            elif succ_tok > 0:
                # All successes — write SR in the success bar
                ax.text(x[i], succ_tok * 0.5, sr_label,
                        ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold",
                        rotation=90)

        ax.set_title(TASK_LABELS[task], fontsize=8, fontweight="bold", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(COND_SHORT, fontsize=6.5)
        ax.tick_params(length=2, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.6, n_conds - 0.4)

    axes[0].set_ylabel("Mean tokens / episode", fontsize=8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="grey", alpha=0.85,
                       label="Failed episodes"),
        mpatches.Patch(facecolor="grey", alpha=0.40, hatch="///",
                       edgecolor="grey", label="Successful episodes"),
        mpatches.Patch(facecolor="white", edgecolor="none",
                       label="SR% shown in bar"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.54, 0.0), ncol=3,
               fontsize=7, framealpha=0.9, edgecolor="0.8")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading fresh eval data ...")
    data = load_efficiency_data(FRESH_EVAL_DIR)
    fig  = plot_efficiency(data)
    _save(fig, "efficiency_nonopt")
    plt.show()
