"""
Cross-task transfer heatmaps — one 5×5 figure per optimised condition.

Rows    = source task (which task the prompt was optimised for)
Columns = eval task   (which task it was evaluated on)
Diagonal = on-task performance (sanity check vs fresh-eval heatmap)

◦ on the row label indicates the source task had zero accepted mutations —
  the prompt is identical to the non-optimised baseline.

Usage:
    cd final_paper_plotting
    python plot_crosstask_heatmap.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import (
    AGENT_NAME, TASKS, TASK_LABELS, TASK_OPT_NAMES,
    FIGURES_DIR,
    VARIANT_LABELS, AA_LABEL, T_STAR, FINAL_INC, SP_LABELS, DELTA,
    ENV_NAME, MODEL_NAME,
    SLUGS, opt_dir, latest_campaign,
    apply_neurips_style,
)
from pathlib import Path as _Path

_V = VARIANT_LABELS

CROSSTASK_FIGURES_DIR = FIGURES_DIR / "crosstask"

def save(fig, name: str) -> None:
    CROSSTASK_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = CROSSTASK_FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")

apply_neurips_style()

# Cross-task eval root follows same {env}/{model}/{campaign}/{slug} structure
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
CROSSTASK_ROOT = _PROJECT_ROOT / "logs_crosstask_eval" / ENV_NAME / MODEL_NAME

CAMPAIGN_PRIMARY = latest_campaign(SLUGS["hsp_rich"])

_CK_OURS_MIN  = ("with_descriptor", "minimal", "single_turn")
_CK_OURS_RICH = ("with_descriptor", "rich",    "single_turn")
_CK_BALROG    = ("balrog_baseline", "minimal", "history_16step")

# ── Condition definitions ─────────────────────────────────────────────────────
# log_root: directory containing one subdir per source task
# cond_key: (pipeline, variant, conv_mode) key used by load_fresh_eval
# opt_run_id / variant_filter: for zero-mutation row detection

_HSP = f"HSP, {DELTA}=0.05"
_LSP = f"LSP, {DELTA}=0.05"
_AA_BT  = f"{AA_LABEL}, {T_STAR}"
_AA_FIN = f"{AA_LABEL}, {FINAL_INC}"

def _ct(slug_key):
    """Cross-task log root for a slug (campaign/slug level)."""
    return CROSSTASK_ROOT / CAMPAIGN_PRIMARY / SLUGS[slug_key] if CAMPAIGN_PRIMARY else None

CONDITIONS = [
    dict(label=f"{AGENT_NAME} {_V['minimal']} — {_HSP}", fname="crosstask_ours_min_gated",
         log_root=_ct("hsp_minimal"), cond_key=_CK_OURS_MIN,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["hsp_minimal"]) if CAMPAIGN_PRIMARY else None),
    dict(label=f"{AGENT_NAME} {_V['rich']} — {_HSP}", fname="crosstask_ours_rich_gated",
         log_root=_ct("hsp_rich"), cond_key=_CK_OURS_RICH,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["hsp_rich"]) if CAMPAIGN_PRIMARY else None),
    dict(label=f"BALROG — {_HSP}", fname="crosstask_balrog_gated",
         log_root=_ct("balrog_minimal"), cond_key=_CK_BALROG,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["balrog_minimal"]) if CAMPAIGN_PRIMARY else None),
    dict(label=f"{AGENT_NAME} {_V['minimal']} — {_LSP}", fname="crosstask_ours_min_gated_trainsig",
         log_root=_ct("lsp_minimal"), cond_key=_CK_OURS_MIN,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["lsp_minimal"]) if CAMPAIGN_PRIMARY else None),
    dict(label=f"{AGENT_NAME} {_V['rich']} — {_LSP}", fname="crosstask_ours_rich_gated_trainsig",
         log_root=_ct("lsp_rich"), cond_key=_CK_OURS_RICH,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["lsp_rich"]) if CAMPAIGN_PRIMARY else None),
    dict(label=f"{AGENT_NAME} {_V['minimal']} — {_AA_FIN}", fname="crosstask_ours_min_s2_inc",
         log_root=_ct("always_accept_minimal"), cond_key=_CK_OURS_MIN,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["always_accept_minimal"]) if CAMPAIGN_PRIMARY else None),
    dict(label=f"{AGENT_NAME} {_V['rich']} — {_AA_FIN}", fname="crosstask_ours_rich_s2_inc",
         log_root=_ct("always_accept_rich"), cond_key=_CK_OURS_RICH,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["always_accept_rich"]) if CAMPAIGN_PRIMARY else None),
    dict(label=f"BALROG — {_AA_FIN}", fname="crosstask_balrog_s2_inc",
         log_root=_ct("balrog_minimal"), cond_key=_CK_BALROG,
         opt_dir=opt_dir(CAMPAIGN_PRIMARY, SLUGS["balrog_minimal"]) if CAMPAIGN_PRIMARY else None),
]

ROWS = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_source_task(src_dir: Path, cond_key: tuple) -> dict | None:
    """
    Load fresh-eval stats for one source-task directory.
    Returns {eval_task: {sr, std, n}} or None if the directory is missing.
    """
    if not src_dir.exists():
        return None

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for summary_path in sorted(src_dir.rglob("run_summary.json")):
        parts = summary_path.parts
        try:
            anchor = next(i for i, p in enumerate(parts) if p == src_dir.name)
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

    cond_data = data.get(cond_key, {})
    if not cond_data:
        return None

    result = {}
    for task, seed_eps in cond_data.items():
        all_eps  = [e for eps in seed_eps.values() for e in eps]
        n        = len(all_eps)
        seed_srs = [
            sum(1 for e in eps if e["success"]) / len(eps)
            for eps in seed_eps.values() if eps
        ]
        result[task] = dict(
            sr  = sum(1 for e in all_eps if e["success"]) / n,
            std = float(np.std(seed_srs)) if len(seed_srs) > 1 else 0.0,
            n   = n,
        )
    return result


def build_matrix(cond: dict) -> tuple:
    """Build SR / std matrices (n_src × n_eval) for one condition."""
    n = len(ROWS)
    sr_mat  = np.full((n, n), np.nan)
    std_mat = np.full((n, n), np.nan)

    for i, src_task in enumerate(ROWS):
        src_dir = cond["log_root"] / src_task
        task_data = load_source_task(src_dir, cond["cond_key"])
        if task_data is None:
            continue
        for j, eval_task in enumerate(ROWS):
            cell = task_data.get(eval_task)
            if cell:
                sr_mat[i, j]  = cell["sr"]
                std_mat[i, j] = cell["std"]

    return sr_mat, std_mat


def zero_mutation_rows(cond: dict) -> set:
    """Return set of source task names (short eval form) where zero mutations were accepted."""
    _OPT_TO_EVAL = {v: k for k, v in TASK_OPT_NAMES.items()}
    run_dir = cond.get("opt_dir")
    zero = set()
    if not run_dir or not run_dir.exists():
        return zero
    for log in run_dir.rglob("optimisation_log.jsonl"):
        if any(p.startswith(("opt_cycle_", "env_round_", "eval_"))
               for p in log.parts):
            continue
        task = _OPT_TO_EVAL.get(log.parent.name, log.parent.name)
        try:
            records = [json.loads(l) for l in open(log)]
            n = sum(1 for r in records
                    if r.get("record_type") == "opt_cycle"
                    and r.get("opt_cycle_outcome") == "accepted")
            if n == 0:
                zero.add(task)
        except Exception:
            pass
    return zero


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_crosstask(sr_mat, std_mat, label, zero_rows):
    n = len(ROWS)
    CELL = 0.52
    fig_size = 0.9 + n * CELL
    fig, ax = plt.subplots(figsize=(fig_size + 0.6, fig_size))
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.22, right=0.90)

    cmap = plt.cm.viridis
    cmap.set_bad("0.93")

    im = ax.imshow(sr_mat, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")

    # Best per column (eval task) underlined
    col_best = np.nanmax(sr_mat, axis=0, keepdims=True)
    is_best  = np.isclose(sr_mat, col_best) & ~np.isnan(sr_mat)

    for i in range(n):
        for j in range(n):
            v   = sr_mat[i, j]
            std = std_mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="0.5")
                continue
            text_col = "white" if v < 0.25 else "0.1"
            std_col  = "white" if v < 0.25 else "0.35"
            ax.text(j, i - 0.13, f"{100*v:.0f}%",
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=text_col)
            if not np.isnan(std):
                ax.text(j, i + 0.22, f"±{100*std:.0f}%",
                        ha="center", va="center",
                        fontsize=6.0, color=std_col)
            if is_best[i, j]:
                ax.plot([j - 0.28, j + 0.28], [i + 0.38, i + 0.38],
                        color=text_col, linewidth=1.2,
                        solid_capstyle="round", clip_on=False)

    # Row labels: task name + ◦ if zero-mutation source
    row_labels = []
    for t in ROWS:
        lbl = TASK_LABELS[t]
        if t in zero_rows:
            lbl = f"{lbl} ◦"
        row_labels.append(lbl)

    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n))
    ax.set_xticklabels([TASK_LABELS[t] for t in ROWS], fontsize=8,
                       rotation=30, ha="right")
    ax.tick_params(length=0, pad=3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.spines[:].set_visible(False)

    # Axis labels
    ax.set_ylabel("Source task (prompt optimised for)", fontsize=8, labelpad=6)
    ax.set_xlabel("Eval task", fontsize=8, labelpad=6)

    # Title
    ax.set_title(label, fontsize=8.5, fontweight="bold", pad=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Success rate", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_generated = 0
    for cond in CONDITIONS:
        if not cond["log_root"].exists():
            print(f"  SKIP {cond['fname']} — log root missing")
            continue

        sr_mat, std_mat = build_matrix(cond)
        n_cells = np.sum(~np.isnan(sr_mat))

        if n_cells == 0:
            print(f"  SKIP {cond['fname']} — no data yet")
            continue

        zero_rows = zero_mutation_rows(cond)
        fig = plot_crosstask(sr_mat, std_mat, cond["label"], zero_rows)

        save(fig, cond["fname"])
        plt.close(fig)
        n_generated += 1

        filled = f"{n_cells}/{len(ROWS)**2}"
        print(f"  {cond['fname']}  ({filled} cells filled)")

    print(f"\nDone. Generated {n_generated}/{len(CONDITIONS)} figures.")
