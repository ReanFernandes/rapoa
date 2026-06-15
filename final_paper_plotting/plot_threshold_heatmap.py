"""
Threshold sensitivity heatmap — success rate by task × threshold.

Three figures saved automatically:
  figures/threshold_heatmap_plain.pdf         plain/minimal prompt (δ=0.05 only; other δ not run)
  figures/threshold_heatmap_guided.pdf        guided/rich prompt, 1-step history
  figures/threshold_heatmap_guided_h16.pdf    guided/rich prompt, 16-step history

Each figure:
  Rows    = 5 tasks
  Columns = HSP × {δ=0.00, 0.02, 0.05, 0.10} | LSP × {δ=0.00, 0.02, 0.05, 0.10}
  Groups  = HSP | LSP

  Note: threshold_heatmap_plain has δ=0.05 populated only (thresh_sweep is rich-only).
  All other threshold columns for plain will render as "—".

Usage:
    cd final_paper_plotting
    python plot_threshold_heatmap.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import (
    TASKS, TASK_LABELS,
    FRESH_EVAL_DIR,
    VARIANT_LABELS, SP_LABELS, DELTA,
    FIG_WIDTH_FULL,
    CAMPAIGN_IDS,
    SLUGS, eval_dir as _eval_dir,
    apply_neurips_style, save,
)

apply_neurips_style()

_V = VARIANT_LABELS

_CK_MIN  = ("with_descriptor", "minimal", "single_turn")
_CK_RICH = ("with_descriptor", "rich",    "single_turn")

# ── Campaign selection ────────────────────────────────────────────────────────
CP_PRIMARY = CAMPAIGN_IDS["primary_20"]
CP_MINIMAL = CAMPAIGN_IDS["primary_minimal"]
CP_THRESH  = CAMPAIGN_IDS["thresh_sweep"]
CP_H16     = CAMPAIGN_IDS["thresh_sweep_h16"]

# ── Column definitions ────────────────────────────────────────────────────────
_THRESHOLDS = [
    ("000", f"{DELTA}=0.00"),
    ("002", f"{DELTA}=0.02"),
    ("default", f"{DELTA}=0.05"),
    ("010", f"{DELTA}=0.10"),
]

# (val_strat, variant, key) → (slug_key, campaign)
# thresh_sweep is rich-only; minimal only has the default (t005) from primary_minimal.
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

# h16 variant: thresh_sweep_h16 campaign, rich only (always_accept is the default)
_THRESH_MAP_H16 = {
    ("valbag",   "000"):     ("thresh000_hsp_h16", CP_H16),
    ("valbag",   "002"):     ("thresh002_hsp_h16", CP_H16),
    ("valbag",   "default"): ("hsp_rich_h16",      CP_H16),
    ("valbag",   "010"):     ("thresh010_hsp_h16", CP_H16),
    ("trainsig", "000"):     ("thresh000_lsp_h16", CP_H16),
    ("trainsig", "002"):     ("thresh002_lsp_h16", CP_H16),
    ("trainsig", "default"): ("lsp_rich_h16",      CP_H16),
    ("trainsig", "010"):     ("thresh010_lsp_h16", CP_H16),
}

def _make_cols(variant: str) -> list:
    cond_key = _CK_MIN if variant == "minimal" else _CK_RICH
    cols = []
    for val_strat, group_label in [("valbag", SP_LABELS["valbag"]),
                                    ("trainsig", SP_LABELS["trainsig"])]:
        for key, label in _THRESHOLDS:
            entry = _THRESH_MAP.get((val_strat, variant, key))
            if entry is None:
                cols.append((cond_key, label, group_label, None))
            else:
                sk, cam = entry
                cols.append((cond_key, label, group_label, _eval_dir(cam, SLUGS[sk])))
    return cols

def _make_cols_h16() -> list:
    cols = []
    for val_strat, group_label in [("valbag", SP_LABELS["valbag"]),
                                    ("trainsig", SP_LABELS["trainsig"])]:
        for key, label in _THRESHOLDS:
            sk, cam = _THRESH_MAP_H16[(val_strat, key)]
            cols.append((_CK_RICH, label, group_label, _eval_dir(cam, SLUGS[sk])))
    return cols

COLS_MIN  = _make_cols("minimal")
COLS_RICH = _make_cols("rich")
COLS_H16  = _make_cols_h16()

ROWS = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]

GROUP_COLORS = {
    SP_LABELS["valbag"]:   "#CC79A7",   # Okabe reddish purple
    SP_LABELS["trainsig"]: "#882255",   # dark magenta
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_fresh_eval(log_dir: Path) -> dict:
    if not log_dir.exists():
        return {}
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
            )
    return result


def build_matrix(rows, cols):
    _cache: dict[Path, dict] = {}

    def _get(eval_dir):
        if eval_dir not in _cache:
            _cache[eval_dir] = load_fresh_eval(eval_dir)
        return _cache[eval_dir]

    sr_mat  = np.full((len(rows), len(cols)), np.nan)
    std_mat = np.full((len(rows), len(cols)), np.nan)

    for i, task in enumerate(rows):
        for j, (cond_key, _, _, eval_dir) in enumerate(cols):
            cell = _get(eval_dir).get(cond_key, {}).get(task)
            if cell:
                sr_mat[i, j]  = cell["sr"]
                std_mat[i, j] = cell["std"]

    return sr_mat, std_mat


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(sr_mat, std_mat, rows, cols):
    n_rows = len(rows)
    n_cols = len(cols)

    HEADER_H = 0.60
    CELL_H   = 0.44
    CELL_W   = 0.58
    fig_h    = HEADER_H + n_rows * CELL_H + 0.4
    fig_w    = max(4.0, 1.35 + n_cols * CELL_W)

    left_frac  = max(0.08, 1.5 / fig_w)
    right_frac = min(0.96, 1 - 0.8 / fig_w)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(top=0.82, bottom=0.12,
                        left=left_frac, right=right_frac)

    cmap = plt.cm.viridis
    cmap.set_bad("0.93")
    im = ax.imshow(sr_mat, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")

    # Underline best value per row
    row_best = np.nanmax(sr_mat, axis=1, keepdims=True)
    is_best  = np.isclose(sr_mat, row_best) & ~np.isnan(sr_mat)

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

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([TASK_LABELS[t] for t in rows], fontsize=8.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([lbl for _, lbl, _, _ in cols], fontsize=7.0)
    ax.tick_params(length=0, pad=3)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Divider between variant groups
    groups = [g for _, _, g, _ in cols]
    for j in range(1, n_cols):
        if groups[j] != groups[j-1]:
            ax.axvline(j - 0.5, color="0.25", linewidth=1.5)

    ax.spines[:].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Success rate", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Group brackets
    trans = ax.get_xaxis_transform()
    group_spans: dict[str, list] = {}
    for j, (_, _, g, _) in enumerate(cols):
        group_spans.setdefault(g, []).append(j)

    Y_BRACKET, Y_LINE, Y_TEXT = 1.05, 1.09, 1.13

    for g, idxs in group_spans.items():
        x0  = idxs[0]  - 0.4
        x1  = idxs[-1] + 0.4
        xm  = (x0 + x1) / 2
        col = GROUP_COLORS.get(g, "0.4")
        ax.plot([x0, x0, x1, x1], [Y_BRACKET, Y_LINE, Y_LINE, Y_BRACKET],
                transform=trans, color=col, linewidth=1.2,
                clip_on=False, solid_capstyle="round")
        ax.text(xm, Y_TEXT, g, transform=trans,
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=col, clip_on=False)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for cols, fname in [
        (COLS_MIN,  "threshold_heatmap_plain"),
        (COLS_RICH, "threshold_heatmap_guided"),
        (COLS_H16,  "threshold_heatmap_guided_h16"),
    ]:
        sr_mat, std_mat = build_matrix(ROWS, cols)
        fig = plot(sr_mat, std_mat, ROWS, cols)
        save(fig, fname)

    plt.show()
