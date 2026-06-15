"""
Optimisation trajectory plot.

Three subplots (one per optimisation condition):
  Ours/minimal  |  Ours/rich  |  BALROG/minimal

Each subplot shows 5 task lines on the T-pool success rate axis:
  — Solid step function  : gated (stage 3) — only jumps at accepted cycles
  — Dashed continuous    : always_accept (stage 2) — backfill trajectory

Stage 2 dashed lines are omitted if backfill data is not yet available.

Usage:
    cd final_paper_plotting
    python plot_opt_trajectory.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from config import (
    AGENT_NAME, TASKS, TASK_LABELS, TASK_OPT_NAMES,
    FIG_WIDTH_FULL, CAMPAIGN_IDS,
    VARIANT_LABELS, AA_LABEL, T_STAR, FINAL_INC, SP_LABELS,
    SLUGS, opt_dir,
    apply_neurips_style, save,
)

_V = VARIANT_LABELS

apply_neurips_style()

GOLD = "#FFD700"   # eval-marker fill colour

# ── Campaign selection ────────────────────────────────────────────────────────
CP_R = CAMPAIGN_IDS["primary_20"]       # primary: rich conditions
CP_M = CAMPAIGN_IDS["primary_minimal"]  # primary: minimal conditions
CP_B = CAMPAIGN_IDS["balrog_all"]       # balrog conditions
CAMPAIGN_THRESH = CAMPAIGN_IDS["thresh_sweep"]

# ── Validation strategy toggle ────────────────────────────────────────────────
# "valbag"   : spa_mean_valbag_t005_rich / balrog_h16_minimal
# "trainsig" : spa_mean_trainsig_t005_rich / balrog_h16_minimal
VAL_STRAT = "trainsig"

_GATED_SLUGS = {
    "valbag":   {"ours_min": "hsp_minimal",  "ours_rich": "hsp_rich",  "balrog": "balrog_minimal"},
    "trainsig": {"ours_min": "lsp_minimal",  "ours_rich": "lsp_rich",  "balrog": "balrog_minimal"},
}

# Threshold sweep slugs (Ours only — no BALROG threshold variants)
_THRESH_SLUGS = {
    ("valbag",   "000"): "thresh000_hsp",
    ("valbag",   "002"): "thresh002_hsp",
    ("valbag",   "010"): "thresh010_hsp",
    ("trainsig", "000"): "thresh000_lsp",
    ("trainsig", "002"): "thresh002_lsp",
    ("trainsig", "010"): "thresh010_lsp",
}

# ── Metric toggle ─────────────────────────────────────────────────────────────
# "sr"     : success rate (binary success per episode)
# "reward" : mean episode reward (encodes both success and step efficiency)
METRIC = "reward"

# ── Condition colour palette ──────────────────────────────────────────────────
# One colour per optimization condition; solid = stage 3 (gated), dashed = stage 2
COND_COLORS = {
    f"{AGENT_NAME}/{_V['minimal']}":        "#E69F00",   # Okabe orange
    f"{AGENT_NAME}/{_V['rich']}":           "#D55E00",   # Okabe vermillion
    f"BALROG/{_V['minimal']}":              "#0072B2",   # Okabe blue
    f"BALROG/{_V['minimal']}/16-step":      "#0072B2",   # same — condition label includes /16-step
}

# Display order (same as heatmap rows)
TASK_ORDER = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]

MAX_CYCLES = 20

# ── Condition definitions ─────────────────────────────────────────────────────
def _make_thresh_conditions(val_strat: str, thresh_key: str) -> list:
    """Ours plain + Ours guided only — no BALROG threshold variants."""
    slug_key = _THRESH_SLUGS[(val_strat, thresh_key)]
    return [
        dict(label=f"{AGENT_NAME}/{_V['minimal']}", gated_dir=opt_dir(CAMPAIGN_THRESH, SLUGS[slug_key]), aa_dir=None),
        dict(label=f"{AGENT_NAME}/{_V['rich']}",    gated_dir=opt_dir(CAMPAIGN_THRESH, SLUGS[slug_key]), aa_dir=None),
    ]


def _make_conditions(val_strat: str) -> list:
    slugs = _GATED_SLUGS[val_strat]
    return [
        dict(
            label    = f"{AGENT_NAME}/{_V['minimal']}",
            gated_dir = opt_dir(CP_M, SLUGS[slugs["ours_min"]]),
            aa_dir    = opt_dir(CP_M, SLUGS["always_accept_minimal"]),
        ),
        dict(
            label    = f"{AGENT_NAME}/{_V['rich']}",
            gated_dir = opt_dir(CP_R, SLUGS[slugs["ours_rich"]]),
            aa_dir    = opt_dir(CP_R, SLUGS["always_accept_rich"]),
        ),
        dict(
            label    = f"BALROG/{_V['minimal']}/16-step",
            gated_dir = opt_dir(CP_B, SLUGS[slugs["balrog"]]),
            aa_dir    = opt_dir(CP_M, SLUGS["always_accept_minimal"]),
        ),
    ]

CONDITIONS = _make_conditions(VAL_STRAT)


# ── Data loading ──────────────────────────────────────────────────────────────

def _extract(lst):
    """Return (success_rate, mean_reward) from a list of per-episode rewards."""
    if not lst:
        return None, None
    vals = []
    for v in lst:
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, dict):
            vals.append(float(v.get("reward", v.get("success", 0))))
    if not vals:
        return None, None
    return sum(1 for v in vals if v > 0) / len(vals), sum(vals) / len(vals)


def _metric(lst):
    """Return the currently selected metric value for a reward list."""
    sr, mr = _extract(lst)
    if METRIC == "reward":
        return mr
    return sr


def load_stage3_trajectory(run_dir: Path, task: str) -> list:
    """
    Returns list of (cycle, t_sr) defining the step function.
    First point is always (0, T_init). Subsequent points at accepted cycles.
    Empty list if run dir or log is missing.

    run_dir: slug-level opt dir (from opt_dir(campaign, slug)).
    task: short eval name (e.g. 'goto') — mapped to full opt task name internally.
    """
    task_dir = run_dir / TASK_OPT_NAMES.get(task, task)

    log = task_dir / "optimisation_log.jsonl"
    if not log.exists():
        return []

    records = [json.loads(l) for l in open(log)]
    cycles  = [r for r in records if r.get("record_type") == "opt_cycle"]
    if not cycles:
        return []

    # Initial T from cycle 1's incumbent (no separate init record in these runs)
    t_init = _metric(cycles[0].get("t_incumbent_rewards", []))
    if t_init is None:
        return []

    points = [(0, t_init)]

    for c in cycles:
        if c.get("opt_cycle_outcome") != "accepted":
            continue
        cn = c.get("opt_cycle")
        for cand in c.get("candidates_tried", []):
            t_res = cand.get("t_result", {})
            if not isinstance(t_res, dict):
                continue
            if t_res.get("verdict") != "accepted":
                continue
            t_chal = _metric(t_res.get("challenger_rewards", []))
            if t_chal is not None:
                points.append((cn, t_chal))
            break   # only the first accepted candidate per cycle

    return points


def load_stage2_trajectory(run_dir: Path, task: str) -> list:
    """
    Returns list of (cycle, metric_value) for cycles 0..N for an always-accept run.

    Reads optimisation_log.jsonl with T baked in.
    run_dir: slug-level opt dir; task: short eval name.
    """
    task_dir = run_dir / TASK_OPT_NAMES.get(task, task)

    # ── Source 1: inline log ──────────────────────────────────────────────────
    log = task_dir / "optimisation_log.jsonl"
    if log.exists():
        records = [json.loads(l) for l in open(log)]
        cycles  = [r for r in records if r.get("record_type") == "opt_cycle"]
        if cycles:
            # cycle 0: T of the initial prompt = incumbent at start of first cycle
            t0 = _metric(cycles[0].get("t_incumbent_rewards", []))
            if t0 is not None:
                points = [(0, t0)]
                for c in cycles:
                    cn = c.get("opt_cycle")
                    # Find the accepted candidate's T rewards (always_accept → first cand)
                    for cand in c.get("candidates_tried", []):
                        t_res = cand.get("t_result", {})
                        if not isinstance(t_res, dict):
                            continue
                        t_chal = _metric(t_res.get("challenger_rewards", []))
                        if t_chal is not None:
                            points.append((cn, t_chal))
                        break  # one candidate per cycle for always_accept
                if len(points) > 1:   # at least one cycle has inline T data
                    return sorted(points)

    # ── Source 2: backfill directories ────────────────────────────────────────
    bf_dir = task_dir / "t_pool_backfill"
    if not bf_dir.exists():
        return []

    points = []
    for cycle_dir in sorted(bf_dir.glob("cycle_*")):
        try:
            cn = int(cycle_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        summaries = list(cycle_dir.rglob("run_summary.json"))
        episodes  = []
        for s in summaries:
            d = json.loads(s.read_text())
            episodes.extend(d.get("episodes", []))
        if episodes:
            rewards = [e.get("reward", float(e["success"])) for e in episodes]
            val = (sum(1 for v in rewards if v > 0) / len(rewards)
                   if METRIC == "sr"
                   else sum(rewards) / len(rewards))
            points.append((cn, val))

    return sorted(points)


def step_xy(points: list, max_cycle: int = MAX_CYCLES):
    """
    Convert sparse (cycle, t_sr) points into arrays suitable for ax.step(..., where='post').
    The last point is extended to max_cycle so the line reaches the right edge.
    """
    if not points:
        return np.array([]), np.array([])
    pts = sorted(points)
    xs = [p[0] for p in pts] + [max_cycle]
    ys = [p[1] for p in pts] + [pts[-1][1]]
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


# ── Eval-selection markers ────────────────────────────────────────────────────

def _draw_stage3_markers(ax, pts, cond_color, normalise=False):
    """Gold ★ at the start of the last highest step — skipped for zero-mutation tasks."""
    if not pts:
        return
    draw_pts = _normalise(pts) if normalise else pts
    last     = sorted(draw_pts)[-1]   # (cycle, value) of last accepted mutation
    if last[0] == 0:                  # no mutations accepted — nothing to mark
        return
    ax.vlines(last[0], 0, last[1], colors=cond_color,
              linewidths=0.7, linestyles=":", alpha=0.45, zorder=2)
    ax.scatter([last[0]], [last[1]], marker="*", s=80,
               facecolors=GOLD, edgecolors=cond_color, linewidths=1.2, zorder=8)


def _draw_stage2_markers(ax, pts, cond_color):
    """Gold ★ at best-T cycle (solid border), gold ◆ at end-of-run (grey border).
    If both coincide only the star is drawn."""
    if not pts:
        return
    best_t = max(pts, key=lambda p: p[1])
    end    = max(pts, key=lambda p: p[0])

    # Best-T: gold star, condition-colour border
    ax.vlines(best_t[0], 0, best_t[1], colors=cond_color,
              linewidths=0.7, linestyles=":", alpha=0.45, zorder=2)
    ax.scatter([best_t[0]], [best_t[1]], marker="*", s=80,
               facecolors=GOLD, edgecolors=cond_color, linewidths=1.2, zorder=8)

    # End-of-run: gold diamond, grey border — only when different from best-T
    if end[0] != best_t[0]:
        ax.vlines(end[0], 0, end[1], colors=cond_color,
                  linewidths=0.7, linestyles=":", alpha=0.45, zorder=2)
        ax.scatter([end[0]], [end[1]], marker="D", s=30,
                   facecolors=GOLD, edgecolors="#888888", linewidths=0.9, zorder=8)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _normalise(points: list) -> list:
    """Subtract the cycle-0 value so every series starts at 0."""
    if not points:
        return []
    t0 = sorted(points)[0][1]
    return [(c, v - t0) for c, v in points]


def _draw_stage3(ax, pts, color, lw=1.6, zero_lw=1.4, normalise=False):
    """
    Draw a stage3 step function (absolute values by default).
    normalise=True subtracts the cycle-0 value (delta view).
    Zero-mutation tasks drawn as thin dotted lines to de-weight them.
    """
    draw_pts = _normalise(pts) if normalise else pts
    xs, ys = step_xy(draw_pts)
    has_mutations = len(draw_pts) > 1
    ax.step(xs, ys, where="post", color=color,
            linewidth=lw if has_mutations else zero_lw,
            linestyle="-" if has_mutations else ":",
            alpha=1.0 if has_mutations else 0.45,
            zorder=3 if has_mutations else 1)
    for cx, cy in draw_pts[1:]:
        ax.scatter([cx], [cy], color=color, s=20, zorder=5,
                   edgecolors="white", linewidths=0.5)


def _apply_ax_style(ax, title, fmt, zero_line=False):
    if zero_line:
        ax.axhline(0, color="0.70", linewidth=0.7, linestyle=":", zorder=0)
    ax.grid(True, color="0.88", linewidth=0.45, alpha=0.8, zorder=0)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    ax.set_xlabel("Cycle", fontsize=7.5)
    ax.set_xlim(0, MAX_CYCLES + 1.2)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=7, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([i * 0.125 for i in range(9)])


def plot_trajectories(conditions, task_order, gated_only: bool = False,
                      row1_title: str = "Gated (δ=0.05)"):
    """
    Two-row layout (default), ABSOLUTE T-pool values:

    Row 1 — Stage 3 (gated) only.  All conditions as coloured step functions.
             Starting heights reflect baseline advantage from decomposition.
             Zero-mutation tasks: thin dotted flat line (present but de-weighted).

    Row 2 — Ablation: Ours/minimal only, stage 3 solid vs stage 2 dashed.
             No acceptance dots — clean two-line strategy comparison.

    gated_only=True: single-row layout (row 1 only), used for threshold sweep plots.
    """
    n_tasks = len(task_order)
    fig_w   = FIG_WIDTH_FULL
    n_rows  = 1 if gated_only else 2
    fig_h   = 2.4 if gated_only else 4.2

    fig, axes_raw = plt.subplots(n_rows, n_tasks, figsize=(fig_w, fig_h),
                                 sharey=True, constrained_layout=False)
    # Normalise axes to always be 2-D array shape (n_rows, n_tasks)
    if n_rows == 1:
        import numpy as _np
        axes = _np.array([axes_raw])
    else:
        axes = axes_raw

    top    = 0.84 if gated_only else 0.87
    bottom = 0.22 if gated_only else 0.16
    fig.subplots_adjust(left=0.10, right=0.98, top=top, bottom=bottom,
                        hspace=0.60, wspace=0.12)
    unit = "SR" if METRIC == "sr" else "reward"
    fmt  = (plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%")
            if METRIC == "sr"
            else plt.FuncFormatter(lambda v, _: f"{v:.3f}"))

    # ── Row 1: stage 3 absolute — all conditions ──────────────────────────
    for col, task in enumerate(task_order):
        ax = axes[0][col]
        for cond in conditions:
            s3_pts = load_stage3_trajectory(cond["gated_dir"], task)
            if s3_pts:
                _draw_stage3(ax, s3_pts, COND_COLORS[cond["label"]])
                _draw_stage3_markers(ax, s3_pts, COND_COLORS[cond["label"]])
        _apply_ax_style(ax, TASK_LABELS[task], fmt)

    axes[0][0].set_ylabel(f"Selection score ({unit})", fontsize=8)
    fig.text(0.50, 0.920, row1_title,
             ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="0.2")

    if not gated_only:
        # ── Row 2: always-accept — all conditions, no stage 3 ────────────
        for col, task in enumerate(task_order):
            ax = axes[1][col]
            for cond in conditions:
                if not cond["aa_dir"]:
                    continue
                s2_pts = load_stage2_trajectory(cond["aa_dir"], task)
                if s2_pts:
                    ax.plot([p[0] for p in s2_pts], [p[1] for p in s2_pts],
                            color=COND_COLORS[cond["label"]], linewidth=1.1,
                            alpha=0.85, zorder=2)
                    _draw_stage2_markers(ax, s2_pts, COND_COLORS[cond["label"]])
            _apply_ax_style(ax, "", fmt)

        axes[1][0].set_ylabel(f"Selection score ({unit})", fontsize=8)
        fig.text(0.50, 0.465, "Always-accept (δ=-∞)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="0.2")

    # ── Legend: condition lines + eval-selection markers ─────────────────
    line_handles = [
        mlines.Line2D([0], [0], color=COND_COLORS[c["label"]],
                      linewidth=1.6, label=c["label"])
        for c in conditions
    ]
    marker_handles = [
        mlines.Line2D([0], [0], marker="*", color="w", markersize=9,
                      markerfacecolor=GOLD, markeredgecolor="black",
                      markeredgewidth=1.2,
                      label=f"Final incumbent / {T_STAR} (eval prompt)"),
        mlines.Line2D([0], [0], marker="D", color="w", markersize=6,
                      markerfacecolor=GOLD, markeredgecolor="#888888",
                      markeredgewidth=1.0,
                      label=f"{AA_LABEL}, {FINAL_INC} (eval prompt)"),
    ]
    # fig.legend(handles=line_handles + marker_handles,
    #            loc="lower center", bbox_to_anchor=(0.50, 0.005),
    #            ncol=len(conditions) + 2, fontsize=8.5,
    #            framealpha=0.9, edgecolor="0.8",
    #            handlelength=2.0, columnspacing=1.4)
    all_handles = line_handles + (marker_handles[:1] if gated_only else marker_handles)
    n_items = len(all_handles)
    legend_y = -0.18 if gated_only else -0.06

    fig.legend(handles=all_handles,
           loc="lower center", bbox_to_anchor=(0.50, legend_y),
           ncol=(n_items + 1) // 2, fontsize=8.5,
           framealpha=0.9, edgecolor="0.8",
           handlelength=2.0, columnspacing=1.2)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-strat", choices=["valbag", "trainsig"], default=VAL_STRAT)
    parser.add_argument("--thresh", choices=["000", "002", "010"], default=None,
                        help="Threshold sweep key; omit for default δ=0.05 run")
    args = parser.parse_args()

    if args.thresh:
        conditions  = _make_thresh_conditions(args.val_strat, args.thresh)
        gated_only  = True
        fname       = f"opt_trajectory_{args.val_strat}_thresh{args.thresh}"
        sp          = SP_LABELS[args.val_strat]
        delta_val   = {"000": "0.00", "002": "0.02", "010": "0.10"}[args.thresh]
        row1_title  = f"Gated, {sp} (δ={delta_val})"
    else:
        conditions  = _make_conditions(args.val_strat)
        gated_only  = False
        fname       = f"opt_trajectory_{args.val_strat}"
        row1_title  = "Gated (δ=0.05)"

    sp = {"valbag": "HSP", "trainsig": "LSP"}[args.val_strat]
    thresh_label = (f", δ=0.{'0' + args.thresh[1:] if args.thresh else '05'}"
                    if args.thresh else "")
    print(f"Val strategy: {args.val_strat}{thresh_label}\nData availability:")
    for cond in conditions:
        print(f"\n  {cond['label']}")
        for task in TASK_ORDER:
            s3 = load_stage3_trajectory(cond["gated_dir"], task)
            s3_str = f"{len(s3)} pts, {len(s3)-1} accepted" if s3 else "MISSING"
            s2_str = "n/a"
            if cond["aa_dir"]:
                s2 = load_stage2_trajectory(cond["aa_dir"], task)
                s2_str = f"{len(s2)} cycles" if s2 else "MISSING (pending)"
            print(f"    {task:<22} gated={s3_str:<30} always_accept={s2_str}")

    fig = plot_trajectories(conditions, TASK_ORDER, gated_only=gated_only,
                            row1_title=row1_title)
    save(fig, fname)
    plt.show()
