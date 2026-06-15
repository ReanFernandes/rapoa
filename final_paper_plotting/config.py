"""
Shared configuration for all paper figures.

To rename the agent before submission: change AGENT_NAME here.
Everything else — labels, colours, legend entries — updates automatically.
"""

# ── Agent name ────────────────────────────────────────────────────────────────
# Change this once to rename the agent across every plot.
AGENT_NAME = "SPA"

# ── Task display names ────────────────────────────────────────────────────────
TASK_LABELS = {
    "goto":               "GoTo",
    "pickup":             "PickUp",
    "open":               "Open",
    "putnext":            "PutNext",
    "pick_up_seq_go_to":  "PickUp→GoTo",
}
TASKS = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]

# ── Condition labels ──────────────────────────────────────────────────────────
# Derived from AGENT_NAME so renaming propagates automatically.
def _cond_labels(agent: str) -> dict:
    return {
        # Fresh eval / non-optimised
        ("balrog_baseline", "minimal", "history_1step"):   "BALROG/min/1s",
        ("balrog_baseline", "minimal", "history_16step"):  "BALROG/min/16s",
        ("balrog_baseline", "rich",    "history_1step"):   "BALROG/rich/1s",
        ("balrog_baseline", "rich",    "history_16step"):  "BALROG/rich/16s",
        ("with_descriptor", "minimal", "single_turn"):     f"{agent}/minimal",
        ("with_descriptor", "rich",    "single_turn"):     f"{agent}/rich",
        # Optimised conditions (for headliner and optimisation plots)
        "balrog_opt_minimal":    "BALROG/min (opt)",
        "ours_opt_minimal":      f"{agent}/min (opt)",
        "ours_opt_rich":         f"{agent}/rich (opt)",
    }

CONDITION_LABELS = _cond_labels(AGENT_NAME)

# Short labels for axes / heatmap cells
CONDITION_SHORT = {
    "BALROG/min/1s":            "B-min-1s",
    "BALROG/min/16s":           "B-min-16s",
    "BALROG/rich/1s":           "B-rich-1s",
    "BALROG/rich/16s":          "B-rich-16s",
    f"{AGENT_NAME}/minimal":    f"{AGENT_NAME}/min",
    f"{AGENT_NAME}/rich":       f"{AGENT_NAME}/rich",
    "BALROG/min (opt)":         "B-min (opt)",
    f"{AGENT_NAME}/min (opt)":  f"{AGENT_NAME}/min (opt)",
    f"{AGENT_NAME}/rich (opt)": f"{AGENT_NAME}/rich (opt)",
}

# Display order for conditions in plots
CONDITION_ORDER = [
    "BALROG/min/1s",
    "BALROG/min/16s",
    "BALROG/rich/1s",
    "BALROG/rich/16s",
    f"{AGENT_NAME}/minimal",
    f"{AGENT_NAME}/rich",
]

# ── Paper terminology (2026-05-06) ───────────────────────────────────────────
# Rename these here; all plotting scripts import and use these constants.
VARIANT_LABELS = {"minimal": "plain", "rich": "guided"}   # prompt variant display names
BALROG_OPT_LABEL  = f"BALROG\n{VARIANT_LABELS['minimal']}\n16-step"  # the BALROG variant we optimise
OURS_OPT_LABEL_MIN  = f"{AGENT_NAME}\n{VARIANT_LABELS['minimal']}"
OURS_OPT_LABEL_RICH = f"{AGENT_NAME}\n{VARIANT_LABELS['rich']}"
DELTA      = "δ"          # threshold symbol
AA_LABEL   = "δ=-∞"       # always-accept condition
T_STAR     = "T*"          # best-T incumbent
FINAL_INC  = "final"       # end-of-run incumbent
SP_LABELS  = {"valbag": "HSP", "trainsig": "LSP"}         # selection pressure abbreviations

CONDITION_ORDER_OPT = [
    "BALROG/min/1s",
    "BALROG/rich/16s",       # strongest BALROG baseline
    f"{AGENT_NAME}/minimal",
    f"{AGENT_NAME}/rich",
    "BALROG/min (opt)",
    f"{AGENT_NAME}/min (opt)",
    f"{AGENT_NAME}/rich (opt)",
]

# ── Heatmap display label overrides ──────────────────────────────────────────
# Change any column header or group bracket in plot_heatmap.py from here.
# Keys must match the strings in the COLS tuples exactly (including \n).
# Omit a key to keep the default.  Supports \n for two-line labels.

COL_LABEL_OVERRIDES: dict[str, str] = {
    # Column headers (label field in COLS):
    # "min\n1-step":   "min\n1-step",
    # "min\n16-step":  "min\n16-step",
    # "rich\n1-step":  "rich\n1-step",
    # "rich\n16-step": "rich\n16-step",
    # "min":           "min",
    # "rich":          "rich",
    # "BALROG":        "BALROG",
}

GROUP_LABEL_OVERRIDES: dict[str, str] = {
    # Group bracket headers (group field in COLS):
    # "BALROG":                    "BALROG",
    # "Ours":                      "Ours",          # matches AGENT_NAME
    # "Gated (V-bag)":             "Gated (V-bag)",
    # "Gated (T-sig)":             "Gated (T-sig)",
    # "Always-accept\nbest-T":     "Always-accept\nbest-T",
    # "Always-accept\nend-of-run": "Always-accept\nend-of-run",
    # "V-bag t=0.02":              "V-bag t=0.02",
    # "V-bag t=0.10":              "V-bag t=0.10",
    # "T-sig t=0.02":              "T-sig t=0.02",
    # "T-sig t=0.10":              "T-sig t=0.10",
    # "Ablation\nAgent-only":      "Ablation\nAgent-only",
    # "Ablation\nDescriptor-only": "Ablation\nDescriptor-only",
}

# ── Colours ───────────────────────────────────────────────────────────────────
# Okabe-Ito palette — distinguishable under deuteranopia, protanopia, tritanopia.
# Reference: Okabe & Ito (2008) "Color Universal Design"
COLOURS = {
    "BALROG/min/1s":            "#56B4E9",   # Okabe sky blue (light)
    "BALROG/min/16s":           "#0072B2",   # Okabe blue
    "BALROG/rich/1s":           "#56B4E9",   # Okabe sky blue
    "BALROG/rich/16s":          "#009E73",   # Okabe bluish green
    f"{AGENT_NAME}/minimal":    "#E69F00",   # Okabe orange
    f"{AGENT_NAME}/rich":       "#D55E00",   # Okabe vermillion
    "BALROG/min (opt)":         "#0072B2",   # Okabe blue
    f"{AGENT_NAME}/min (opt)":  "#E69F00",   # Okabe orange
    f"{AGENT_NAME}/rich (opt)": "#D55E00",   # Okabe vermillion
}

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR  = Path(__file__).resolve().parent / "figures"

# Non-optimised baseline eval (unchanged structure)
FRESH_EVAL_DIR = PROJECT_ROOT / "logs_fresh_eval"
ORIG_EVAL_DIR  = PROJECT_ROOT / "logs"

# New-system roots — all campaigns live under {env}/{model}/
ENV_NAME   = "babyai"
MODEL_NAME = "gpt-oss-20b"

OPT_RUNS_ROOT  = PROJECT_ROOT / "optimization_runs"         / ENV_NAME / MODEL_NAME
EVAL_RUNS_ROOT = PROJECT_ROOT / "logs_fresh_eval_optimised" / ENV_NAME / MODEL_NAME

# Backward-compat alias
OPT_RUNS_DIR = OPT_RUNS_ROOT

# ── Campaign IDs ──────────────────────────────────────────────────────────────
# Hardcoded campaign IDs — use instead of latest_campaign() for reproducibility.
# Campaigns are split by config: primary_20 has rich conditions + ablations;
# primary_minimal has minimal conditions; balrog_all has all BALROG variants.
CAMPAIGN_IDS: dict[str, str] = {
    "primary_20":       "primary_20_20260522_110326",
    "primary_minimal":  "primary_20_minimal_20260522_115028",
    "balrog_all":       "balrog_all_20260522_115304",
    "thresh_sweep":     "thresh_sweep_20260603_145707",
    "thresh_sweep_h16": "thresh_sweep_h16_20260605_124627",
    "random_module":    "random_module_20260605_150125",
    "balrog_h1":        "balrog_h1_20260608_120040",
}

# ── Task name mapping ──────────────────────────────────────────────────────────
# Eval paths use short names (from gym_id suffix); opt paths use full task names.
TASK_OPT_NAMES = {
    "goto":              "mixed_train_goto",
    "pickup":            "mixed_train_pickup",
    "open":              "mixed_train_open",
    "putnext":           "mixed_train_putnext",
    "pick_up_seq_go_to": "mixed_train_pick_up_seq_go_to",
}

# ── Standard slugs ─────────────────────────────────────────────────────────────
# Deterministic names generated by run_pipeline._slug() for each experiment type.
SLUGS: dict[str, str] = {
    "hsp_rich":                    "spa_mean_valbag_t005_rich",
    "hsp_minimal":                 "spa_mean_valbag_t005_minimal",
    "lsp_rich":                    "spa_mean_trainsig_t005_rich",
    "lsp_minimal":                 "spa_mean_trainsig_t005_minimal",
    "always_accept_rich":          "spa_always_accept_rich",
    "always_accept_minimal":       "spa_always_accept_minimal",
    "actor_ablation_rich":         "module_ablation_actor_rich",
    "actor_ablation_minimal":      "module_ablation_actor_minimal",
    "descriptor_ablation_rich":    "module_ablation_descriptor_rich",
    "descriptor_ablation_minimal": "module_ablation_descriptor_minimal",
    "balrog_rich":                 "balrog_h16_rich",
    "balrog_rich_hsp":             "balrog_h16_valbag_t005_rich",
    "balrog_rich_lsp":             "balrog_h16_trainsig_t005_rich",
    "balrog_h1_minimal":           "balrog_h1_minimal",
    "balrog_h1_rich":              "balrog_h1_rich",
    "balrog_h1_rich_hsp":          "balrog_h1_valbag_t005_rich",
    "balrog_h1_rich_lsp":          "balrog_h1_trainsig_t005_rich",
    "balrog_h1_minimal_hsp":       "balrog_h1_valbag_t005_minimal",
    "balrog_h1_minimal_lsp":       "balrog_h1_trainsig_t005_minimal",
    "hsp_rich_h16":                "spa_mean_valbag_t005_rich_h16",
    "lsp_rich_h16":                "spa_mean_trainsig_t005_rich_h16",
    "always_accept_rich_h16":      "spa_always_accept_rich_h16",
    "hsp_minimal_h16":             "spa_mean_valbag_t005_minimal_h16",
    "lsp_minimal_h16":             "spa_mean_trainsig_t005_minimal_h16",
    "balrog_minimal":              "balrog_h16_minimal",
    "balrog_minimal_hsp":          "balrog_h16_valbag_t005_minimal",
    "balrog_minimal_lsp":          "balrog_h16_trainsig_t005_minimal",
    "thresh000_hsp":               "spa_mean_valbag_t000_rich",
    "thresh002_hsp":               "spa_mean_valbag_t002_rich",
    "thresh010_hsp":               "spa_mean_valbag_t010_rich",
    "thresh000_lsp":               "spa_mean_trainsig_t000_rich",
    "thresh002_lsp":               "spa_mean_trainsig_t002_rich",
    "thresh010_lsp":               "spa_mean_trainsig_t010_rich",
    # h16 threshold sweep (rich only)
    "thresh000_hsp_h16":           "spa_mean_valbag_t000_rich_h16",
    "thresh002_hsp_h16":           "spa_mean_valbag_t002_rich_h16",
    "thresh010_hsp_h16":           "spa_mean_valbag_t010_rich_h16",
    "thresh000_lsp_h16":           "spa_mean_trainsig_t000_rich_h16",
    "thresh002_lsp_h16":           "spa_mean_trainsig_t002_rich_h16",
    "thresh010_lsp_h16":           "spa_mean_trainsig_t010_rich_h16",
    # random module ablation
    "random_hsp_rich":             "spa_random_valbag_t005_rich",
    "random_hsp_minimal":          "spa_random_valbag_t005_minimal",
    "random_lsp_rich":             "spa_random_trainsig_t000_rich",
    "random_lsp_minimal":          "spa_random_trainsig_t000_minimal",
}

# ── Path helpers ───────────────────────────────────────────────────────────────

def opt_dir(campaign: str, slug: str, task: str | None = None) -> Path:
    """Path to a run's optimisation directory.

    task: short eval name ('goto') or full opt name ('mixed_train_goto') — both accepted.
    """
    p = OPT_RUNS_ROOT / campaign / slug
    if task:
        p = p / TASK_OPT_NAMES.get(task, task)
    return p


def eval_dir(campaign: str, slug: str) -> Path:
    """Root of a slug's fresh-eval directory (pass to rglob for run_summary.json)."""
    return EVAL_RUNS_ROOT / campaign / slug


def list_campaigns() -> list[str]:
    """All available campaigns under OPT_RUNS_ROOT, sorted by name."""
    if not OPT_RUNS_ROOT.exists():
        return []
    return sorted(d.name for d in OPT_RUNS_ROOT.iterdir() if d.is_dir())


def latest_campaign(slug_required: str | None = None) -> str | None:
    """Most recently modified campaign, optionally requiring it to contain slug_required."""
    if not OPT_RUNS_ROOT.exists():
        return None
    candidates = [d for d in OPT_RUNS_ROOT.iterdir() if d.is_dir()]
    if slug_required:
        candidates = [d for d in candidates if (d / slug_required).exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime).name

# ── NeurIPS matplotlib style ──────────────────────────────────────────────────
import matplotlib as mpl

def apply_neurips_style():
    """Call once at the top of each plotting script."""
    mpl.rcParams.update({
        # Font
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.title_fontsize": 8,
        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        # Axes
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "axes.grid":          False,
        # Lines
        "lines.linewidth":    1.5,
        # Ticks
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        # Legend
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.8",
    })

# NeurIPS figure widths (inches)
FIG_WIDTH_FULL   = 6.75   # full text width
FIG_WIDTH_HALF   = 3.25   # single column
FIG_WIDTH_THIRD  = 2.125  # one third

# ── Save helper ───────────────────────────────────────────────────────────────
def save(fig, name: str) -> None:
    """Save figure as both PNG and PDF into figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")
