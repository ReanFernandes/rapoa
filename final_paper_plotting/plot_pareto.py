"""
Pareto efficiency scatter — success rate vs mean tokens per episode.

Each system has a progenitor (★, non-optimised) connected by arrows to its
optimised descendants. Shape encodes the acceptance criterion; colour encodes
the system. Arrows go progenitor → condition (no sequential chain implied).

Systems (4 chains × 3 SPA = 7 total):
  BALROG plain/16-step  ★ → {h16 AA}
  BALROG plain/1-step   ★ → {h1 AA, h1 HSP, h1 LSP}
  BALROG guided/16-step ★ → {h16 AA, h16 HSP, h16 LSP}
  BALROG guided/1-step  ★ → {h1 AA, h1 HSP, h1 LSP}
  SPA plain             ★ → {AA, HSP, LSP}
  SPA guided            ★ → {AA, HSP, LSP}
  SPA guided h16        ★ → {AA, HSP, LSP}

Campaigns:
  h16 plain opt  : primary_20    (balrog_h16_minimal)
  h16 guided opt : balrog_all    (balrog_h16_rich/valbag/trainsig)
  h1 all opt     : balrog_h1     (balrog_h1_{minimal,rich}/valbag/trainsig)
  SPA            : primary_20 + primary_minimal

Markers:
  ★  progenitor (non-optimised)
  ●  always-accept, end-of-run incumbent
  ▲  gated, HSP (validation bag, δ=0.05)
  ▼  gated, LSP (train signal, δ=0.05)
  X/P/s  threshold sweep δ=0.00/0.02/0.10 HSP  — appendix only
  D/p/h  threshold sweep δ=0.00/0.02/0.10 LSP  — appendix only
  </>    module ablation (actor-only, descriptor-only)  — appendix only
  +/d    random-module ablation (HSP/LSP)  — appendix only

Usage:
    cd final_paper_plotting

    # Main paper figure — all 6 chains, full BALROG h1+h16 conditions
    #   Output: figures/efficiency/pareto_efficiency_main.pdf
    python plot_pareto.py

    # Appendix figure — main conditions dimmed + threshold sweep + ablations on top
    #   Adds: δ=0.00/0.02/0.10 × HSP/LSP, actor/descriptor/random-module ablations
    #   Output: figures/efficiency/pareto_efficiency_appendix.pdf
    python plot_pareto.py --version appendix

Note: Threshold-sweep plain conditions in the appendix version are absent
  (thresh_sweep ran rich-only). Missing eval dirs render as absent points with
  a "SKIP ... — no data" console message.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from config import (
    AGENT_NAME,
    TASKS,
    FRESH_EVAL_DIR, CAMPAIGN_IDS,
    FIG_WIDTH_FULL,
    VARIANT_LABELS, AA_LABEL, T_STAR, FINAL_INC, SP_LABELS, DELTA,
    SLUGS, eval_dir as _eval_dir,
    apply_neurips_style,
)

_V = VARIANT_LABELS

apply_neurips_style()

CP_R   = CAMPAIGN_IDS["primary_20"]       # primary: rich conditions
CP_M   = CAMPAIGN_IDS["primary_minimal"]  # primary: minimal conditions
CP_B   = CAMPAIGN_IDS["balrog_all"]       # balrog h16 rich opt conditions
CP_H1  = CAMPAIGN_IDS["balrog_h1"]        # balrog h1 re-run (correctly at h1)
CP_H16 = CAMPAIGN_IDS["thresh_sweep_h16"] # spa h16 conditions
CP_RND = CAMPAIGN_IDS["random_module"]    # random module ablation
CAMPAIGN_THRESH   = CAMPAIGN_IDS["thresh_sweep"]
CAMPAIGN_ABLATION = CP_R

def _e(campaign, sk): return _eval_dir(campaign, SLUGS[sk]) if campaign else None

PARETO_FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "efficiency"


def _save(fig, name):
    PARETO_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = PARETO_FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")


# ── Toggle ────────────────────────────────────────────────────────────────────
SHOW_SPREAD = False   # set True to show per-seed 95% CI bars

# ── Condition keys ────────────────────────────────────────────────────────────
_CK_B_MIN_1  = ("balrog_baseline", "minimal", "history_1step")
_CK_B_MIN_16 = ("balrog_baseline", "minimal", "history_16step")
_CK_B_R_1    = ("balrog_baseline", "rich",    "history_1step")
_CK_B_R_16   = ("balrog_baseline", "rich",    "history_16step")
_CK_O_MIN      = ("with_descriptor", "minimal", "single_turn")
_CK_O_RICH     = ("with_descriptor", "rich",    "single_turn")
_CK_O_RICH_H16 = ("with_descriptor", "rich",    "agent_multi_turn")

# ── Base colours (Okabe-Ito) ──────────────────────────────────────────────────
_BASE_B_MIN      = "#0072B2"   # Okabe blue            — BALROG plain chain
_BASE_B_RICH     = "#009E73"   # Okabe green           — BALROG guided chain
_BASE_O_MIN      = "#E69F00"   # Okabe orange          — Ours plain chain
_BASE_O_RICH     = "#D55E00"   # Okabe vermillion      — Ours guided chain
_BASE_O_RICH_H16 = "#CC79A7"   # Okabe reddish purple  — Ours guided h16 chain
_BASE_ALONE      = "#56B4E9"   # Okabe sky blue        — standalone progenitors (no descendants)

# ── Lightness levels: progenitor is lighter, all optimised use full colour ────
_LIGHTNESS = {
    "prog":     0.67,
    "s2-end":   1.00,
    "s2-bt":    1.00,
    "gated-vb": 1.00,
    "gated-ts": 1.00,
    "thresh":   0.80,   # threshold sweep extras (slightly lighter)
    "ablation": 0.70,   # module ablation extras
}


def _shade(base_hex: str, lightness: float) -> tuple:
    """Blend base colour with white at the given lightness (0=white, 1=full)."""
    r, g, b = mcolors.to_rgb(base_hex)
    return (1 - lightness*(1-r), 1 - lightness*(1-g), 1 - lightness*(1-b))


# ── System definitions ────────────────────────────────────────────────────────
# Each system: one progenitor (★) + N conditions each arrow'd from the progenitor.
# conditions: list of (label, cond_key, eval_dir, marker, lightness_key)

_pln = _V["minimal"]
_gdd = _V["rich"]
_HSP = f"HSP, {DELTA}=0.05"
_LSP = f"LSP, {DELTA}=0.05"
_AA_BT  = f"{AA_LABEL}, {T_STAR}"
_AA_FIN = f"{AA_LABEL}, {FINAL_INC}"

SYSTEMS = [
    # BALROG plain / 16-step — non-opt progenitor + h16-optimised conditions.
    dict(
        name       = f"BALROG {_pln}",
        base_color = _BASE_B_MIN,
        progenitor = (f"BALROG {_pln}/16-step", _CK_B_MIN_16, FRESH_EVAL_DIR),
        conditions = [
            (f"BALROG {_pln} h16 {_AA_FIN}", _CK_B_MIN_16, _e(CP_R, "balrog_minimal"),     "o", "s2-end"),
            (f"BALROG {_pln} h16 {_HSP}",    _CK_B_MIN_16, _e(CP_B, "balrog_minimal_hsp"), "^", "gated-vb"),
            (f"BALROG {_pln} h16 {_LSP}",    _CK_B_MIN_16, _e(CP_B, "balrog_minimal_lsp"), "v", "gated-ts"),
        ],
    ),
    # BALROG plain / 1-step — own non-opt progenitor + h1-optimised conditions.
    dict(
        name       = f"BALROG {_pln}",
        base_color = _BASE_B_MIN,
        progenitor = (f"BALROG {_pln}/1-step", _CK_B_MIN_1, FRESH_EVAL_DIR),
        conditions = [
            (f"BALROG {_pln} h1 {_AA_FIN}", _CK_B_MIN_1, _e(CP_H1, "balrog_h1_minimal"),     "o", "s2-end"),
            (f"BALROG {_pln} h1 {_HSP}",    _CK_B_MIN_1, _e(CP_H1, "balrog_h1_minimal_hsp"), "^", "gated-vb"),
            (f"BALROG {_pln} h1 {_LSP}",    _CK_B_MIN_1, _e(CP_H1, "balrog_h1_minimal_lsp"), "v", "gated-ts"),
        ],
    ),
    # BALROG guided / 16-step — own non-opt progenitor + h16-optimised conditions.
    dict(
        name       = f"BALROG {_gdd}",
        base_color = _BASE_B_RICH,
        progenitor = (f"BALROG {_gdd}/16-step", _CK_B_R_16, FRESH_EVAL_DIR),
        conditions = [
            (f"BALROG {_gdd} h16 {_AA_FIN}", _CK_B_R_16, _e(CP_B, "balrog_rich"),     "o", "s2-end"),
            (f"BALROG {_gdd} h16 {_HSP}",    _CK_B_R_16, _e(CP_B, "balrog_rich_hsp"), "^", "gated-vb"),
            (f"BALROG {_gdd} h16 {_LSP}",    _CK_B_R_16, _e(CP_B, "balrog_rich_lsp"), "v", "gated-ts"),
        ],
    ),
    # BALROG guided / 1-step — own non-opt progenitor + h1-optimised conditions.
    dict(
        name       = f"BALROG {_gdd}",
        base_color = _BASE_B_RICH,
        progenitor = (f"BALROG {_gdd}/1-step", _CK_B_R_1, FRESH_EVAL_DIR),
        conditions = [
            (f"BALROG {_gdd} h1 {_AA_FIN}", _CK_B_R_1, _e(CP_H1, "balrog_h1_rich"),     "o", "s2-end"),
            (f"BALROG {_gdd} h1 {_HSP}",    _CK_B_R_1, _e(CP_H1, "balrog_h1_rich_hsp"), "^", "gated-vb"),
            (f"BALROG {_gdd} h1 {_LSP}",    _CK_B_R_1, _e(CP_H1, "balrog_h1_rich_lsp"), "v", "gated-ts"),
        ],
    ),
    dict(
        name       = f"{AGENT_NAME} {_pln}",
        base_color = _BASE_O_MIN,
        progenitor = (f"{AGENT_NAME} {_pln}", _CK_O_MIN, FRESH_EVAL_DIR),
        conditions = [
            (f"{AGENT_NAME} {_pln} {_AA_FIN}", _CK_O_MIN, _e(CP_M, "always_accept_minimal"), "o", "s2-end"),
            (f"{AGENT_NAME} {_pln} {_HSP}",    _CK_O_MIN, _e(CP_M, "hsp_minimal"),           "^", "gated-vb"),
            (f"{AGENT_NAME} {_pln} {_LSP}",    _CK_O_MIN, _e(CP_M, "lsp_minimal"),           "v", "gated-ts"),
        ],
    ),
    dict(
        name       = f"{AGENT_NAME} {_gdd}",
        base_color = _BASE_O_RICH,
        progenitor = (f"{AGENT_NAME} {_gdd}", _CK_O_RICH, FRESH_EVAL_DIR),
        conditions = [
            (f"{AGENT_NAME} {_gdd} {_AA_FIN}", _CK_O_RICH, _e(CP_R, "always_accept_rich"), "o", "s2-end"),
            (f"{AGENT_NAME} {_gdd} {_HSP}",    _CK_O_RICH, _e(CP_R, "hsp_rich"),           "^", "gated-vb"),
            (f"{AGENT_NAME} {_gdd} {_LSP}",    _CK_O_RICH, _e(CP_R, "lsp_rich"),           "v", "gated-ts"),
        ],
    ),
    dict(
        name       = f"{AGENT_NAME} {_gdd} h16",
        base_color = _BASE_O_RICH_H16,
        progenitor = (f"{AGENT_NAME} {_gdd}/h16", _CK_O_RICH_H16, FRESH_EVAL_DIR),
        conditions = [
            (f"{AGENT_NAME} {_gdd} h16 {_AA_FIN}", _CK_O_RICH_H16, _e(CP_H16, "always_accept_rich_h16"), "o", "s2-end"),
            (f"{AGENT_NAME} {_gdd} h16 {_HSP}",    _CK_O_RICH_H16, _e(CP_H16, "hsp_rich_h16"),           "^", "gated-vb"),
            (f"{AGENT_NAME} {_gdd} h16 {_LSP}",    _CK_O_RICH_H16, _e(CP_H16, "lsp_rich_h16"),           "v", "gated-ts"),
        ],
    ),
]

# Standalone progenitors — no optimised descendants
# Note: BALROG plain/1-step and guided/1-step are now progenitors in SYSTEMS above.
STANDALONE = []

# ── Appendix-only extra conditions ───────────────────────────────────────────
# Threshold sweep + module ablation — not shown in main paper pareto plot.
# Drawn on top of dimmed main conditions in the appendix version.
SYSTEMS_EXTRA = [
    dict(
        name       = f"{AGENT_NAME} {_pln} (extra)",
        base_color = _BASE_O_MIN,
        progenitor = None,
        conditions = [
            (f"{AGENT_NAME} {_pln} HSP, {DELTA}=0.00", _CK_O_MIN, _e(CAMPAIGN_THRESH,   "thresh000_hsp"),             "X", "thresh"),
            (f"{AGENT_NAME} {_pln} HSP, {DELTA}=0.02", _CK_O_MIN, _e(CAMPAIGN_THRESH,   "thresh002_hsp"),             "P", "thresh"),
            (f"{AGENT_NAME} {_pln} HSP, {DELTA}=0.10", _CK_O_MIN, _e(CAMPAIGN_THRESH,   "thresh010_hsp"),             "s", "thresh"),
            (f"{AGENT_NAME} {_pln} LSP, {DELTA}=0.00", _CK_O_MIN, _e(CAMPAIGN_THRESH,   "thresh000_lsp"),             "D", "thresh"),
            (f"{AGENT_NAME} {_pln} LSP, {DELTA}=0.02", _CK_O_MIN, _e(CAMPAIGN_THRESH,   "thresh002_lsp"),             "p", "thresh"),
            (f"{AGENT_NAME} {_pln} LSP, {DELTA}=0.10", _CK_O_MIN, _e(CAMPAIGN_THRESH,   "thresh010_lsp"),             "h", "thresh"),
            (f"{AGENT_NAME} {_pln} actor-only",          _CK_O_MIN, _e(CP_M,              "actor_ablation_minimal"),      "<", "ablation"),
            (f"{AGENT_NAME} {_pln} descriptor-only",     _CK_O_MIN, _e(CP_M,              "descriptor_ablation_minimal"), ">", "ablation"),
            (f"{AGENT_NAME} {_pln} random-module HSP",   _CK_O_MIN, _e(CP_RND,            "random_hsp_minimal"),          "+", "ablation"),
            (f"{AGENT_NAME} {_pln} random-module LSP",   _CK_O_MIN, _e(CP_RND,            "random_lsp_minimal"),          "d", "ablation"),
        ],
    ),
    dict(
        name       = f"{AGENT_NAME} {_gdd} (extra)",
        base_color = _BASE_O_RICH,
        progenitor = None,
        conditions = [
            (f"{AGENT_NAME} {_gdd} HSP, {DELTA}=0.00", _CK_O_RICH, _e(CAMPAIGN_THRESH,   "thresh000_hsp"),             "X", "thresh"),
            (f"{AGENT_NAME} {_gdd} HSP, {DELTA}=0.02", _CK_O_RICH, _e(CAMPAIGN_THRESH,   "thresh002_hsp"),             "P", "thresh"),
            (f"{AGENT_NAME} {_gdd} HSP, {DELTA}=0.10", _CK_O_RICH, _e(CAMPAIGN_THRESH,   "thresh010_hsp"),             "s", "thresh"),
            (f"{AGENT_NAME} {_gdd} LSP, {DELTA}=0.00", _CK_O_RICH, _e(CAMPAIGN_THRESH,   "thresh000_lsp"),             "D", "thresh"),
            (f"{AGENT_NAME} {_gdd} LSP, {DELTA}=0.02", _CK_O_RICH, _e(CAMPAIGN_THRESH,   "thresh002_lsp"),             "p", "thresh"),
            (f"{AGENT_NAME} {_gdd} LSP, {DELTA}=0.10", _CK_O_RICH, _e(CAMPAIGN_THRESH,   "thresh010_lsp"),             "h", "thresh"),
            (f"{AGENT_NAME} {_gdd} actor-only",          _CK_O_RICH, _e(CAMPAIGN_ABLATION, "actor_ablation_rich"),      "<", "ablation"),
            (f"{AGENT_NAME} {_gdd} descriptor-only",     _CK_O_RICH, _e(CAMPAIGN_ABLATION, "descriptor_ablation_rich"), ">", "ablation"),
            (f"{AGENT_NAME} {_gdd} random-module HSP",   _CK_O_RICH, _e(CP_RND,            "random_hsp_rich"),           "+", "ablation"),
            (f"{AGENT_NAME} {_gdd} random-module LSP",   _CK_O_RICH, _e(CP_RND,            "random_lsp_rich"),           "d", "ablation"),
        ],
    ),
    dict(
        name       = f"{AGENT_NAME} {_gdd} h16 (extra)",
        base_color = _BASE_O_RICH_H16,
        progenitor = None,
        conditions = [
            (f"{AGENT_NAME} {_gdd} h16 HSP, {DELTA}=0.00", _CK_O_RICH_H16, _e(CP_H16, "thresh000_hsp_h16"), "X", "thresh"),
            (f"{AGENT_NAME} {_gdd} h16 HSP, {DELTA}=0.02", _CK_O_RICH_H16, _e(CP_H16, "thresh002_hsp_h16"), "P", "thresh"),
            (f"{AGENT_NAME} {_gdd} h16 HSP, {DELTA}=0.10", _CK_O_RICH_H16, _e(CP_H16, "thresh010_hsp_h16"), "s", "thresh"),
            (f"{AGENT_NAME} {_gdd} h16 LSP, {DELTA}=0.00", _CK_O_RICH_H16, _e(CP_H16, "thresh000_lsp_h16"), "D", "thresh"),
            (f"{AGENT_NAME} {_gdd} h16 LSP, {DELTA}=0.02", _CK_O_RICH_H16, _e(CP_H16, "thresh002_lsp_h16"), "p", "thresh"),
            (f"{AGENT_NAME} {_gdd} h16 LSP, {DELTA}=0.10", _CK_O_RICH_H16, _e(CP_H16, "thresh010_lsp_h16"), "h", "thresh"),
        ],
    ),
]

# Inline text labels for progenitor points only
LABEL_OFFSETS = {
    f"BALROG {_pln}/16-step":     (  -56,   0, "left",  "bottom"),
    f"BALROG {_pln}/1-step":      (    1,  12, "right", "bottom"),
    f"BALROG {_gdd}/1-step":      (   -6,   0, "right", "top"),
    f"BALROG {_gdd}/16-step":     (  -56,  12, "left",  "top"),
    f"{AGENT_NAME} {_pln}":       (   44, -10, "right", "bottom"),
    f"{AGENT_NAME} {_gdd}":       (    6, -10, "left",  "bottom"),
    f"{AGENT_NAME} {_gdd}/h16":   (    6,   8, "left",  "bottom"),
}
LABEL_SHORT = {
    f"BALROG {_pln}/16-step":     f"B-{_pln} 16 step",
    f"BALROG {_pln}/1-step":      f"B-{_pln} 1 step",
    f"BALROG {_gdd}/1-step":      f"B-{_gdd} 1 step",
    f"BALROG {_gdd}/16-step":     f"B-{_gdd} 16 step",
    f"{AGENT_NAME} {_pln}":       f"{AGENT_NAME} {_pln}",
    f"{AGENT_NAME} {_gdd}":       f"{AGENT_NAME} {_gdd}",
    f"{AGENT_NAME} {_gdd}/h16":   f"{AGENT_NAME} {_gdd}/h16",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _episode_tokens(ep: dict, pipeline: str) -> int:
    agent = ep.get("agent_prompt_tokens", 0) + ep.get("agent_completion_tokens", 0)
    desc  = (ep.get("descriptor_prompt_tokens", 0) + ep.get("descriptor_completion_tokens", 0)
             if pipeline == "with_descriptor" else 0)
    return agent + desc


_cache:      dict[Path, dict] = {}
_seed_cache: dict[Path, dict] = {}


def _load(eval_dir: Path) -> dict:
    if eval_dir not in _cache:
        data = defaultdict(lambda: defaultdict(list))
        for s in sorted(eval_dir.rglob("run_summary.json")):
            parts = s.parts
            try:
                anchor = next(i for i, p in enumerate(parts) if p == eval_dir.name)
                _, task, _, pipeline, variant, conv_mode, *_ = parts[anchor+1:]
            except (StopIteration, ValueError):
                continue
            if task not in TASKS:
                continue
            d = json.load(open(s))
            data[(pipeline, variant, conv_mode)][task].extend(d["episodes"])
        _cache[eval_dir] = data
    return _cache[eval_dir]


def _load_by_seed(eval_dir: Path) -> dict:
    if eval_dir not in _seed_cache:
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for s in sorted(eval_dir.rglob("run_summary.json")):
            parts = s.parts
            try:
                anchor = next(i for i, p in enumerate(parts) if p == eval_dir.name)
                _, task, _, pipeline, variant, conv_mode, _, iseed, _ = parts[anchor+1:anchor+10]
            except (StopIteration, ValueError):
                continue
            if task not in TASKS:
                continue
            d = json.load(open(s))
            data[(pipeline, variant, conv_mode)][task][iseed].extend(d["episodes"])
        _seed_cache[eval_dir] = data
    return _seed_cache[eval_dir]


def get_point(cond_key: tuple, eval_dir: Path) -> tuple | None:
    if not eval_dir.exists():
        return None
    data     = _load(eval_dir)
    pipeline = cond_key[0]
    task_srs, task_toks = [], []
    for task in TASKS:
        eps = data.get(cond_key, {}).get(task, [])
        if not eps:
            continue
        task_srs.append(sum(1 for e in eps if e["success"]) / len(eps))
        task_toks.append(sum(_episode_tokens(e, pipeline) for e in eps) / len(eps))
    if not task_srs:
        return None
    if len(task_srs) < len(TASKS):
        print(f"  NOTE {eval_dir.name}/{cond_key[1]}: {len(task_srs)}/{len(TASKS)} tasks have data")
    return float(np.mean(task_srs)), float(np.mean(task_toks))


def get_seed_points(cond_key: tuple, eval_dir: Path) -> list[tuple]:
    if not eval_dir.exists():
        return []
    data      = _load_by_seed(eval_dir)
    pipeline  = cond_key[0]
    all_seeds: set = set()
    for task in TASKS:
        all_seeds |= set(data.get(cond_key, {}).get(task, {}).keys())
    points = []
    for iseed in sorted(all_seeds):
        srs, toks = [], []
        for task in TASKS:
            eps = data.get(cond_key, {}).get(task, {}).get(iseed, [])
            if not eps:
                continue
            srs.append(sum(1 for e in eps if e["success"]) / len(eps))
            toks.append(sum(_episode_tokens(e, pipeline) for e in eps) / len(eps))
        if srs and toks:
            points.append((float(np.mean(srs)), float(np.mean(toks))))
    return points


# ── Pareto frontier ───────────────────────────────────────────────────────────

def pareto_frontier(points: list[tuple]) -> list[tuple]:
    dominated = set()
    for i, (sr_i, tok_i) in enumerate(points):
        for j, (sr_j, tok_j) in enumerate(points):
            if i != j and sr_j >= sr_i and tok_j <= tok_i and (sr_j > sr_i or tok_j < tok_i):
                dominated.add(i)
    return [p for k, p in enumerate(points) if k not in dominated]


def draw_pareto_line(ax, pts: list[tuple], **kw):
    pts = sorted(pts, key=lambda p: p[0])
    ax.step([p[0] for p in pts], [p[1] for p in pts], where="pre", **kw)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_pareto(version: str = "main"):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL * 1.6, 4.2))
    fig.subplots_adjust(left=0.08, right=0.65, top=0.93, bottom=0.13)

    # ── Resolve all points ────────────────────────────────────────────────────
    # prog_pts[si]         → (sr, tok) for system si progenitor
    # cond_pts[(si, ci)]   → (sr, tok) for system si, condition ci
    prog_pts: dict[int, tuple] = {}
    cond_pts: dict[tuple, tuple] = {}

    for si, sys in enumerate(SYSTEMS):
        pt = get_point(sys["progenitor"][1], sys["progenitor"][2])
        if pt is not None:
            prog_pts[si] = pt
        for ci, (label, cond_key, eval_dir, _, _) in enumerate(sys["conditions"]):
            pt = get_point(cond_key, eval_dir)
            if pt is None:
                print(f"  SKIP {label} — no data")
                continue
            cond_pts[(si, ci)] = pt

    standalone_pts: dict[int, tuple] = {}
    for si, (label, cond_key, eval_dir, _) in enumerate(STANDALONE):
        pt = get_point(cond_key, eval_dir)
        if pt is None:
            print(f"  SKIP {label} — no data")
            continue
        standalone_pts[si] = pt

    # ── Fan arrows: progenitor → each condition (behind points) ──────────────
    for si, sys in enumerate(SYSTEMS):
        if si not in prog_pts:
            continue
        x0, y0 = prog_pts[si]
        for ci, (_, _, _, marker, lk) in enumerate(sys["conditions"]):
            if (si, ci) not in cond_pts:
                continue
            x1, y1 = cond_pts[(si, ci)]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=_shade(sys["base_color"], _LIGHTNESS[lk]),
                            lw=0.9, mutation_scale=7,
                            connectionstyle="arc3,rad=0",
                            shrinkA=5, shrinkB=5,
                        ))

    # ── Pareto frontier(s) ────────────────────────────────────────────────────
    all_pts = list(prog_pts.values()) + list(cond_pts.values()) + list(standalone_pts.values())

    # Collect appendix extra points now so the full frontier can be computed
    extra_pts = []
    if version == "appendix":
        for sys in SYSTEMS_EXTRA:
            for (_, cond_key, eval_dir, _, _) in sys["conditions"]:
                pt = get_point(cond_key, eval_dir)
                if pt is not None:
                    extra_pts.append(pt)

    main_pareto = set(pareto_frontier(all_pts)) if len(all_pts) >= 2 else set()

    if version == "appendix":
        if main_pareto:
            draw_pareto_line(ax, list(main_pareto),
                             color="0.55", lw=0.8, ls="--", zorder=1,
                             label="Pareto frontier (main)")
        full_pts = all_pts + extra_pts
        full_pareto = set(pareto_frontier(full_pts)) if len(full_pts) >= 2 else set()
        if full_pareto:
            draw_pareto_line(ax, list(full_pareto),
                             color="0.30", lw=1.0, ls="-", zorder=1,
                             label="Pareto frontier (all)")
        pareto_set = full_pareto
    else:
        pareto_set = main_pareto
        if pareto_set:
            draw_pareto_line(ax, list(pareto_set),
                             color="0.30", lw=1.0, ls="-", zorder=1,
                             label="Pareto frontier")

    # ── CI error bars ─────────────────────────────────────────────────────────
    T95 = 2.571
    if SHOW_SPREAD:
        for si, sys in enumerate(SYSTEMS):
            # progenitor CI
            if si in prog_pts:
                pts = get_seed_points(sys["progenitor"][1], sys["progenitor"][2])
                if len(pts) >= 2:
                    srs  = np.array([p[0] for p in pts])
                    toks = np.array([p[1] for p in pts])
                    n    = len(pts)
                    ax.errorbar(*prog_pts[si],
                                xerr=T95*srs.std(ddof=1)/np.sqrt(n),
                                yerr=T95*toks.std(ddof=1)/np.sqrt(n),
                                fmt="none",
                                color=_shade(sys["base_color"], _LIGHTNESS["prog"]),
                                capsize=2.5, capthick=0.7,
                                elinewidth=0.7, alpha=0.55, zorder=4)
            # condition CIs
            for ci, (_, cond_key, eval_dir, _, lk) in enumerate(sys["conditions"]):
                if (si, ci) not in cond_pts:
                    continue
                pts = get_seed_points(cond_key, eval_dir)
                if len(pts) < 2:
                    continue
                srs  = np.array([p[0] for p in pts])
                toks = np.array([p[1] for p in pts])
                n    = len(pts)
                ax.errorbar(*cond_pts[(si, ci)],
                            xerr=T95*srs.std(ddof=1)/np.sqrt(n),
                            yerr=T95*toks.std(ddof=1)/np.sqrt(n),
                            fmt="none",
                            color=_shade(sys["base_color"], _LIGHTNESS[lk]),
                            capsize=2.5, capthick=0.7,
                            elinewidth=0.7, alpha=0.55, zorder=4)

    bg_alpha = 0.30 if version == "appendix" else 1.0

    # ── System points ─────────────────────────────────────────────────────────
    for si, sys in enumerate(SYSTEMS):
        if si in prog_pts:
            sr, tok = prog_pts[si]
            c = _shade(sys["base_color"], _LIGHTNESS["prog"])
            label = sys["progenitor"][0]
            ec, lw = ("black", 1.0) if (sr, tok) in pareto_set else ("white", 0.5)
            ax.scatter(sr, tok, color=c, marker="*", s=160, alpha=bg_alpha,
                       zorder=5, edgecolors=ec, linewidths=lw)
            if label in LABEL_OFFSETS:
                dx, dy, ha, va = LABEL_OFFSETS[label]
                ax.annotate(LABEL_SHORT[label], xy=(sr, tok),
                            xytext=(dx, dy), textcoords="offset points",
                            fontsize=6.5, color=c, ha=ha, va=va, zorder=6)
        for ci, (label, _, _, marker, lk) in enumerate(sys["conditions"]):
            if (si, ci) not in cond_pts:
                continue
            sr, tok = cond_pts[(si, ci)]
            c = _shade(sys["base_color"], _LIGHTNESS[lk])
            ec, lw = ("black", 1.0) if (sr, tok) in pareto_set else ("white", 0.5)
            ax.scatter(sr, tok, color=c, marker=marker, s=60, alpha=bg_alpha,
                       zorder=5, edgecolors=ec, linewidths=lw)

    # ── Standalone progenitors ────────────────────────────────────────────────
    for si, (label, _, _, base_color) in enumerate(STANDALONE):
        if si not in standalone_pts:
            continue
        sr, tok = standalone_pts[si]
        c = _shade(base_color, 0.55)
        ec, lw = ("black", 1.0) if (sr, tok) in pareto_set else ("white", 0.5)
        ax.scatter(sr, tok, color=c, marker="*", s=160, alpha=bg_alpha,
                   zorder=5, edgecolors=ec, linewidths=lw)
        if label in LABEL_OFFSETS:
            dx, dy, ha, va = LABEL_OFFSETS[label]
            ax.annotate(LABEL_SHORT[label], xy=(sr, tok),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.5, color=c, ha=ha, va=va, zorder=6)

    # ── Appendix extra conditions ──────────────────────────────────────────────
    if version == "appendix":
        for sys in SYSTEMS_EXTRA:
            for (label, cond_key, eval_dir, marker, lk) in sys["conditions"]:
                pt = get_point(cond_key, eval_dir)
                if pt is None:
                    continue
                sr, tok = pt
                c = _shade(sys["base_color"], _LIGHTNESS[lk])
                ec, lw = ("black", 1.0) if (sr, tok) in pareto_set else ("white", 0.5)
                ax.scatter(sr, tok, color=c, marker=marker, s=60,
                           zorder=6, edgecolors=ec, linewidths=lw)


    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel("Mean success rate (avg over 5 tasks)", fontsize=8.5)
    ax.set_ylabel("Mean tokens per episode\n(avg over all episodes, success and failure)",
                  fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([i/20 for i in range(21)])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(10_000))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.90", lw=0.5, zorder=0)

    # ── Legends ───────────────────────────────────────────────────────────────
    shape_handles = [
        plt.scatter([], [], marker="*", color="0.4", s=120, label="Non-optimised"),
        plt.scatter([], [], marker="o", color="0.4", s=50,  label=f"{AA_LABEL}, {FINAL_INC}"),
        plt.scatter([], [], marker="^", color="0.4", s=50,  label=_HSP),
        plt.scatter([], [], marker="v", color="0.4", s=50,  label=_LSP),
        plt.Line2D([0],[0], color="0.30", lw=1.0, ls="-",
                   label="Pareto frontier (all)" if version == "appendix" else "Pareto frontier"),
        plt.scatter([], [], marker="o", color="0.4", s=50,
                    edgecolors="black", linewidths=1.0, label="Pareto dominant"),
    ]
    if version == "appendix":
        shape_handles.insert(-1, plt.Line2D([0],[0], color="0.55", lw=0.8, ls="--",
                                            label="Pareto frontier (main)"))
        shape_handles += [
            plt.scatter([], [], marker="X", color="0.4", s=50, label=f"HSP, {DELTA}=0.00"),
            plt.scatter([], [], marker="P", color="0.4", s=50, label=f"HSP, {DELTA}=0.02"),
            plt.scatter([], [], marker="s", color="0.4", s=45, label=f"HSP, {DELTA}=0.10"),
            plt.scatter([], [], marker="D", color="0.4", s=45, label=f"LSP, {DELTA}=0.00"),
            plt.scatter([], [], marker="p", color="0.4", s=50, label=f"LSP, {DELTA}=0.02"),
            plt.scatter([], [], marker="h", color="0.4", s=50, label=f"LSP, {DELTA}=0.10"),
            plt.scatter([], [], marker="<", color="0.4", s=50, label="Actor-only ablation"),
            plt.scatter([], [], marker=">", color="0.4", s=50, label="Descriptor-only ablation"),
            plt.scatter([], [], marker="+", color="0.4", s=60, label="Random-module, HSP"),
            plt.scatter([], [], marker="d", color="0.4", s=60, label="Random-module, LSP"),
        ]
    color_handles = [
        mpatches.Patch(color=_shade(_BASE_B_MIN,      0.70), label=f"BALROG {_pln}"),
        mpatches.Patch(color=_shade(_BASE_B_RICH,     0.70), label=f"BALROG {_gdd}"),
        mpatches.Patch(color=_shade(_BASE_O_MIN,      0.70), label=f"{AGENT_NAME} {_pln}"),
        mpatches.Patch(color=_shade(_BASE_O_RICH,     0.70), label=f"{AGENT_NAME} {_gdd}"),
        mpatches.Patch(color=_shade(_BASE_O_RICH_H16, 0.70), label=f"{AGENT_NAME} {_gdd} h16"),
        mpatches.Patch(color=_shade(_BASE_ALONE,      0.55), label="Non-opt only (no descendants)"),
    ]
    leg1 = ax.legend(handles=shape_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=7,
                     framealpha=0.9, edgecolor="0.8",
                     title="Opt stage / method", title_fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, loc="upper left",
              bbox_to_anchor=(1.02, 0.2), fontsize=7,
              framealpha=0.9, edgecolor="0.8",
              title="System", title_fontsize=7.5)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="main", choices=["main", "appendix"],
                        help="main = primary paper figure; appendix = all conditions")
    args = parser.parse_args()

    fig = plot_pareto(version=args.version)
    if fig:
        _save(fig, f"pareto_efficiency_{args.version}")
        plt.show()
