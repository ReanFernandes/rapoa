"""
Generate BA attribution and constraint violation tables for the module ablation.

Produces one combined table showing, per condition:
  - BA module attribution distribution (% agent / descriptor / skip)
  - Constraint violations (ba_skips) for constrained runs only

Also produces the detail table (per task, violations only).

Output:
    tables/ba_constraint_violations.md
    tables/ba_constraint_violations.tex

Usage:
    cd final_paper_plotting
    python gen_ba_constraint_table.py
"""

import json
from collections import Counter
from pathlib import Path

from config import (
    TASK_LABELS, TASK_OPT_NAMES, VARIANT_LABELS,
    SLUGS, opt_dir, latest_campaign,
)

_V = VARIANT_LABELS
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)

CAMPAIGN_PRIMARY  = latest_campaign(SLUGS["hsp_rich"])
CAMPAIGN_ABLATION = latest_campaign(SLUGS["actor_ablation_rich"])

TASKS    = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]
VARIANTS = ["minimal", "rich"]

VLABEL = {"minimal": _V["minimal"], "rich": _V["rich"]}


# ── Data helpers ──────────────────────────────────────────────────────────────

def collect_attribution(base: Path) -> Counter:
    """Count ba_output.module across all opt_cycle records under base."""
    c = Counter()
    for log in base.rglob("optimisation_log.jsonl"):
        if any(p.startswith(("opt_cycle_", "env_round_", "eval_")) for p in log.parts):
            continue
        for line in open(log):
            r = json.loads(line)
            if r.get("record_type") != "opt_cycle":
                continue
            mod = (r.get("ba_output") or {}).get("module") or "none"
            c[mod] += 1
    return c


def collect_violations(slug_dir: Path) -> tuple[int, int]:
    """Return (total_cycles, n_ba_skips) for all tasks in a slug dir."""
    total, skips = 0, 0
    for task in TASKS:
        log = slug_dir / TASK_OPT_NAMES.get(task, task) / "optimisation_log.jsonl"
        if not log.exists():
            continue
        for line in open(log):
            r = json.loads(line)
            if r.get("record_type") != "opt_cycle":
                continue
            total += 1
            if r.get("opt_cycle_outcome") == "ba_skip":
                skips += 1
    return total, skips


def collect_violations_per_task(slug_dir: Path) -> list:
    """Return [(task, n_cycles, n_skips)] for tasks with at least one skip."""
    out = []
    for task in TASKS:
        log = slug_dir / TASK_OPT_NAMES.get(task, task) / "optimisation_log.jsonl"
        if not log.exists():
            continue
        cycles = [json.loads(l) for l in open(log)
                  if '"record_type": "opt_cycle"' in l]
        n_s = sum(1 for r in cycles if r.get("opt_cycle_outcome") == "ba_skip")
        if n_s > 0:
            out.append((task, len(cycles), n_s))
    return out


# ── Build combined rows ───────────────────────────────────────────────────────
# Each row: (label, variant, agent%, desc%, skip%, n_cycles, n_violations, constrained)

combined = []

# Unconstrained gated runs
for slug_key, sp_label in [
    ("hsp_rich",    "HSP"),
    ("hsp_minimal", "HSP"),
    ("lsp_rich",    "LSP"),
    ("lsp_minimal", "LSP"),
]:
    var = "rich" if "rich" in slug_key else "minimal"
    sd  = opt_dir(CAMPAIGN_PRIMARY, SLUGS[slug_key]) if CAMPAIGN_PRIMARY else None
    if not sd:
        continue
    c   = collect_attribution(sd)
    tot = sum(c.values()) or 1
    combined.append((
        sp_label, VLABEL[var],
        c.get("actor", 0) / tot,
        c.get("descriptor", 0) / tot,
        (c.get("none", 0) + c.get(None, 0)) / tot,
        sum(c.values()), None, False,
    ))

# Constrained ablation runs
for mod, mod_label in [("actor", "Actor-only"), ("descriptor", "Descriptor-only")]:
    for var in VARIANTS:
        sk  = f"{mod}_ablation_{var}"
        sd  = opt_dir(CAMPAIGN_ABLATION, SLUGS[sk]) if CAMPAIGN_ABLATION else None
        if not sd:
            continue
        c            = collect_attribution(sd)
        tot          = sum(c.values()) or 1
        n_cyc, n_vio = collect_violations(sd)
        combined.append((
            mod_label, VLABEL[var],
            c.get("actor", 0) / tot,
            c.get("descriptor", 0) / tot,
            (c.get("none", 0) + c.get(None, 0)) / tot,
            n_cyc, n_vio, True,
        ))

# Detail violation rows
detail = []
for mod, mod_label in [("actor", "Actor-only"), ("descriptor", "Descriptor-only")]:
    for var in VARIANTS:
        sk = f"{mod}_ablation_{var}"
        sd = opt_dir(CAMPAIGN_ABLATION, SLUGS[sk]) if CAMPAIGN_ABLATION else None
        if not sd:
            continue
        for task, n_c, n_s in collect_violations_per_task(sd):
            detail.append((mod_label, VLABEL[var], task, n_c, n_s))

grand_cycles = sum(r[5] for r in combined if r[7])
grand_vio    = sum(r[6] for r in combined if r[7])


# ── Formatting helpers ────────────────────────────────────────────────────────

def fp(v):
    return f"{100*v:.0f}\\%" if isinstance(v, float) else "---"

def fp_md(v):
    return f"{100*v:.0f}%" if isinstance(v, float) else "—"

def fv(n):
    return str(n) if n is not None else "—"

def fvr(n, tot):
    return f"{100*n/tot:.1f}\\%" if n is not None and tot else "---"

def fvr_md(n, tot):
    return f"{100*n/tot:.1f}%" if n is not None and tot else "—"


# ── Markdown ──────────────────────────────────────────────────────────────────

md = []
md.append("## BA Attribution and Constraint Violations — Module Ablation\n")
md.append("### Combined table\n")
md.append("| Condition | Variant | Agent | Descriptor | Skip | Cycles | Violations | Viol. rate |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|")

prev_constrained = False
for label, var, fa, fd, fs, nc, nv, constrained in combined:
    if constrained and not prev_constrained:
        md.append("|---|---|---|---|---|---|---|---|")
    prev_constrained = constrained
    md.append(
        f"| {label} | {var} "
        f"| {fp_md(fa)} | {fp_md(fd)} | {fp_md(fs)} "
        f"| {nc} | {fv(nv)} | {fvr_md(nv, nc)} |"
    )
md.append(f"| **Total (constrained)** | | | | | **{grand_cycles}** | **{grand_vio}** | **{fvr_md(grand_vio, grand_cycles)}** |")

md.append("\n### Detail — violation runs only\n")
md.append("| Constraint | Variant | Task | Cycles | Violations | Viol. rate |")
md.append("|---|---|---|---:|---:|---:|")
for lbl, var, task, nc, ns in detail:
    md.append(f"| {lbl} | {var} | {TASK_LABELS[task]} | {nc} | {ns} | {fvr_md(ns, nc)} |")
md.append("\n_All violations are in descriptor-only runs. Agent-only runs produced zero violations._")

md_path = OUT / "ba_constraint_violations.md"
md_path.write_text("\n".join(md))
print(f"Saved: {md_path}")


# ── LaTeX ─────────────────────────────────────────────────────────────────────

tex = []
tex.append("% BA attribution + constraint violation table — auto-generated by gen_ba_constraint_table.py")
tex.append("")
tex.append("\\begin{table}[h]")
tex.append("\\centering")
tex.append("\\small")
tex.append("\\begin{tabular}{llrrrrrr}")
tex.append("\\toprule")
tex.append("Condition & Variant & Agent & Descriptor & Skip & Cycles & Violations & Viol.\\ rate \\\\")
tex.append("\\midrule")

prev_constrained = False
for label, var, fa, fd, fs, nc, nv, constrained in combined:
    if constrained and not prev_constrained:
        tex.append("\\midrule")
    prev_constrained = constrained
    tex.append(
        f"{label} & {var} & {fp(fa)} & {fp(fd)} & {fp(fs)} "
        f"& {nc} & {fv(nv)} & {fvr(nv, nc)} \\\\"
    )

tex.append("\\midrule")
tex.append(
    f"\\textbf{{Total (constrained)}} & & & & & "
    f"\\textbf{{{grand_cycles}}} & \\textbf{{{grand_vio}}} & \\textbf{{{fvr(grand_vio, grand_cycles)}}} \\\\"
)
tex.append("\\bottomrule")
tex.append("\\end{tabular}")
tex.append(
    "\\caption{BA module attribution and constraint violations for unconstrained gated runs "
    "and module-ablation constrained runs. Attribution columns show the fraction of "
    "optimisation cycles where the BA diagnosed a failure in each module. Violations count "
    "cycles where the BA attributed the locked module, requiring a re-sample. All violations "
    "occur in descriptor-only runs; agent-only runs produced zero violations.}"
)
tex.append("\\label{tab:ba_attribution}")
tex.append("\\end{table}")
tex.append("")

# Detail table
tex.append("\\begin{table}[h]")
tex.append("\\centering")
tex.append("\\small")
tex.append("\\begin{tabular}{lllrrr}")
tex.append("\\toprule")
tex.append("Constraint & Variant & Task & Cycles & Violations & Viol.\\ rate \\\\")
tex.append("\\midrule")
for lbl, var, task, nc, ns in detail:
    tex.append(f"{lbl} & {var} & {TASK_LABELS[task]} & {nc} & {ns} & {fvr(ns, nc)} \\\\")
tex.append("\\bottomrule")
tex.append("\\end{tabular}")
tex.append(
    "\\caption{Constraint violation detail — runs with at least one violation shown. "
    "Violations are concentrated in descriptor-only / \\textsc{GoTo}, where the BA "
    "is tempted toward agent suggestions on a near-ceiling task.}"
)
tex.append("\\label{tab:ba_attribution_detail}")
tex.append("\\end{table}")

tex_path = OUT / "ba_constraint_violations.tex"
tex_path.write_text("\n".join(tex))
print(f"Saved: {tex_path}")
