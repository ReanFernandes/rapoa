#!/usr/bin/env python3
"""
gen_ba_attribution_tables.py
Extract BA attribution data from optimisation logs and produce three tables
that together support the claim that the BA is behaviorally intelligent.

Scope: SPA plain + SPA guided, gated runs only (δ=0.00/0.02/0.05/0.10 × HSP/LSP).
Always-accept excluded (acceptance is trivial, hereditary context is uninformative).
BALROG excluded (no threshold sweep; different pipeline, different BA context).

Tables produced (in tables/):
  table_ba_type_by_task.{tex,md}
      RQ6: F/I/S counts and % per task and variant, aggregated across all conditions.
      Supports: "the BA produces diverse output types — not just failures."

  table_ba_type_by_condition.{tex,md}
      RQ5+RQ6: per (threshold × SP), F/I/S counts + F→accepted / I→accepted rates.
      Supports: "different output types have different gate-acceptance rates."

  table_ba_rank.{tex,md}
      RQ4: which candidate rank was accepted, per condition.
      Supports: "the BA's top-ranked proposal is usually the one that clears the gate."

  table_ba_fresh_sr.{tex,md}
      RQ5 (fresh eval link): per (task × variant × condition), composition of
      accepted mutations (n_F_acc, n_I_acc) alongside final fresh SR.
      Supports: "runs where the BA found insights correlate with higher final performance."

Usage:
    cd final_paper_plotting
    python gen_ba_attribution_tables.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TASKS, TASK_LABELS, TASK_OPT_NAMES, FRESH_EVAL_DIR,
    VARIANT_LABELS, SP_LABELS, DELTA,
    SLUGS, opt_dir, eval_dir as _eval_dir, latest_campaign,
)

TABLES_DIR = Path(__file__).resolve().parent / "tables"
TABLES_DIR.mkdir(exist_ok=True)

_V = VARIANT_LABELS

CAMPAIGN_PRIMARY = latest_campaign(SLUGS["hsp_rich"])
CAMPAIGN_THRESH  = latest_campaign(SLUGS["thresh002_hsp"])

def _cam(slug_key): return CAMPAIGN_PRIMARY if slug_key in ("hsp_rich","hsp_minimal","lsp_rich","lsp_minimal") else CAMPAIGN_THRESH

# ── Run set ────────────────────────────────────────────────────────────────────
# (sp_label, delta_str, slug_key)
RUNS = [
    ("HSP", "0.00", "thresh000_hsp"),
    ("HSP", "0.02", "thresh002_hsp"),
    ("HSP", "0.05", "hsp_rich"),
    ("HSP", "0.10", "thresh010_hsp"),
    ("LSP", "0.00", "thresh000_lsp"),
    ("LSP", "0.02", "thresh002_lsp"),
    ("LSP", "0.05", "lsp_rich"),
    ("LSP", "0.10", "thresh010_lsp"),
]
VARIANTS  = [("minimal", _V["minimal"]), ("rich", _V["rich"])]
TASK_ORDER = ["goto", "pickup", "open", "pick_up_seq_go_to", "putnext"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ba_data(slug_key: str, variant: str, task: str) -> dict | None:
    """
    Returns per-cycle BA attribution data for one (run, variant, task) triple.
    Returns None if the optimisation log does not exist (run incomplete).

    Fields returned:
      n_cycles                  total opt_cycles in log
      n_failure/insight/skip    BA output type counts
      n_failure_accepted        failure-type cycles that ended in T acceptance
      n_insight_accepted        insight-type cycles that ended in T acceptance
      n_failure_rejected        failure-type cycles that were rejected
      n_insight_rejected        insight-type cycles that were rejected
      n_constraint_skip         cycles where signal was insufficient (not BA-driven)
      module_dist               {"agent": N, "descriptor": M}
      rank_dist                 {0: N, 1: M, 2: K} for accepted-cycle ranks
    """
    run_dir  = opt_dir(_cam(slug_key), SLUGS[slug_key])
    task_dir = run_dir / TASK_OPT_NAMES.get(task, task)
    log      = task_dir / "optimisation_log.jsonl"
    if not log.exists():
        return None

    records = [json.loads(l) for l in open(log)]
    cycles  = [r for r in records if r.get("record_type") == "opt_cycle"]

    d = dict(
        n_cycles=len(cycles),
        n_failure=0, n_insight=0, n_skip=0,
        n_failure_accepted=0, n_insight_accepted=0,
        n_failure_rejected=0, n_insight_rejected=0,
        n_constraint_skip=0,
        module_dist={}, rank_dist={},
    )

    for c in cycles:
        ba      = c.get("ba_output") or {}
        ba_type = ba.get("type")          # "failure" | "insight" | "skip"
        module  = ba.get("module")        # "agent" | "descriptor"
        outcome = c.get("opt_cycle_outcome")

        # Count BA output type
        if ba_type == "failure":
            d["n_failure"] += 1
        elif ba_type == "insight":
            d["n_insight"] += 1
        elif ba_type == "skip" or outcome == "ba_skip":
            d["n_skip"] += 1

        # Count module attribution
        if module:
            d["module_dist"][module] = d["module_dist"].get(module, 0) + 1

        # Count constraint_skip (insufficient signal — not BA-driven)
        if outcome == "constraint_skip":
            d["n_constraint_skip"] += 1

        # Acceptance outcome by BA type
        if outcome == "accepted":
            # Find the rank of the accepted candidate
            for cand in c.get("candidates_tried", []):
                t_res = cand.get("t_result") or {}
                if isinstance(t_res, dict) and t_res.get("verdict") == "accepted":
                    rank = cand.get("candidate_rank", 0)
                    d["rank_dist"][rank] = d["rank_dist"].get(rank, 0) + 1
                    break
            if ba_type == "failure":
                d["n_failure_accepted"] += 1
            elif ba_type == "insight":
                d["n_insight_accepted"] += 1

        elif outcome in ("rejected_all", "constraint_skip"):
            if ba_type == "failure":
                d["n_failure_rejected"] += 1
            elif ba_type == "insight":
                d["n_insight_rejected"] += 1

    return d


def load_fresh_sr(slug_key: str, variant: str, task: str) -> float | None:
    """SR from the fresh eval for the final incumbent of this run."""
    from collections import defaultdict as _dd

    ck      = ("with_descriptor", variant, "single_turn")
    ed      = _eval_dir(_cam(slug_key), SLUGS[slug_key])
    if not ed or not ed.exists():
        return None

    seed_eps: dict[str, list] = _dd(list)
    for s in sorted(ed.rglob("run_summary.json")):
        parts = s.parts
        try:
            anchor   = next(i for i, p in enumerate(parts) if p == ed.name)
            _, t, _, pl, va, cm, _, iseed, _ = parts[anchor + 1:anchor + 10]
        except (StopIteration, ValueError):
            continue
        if t != task or (pl, va, cm) != ck:
            continue
        d = json.load(open(s))
        seed_eps[iseed].extend(d["episodes"])

    if not seed_eps:
        return None
    all_eps = [e for eps in seed_eps.values() for e in eps]
    return sum(1 for e in all_eps if e["success"]) / len(all_eps)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pct(n, total):
    return f"{100 * n / total:.0f}%" if total else "—"


def _write(stem: str, tex: str, md: str) -> None:
    (TABLES_DIR / f"{stem}.tex").write_text(tex)
    (TABLES_DIR / f"{stem}.md").write_text(md)
    print(f"  {stem}.tex + .md")


# ── Pre-load all data ──────────────────────────────────────────────────────────

print("Loading BA data...")
# corpus[sp][delta][var_key][task] = ba_data dict or None
corpus: dict = {}
for sp, delta, slug_key in RUNS:
    corpus.setdefault(sp, {}).setdefault(delta, {})
    for var_key, _ in VARIANTS:
        corpus[sp][delta].setdefault(var_key, {})
        for task in TASK_ORDER:
            corpus[sp][delta][var_key][task] = load_ba_data(slug_key, var_key, task)

# fresh_sr[sp][delta][var_key][task] = float or None
print("Loading fresh SR...")
fresh_sr: dict = {}
for sp, delta, slug_key in RUNS:
    fresh_sr.setdefault(sp, {}).setdefault(delta, {})
    for var_key, _ in VARIANTS:
        fresh_sr[sp][delta].setdefault(var_key, {})
        for task in TASK_ORDER:
            fresh_sr[sp][delta][var_key][task] = load_fresh_sr(slug_key, var_key, task)


# ══════════════════════════════════════════════════════════════════════════════
# Table 1: BA type distribution by task (aggregated across all conditions)
# RQ6 — "the BA is not just outputting failures"
# ══════════════════════════════════════════════════════════════════════════════

def gen_type_by_task() -> None:
    """
    For each (task, variant): total cycles, n_failure, n_insight, n_skip,
    aggregated across all 8 (sp × delta) conditions.
    Shows that insight and skip rates differ meaningfully across tasks.
    """
    tex = [
        r"\begin{table}[!h]",
        r"\centering",
        (r"\caption{BA output type distribution per task and prompt variant, "
         r"aggregated across all gated conditions (HSP/LSP, $\delta$=0.00/0.02/0.05/0.10). "
         r"Values are total cycle counts; percentages in parentheses.}"),
        r"\label{tab:ba_type_by_task}",
        r"\begin{tabular}{ll rrrr}",
        r"\toprule",
        r"Task & Variant & Cycles & Failure & Insight & Skip \\",
        r"\midrule",
    ]
    md = [
        "**BA type distribution per task** — aggregated across all gated conditions.\n",
        "| Task | Variant | Cycles | Failure | Insight | Skip |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    totals = dict(cycles=0, failure=0, insight=0, skip=0)

    for ti, task in enumerate(TASK_ORDER):
        for vi, (var_key, var_lbl) in enumerate(VARIANTS):
            cyc = fail = ins = skip = 0
            for sp, delta, _ in RUNS:
                d = corpus[sp][delta][var_key][task]
                if d is None:
                    continue
                cyc  += d["n_cycles"]
                fail += d["n_failure"]
                ins  += d["n_insight"]
                skip += d["n_skip"]

            totals["cycles"]  += cyc
            totals["failure"] += fail
            totals["insight"] += ins
            totals["skip"]    += skip

            task_cell = (rf"\multirow{{2}}{{*}}{{{TASK_LABELS[task]}}}"
                         if vi == 0 else "")
            row_tex = (f"{task_cell} & {var_lbl} & {cyc} & "
                       f"{fail} ({_pct(fail, cyc)}) & "
                       f"{ins} ({_pct(ins, cyc)}) & "
                       f"{skip} ({_pct(skip, cyc)}) \\\\")
            tex.append(row_tex)
            md.append(
                f"| {'**' + TASK_LABELS[task] + '**' if vi == 0 else ''} "
                f"| {var_lbl} | {cyc} "
                f"| {fail} ({_pct(fail, cyc)}) "
                f"| {ins} ({_pct(ins, cyc)}) "
                f"| {skip} ({_pct(skip, cyc)}) |"
            )

        if ti < len(TASK_ORDER) - 1:
            tex.append(r"\midrule")

    c = totals["cycles"]
    tex += [
        r"\midrule",
        (rf"\textbf{{All}} & & \textbf{{{c}}} & "
         rf"\textbf{{{totals['failure']}}} ({_pct(totals['failure'], c)}) & "
         rf"\textbf{{{totals['insight']}}} ({_pct(totals['insight'], c)}) & "
         rf"\textbf{{{totals['skip']}}} ({_pct(totals['skip'], c)}) \\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    md.append(
        f"| **All** | | **{c}** "
        f"| **{totals['failure']} ({_pct(totals['failure'], c)})** "
        f"| **{totals['insight']} ({_pct(totals['insight'], c)})** "
        f"| **{totals['skip']} ({_pct(totals['skip'], c)})** |"
    )

    _write("table_ba_type_by_task", "\n".join(tex), "\n".join(md))


# ══════════════════════════════════════════════════════════════════════════════
# Table 2: BA type × acceptance outcome per condition
# RQ5 + RQ6 — "which BA output types cleared the gate, and how often"
# ══════════════════════════════════════════════════════════════════════════════

def gen_type_by_condition() -> None:
    """
    For each (sp, delta): F total, F→accepted, F→accepted%, I total, I→accepted%,
    S total (no gate possible). Aggregated across all tasks and variants.
    Shows how gate permissiveness interacts with BA output type.
    """
    tex = [
        r"\begin{table}[!h]",
        r"\centering",
        (r"\caption{BA output type and gate-acceptance rate per condition. "
         r"F = failure, I = insight, S = skip. "
         r"Acc\% = proportion of that type accepted by the T gate. "
         r"Aggregated across all 5 tasks and both prompt variants.}"),
        r"\label{tab:ba_type_by_condition}",
        r"\begin{tabular}{ll rr r rr r r}",
        r"\toprule",
        (r"SP & $\delta$ & "
         r"F & F acc & F acc\% & "
         r"I & I acc & I acc\% & "
         r"S \\"),
        r"\midrule",
    ]
    md = [
        "**BA type × gate acceptance per condition** — aggregated across tasks and variants.\n",
        "| SP | δ | F total | F acc | F acc% | I total | I acc | I acc% | S |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    prev_sp = None
    for sp, delta, slug_key in RUNS:
        if prev_sp and sp != prev_sp:
            tex.append(r"\midrule")
        prev_sp = sp

        f_tot = f_acc = i_tot = i_acc = s_tot = 0
        for var_key, _ in VARIANTS:
            for task in TASK_ORDER:
                d = corpus[sp][delta][var_key][task]
                if d is None:
                    continue
                f_tot += d["n_failure"]
                f_acc += d["n_failure_accepted"]
                i_tot += d["n_insight"]
                i_acc += d["n_insight_accepted"]
                s_tot += d["n_skip"]

        f_pct = _pct(f_acc, f_tot)
        i_pct = _pct(i_acc, i_tot)

        tex.append(
            f"{sp} & $\\delta$={delta} & "
            f"{f_tot} & {f_acc} & {f_pct} & "
            f"{i_tot} & {i_acc} & {i_pct} & "
            f"{s_tot} \\\\"
        )
        md.append(
            f"| {sp} | {delta} | "
            f"{f_tot} | {f_acc} | {f_pct} | "
            f"{i_tot} | {i_acc} | {i_pct} | "
            f"{s_tot} |"
        )

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("table_ba_type_by_condition", "\n".join(tex), "\n".join(md))


# ══════════════════════════════════════════════════════════════════════════════
# Table 3: Candidate rank at acceptance per condition
# RQ4 — "the BA's top proposal is usually the one that clears the gate"
# ══════════════════════════════════════════════════════════════════════════════

def gen_rank_table() -> None:
    """
    For each condition: total accepted cycles, and how many were rank 0 / 1 / 2.
    Aggregated across all tasks and variants.
    """
    tex = [
        r"\begin{table}[!h]",
        r"\centering",
        (r"\caption{Candidate rank at T-pool acceptance per condition. "
         r"Rank 0 = BA's highest-priority proposal; rank 1/2 = fallback candidates. "
         r"Aggregated across all 5 tasks and both prompt variants.}"),
        r"\label{tab:ba_rank}",
        r"\begin{tabular}{ll rrrr}",
        r"\toprule",
        r"SP & $\delta$ & Accepted & Rank 1 (primary) & Rank 2 & Rank 3 \\",
        r"\midrule",
    ]
    md = [
        "**Candidate rank at acceptance** — aggregated across tasks and variants.\n",
        "| SP | δ | Accepted | Rank 1 (primary) | Rank 2 | Rank 3 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    prev_sp = None
    for sp, delta, slug_key in RUNS:
        if prev_sp and sp != prev_sp:
            tex.append(r"\midrule")
        prev_sp = sp

        rank_dist: dict[int, int] = {}
        for var_key, _ in VARIANTS:
            for task in TASK_ORDER:
                d = corpus[sp][delta][var_key][task]
                if d is None:
                    continue
                for rank, cnt in d["rank_dist"].items():
                    rank_dist[rank] = rank_dist.get(rank, 0) + cnt

        total = sum(rank_dist.values())
        r0 = rank_dist.get(1, 0)   # rank 1 = primary candidate
        r1 = rank_dist.get(2, 0)   # rank 2 = first fallback
        r2 = rank_dist.get(3, 0)   # rank 3 = second fallback

        def _rc(n):
            return f"{n} ({_pct(n, total)})" if total else "—"

        tex.append(
            f"{sp} & $\\delta$={delta} & {total} & "
            f"{_rc(r0)} & {_rc(r1)} & {_rc(r2)} \\\\"
        )
        md.append(
            f"| {sp} | {delta} | {total} | "
            f"{_rc(r0)} | {_rc(r1)} | {_rc(r2)} |"
        )

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("table_ba_rank", "\n".join(tex), "\n".join(md))


# ══════════════════════════════════════════════════════════════════════════════
# Table 4: Accepted mutation composition × fresh SR
# RQ5 (fresh eval link)
# "runs where insights were accepted correspond to higher final performance"
# ══════════════════════════════════════════════════════════════════════════════

def gen_fresh_sr_link() -> None:
    """
    For each (task, variant): per condition, show n_F_acc / n_I_acc alongside
    fresh SR. Rows grouped by task. Supports the claim that insight-type
    accepted mutations correlate with higher final performance.
    """
    n_conds = len(RUNS)
    cond_hdrs = [f"{sp} {DELTA}={d}" for sp, d, _ in RUNS]

    tex = [
        r"\begin{table}[!h]",
        r"\centering",
        (r"\caption{Accepted mutation composition and final fresh-eval SR per "
         r"task, variant, and condition. "
         r"Each cell: F\,/\,I accepted (n\_failure\_accepted / n\_insight\_accepted) "
         r"$\mid$ fresh SR\%. "
         r"--- = run incomplete.}"),
        r"\label{tab:ba_fresh_sr}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll " + "c " * n_conds + "}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{HSP} & \multicolumn{4}{c}{LSP} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        "Task & Variant & "
            + " & ".join(rf"$\delta$={d}" for _, d, _ in RUNS) + r" \\",
        r"\midrule",
    ]
    md = [
        "**Accepted mutation composition × fresh SR** — cell: F acc / I acc | SR%.\n",
        "| Task | Variant | " + " | ".join(cond_hdrs) + " |",
        "| --- | --- |" + " --- |" * n_conds,
    ]

    for ti, task in enumerate(TASK_ORDER):
        for vi, (var_key, var_lbl) in enumerate(VARIANTS):
            cells_tex = [
                rf"\multirow{{2}}{{*}}{{{TASK_LABELS[task]}}}" if vi == 0 else "",
                var_lbl,
            ]
            cells_md = [
                f"{'**' + TASK_LABELS[task] + '**' if vi == 0 else ''}",
                var_lbl,
            ]

            for sp, delta, _ in RUNS:
                d   = corpus[sp][delta][var_key][task]
                sr  = fresh_sr[sp][delta][var_key][task]
                if d is None:
                    cells_tex.append("—")
                    cells_md.append("—")
                    continue

                f_a = d["n_failure_accepted"]
                i_a = d["n_insight_accepted"]
                sr_str = f"{100 * sr:.0f}\\%" if sr is not None else "—"
                sr_md  = f"{100 * sr:.0f}%"   if sr is not None else "—"
                cells_tex.append(rf"{f_a}/{i_a} \textbar\ {sr_str}")
                cells_md.append(f"{f_a}/{i_a} | {sr_md}")

            tex.append(" & ".join(cells_tex) + r" \\")
            md.append("| " + " | ".join(cells_md) + " |")

        if ti < len(TASK_ORDER) - 1:
            tex.append(r"\midrule")

    tex += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\\[2pt]",
        r"{\footnotesize F/I = failure-type / insight-type accepted mutations; "
        r"SR = fresh-eval success rate of final incumbent.}",
        r"\end{table}",
    ]
    _write("table_ba_fresh_sr_link", "\n".join(tex), "\n".join(md))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nTable 1: BA type by task...")
    gen_type_by_task()

    print("\nTable 2: BA type × acceptance by condition...")
    gen_type_by_condition()

    print("\nTable 3: Candidate rank at acceptance...")
    gen_rank_table()

    print("\nTable 4: Accepted composition × fresh SR...")
    gen_fresh_sr_link()

    print("\nDone.")
