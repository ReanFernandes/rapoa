#!/usr/bin/env python3
"""
gen_paper_tables.py — Generate paper tables (LaTeX + Markdown) from all plot data.

Run from final_paper_plotting/:
    python gen_paper_tables.py

Outputs written to tables/:
    table_heatmap_{section1,section3,section4}.{tex,md}
    table_threshold_sensitivity.{tex,md}
    table_opt_trajectory_{lsp,hsp}.{tex,md}
    table_pareto{,_appendix}.{tex,md}
    table_crosstask_{fname}.{tex,md}   (11 files)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    AGENT_NAME, TASKS, TASK_LABELS, FRESH_EVAL_DIR, OPT_RUNS_DIR,
    VARIANT_LABELS, DELTA, AA_LABEL, T_STAR, FINAL_INC, SP_LABELS,
)
from plot_heatmap import (
    COLS, NAMED_COLS, ROWS,
    build_matrix as _hm_build,
    load_mutation_counts as _hm_zero_muts,
    OPT_EVAL_ROOT,
)
from plot_pareto import (
    SYSTEMS, STANDALONE, SYSTEMS_EXTRA,
    get_point, pareto_frontier,
)
from plot_opt_trajectory import (
    _make_conditions, TASK_ORDER,
    load_stage3_trajectory, load_stage2_trajectory,
    METRIC, MAX_CYCLES,
)
from plot_threshold_sensitivity import (
    COND_DEFS, THRESH_RUN_IDS, AA_EVAL_DIR,
    load_sr, load_zero_mutations,
    VARIANTS, VAL_STRATS,
    _CK_MIN as _TH_CK_MIN,
    _CK_RICH as _TH_CK_RICH,
)
from plot_crosstask_heatmap import (
    CONDITIONS as CROSSTASK_CONDITIONS,
    build_matrix as _ct_build,
    zero_mutation_rows as _ct_zero_rows,
)

TABLES_DIR = Path(__file__).resolve().parent / "tables"
TABLES_DIR.mkdir(exist_ok=True)

_V = VARIANT_LABELS   # {"minimal": "plain", "rich": "guided"}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _flat(s: str) -> str:
    """Collapse \\n in plot labels to a single space."""
    return " ".join(s.split("\n"))


def _ltx(s: str) -> str:
    """Minimal LaTeX escaping for label strings."""
    return s.replace("δ", r"$\delta$").replace("−∞", r"$-\infty$").replace("∞", r"$\infty$")


def _cell_sr(sr: float, std: float, zero: bool = False) -> tuple[str, str]:
    """(latex, markdown) for a SR±std cell. Inputs are fractions [0,1]."""
    if np.isnan(sr):
        return "—", "—"
    s, d = f"{100 * sr:.1f}", f"{100 * std:.1f}"
    zm_tex = r"$^{\circ}$" if zero else ""
    zm_md  = " ◦" if zero else ""
    tex = rf"{s}\%{{\scriptsize$\pm${d}\%}}{zm_tex}"
    md  = f"{s}% ±{d}%{zm_md}"
    return tex, md


def _fmt_metric(v: float | None) -> tuple[str, str]:
    """(latex, md) for a trajectory metric value (reward or SR)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—", "—"
    if METRIC == "sr":
        t = f"{100 * v:.1f}\\%"
        m = f"{100 * v:.1f}%"
    else:
        t = f"{v:.4f}"
        m = f"{v:.4f}"
    return t, m


def _tok_str(tokens: float) -> str:
    if tokens >= 1e6:
        return f"{tokens / 1e6:.2f}M"
    return f"{tokens / 1e3:.0f}k"


def _write(stem: str, tex: str, md: str) -> None:
    (TABLES_DIR / f"{stem}.tex").write_text(tex)
    (TABLES_DIR / f"{stem}.md").write_text(md)
    print(f"  {stem}.tex + .md")


# ── Group-span helper for multicolumn headers ──────────────────────────────────

def _group_spans(cols: list) -> list[tuple[str, int, int]]:
    """
    Returns [(group_label, start_1based, end_1based)] where col 1 = Task label.
    Consecutive columns with the same group are merged.
    """
    spans: list[tuple[str, int, int]] = []
    prev_g = cols[0][2]
    start, count = 2, 1
    for j in range(1, len(cols)):
        g = cols[j][2]
        if g == prev_g:
            count += 1
        else:
            spans.append((prev_g, start, start + count - 1))
            start += count
            prev_g = g
            count = 1
    spans.append((prev_g, start, start + count - 1))
    return spans


# ══════════════════════════════════════════════════════════════════════════════
# 1–3. Heatmap tables (sections 1, 3, 4)
# ══════════════════════════════════════════════════════════════════════════════

_HEATMAP_CAPTIONS = {
    "section1": (
        r"Success rate (\%) by task and condition (\S1: decomposition performance). "
        r"Each cell shows mean SR $\pm$ std across 6 inference seeds. "
        r"$^{\circ}$ = zero accepted mutations; performance equals the non-optimised baseline."
    ),
    "section3": (
        r"Module ablation: success rate (\%) by task and condition (\S3: role of decomposition). "
        r"$^{\circ}$ = zero accepted mutations."
    ),
    "section4": (
        r"Always-accept vs.\ gated: success rate (\%) by task and condition (\S4). "
        r"$^{\circ}$ = zero accepted mutations."
    ),
}


def gen_heatmap_table(sec: str) -> None:
    active   = [COLS[i] for i in NAMED_COLS[sec]]
    sr_mat, std_mat, _ = _hm_build(ROWS, active)
    zero_mut = _hm_zero_muts(active, ROWS)

    n_cols = len(active)
    groups = _group_spans(active)
    caption = _HEATMAP_CAPTIONS[sec]

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    col_spec = "l " + " ".join(["c"] * n_cols)
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:heatmap_{sec}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]
    # Group-header row + cmidrule separators
    gcells = [""] + [
        rf"\multicolumn{{{e - s + 1}}}{{c}}{{{_ltx(_flat(g))}}}"
        for g, s, e in groups
    ]
    cmidrules = " ".join(rf"\cmidrule(lr){{{s}-{e}}}" for _, s, e in groups)
    col_hdrs = ["Task"] + [_ltx(_flat(lbl)) for _, lbl, *_ in active]
    lines += [
        " & ".join(gcells) + r" \\",
        cmidrules,
        " & ".join(col_hdrs) + r" \\",
        r"\midrule",
    ]
    for i, task in enumerate(ROWS):
        cells = [TASK_LABELS[task]]
        for j in range(n_cols):
            zm = zero_mut.get((j, i), False)
            tex_c, _ = _cell_sr(sr_mat[i, j], std_mat[i, j], zm)
            cells.append(tex_c)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\\[2pt]",
        r"{\footnotesize $^{\circ}$ Zero accepted mutations; prompt identical to non-optimised baseline.}",
        r"\end{table}",
    ]
    tex = "\n".join(lines)

    # ── Markdown ────────────────────────────────────────────────────────────────
    sec_label = sec.replace("section", "Section ")
    hdr  = ["Task"] + [f"[{_flat(grp)}] {_flat(lbl)}" for _, lbl, grp, *_ in active]
    sep  = ["---"] * (n_cols + 1)
    rows_md = [
        f"**Heatmap {sec_label}** — SR% ± std%. ◦ = zero accepted mutations.\n",
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for i, task in enumerate(ROWS):
        row = [TASK_LABELS[task]]
        for j in range(n_cols):
            zm = zero_mut.get((j, i), False)
            _, md_c = _cell_sr(sr_mat[i, j], std_mat[i, j], zm)
            row.append(md_c)
        rows_md.append("| " + " | ".join(row) + " |")
    md = "\n".join(rows_md)

    _write(f"table_heatmap_{sec}", tex, md)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Threshold sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def gen_threshold_table() -> None:
    """4 sub-tables: (HSP | LSP) × (plain | guided), each 5 tasks × 5 thresholds."""
    cond_labels = [lbl for lbl, _, _ in COND_DEFS]   # non-opt / δ=-∞ / δ=0.02 / δ=0.05 / δ=0.10

    all_tex: list[str] = []
    all_md:  list[str] = []

    for val_strat in VAL_STRATS:
        sp  = SP_LABELS[val_strat]                    # "HSP" / "LSP"
        for variant in VARIANTS:
            var     = _V[variant]                     # "plain" / "guided"
            ck      = _TH_CK_MIN if variant == "minimal" else _TH_CK_RICH
            title   = f"{sp}, {var} prompt"

            base_sr = load_sr(FRESH_EVAL_DIR, ck)
            aa_sr   = load_sr(AA_EVAL_DIR, ck) if AA_EVAL_DIR.exists() else {}

            tdata: dict[str, dict] = {}
            tzero: dict[str, set]  = {}
            for _, _, key in COND_DEFS:
                if key in ("baseline", "aa"):
                    continue
                run_id = THRESH_RUN_IDS.get((val_strat, key), "")
                if run_id:
                    ed = OPT_EVAL_ROOT / run_id
                    tdata[key] = load_sr(ed, ck) if ed.exists() else {}
                    tzero[key] = load_zero_mutations(run_id, variant)
                else:
                    tdata[key] = {}
                    tzero[key] = set()

            # LaTeX sub-table
            col_spec_t = "l ccccc"
            all_tex += [
                rf"\noindent\textbf{{{title}}}\\[2pt]",
                rf"\begin{{tabular}}{{{col_spec_t}}}",
                r"\toprule",
                r"Task & non-opt & $\delta=-\infty$ & $\delta=0.02$ & $\delta=0.05$ & $\delta=0.10$ \\",
                r"\midrule",
            ]
            for task in ROWS:
                cells = [TASK_LABELS[task]]
                for _, _, key in COND_DEFS:
                    if key == "baseline":
                        sr = base_sr.get(task, float("nan"))
                    elif key == "aa":
                        sr = aa_sr.get(task, float("nan"))
                    else:
                        sr = tdata[key].get(task, float("nan"))
                    zm = (key not in ("baseline", "aa")) and (task in tzero.get(key, set()))
                    if np.isnan(sr):
                        cells.append("—")
                    else:
                        cells.append(_cell_sr(sr, 0.0, zm)[0])
                all_tex.append(" & ".join(cells) + r" \\")
            all_tex += [r"\bottomrule", r"\end{tabular}", r"\\[6pt]"]

            # Markdown sub-table
            all_md += [
                f"\n### {title}\n",
                "| Task | non-opt | δ=−∞ | δ=0.02 | δ=0.05 | δ=0.10 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for task in ROWS:
                cells = [TASK_LABELS[task]]
                for _, _, key in COND_DEFS:
                    if key == "baseline":
                        sr = base_sr.get(task, float("nan"))
                    elif key == "aa":
                        sr = aa_sr.get(task, float("nan"))
                    else:
                        sr = tdata[key].get(task, float("nan"))
                    zm = (key not in ("baseline", "aa")) and (task in tzero.get(key, set()))
                    if np.isnan(sr):
                        cells.append("—")
                    else:
                        cells.append(f"{100*sr:.1f}%" + (" ◦" if zm else ""))
                all_md.append("| " + " | ".join(cells) + " |")

    header_tex = [
        r"\begin{table}[!h]",
        r"\centering",
        (r"\caption{Threshold sensitivity: SR\% by task, selection pressure, prompt variant, "
         r"and acceptance threshold $\delta$. "
         r"$^{\circ}$ = zero accepted mutations. std not shown (no seed-level breakdown here).}"),
        r"\label{tab:threshold_sensitivity}",
    ]
    footer_tex = [
        r"{\footnotesize $^{\circ}$ Zero accepted mutations; prompt identical to non-optimised baseline.}",
        r"\end{table}",
    ]
    tex = "\n".join(header_tex + all_tex + footer_tex)
    md  = ("**Threshold sensitivity** — SR% per task, selection pressure, variant, and δ. "
           "◦ = zero accepted mutations. std omitted (no per-seed breakdown for individual thresholds).\n"
           + "\n".join(all_md))
    _write("table_threshold_sensitivity", tex, md)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Optimisation trajectory summary
#
# Compact cell format:
#   Gated:    "c0 → final (N)"    where N = n accepted mutations
#   δ=-∞:     "T*(cyc) → final [Δ]"
# Each table: 5 tasks × 3 conditions (Ours plain | Ours guided | BALROG plain)
# ══════════════════════════════════════════════════════════════════════════════

def _n_accepted(run_id: str, variant: str | None, task: str) -> int:
    import json
    base     = OPT_RUNS_DIR / run_id
    task_dir = (base / variant / task) if variant else (base / task)
    log      = task_dir / "optimisation_log.jsonl"
    if not log.exists():
        return -1
    records = [json.loads(l) for l in open(log)]
    return sum(
        1 for r in records
        if r.get("record_type") == "opt_cycle"
        and r.get("opt_cycle_outcome") == "accepted"
    )


def _gated_cell(pts: list) -> tuple[str, str]:
    """Compact gated cell: 'initial → final (n_accepted)'."""
    if not pts:
        return "—", "—"
    pts = sorted(pts)
    c0t, c0m = _fmt_metric(pts[0][1])
    ft,  fm  = _fmt_metric(pts[-1][1])
    n_acc    = len(pts) - 1
    return rf"{c0t}\,$\to$\,{ft} ({n_acc})", f"{c0m} → {fm} ({n_acc})"


def gen_trajectory_table(val_strat: str) -> None:
    sp_label   = SP_LABELS[val_strat]
    sp_key     = "lsp" if val_strat == "trainsig" else "hsp"
    conditions = _make_conditions(val_strat)
    metric_lbl = "SR" if METRIC == "sr" else "reward"

    # Short condition headers (strip "Ours/" / "BALROG/" prefix)
    def _short(label: str) -> str:
        parts = label.split("/")
        return "/".join(p.strip() for p in parts[:2])

    cond_headers = [_short(c["label"]) for c in conditions]

    def _aa_cell(pts: list) -> tuple[str, str]:
        if not pts:
            return "—", "—"
        pts       = sorted(pts)
        best_cyc, best_val = max(pts, key=lambda p: p[1])
        fin_val   = pts[-1][1]
        gap       = best_val - fin_val
        bt, bm    = _fmt_metric(best_val)
        ft, fm    = _fmt_metric(fin_val)
        gt, gm    = _fmt_metric(abs(gap))
        sign      = "+" if gap >= 0 else "−"
        tex = rf"{bt}\,({best_cyc})\,$\to$\,{ft}\,[{sign}{gt}]"
        md  = f"{bm} ({best_cyc}) → {fm} [{sign}{gm}]"
        return tex, md

    # Build two sub-tables: gated then δ=-∞
    all_tex: list[str] = []
    all_md:  list[str] = []

    for stage, stage_title_tex, stage_title_md, cell_fn, run_attr, var_attr in [
        ("stage3",
         rf"Gated, {sp_label} ($\delta$=0.05) --- cell format: initial $\to$ final (n\,accepted)",
         f"Gated, {sp_label} (δ=0.05) — cell: initial → final (n accepted)",
         _gated_cell, "stage3_run", "stage3_variant"),
        ("stage2",
         r"Always-accept ($\delta=-\infty$) --- cell format: T$^*$(cycle) $\to$ final [$\Delta$]",
         "Always-accept (δ=−∞) — cell: T*(cycle) → final [Δ(T*−final)]",
         _aa_cell, "stage2_run", "stage2_variant"),
    ]:
        loader = (load_stage3_trajectory if stage == "stage3"
                  else load_stage2_trajectory)

        col_spec = "l " + " ".join(["c"] * len(conditions))
        all_tex += [
            rf"\noindent\textbf{{{stage_title_tex}}}\\[2pt]",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(["Task"] + cond_headers) + r" \\",
            r"\midrule",
        ]
        all_md += [f"\n### {stage_title_md}\n",
                   "| Task | " + " | ".join(cond_headers) + " |",
                   "| --- |" + " --- |" * len(conditions)]

        for task in TASK_ORDER:
            row_tex = [TASK_LABELS[task]]
            row_md  = [TASK_LABELS[task]]
            for cond in conditions:
                run_id  = cond[run_attr]
                variant = cond[var_attr]
                if not run_id:
                    row_tex.append("—")
                    row_md.append("—")
                    continue
                pts = loader(run_id, variant, task)
                t, m = cell_fn(pts)
                row_tex.append(t)
                row_md.append(m)
            all_tex.append(" & ".join(row_tex) + r" \\")
            all_md.append("| " + " | ".join(row_md) + " |")

        all_tex += [r"\bottomrule", r"\end{tabular}", r"\\[6pt]"]

    header = [
        r"\begin{table}[!h]",
        r"\centering",
        (rf"\caption{{Optimisation trajectory summary ({sp_label}, $\delta$=0.05 gated). "
         rf"Metric: {metric_lbl}. "
         r"Gated cells: initial $\to$ final (n\,accepted mutations). "
         r"Always-accept cells: T$^*$(cycle\,of\,peak) $\to$ end-of-run [$\Delta$ = T$^*$ $-$ final].}}"),
        rf"\label{{tab:trajectory_{sp_key}}}",
    ]
    footer = [r"\end{table}"]

    tex = "\n".join(header + all_tex + footer)
    md  = (f"**Trajectory summary ({sp_label})** — metric: {metric_lbl}.\n"
           + "\n".join(all_md))
    _write(f"table_opt_trajectory_{sp_key}", tex, md)


# ── 5b. Threshold sweep trajectory ────────────────────────────────────────────

def gen_threshold_trajectory_table() -> None:
    """
    Trajectory summary for the threshold sweep (δ=0.02 / 0.05 / 0.10).
    4 sub-tables: (HSP | LSP) × (plain | guided).
    Rows: 5 tasks. Columns: δ=0.02 | δ=0.05 | δ=0.10.
    Cell: initial → final (n_accepted).  Same format as gated rows in gen_trajectory_table.
    """
    thresh_keys      = ["002", "default", "010"]
    thresh_hdrs_tex  = [r"$\delta=0.02$", r"$\delta=0.05$", r"$\delta=0.10$"]
    thresh_hdrs_md   = ["δ=0.02", "δ=0.05", "δ=0.10"]
    metric_lbl       = "SR" if METRIC == "sr" else "reward"

    all_tex: list[str] = []
    all_md:  list[str] = []

    for val_strat in VAL_STRATS:
        sp = SP_LABELS[val_strat]
        for variant in VARIANTS:
            var   = _V[variant]
            title = f"{sp}, {var} prompt"

            all_tex += [
                rf"\noindent\textbf{{{title}}}\\[2pt]",
                r"\begin{tabular}{l ccc}",
                r"\toprule",
                " & ".join(["Task"] + thresh_hdrs_tex) + r" \\",
                r"\midrule",
            ]
            all_md += [
                f"\n### {title}\n",
                "| Task | " + " | ".join(thresh_hdrs_md) + " |",
                "| --- | --- | --- | --- |",
            ]

            for task in TASK_ORDER:
                row_tex = [TASK_LABELS[task]]
                row_md  = [TASK_LABELS[task]]
                for key in thresh_keys:
                    run_id = THRESH_RUN_IDS.get((val_strat, key), "")
                    if not run_id:
                        row_tex.append("—"); row_md.append("—")
                        continue
                    pts = load_stage3_trajectory(run_id, variant, task)
                    t, m = _gated_cell(pts)
                    row_tex.append(t); row_md.append(m)
                all_tex.append(" & ".join(row_tex) + r" \\")
                all_md.append("| " + " | ".join(row_md) + " |")

            all_tex += [r"\bottomrule", r"\end{tabular}", r"\\[6pt]"]

    header = [
        r"\begin{table}[!h]",
        r"\centering",
        (rf"\caption{{Threshold sweep trajectory summary. "
         rf"Rows = tasks, columns = acceptance threshold $\delta$. "
         rf"Cell format: initial $\to$ final (n\,accepted mutations). Metric: {metric_lbl}.}}"),
        r"\label{tab:trajectory_threshold}",
    ]
    footer = [r"\end{table}"]

    tex = "\n".join(header + all_tex + footer)
    md  = (f"**Threshold sweep trajectory** — metric: {metric_lbl}. "
           f"Cell: initial → final (n accepted).\n" + "\n".join(all_md))
    _write("table_opt_trajectory_threshold", tex, md)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pareto efficiency table
# ══════════════════════════════════════════════════════════════════════════════

def gen_pareto_table(version: str = "main") -> None:
    """One row per system × condition. Cols: label, SR%, tokens, Pareto."""
    # Collect all points
    all_pts: list[tuple[float, float]] = []

    for sys in SYSTEMS:
        pt = get_point(sys["progenitor"][1], sys["progenitor"][2])
        if pt:
            all_pts.append(pt)
        for _, ck, ed, _, _ in sys["conditions"]:
            pt = get_point(ck, ed)
            if pt:
                all_pts.append(pt)
    for _, ck, ed, _ in STANDALONE:
        pt = get_point(ck, ed)
        if pt:
            all_pts.append(pt)

    extra_pts: list[tuple[float, float]] = []
    if version == "appendix":
        for sys in SYSTEMS_EXTRA:
            for _, ck, ed, _, _ in sys["conditions"]:
                pt = get_point(ck, ed)
                if pt:
                    extra_pts.append(pt)

    pareto_set = set(map(tuple, pareto_frontier(all_pts + extra_pts)))

    # Build ordered row list
    RowEntry = tuple  # (label, sr, tokens, is_pareto, is_progenitor)
    rows: list[RowEntry] = []

    def _add(label: str, ck, ed, is_prog: bool) -> None:
        pt = get_point(ck, ed)
        if pt is None:
            return
        rows.append((_flat(label), pt[0], pt[1],
                     tuple(pt) in pareto_set, is_prog))

    for _, ck, ed, _ in STANDALONE:
        label = next(lbl for lbl, c, d, _ in STANDALONE if c == ck and d == ed)
        _add(label, ck, ed, True)

    for sys in SYSTEMS:
        _add(sys["progenitor"][0], sys["progenitor"][1], sys["progenitor"][2], True)
        for label, ck, ed, _, _ in sys["conditions"]:
            _add(label, ck, ed, False)

    if version == "appendix":
        for sys in SYSTEMS_EXTRA:
            for label, ck, ed, _, _ in sys["conditions"]:
                _add(label, ck, ed, False)

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    tex = "\n".join([
        r"\begin{table}[!h]",
        r"\centering",
        (rf"\caption{{Efficiency: mean SR\% and mean tokens per episode averaged over all five tasks "
         rf"({'all conditions' if version == 'appendix' else 'main paper conditions'}). "
         rf"$\star$ = non-optimised progenitor. \checkmark = Pareto dominant.}}"),
        rf"\label{{tab:pareto_{version}}}",
        r"\begin{tabular}{l r r c}",
        r"\toprule",
        r"System / condition & SR (\%) & Tokens & Pareto \\",
        r"\midrule",
        *[
            (r"$\star$ " if is_prog else "") + rf"{label} & {100*sr:.1f} & {_tok_str(tok)} & "
            + (r"\checkmark" if par else "") + r" \\"
            for label, sr, tok, par, is_prog in rows
        ],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    # ── Markdown ────────────────────────────────────────────────────────────────
    md = "\n".join([
        f"**Pareto efficiency ({'all conditions' if version == 'appendix' else 'main'})** "
        f"— mean SR% and tokens/episode, averaged over 5 tasks. ★ = progenitor. ✓ = Pareto dominant.\n",
        "| System / condition | SR% | Tokens | Pareto |",
        "| --- | --- | --- | --- |",
        *[
            f"| {'★ ' if is_prog else ''}{label} | {100*sr:.1f}% | {_tok_str(tok)} | {'✓' if par else ''} |"
            for label, sr, tok, par, is_prog in rows
        ],
    ])

    _write(f"table_pareto{'_appendix' if version == 'appendix' else ''}", tex, md)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Cross-task heatmap tables
# ══════════════════════════════════════════════════════════════════════════════

def gen_crosstask_table(cond: dict) -> None:
    sr_mat, std_mat = _ct_build(cond)
    if int(np.sum(~np.isnan(sr_mat))) == 0:
        print(f"  SKIP {cond['fname']} — no data")
        return

    zero_rows = _ct_zero_rows(cond)
    n         = len(ROWS)

    col_best = np.nanmax(sr_mat, axis=0, keepdims=True)
    is_best  = np.isclose(sr_mat, col_best) & ~np.isnan(sr_mat)

    eval_lbls = [TASK_LABELS[t] for t in ROWS]
    col_spec  = "l " + " ".join(["c"] * n)

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        (rf"\caption{{Cross-task transfer: {cond['label']}. "
         r"Rows = source task (prompt optimised for); columns = eval task. "
         r"Underline = best source for that eval task. "
         r"$^{\circ}$ on row label = zero accepted mutations (prompt = baseline).}}"),
        rf"\label{{tab:{cond['fname']}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Source $\backslash$ Eval & " + " & ".join(eval_lbls) + r" \\",
        r"\midrule",
    ]
    for i, src in enumerate(ROWS):
        src_lbl = TASK_LABELS[src] + (r" $^{\circ}$" if src in zero_rows else "")
        cells   = [src_lbl]
        for j in range(n):
            v, s = sr_mat[i, j], std_mat[i, j]
            if np.isnan(v):
                c = "—"
            else:
                c = rf"{100*v:.1f}\%{{\scriptsize$\pm${100*s:.1f}\%}}"
                if is_best[i, j]:
                    c = r"\underline{" + c + "}"
            cells.append(c)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)

    # ── Markdown ────────────────────────────────────────────────────────────────
    md_lines = [
        f"**Cross-task: {cond['label']}** — SR% ± std%. "
        f"**Bold** = best source for that eval task. ◦ = zero accepted mutations.\n",
        "| Source \\\\ Eval | " + " | ".join(eval_lbls) + " |",
        "| --- |" + " --- |" * n,
    ]
    for i, src in enumerate(ROWS):
        src_lbl = TASK_LABELS[src] + (" ◦" if src in zero_rows else "")
        row     = [src_lbl]
        for j in range(n):
            v, s = sr_mat[i, j], std_mat[i, j]
            if np.isnan(v):
                row.append("—")
            else:
                val = f"{100*v:.1f}% ±{100*s:.1f}%"
                row.append(f"**{val}**" if is_best[i, j] else val)
        md_lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(md_lines)

    _write(f"table_{cond['fname']}", tex, md)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Accepted-mutation counts across thresholds
# ══════════════════════════════════════════════════════════════════════════════

def gen_mutation_counts_table() -> None:
    """
    n_accepted per task × variant × (selection pressure × threshold δ).
    Columns: HSP δ=0.00/0.02/0.05/0.10, LSP δ=0.00/0.02/0.05/0.10.
    Rows: 5 tasks × 2 variants (plain / guided), grouped by task.
    Summary row: zero-mutation runs / available runs per column.
    Always-accept excluded by design (it always accepts).
    δ=0.00 cells show — where that run is still incomplete.
    """
    COLUMNS = [
        ("HSP", "0.00", "stage3_valbag_thresh000_20260505"),
        ("HSP", "0.02", "stage3_valbag_thresh002_20260501"),
        ("HSP", "0.05", "stage3_mean_valbag_20260430"),
        ("HSP", "0.10", "stage3_valbag_thresh010_20260501"),
        ("LSP", "0.00", "stage3_trainsig_thresh000_20260505"),
        ("LSP", "0.02", "stage3_trainsig_thresh002_20260501"),
        ("LSP", "0.05", "stage3_mean_trainsig_20260430"),
        ("LSP", "0.10", "stage3_trainsig_thresh010_20260501"),
    ]
    VORD   = [("minimal", _V["minimal"]), ("rich", _V["rich"])]
    n_cols = len(COLUMNS)

    # data[task][var_key][col_idx] = n_accepted, or -1 if run missing
    data = {
        task: {
            var_key: [_n_accepted(run_id, var_key, task) for _, _, run_id in COLUMNS]
            for var_key, _ in VORD
        }
        for task in ROWS
    }

    def _cell(n: int) -> tuple[str, str]:
        if n < 0:  return "—", "—"
        if n == 0: return r"\textbf{0}", "**0**"
        return str(n), str(n)

    def _summary(ci: int) -> str:
        n_zero  = sum(1 for t in ROWS for k, _ in VORD if data[t][k][ci] == 0)
        n_avail = sum(1 for t in ROWS for k, _ in VORD if data[t][k][ci] >= 0)
        return f"{n_zero}/{n_avail}"

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    col_spec = "ll " + "c" * n_cols
    tex = [
        r"\begin{table}[!h]",
        r"\centering",
        (r"\caption{Accepted mutation counts per task, prompt variant, selection pressure, "
         r"and threshold $\delta$. "
         r"\textbf{0} = gate never fired; prompt identical to non-optimised baseline. "
         r"--- = run incomplete. "
         r"Bottom row: zero-mutation runs out of available (max 10 per column).}"),
        r"\label{tab:mutation_counts}",
        r"\renewcommand{\arraystretch}{1.1}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{HSP} & \multicolumn{4}{c}{LSP} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        "Task & Variant & "
            + " & ".join(rf"$\delta$={d}" for _, d, _ in COLUMNS)
            + r" \\",
        r"\midrule",
    ]
    for ti, task in enumerate(ROWS):
        for vi, (var_key, var_lbl) in enumerate(VORD):
            cells = [
                rf"\multirow{{2}}{{*}}{{{TASK_LABELS[task]}}}" if vi == 0 else "",
                var_lbl,
            ]
            for ci in range(n_cols):
                t, _ = _cell(data[task][var_key][ci])
                cells.append(t)
            tex.append(" & ".join(cells) + r" \\")
        if ti < len(ROWS) - 1:
            tex.append(r"\midrule")
    tex += [
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{zeros / available}} & "
            + " & ".join(_summary(ci) for ci in range(n_cols)) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_out = "\n".join(tex)

    # ── Markdown ────────────────────────────────────────────────────────────────
    hdr = ["Task", "Variant"] + [f"[{sp}] δ={d}" for sp, d, _ in COLUMNS]
    sep = ["---"] * (2 + n_cols)
    md  = [
        "**Accepted mutations per task, variant, selection pressure, and threshold.**  "
        "**0** = gate never fired. — = run incomplete. "
        "Bottom row: zeros / available.\n",
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for task in ROWS:
        for vi, (var_key, var_lbl) in enumerate(VORD):
            row = [TASK_LABELS[task] if vi == 0 else "", var_lbl]
            for ci in range(n_cols):
                _, m = _cell(data[task][var_key][ci])
                row.append(m)
            md.append("| " + " | ".join(row) + " |")
    md.append(
        "| **zeros / available** | | "
        + " | ".join(_summary(ci) for ci in range(n_cols)) + " |"
    )
    _write("table_mutation_counts", latex_out, "\n".join(md))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Heatmap tables...")
    for sec in ("section1", "section3", "section4"):
        gen_heatmap_table(sec)

    print("\nThreshold sensitivity table...")
    gen_threshold_table()

    print("\nTrajectory tables...")
    for vs in ("trainsig", "valbag"):
        gen_trajectory_table(vs)

    print("\nThreshold trajectory table...")
    gen_threshold_trajectory_table()

    print("\nPareto tables...")
    gen_pareto_table("main")
    gen_pareto_table("appendix")

    print("\nCross-task tables...")
    for cond in CROSSTASK_CONDITIONS:
        gen_crosstask_table(cond)

    print("\nMutation counts table...")
    gen_mutation_counts_table()

    print("\nDone.")
