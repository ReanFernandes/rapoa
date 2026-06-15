"""Optimisation run analysis — complements opt_progress.py with what it doesn't show.

Shows per-run:
  - V trajectory as a sparkline across all cycles
  - V_start, V_now, V_max, and delta from start
  - Acceptance rate
  - constraint_skip count (acceptance signal failures)
  - T_fail count (V-passed but T-rejected — gated runs only)
  - BA module attribution (agent % vs descriptor %)

Usage:
    python tools/analyse_runs.py                              # all experiments
    python tools/analyse_runs.py --run-id stage3_mean_valbag_20260430
    python tools/analyse_runs.py --detail minimal/goto       # full cycle breakdown for one run
    python tools/analyse_runs.py --run-id stage3_mean_valbag_20260430 --detail minimal/open
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPARK = " ▁▂▃▄▅▆▇█"   # index 0 = no data, 1–8 = low–high


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return records


def _sparkline(values: list[float], lo: float = 0.0, hi: float = 1.0) -> str:
    if not values:
        return ""
    span = hi - lo if hi > lo else 1.0
    chars = []
    for v in values:
        idx = int(((v - lo) / span) * 7) + 1
        idx = max(1, min(8, idx))
        chars.append(SPARK[idx])
    return "".join(chars)


def _load_run(run_dir: Path) -> dict | None:
    log_path = run_dir / "optimisation_log.jsonl"
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None

    with open(cfg_path) as f:
        cfg = json.load(f)

    records = _read_jsonl(log_path)
    cycles = [r for r in records if r.get("record_type") == "opt_cycle"]

    if not cycles:
        return {
            "cycles": [],
            "v_traj": [],
            "v_start": None,
            "acceptance_rule": cfg.get("acceptance_rule", "?"),
            "validation_strategy": cfg.get("validation_strategy", "?"),
            "opt_cycles": cfg.get("opt_cycles", 20),
        }

    # V trajectory from v_incumbent_rewards each cycle
    v_traj = []
    for c in cycles:
        rewards = c.get("v_incumbent_rewards") or []
        v_traj.append(sum(rewards) / len(rewards) if rewards else None)

    # V_start from env_round_setup (round 0)
    round0 = next(
        (r for r in records
         if r.get("record_type") == "env_round_setup" and r.get("env_round") == 0),
        None,
    )
    v_start = None
    if round0:
        rewards = round0.get("v_rewards") or []
        if rewards:
            v_start = sum(rewards) / len(rewards)

    # Outcome breakdown
    outcomes = {"accepted": 0, "rejected": 0, "ba_skip": 0,
                "constraint_skip": 0, "insufficient_signal": 0, "other": 0}
    for c in cycles:
        o = c.get("opt_cycle_outcome", "")
        if o in outcomes:
            outcomes[o] += 1
        elif o in ("all_candidates_failed", "rejected_all"):
            outcomes["rejected"] += 1
        else:
            outcomes["other"] += 1

    # T_fail: cycles that were ultimately rejected where at least one candidate
    # passed V but was blocked by T. Excludes cycles that succeeded via a later
    # ranked candidate (candidate queue working as designed).
    t_fail = 0
    for c in cycles:
        if c.get("opt_cycle_outcome") == "accepted":
            continue
        for cand in (c.get("candidates_tried") or []):
            v_v = (cand.get("v_result") or {}).get("verdict")
            t_v = (cand.get("t_result") or {}).get("verdict")
            if v_v == "accepted" and t_v == "rejected":
                t_fail += 1
                break

    # BA module attribution
    agent_count = 0
    desc_count = 0
    for c in cycles:
        ba = c.get("ba_output") or {}
        mod = ba.get("module")
        if mod == "agent":
            agent_count += 1
        elif mod == "descriptor":
            desc_count += 1

    # Cumulative paired Δ across accepted cycles.
    # Each entry is (cycle_num, paired_delta, running_total).
    # paired_delta = net_mean_reward from that acceptance eval — challenger minus
    # incumbent on the SAME seeds, so seed variance is cancelled within each event.
    # The running total sums deltas across DIFFERENT seed batches, so magnitude is
    # approximate. DIAGNOSTIC ONLY — not suitable for paper reporting.
    cum_delta_trace = []
    cum = 0.0
    for c in cycles:
        if c.get("opt_cycle_outcome") != "accepted":
            continue
        for cand in (c.get("candidates_tried") or []):
            net = (cand.get("v_result") or {}).get("net_mean_reward")
            if net is not None:
                cum += net
                cum_delta_trace.append((c.get("opt_cycle"), net, cum))
                break

    return {
        "cycles":              cycles,
        "v_traj":              [v for v in v_traj if v is not None],
        "v_start":             v_start,
        "cycles_done":         len(cycles),
        "opt_cycles":          cfg.get("opt_cycles", 20),
        "acceptance_rule":     cfg.get("acceptance_rule", "?"),
        "validation_strategy": cfg.get("validation_strategy", "?"),
        "outcomes":            outcomes,
        "t_fail":              t_fail,
        "agent_attr":          agent_count,
        "desc_attr":           desc_count,
        "cum_delta_trace":     cum_delta_trace,
        "cum_delta_total":     cum,
    }


def _summary_line(label: str, info: dict) -> str:
    vt = info["v_traj"]
    v_start = info.get("v_start")
    v_now   = vt[-1]  if vt else None
    v_max   = max(vt) if vt else None

    def fmt(v): return f"{v:.3f}" if v is not None else "  — "

    # NOTE: V_start, V_now, V_max all come from different env seeds each cycle.
    # No delta is computed — comparing across different seeds is misleading.
    # Use the sparkline to read trajectory shape; use acceptance count for
    # whether the prompt actually changed.

    n_done  = info.get("cycles_done", 0)
    n_total = info.get("opt_cycles", 20)

    o = info.get("outcomes", {})
    n_accepted = o.get("accepted", 0)
    n_gated    = n_accepted + o.get("rejected", 0) + o.get("constraint_skip", 0)
    acc_rate   = f"{100*n_accepted/n_gated:.0f}%" if n_gated > 0 else " — "

    cskip  = o.get("constraint_skip", 0)
    t_fail = info.get("t_fail", 0)

    total_attr = info.get("agent_attr", 0) + info.get("desc_attr", 0)
    if total_attr > 0:
        agt_pct = f"{100*info['agent_attr']//total_attr}%"
        dsc_pct = f"{100*info['desc_attr']//total_attr}%"
    else:
        agt_pct = dsc_pct = " — "

    spark = _sparkline(vt)
    spark_padded = f"{spark:<22}"

    cum_total = info.get("cum_delta_total", 0.0)
    n_acc     = info.get("outcomes", {}).get("accepted", 0)
    cum_str   = f"{cum_total:+.3f}" if n_acc > 0 else "  — "

    return (
        f"  {label:<32}  {n_done:>2}/{n_total}  "
        f"{fmt(v_start)}  {fmt(v_now)}  {fmt(v_max)}  "
        f"{acc_rate:>4}  {cskip:>5}  {t_fail:>6}  "
        f"{agt_pct:>4}/{dsc_pct:<4}  {cum_str:>7}  {spark_padded}"
    )


def _print_detail(label: str, info: dict) -> None:
    cycles = info.get("cycles", [])
    if not cycles:
        print(f"  No cycles recorded for {label}")
        return

    print(f"\n  {'='*80}")
    print(f"  Detail: {label}   "
          f"({info.get('acceptance_rule','?')} / {info.get('validation_strategy','?')})")
    print(f"  {'='*80}")
    print(f"  * pairedΔ and cumΔ: challenger minus incumbent on SAME seeds — DIAGNOSTIC ONLY")
    print(f"  {'cyc':>3}  {'V_mean':>7}  {'outcome':<20}  {'pairedΔ':>8}  {'cumΔ*':>7}  "
          f"{'T':>8}  {'BA':>8}  {'mod':>6}  {'wall':>7}")
    print(f"  {'─'*88}")

    cum = 0.0
    for c in cycles:
        cyc     = c.get("opt_cycle", "?")
        rewards = c.get("v_incumbent_rewards") or []
        vmean   = f"{sum(rewards)/len(rewards):.3f}" if rewards else "  — "
        outcome = c.get("opt_cycle_outcome", "?")
        wall    = c.get("wall_time_s")
        wall_s  = f"{wall:.0f}s" if wall else "  —"

        ba      = c.get("ba_output") or {}
        ba_type = ba.get("type", "?")
        ba_mod  = ba.get("module") or "—"

        paired_str = "       "
        cum_str    = "      "
        t_str      = "  — "
        for cand in (c.get("candidates_tried") or []):
            vr  = cand.get("v_result") or {}
            tr  = cand.get("t_result") or {}
            net = vr.get("net_mean_reward")
            if net is not None:
                paired_str = f"{net:+.3f}  "
                if outcome == "accepted":
                    cum += net
                    cum_str = f"{cum:+.3f}"
            t_verdict = tr.get("verdict", "—")
            t_note    = tr.get("note", "")
            t_str     = t_note if t_note else t_verdict
            break

        acc_marker = " ◀" if outcome == "accepted" else ""
        print(f"  {cyc:>3}  {vmean:>7}  {outcome:<20}  {paired_str:>8}  {cum_str:>7}  "
              f"{t_str:>8}  {ba_type:>8}  {ba_mod:>6}  {wall_s:>7}{acc_marker}")

    print()

    # Accepted mutations summary
    accepted_cycles = [c for c in cycles if c.get("opt_cycle_outcome") == "accepted"]
    if accepted_cycles:
        print(f"  Accepted mutations ({len(accepted_cycles)}):")
        print(f"  {'─'*70}")
        for c in accepted_cycles:
            cyc = c.get("opt_cycle", "?")
            for cand in (c.get("candidates_tried") or []):
                mod    = cand.get("module", "?")
                change = cand.get("mutator_change", "")[:70]
                print(f"  cycle {cyc:>2}  [{mod}]  {change}")
                break
        print()


VARIANTS    = ["minimal", "rich"]
TASKS       = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]
CONSTRAINTS = ["agent", "descriptor"]


def _detect_structure(base_dir: Path) -> str:
    if any((base_dir / c / v / t / "run_config.json").exists()
           for c in CONSTRAINTS for v in VARIANTS for t in TASKS):
        return "module"
    if any((base_dir / v / t / "run_config.json").exists()
           for v in VARIANTS for t in TASKS):
        return "standard"
    if any((base_dir / t / "run_config.json").exists() for t in TASKS):
        return "balrog"
    return "standard"


def _build_rows(base_dir: Path, structure: str) -> list[tuple[str, Path]]:
    if structure == "module":
        return [(f"{c}/{v}/{t}", base_dir / c / v / t)
                for c in CONSTRAINTS for v in VARIANTS for t in TASKS]
    elif structure == "balrog":
        return [(t, base_dir / t) for t in TASKS]
    else:
        return [(f"{v}/{t}", base_dir / v / t) for v in VARIANTS for t in TASKS]


def _print_experiment(base_dir: Path, detail_label: str | None) -> None:
    structure = _detect_structure(base_dir)
    rows      = _build_rows(base_dir, structure)

    loaded = {label: _load_run(run_dir) for label, run_dir in rows}
    loaded = {k: v for k, v in loaded.items() if v is not None}

    if detail_label:
        if detail_label in loaded:
            _print_detail(detail_label, loaded[detail_label])
        else:
            print(f"  Run '{detail_label}' not found in {base_dir.name}")
        return

    print(f"\n{'='*72}")
    print(f"  {base_dir.name}")
    print(f"{'='*72}")
    print(f"  {'run':<32}  {'cyc':>5}  "
          f"{'V_strt':>6}  {'V_now':>5}  {'V_max':>5}  "
          f"{'acc%':>4}  {'cskip':>5}  {'T_fail':>6}  "
          f"{'agt/dsc':<9}  {'cumΔ*':>7}  {'V trajectory (shape only — seeds differ per cycle)→'}")
    print(f"  * cumΔ = sum of paired deltas at each accepted cycle (DIAGNOSTIC ONLY — seeds differ across events)")
    print(f"  {'─'*120}")

    prev_group = None
    for label, run_dir in rows:
        info = loaded.get(label)
        if info is None:
            continue
        group = label.split("/")[0]
        if group != prev_group and prev_group is not None:
            print()
        prev_group = group
        print(_summary_line(label, info))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimisation run analysis")
    parser.add_argument("--opt-dir", default="optimization_runs")
    parser.add_argument("--run-id",  nargs="+", default=None,
                        help="One or more experiment IDs (default: all)")
    parser.add_argument("--detail",  default=None,
                        help="Show full cycle breakdown for one run (e.g. minimal/open)")
    args = parser.parse_args()

    opt_dir = Path(args.opt_dir)

    if args.run_id:
        base_dirs = [opt_dir / rid for rid in args.run_id]
    else:
        def _is_opt_run(d: Path) -> bool:
            return (any((d / v / t / "run_config.json").exists() for v in VARIANTS for t in TASKS)
                    or any((d / c / v / t / "run_config.json").exists()
                           for c in CONSTRAINTS for v in VARIANTS for t in TASKS)
                    or any((d / t / "run_config.json").exists() for t in TASKS))
        base_dirs = sorted(
            [d for d in opt_dir.iterdir() if d.is_dir() and _is_opt_run(d)],
            key=lambda d: d.stat().st_mtime,
        )

    for base_dir in base_dirs:
        if base_dir.exists():
            _print_experiment(base_dir, args.detail)

    print()


if __name__ == "__main__":
    main()
