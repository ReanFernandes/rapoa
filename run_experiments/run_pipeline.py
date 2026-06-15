"""
Master pipeline runner — calls optimise.py for each (task × variant) pair,
then evaluates each experiment on the fresh-seed protocol as soon as it completes.

All experiments within a config run in parallel. Within each experiment, all
(task) subprocesses run in parallel. Eval is chained after opt completes.
Re-running with resume=true skips phases already complete.

Usage:
    python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml
    python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml --dry-run

Config format: see conf/configs/ for examples and conf/config.yaml for the template.

Experiment types (set in the experiments list):
    spa     — SPA descriptor+actor pipeline, configurable acceptance
    balrog  — BALROG baseline pipeline

Directory layout under log_root:
    {campaign}/{slug}/{task}/
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON  = PROJECT_ROOT / ".venv" / "bin" / "python"
RUN_SCRIPTS  = PROJECT_ROOT / "run_experiments"
CONF_ROOT    = PROJECT_ROOT / "conf"

sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_slug(cfg) -> str:
    """Sanitised model name for use in log paths. openai/gpt-oss-20b → openai--gpt-oss-20b."""
    name = OmegaConf.select(cfg, "models.strong.name") or "unknown-model"
    return name.replace("/", "--")


# ── Task config loading ────────────────────────────────────────────────────────

def _load_task_configs(cfg) -> list[dict]:
    """Load task YAML configs for all tasks listed in cfg.tasks."""
    env_name = cfg.env.name
    tasks = []
    for task_name in cfg.tasks:
        path = CONF_ROOT / f"task/{env_name}/{task_name}.yaml"
        tc   = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        tasks.append({"name": task_name, **tc})
    return tasks


# ── Slug generation ────────────────────────────────────────────────────────────

def _slug(exp) -> str:
    """Stable, human-readable run identifier for an experiment entry."""
    exp = OmegaConf.to_container(exp, resolve=True) if not isinstance(exp, dict) else exp
    t = exp["type"]

    if t == "spa":
        acc     = exp.get("acceptance", {})
        thresh  = acc.get("reward_threshold", 0.05)
        vstrat  = acc.get("validation_strategy", "validation_bag")
        mc      = exp.get("module_constraint", "both")
        variant = exp.get("prompt_variant", "rich")

        if mc not in ("both", "random"):
            return f"module_ablation_{mc}_{variant}"

        actor_hw = exp.get("actor_history_window")
        hw_suffix = f"_h{actor_hw}" if actor_hw is not None else ""

        if mc == "random":
            if math.isinf(thresh) and thresh < 0:
                return f"spa_random_always_accept_{variant}{hw_suffix}"
            vstrat_short = "valbag" if vstrat == "validation_bag" else "trainsig"
            tstr = f"{int(round(thresh * 100)):03d}"
            return f"spa_random_{vstrat_short}_t{tstr}_{variant}{hw_suffix}"

        if math.isinf(thresh) and thresh < 0:
            return f"spa_always_accept_{variant}{hw_suffix}"

        vstrat_short = "valbag" if vstrat == "validation_bag" else "trainsig"
        tstr = f"{int(round(thresh * 100)):03d}"
        return f"spa_mean_{vstrat_short}_t{tstr}_{variant}{hw_suffix}"

    if t == "balrog":
        agent   = exp.get("agent", {})
        hl      = agent.get("history_length", 16)
        variant = exp.get("prompt_variant", "minimal")
        acc     = exp.get("acceptance", {})
        thresh  = acc.get("reward_threshold", float("-inf"))
        vstrat  = acc.get("validation_strategy", "validation_bag")
        if math.isinf(float(thresh)) and float(thresh) < 0:
            return f"balrog_h{hl}_{variant}"   # always-accept: no suffix (backward compat)
        vstrat_short = "valbag" if vstrat == "validation_bag" else "trainsig"
        tstr = f"{int(round(float(thresh) * 100)):03d}"
        return f"balrog_h{hl}_{vstrat_short}_t{tstr}_{variant}"

    return t


# ── Command builders ───────────────────────────────────────────────────────────

def _common_opt_flags(cfg) -> list[str]:
    opt = cfg.optimisation
    flags = [
        "--opt-cycles",        str(opt.opt_cycles),
        "--ba-episodes",       str(opt.ba_episodes),
        "--max-skip-resample", str(opt.max_skip_resample),
        "--t-size",            str(opt.t_size),
        "--env-seed",          str(opt.env_seed),
        "--inference-seed",    str(opt.inference_seed),
        "--max-steps",         str(cfg.rollout.max_steps),
        "--workers",           str(cfg.rollout.workers),
        "--log-dir",           cfg.log_root,
    ]
    model = OmegaConf.select(cfg, "models.strong.name")
    if model:
        flags += ["--model", model]
    return flags


def _build_optimise_cmds(exp, campaign: str, cfg, tasks: list[dict]) -> list[tuple[str, list[str]]]:
    """Return list of (label, cmd) — one optimise.py call per task."""
    exp  = OmegaConf.to_container(exp, resolve=True) if not isinstance(exp, dict) else exp
    t    = exp["type"]
    slug = _slug(exp)
    base = [str(VENV_PYTHON), str(PROJECT_ROOT / "experiments" / "optimise.py")]
    common = _common_opt_flags(cfg)
    cmds: list[tuple[str, list[str]]] = []

    if t == "spa":
        acc     = exp.get("acceptance", {})
        thresh  = acc.get("reward_threshold", 0.05)
        vstrat  = acc.get("validation_strategy", "validation_bag")
        rule    = acc.get("rule", "mean")
        min_dp  = str(acc.get("min_discordant_pairs", 4))
        p_thr   = str(acc.get("p_threshold", 0.05))
        mc_cfg  = exp.get("module_constraint", "both")
        variant = exp.get("prompt_variant", "rich")

        mc_cli    = mc_cfg  # "both" | "actor" | "descriptor"
        actor_hw  = exp.get("actor_history_window")

        for task in tasks:
            run_id = f"{campaign}/{slug}/{task['name']}"
            cmd = base + [
                "--env",                  task["gym_id"],
                "--pipeline",             "with_descriptor",
                "--prompt-variant",       variant,
                "--rule",                 rule,
                "--validation-strategy",  vstrat,
                f"--reward-threshold={thresh}",
                "--min-discordant-pairs", min_dp,
                "--p-threshold",          p_thr,
                "--module-constraint",    mc_cli,
                "--run-id",               run_id,
                *common,
            ]
            if actor_hw is not None:
                cmd += ["--actor-history-window", str(actor_hw)]
            cmds.append((f"{slug}/{task['name']}", cmd))

    elif t == "balrog":
        acc     = exp.get("acceptance", {})
        thresh  = acc.get("reward_threshold", float("-inf"))
        vstrat  = acc.get("validation_strategy", "validation_bag")
        rule    = acc.get("rule", "mean")
        min_dp  = str(acc.get("min_discordant_pairs", 4))
        p_thr   = str(acc.get("p_threshold", 0.05))
        variant = exp.get("prompt_variant", "minimal")
        hl      = str(exp.get("agent", {}).get("history_length", 16))

        for task in tasks:
            run_id = f"{campaign}/{slug}/{task['name']}"
            cmds.append((f"{slug}/{task['name']}", base + [
                "--env",                  task["gym_id"],
                "--pipeline",             "balrog_baseline",
                "--prompt-variant",       variant,
                "--history-window",       hl,
                "--rule",                 rule,
                "--validation-strategy",  vstrat,
                f"--reward-threshold={thresh}",
                "--min-discordant-pairs", min_dp,
                "--p-threshold",          p_thr,
                "--module-constraint",    "actor",
                "--run-id",               run_id,
                *common,
            ]))

    else:
        raise ValueError(
            f"Unknown experiment type: {t!r}. Must be one of: spa, balrog"
        )

    return cmds


def _build_eval_cmds(exp, campaign: str, cfg) -> list[list[str]]:
    """Return list of eval commands for this experiment."""
    slug   = _slug(exp)
    script = str(RUN_SCRIPTS / "run_optimised_eval.sh")
    ev     = cfg.evaluation
    seeds  = [str(s) for s in ev.inference_seeds]
    common = [
        "--episodes",        str(ev.episodes),
        "--env-seed",        str(ev.env_seed),
        "--inference-seeds", *seeds,
        "--workers",         str(ev.workers),
        "--opt-runs-dir",    cfg.log_root,
        "--log-root",        cfg.eval_log_root,
    ]
    return [["bash", script, "--run-ids", f"{campaign}/{slug}", *common]]


# ── Completion checks ──────────────────────────────────────────────────────────

def _opt_is_complete(cmds: list[tuple[str, list[str]]], cfg) -> bool:
    log_root  = PROJECT_ROOT / cfg.log_root
    opt_cycles = cfg.optimisation.opt_cycles
    for _, cmd in cmds:
        run_id = cmd[cmd.index("--run-id") + 1]
        log    = log_root / run_id / "optimisation_log.jsonl"
        if not log.exists():
            return False
        with open(log) as f:
            done = sum(1 for line in f
                       if json.loads(line).get("record_type") == "opt_cycle")
        if done < opt_cycles:
            return False
    return True


def _eval_is_complete(eval_run_ids: list[str], cfg) -> bool:
    n_seeds    = len(cfg.evaluation.inference_seeds)
    n_tasks    = len(cfg.tasks)
    n_expected = n_tasks * n_seeds
    for eval_run_id in eval_run_ids:
        eval_root = PROJECT_ROOT / cfg.eval_log_root / eval_run_id
        if not eval_root.exists():
            return False
        found = len(list(eval_root.rglob("run_summary.json")))
        if found < n_expected:
            return False
    return True


# ── Parallel execution ─────────────────────────────────────────────────────────

def _run_parallel(cmds: list[tuple[str, list[str]]], log_dir: Path, dry_run: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for label, cmd in cmds:
            print(f"      [dry-run] {label}:\n        {' '.join(cmd)}")
        return

    def _one(label: str, cmd: list[str]) -> None:
        safe  = label.replace("/", "_")
        lpath = log_dir / f"{safe}.log"
        lpath.parent.mkdir(parents=True, exist_ok=True)
        with open(lpath, "w") as fh:
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                               stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"{label} exited {r.returncode} — see {lpath}")

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        futures = {pool.submit(_one, lbl, cmd): lbl for lbl, cmd in cmds}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                errors.append(str(e))

    if errors:
        raise RuntimeError(
            f"{len(errors)} subprocess(es) failed:\n" + "\n".join(errors)
        )


# ── Experiment runner ──────────────────────────────────────────────────────────

def run_experiment(exp, campaign: str, cfg, tasks: list[dict], log_dir: Path, dry_run: bool) -> None:
    slug      = _slug(exp)
    opt_cmds  = _build_optimise_cmds(exp, campaign, cfg, tasks)
    eval_cmds = _build_eval_cmds(exp, campaign, cfg)
    eval_ids  = [cmd[cmd.index("--run-ids") + 1] for cmd in eval_cmds]

    print(f"\n  [{slug}]  {len(opt_cmds)} optimise subprocess(es)")

    if cfg.resume and _opt_is_complete(opt_cmds, cfg):
        print(f"  [{slug}]  opt complete — skipping")
    else:
        print(f"  [{slug}]  starting optimisation ...")
        t0 = time.perf_counter()
        _run_parallel(opt_cmds, log_dir / "opt_logs", dry_run)
        if not dry_run:
            print(f"  [{slug}]  optimisation done  ({(time.perf_counter() - t0) / 3600:.1f}h)")

    if cfg.resume and _eval_is_complete(eval_ids, cfg):
        print(f"  [{slug}]  eval complete — skipping")
        return

    print(f"  [{slug}]  starting eval ...")
    if dry_run:
        for cmd in eval_cmds:
            print(f"      [dry-run] {' '.join(cmd)}")
        return

    t0 = time.perf_counter()
    for i, cmd in enumerate(eval_cmds):
        elog = log_dir / f"eval_{i}.log"
        with open(elog, "w") as fh:
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                               stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"[{slug}] eval failed — see {elog}")
    print(f"  [{slug}]  eval done  ({(time.perf_counter() - t0) / 3600:.1f}h)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a full optimisation + eval pipeline from a structured config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dotpath overrides (applied after loading the config file):
  optimisation.opt_cycles=5
  rollout.workers=30
  optimisation.env_seed=99

Examples:
  python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml
  python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml --dry-run
  python run_experiments/run_pipeline.py conf/configs/main/spa_guided.yaml optimisation.opt_cycles=5
  python run_experiments/run_pipeline.py conf/configs/smoke_test.yaml --campaign-override babyai/openai--gpt-oss-20b/my_run_20260520_143022
""",
    )
    ap.add_argument("config", help="Path to config YAML (e.g. conf/configs/main/spa_guided.yaml)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without running anything")
    ap.add_argument("--campaign-override",
                    help="Override the auto-generated campaign prefix (useful for re-runs)")
    # parse_known_args collects dotpath overrides without swallowing --dry-run and --campaign-override
    args, overrides = ap.parse_known_args()

    cfg = load_config(args.config)

    # Apply any dotpath overrides from the command line
    if overrides:
        dotlist = [o for o in overrides if "=" in o]
        if dotlist:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
            from src.config.schema import validate_config
            validate_config(cfg)
            print(f"Overrides applied: {dotlist}")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_slug   = cfg.env.name                          # e.g. babyai
    model_slug = _model_slug(cfg)                      # e.g. openai--gpt-oss-20b
    run_slug   = f"{cfg.run_name}_{timestamp}"         # e.g. spa_guided_20260520_143022
    campaign   = args.campaign_override or f"{env_slug}/{model_slug}/{run_slug}"
    tasks      = _load_task_configs(cfg)

    experiments = cfg.experiments

    print(f"\nCampaign     : {campaign}")
    print(f"Env          : {cfg.env.name}")
    print(f"Tasks        : {[t['name'] for t in tasks]}")
    print(f"Opt cycles   : {cfg.optimisation.opt_cycles}")
    print(f"V bag        : {cfg.optimisation.ba_episodes} x {cfg.optimisation.max_skip_resample} = "
          f"{cfg.optimisation.ba_episodes * cfg.optimisation.max_skip_resample}")
    print(f"Resume       : {cfg.resume}")
    print(f"\nExperiments ({len(experiments)}):")
    for exp in experiments:
        print(f"  {_slug(exp)}")
    print()

    pipeline_log_dir = PROJECT_ROOT / "logs_pipeline" / campaign
    pipeline_log_dir.mkdir(parents=True, exist_ok=True)

    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=len(experiments)) as pool:
        futures = {
            pool.submit(
                run_experiment,
                exp,
                campaign,
                cfg,
                tasks,
                pipeline_log_dir / _slug(exp),
                args.dry_run,
            ): _slug(exp)
            for exp in experiments
        }
        for fut in as_completed(futures):
            slug = futures[fut]
            try:
                fut.result()
                print(f"\n  ✓ {slug} complete")
            except Exception as e:
                print(f"\n  ✗ {slug} FAILED: {e}")
                errors.append((slug, str(e)))

    print(f"\n{'=' * 60}")
    if errors:
        print(f"Pipeline finished with {len(errors)} failure(s):")
        for slug, msg in errors:
            print(f"  {slug}: {msg}")
        sys.exit(1)
    else:
        print(f"All {len(experiments)} experiment(s) complete.")
        print(f"Opt results  : {cfg.log_root}/{campaign}/")
        print(f"Eval results : {cfg.eval_log_root}/{campaign}/")


if __name__ == "__main__":
    main()
