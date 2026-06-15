"""Config loader — OmegaConf-based composition for the conf/ directory system.

Usage:
    from src.config import load_config
    cfg = load_config("conf/configs/main/spa_guided.yaml")

    # Access structured fields
    cfg.env.name          # "babyai"
    cfg.rollout.workers   # 20
    cfg.optimisation.opt_cycles  # 20
    cfg.evaluation.inference_seeds  # [2, 3, 4, 5, 6, 7]
    cfg.experiments       # list of experiment dicts
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf, DictConfig

CONF_ROOT = Path(__file__).resolve().parents[2] / "conf"


def load_config(config_path: str | Path) -> DictConfig:
    """Load a runnable config, composing base blocks from conf/ then merging user overrides.

    The user config (e.g. conf/configs/main/spa_guided.yaml) declares which base blocks to use via
    top-level keys: env, models, agent. All other block defaults (rollout,
    optimisation, evaluation) are always loaded from conf/{block}/default.yaml
    and can be overridden inline.

    Interpolations (${rollout.workers} etc.) are resolved after merging.
    """
    user = OmegaConf.load(config_path)

    # These keys select which base configs to load — they are not merged into the tree
    env_name    = user.pop("env",    "babyai")
    models_name = user.pop("models", "cluster")
    agent_name  = user.pop("agent",  "descriptor_actor")

    env_cfg  = OmegaConf.load(CONF_ROOT / f"env/{env_name}.yaml")
    mod_cfg  = OmegaConf.load(CONF_ROOT / f"models/{models_name}.yaml")
    agt_cfg  = OmegaConf.load(CONF_ROOT / f"agent/{agent_name}.yaml")
    rol_cfg  = OmegaConf.load(CONF_ROOT / "rollout/default.yaml")
    opt_cfg  = OmegaConf.load(CONF_ROOT / "optimisation/default.yaml")
    eva_cfg  = OmegaConf.load(CONF_ROOT / "evaluation/default.yaml")

    base = OmegaConf.create({
        "env":          env_cfg,
        "models":       mod_cfg,
        "agent":        agt_cfg,
        "rollout":      rol_cfg,
        "optimisation": opt_cfg,
        "evaluation":   eva_cfg,
    })

    # User keys merged on top — nested dicts are deep-merged, not replaced
    cfg = OmegaConf.merge(base, user)

    # tasks: if not overridden in user config, default to env.tasks
    if "tasks" not in user:
        OmegaConf.update(cfg, "tasks", cfg.env.tasks, merge=False)

    OmegaConf.resolve(cfg)

    from src.config.schema import validate_config
    validate_config(cfg)

    return cfg
