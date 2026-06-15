"""Structured config schema — Python dataclasses for all config blocks.

These dataclasses define the expected types and defaults for every config
field. They serve as documentation and as a validation layer: loading a
config through load_config() will fail early on type mismatches or unknown
keys rather than propagating silent errors into the pipeline.

Usage:
    from src.config.schema import validate_config
    cfg = load_config("conf/configs/main/spa_guided.yaml")
    validate_config(cfg)   # raises on type errors or missing required fields
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Leaf dataclasses ──────────────────────────────────────────────────────────

@dataclass
class LLMRoleConfig:
    """Generation parameters for one LLM role (actor, descriptor, BA, mutator)."""
    model: str = "strong"       # Key into the models registry
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class ModelEntry:
    """One entry in the model registry."""
    name: str                   # Model identifier passed to the OpenAI client
    endpoint: str | None = None # Base URL; None = dynamic discovery


@dataclass
class InventoryConfig:
    type: str                   # single_slot | full_list | none
    source: str | None = None   # How to read inventory (e.g. env.unwrapped.carrying)


@dataclass
class ObsConfig:
    """Observation field paths for an environment."""
    mission: str | None = None
    direction: str | None = None
    scene_text: str | None = None


@dataclass
class AcceptanceConfig:
    rule: str = "mean"                      # mean | wilcoxon
    reward_threshold: float = 0.05          # -inf = always accept
    min_discordant_pairs: int = 4
    p_threshold: float = 0.05              # wilcoxon only
    validation_strategy: str = "validation_bag"  # validation_bag | train_signal


# ── Block dataclasses ─────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """Environment structure — what is structurally true of an environment family."""
    name: str = "babyai"
    adapter_class: str = "src.environment.minigrid.MiniGridAdapter"
    tasks: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    obs: ObsConfig = field(default_factory=ObsConfig)
    inventory: InventoryConfig = field(default_factory=lambda: InventoryConfig(type="none"))


@dataclass
class ModelsConfig:
    """Model registry — maps key names to model identity and endpoint."""
    strong: ModelEntry = field(default_factory=lambda: ModelEntry(name="openai/gpt-oss-20b"))
    # Add more model keys here as needed (e.g. fast, descriptor_only)


@dataclass
class AgentConfig:
    """Agent architecture selection and behavioral parameters."""
    type: str = "descriptor_actor"  # descriptor_actor | balrog
    history_length: int | None = None  # Only for balrog; None for descriptor_actor


@dataclass
class RolloutConfig:
    """Episode execution mechanics."""
    max_steps: int = 64
    workers: int = 20
    actor: LLMRoleConfig = field(default_factory=LLMRoleConfig)
    descriptor: LLMRoleConfig = field(default_factory=LLMRoleConfig)


@dataclass
class OptimisationConfig:
    """Optimisation loop parameters."""
    opt_cycles: int = 20
    ba_episodes: int = 6
    max_skip_resample: int = 3      # v_bag = ba_episodes x max_skip_resample
    t_size: int = 20
    env_seed: int = 42
    inference_seed: int = 1
    ba: LLMRoleConfig = field(default_factory=lambda: LLMRoleConfig(max_tokens=32768))
    mutator: LLMRoleConfig = field(default_factory=lambda: LLMRoleConfig(max_tokens=8092))


@dataclass
class EvaluationConfig:
    """Fresh evaluation phase parameters."""
    env_seed: int = 500
    inference_seeds: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7])
    episodes: int = 20
    workers: int = 20       # Resolved from ${rollout.workers} in YAML
    actor: LLMRoleConfig = field(default_factory=LLMRoleConfig)
    descriptor: LLMRoleConfig = field(default_factory=LLMRoleConfig)


@dataclass
class ExperimentConfig:
    """One experimental condition in the experiments list."""
    type: str = "spa"                       # spa | balrog
    prompt_variant: str = "rich"            # rich | minimal
    module_constraint: str = "both"         # both | actor | descriptor (spa only)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    agent: AgentConfig | None = None        # balrog override only


@dataclass
class PipelineConfig:
    """Top-level config — ties all blocks together."""
    run_name: str = "pipeline_run"
    resume: bool = False
    log_root: str = "optimization_runs"
    eval_log_root: str = "logs_fresh_eval_optimised"
    tasks: list[str] = field(default_factory=list)  # Defaults to env.tasks
    experiments: list[ExperimentConfig] = field(default_factory=list)
    env: EnvConfig = field(default_factory=EnvConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optimisation: OptimisationConfig = field(default_factory=OptimisationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ── Validation ────────────────────────────────────────────────────────────────

_VALID_EXPERIMENT_TYPES   = {"spa", "balrog"}
_VALID_PROMPT_VARIANTS    = {"rich", "minimal"}
_VALID_MODULE_CONSTRAINTS = {"both", "actor", "descriptor", "random"}
_VALID_ACCEPTANCE_RULES   = {"mean", "wilcoxon"}
_VALID_VSTRATS            = {"validation_bag", "train_signal"}
_VALID_AGENT_TYPES        = {"descriptor_actor", "balrog"}
_VALID_PIPELINE_VARIANTS  = {"with_descriptor", "balrog_baseline"}


def validate_config(cfg: Any) -> None:
    """Validate a loaded OmegaConf config against the schema.

    Raises ValueError with a descriptive message on the first problem found.
    Call after load_config() to catch misconfigured YAMLs before the run starts.
    """
    from omegaconf import OmegaConf

    def _get(obj, *keys, default=None):
        for k in keys:
            try:
                obj = obj[k] if isinstance(obj, dict) else getattr(obj, k)
            except (KeyError, AttributeError):
                return default
        return obj

    errors: list[str] = []

    # run_name must be set
    if not _get(cfg, "run_name"):
        errors.append("run_name is required")

    # rollout
    workers = _get(cfg, "rollout", "workers")
    if workers is not None and workers < 1:
        errors.append(f"rollout.workers must be >= 1, got {workers}")

    max_steps = _get(cfg, "rollout", "max_steps")
    if max_steps is not None and max_steps < 1:
        errors.append(f"rollout.max_steps must be >= 1, got {max_steps}")

    # optimisation
    ba_ep  = _get(cfg, "optimisation", "ba_episodes", default=6)
    max_sk = _get(cfg, "optimisation", "max_skip_resample", default=3)
    if ba_ep < 1:
        errors.append(f"optimisation.ba_episodes must be >= 1, got {ba_ep}")
    if max_sk < 1:
        errors.append(f"optimisation.max_skip_resample must be >= 1, got {max_sk}")

    # agent
    agent_type = _get(cfg, "agent", "type")
    if agent_type not in _VALID_AGENT_TYPES:
        errors.append(f"agent.type must be one of {_VALID_AGENT_TYPES}, got {agent_type!r}")

    if agent_type == "balrog":
        hl = _get(cfg, "agent", "history_length")
        if hl is not None and hl < 1:
            errors.append(f"agent.history_length must be >= 1, got {hl}")

    # experiments
    experiments = _get(cfg, "experiments") or []
    if not experiments:
        errors.append("experiments list is empty — nothing to run")

    for i, exp in enumerate(experiments):
        prefix = f"experiments[{i}]"
        exp_type = _get(exp, "type")
        if exp_type not in _VALID_EXPERIMENT_TYPES:
            errors.append(f"{prefix}.type must be one of {_VALID_EXPERIMENT_TYPES}, got {exp_type!r}")

        pv = _get(exp, "prompt_variant")
        if pv not in _VALID_PROMPT_VARIANTS:
            errors.append(f"{prefix}.prompt_variant must be one of {_VALID_PROMPT_VARIANTS}, got {pv!r}")

        if exp_type == "spa":
            mc = _get(exp, "module_constraint", default="both")
            if mc not in _VALID_MODULE_CONSTRAINTS:
                errors.append(f"{prefix}.module_constraint must be one of {_VALID_MODULE_CONSTRAINTS}, got {mc!r}")

            rule = _get(exp, "acceptance", "rule", default="mean")
            if rule not in _VALID_ACCEPTANCE_RULES:
                errors.append(f"{prefix}.acceptance.rule must be one of {_VALID_ACCEPTANCE_RULES}, got {rule!r}")

            vstrat = _get(exp, "acceptance", "validation_strategy", default="validation_bag")
            if vstrat not in _VALID_VSTRATS:
                errors.append(f"{prefix}.acceptance.validation_strategy must be one of {_VALID_VSTRATS}, got {vstrat!r}")

        if exp_type == "balrog":
            exp_agent = _get(exp, "agent")
            if exp_agent is not None:
                at = _get(exp_agent, "type")
                if at and at not in _VALID_AGENT_TYPES:
                    errors.append(f"{prefix}.agent.type must be one of {_VALID_AGENT_TYPES}, got {at!r}")

    if errors:
        raise ValueError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
