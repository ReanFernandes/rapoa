from __future__ import annotations

import json
import math
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Project root: src/optimization/evaluator.py -> src/optimization -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class EvaluatorOutput:
    verdict: Literal["accepted", "rejected", "insufficient_signal"]
    stage: Literal["V", "T"]
    acceptance_rule: str   # "always_accept" | "mean" | "wilcoxon" — computed for logging
    # Statistical fields — always computed when episodes are run; None otherwise
    p_value: float | None
    net_mean_reward: float | None       # mean(challenger) - mean(incumbent)
    n_positive: int | None              # episodes where challenger reward > incumbent
    n_negative: int | None              # episodes where incumbent reward > challenger
    n_tied: int | None                  # d_i == 0, excluded from Wilcoxon test
    # Episode data
    challenger_rewards: list[float]
    incumbent_rewards: list[float]
    run_directory: Path | None
    parse_error: str | None


class Evaluator:
    """
    Evaluates a candidate prompt against the incumbent on a fixed set of seeds.

    Two acceptance rules:
      mean     — simple mean reward improvement threshold only.
      wilcoxon — one-sided exact Wilcoxon signed-rank test on paired rewards,
                 plus a practical reward-improvement threshold.

    Always-accept is not a rule — it is reward_threshold = -inf. Setting
    reward_threshold to float('-inf') forces acceptance regardless of measured
    improvement while still running all episodes and logging full paired stats.

    In all cases, both p_value and net_mean_reward are computed (when episodes
    are run) and logged regardless of the active rule. This makes runs
    retrospectively comparable without re-running episodes.
    """

    def __init__(
        self,
        rule: Literal["mean", "wilcoxon"] = "mean",
        p_threshold: float = 0.05,
        reward_threshold: float = 0.05,   # float('-inf') = always accept
        min_discordant_pairs: int = 4,
    ):
        if rule == "wilcoxon" and not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for the 'wilcoxon' acceptance rule: pip install scipy"
            )
        self.rule = rule
        self.p_threshold = p_threshold
        self.reward_threshold = reward_threshold
        self.min_discordant_pairs = min_discordant_pairs

    def _always_accept(self) -> bool:
        return math.isinf(self.reward_threshold) and self.reward_threshold < 0

    def _rule_label(self) -> str:
        """Human-readable label for logging — preserves 'always_accept' string."""
        return "always_accept" if self._always_accept() else self.rule

    def evaluate(
        self,
        stage: Literal["V", "T"],
        seeds: list[int],
        incumbent_rewards: list[float],
        candidate_prompt: str,
        module: Literal["actor", "descriptor"],
        env: str,
        pipeline: str = "with_descriptor",
        prompt_variant: str = "rich",
        model: str | None = None,
        inference_seed: int | None = None,
        workers: int = 1,
        history_window: int = 16,
        actor_history_window: int | None = None,
        log_dir: Path | str = "logs",
    ) -> EvaluatorOutput:
        """
        Evaluate a candidate prompt against the incumbent on the given seeds.

        Args:
            stage:             "V" for validation pre-filter, "T" for test acceptance.
            seeds:             Env seeds to run the candidate on (must match the seeds
                               used to collect incumbent_rewards).
            incumbent_rewards: Pre-computed per-episode rewards for the incumbent on
                               these exact seeds, in seed-list order. Never re-run.
            candidate_prompt:  Full text of the candidate prompt to evaluate.
            module:            Which prompt this candidate replaces ("actor" or "descriptor").
            env:               Environment ID passed to run.py --env.
            pipeline:          Pipeline variant (default: "with_descriptor").
            prompt_variant:    Prompt variant for the non-overridden prompt file.
            model:             Model override for run.py (None = use env var).
            inference_seed:    LLM sampling seed for run.py.
            workers:           Parallel episode workers for run.py.
            log_dir:           Root log directory; challenger episodes are written to a
                               unique subdirectory inside here.
        """
        # always_accept: fall through for BOTH V and T stages — run episodes and
        # collect full paired stats for logging. Verdict is forced to "accepted"
        # at the decision point below regardless of the measured improvement.

        # --- Run candidate episodes ---
        eval_log_dir = Path(log_dir) / f"eval_{stage}_{uuid.uuid4().hex[:8]}"
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        try:
            challenger_rewards, run_directory = self._run_candidate(
                seeds=seeds,
                candidate_prompt=candidate_prompt,
                module=module,
                env=env,
                pipeline=pipeline,
                prompt_variant=prompt_variant,
                model=model,
                inference_seed=inference_seed,
                workers=workers,
                history_window=history_window,
                actor_history_window=actor_history_window,
                eval_log_dir=eval_log_dir,
            )
        except Exception as exc:
            return EvaluatorOutput(
                verdict="rejected",
                stage=stage,
                acceptance_rule=self._rule_label(),
                p_value=None,
                net_mean_reward=None,
                n_positive=None,
                n_negative=None,
                n_tied=None,
                challenger_rewards=[],
                incumbent_rewards=list(incumbent_rewards),
                run_directory=None,
                parse_error=str(exc),
            )

        if len(challenger_rewards) != len(incumbent_rewards):
            return EvaluatorOutput(
                verdict="rejected",
                stage=stage,
                acceptance_rule=self._rule_label(),
                p_value=None,
                net_mean_reward=None,
                n_positive=None,
                n_negative=None,
                n_tied=None,
                challenger_rewards=challenger_rewards,
                incumbent_rewards=list(incumbent_rewards),
                run_directory=run_directory,
                parse_error=(
                    f"Reward count mismatch: expected {len(incumbent_rewards)}, "
                    f"got {len(challenger_rewards)}"
                ),
            )

        # --- Compute paired statistics ---
        diffs = [c - i for c, i in zip(challenger_rewards, incumbent_rewards)]
        n_positive = sum(1 for d in diffs if d > 0)
        n_negative = sum(1 for d in diffs if d < 0)
        n_tied = sum(1 for d in diffs if d == 0)
        net_mean_reward = sum(diffs) / len(diffs)

        if self.rule == "wilcoxon" and n_positive + n_negative < self.min_discordant_pairs and not self._always_accept():
            return EvaluatorOutput(
                verdict="insufficient_signal",
                stage=stage,
                acceptance_rule=self._rule_label(),
                p_value=None,
                net_mean_reward=net_mean_reward,
                n_positive=n_positive,
                n_negative=n_negative,
                n_tied=n_tied,
                challenger_rewards=challenger_rewards,
                incumbent_rewards=list(incumbent_rewards),
                run_directory=run_directory,
                parse_error=None,
            )

        # --- Compute p-value (always, regardless of active rule, for logging) ---
        p_value: float | None = None
        nonzero_diffs = [d for d in diffs if d != 0]
        try:
            # method="exact" works for n <= 25; fall back to "approx" for larger samples
            method = "exact" if len(nonzero_diffs) <= 25 else "approx"
            result = _scipy_wilcoxon(nonzero_diffs, alternative="greater", method=method)
            p_value = float(result.pvalue)
        except Exception:
            pass  # p_value stays None; mean rule still applies

        # --- Apply acceptance criterion ---
        if self._always_accept():
            accepted = True   # verdict forced; stats above are for observation only
        elif self.rule == "wilcoxon":
            accepted = (
                p_value is not None
                and p_value < self.p_threshold
                and net_mean_reward >= self.reward_threshold
            )
        else:  # mean
            accepted = net_mean_reward >= self.reward_threshold

        return EvaluatorOutput(
            verdict="accepted" if accepted else "rejected",
            stage=stage,
            acceptance_rule=self._rule_label(),
            p_value=p_value,
            net_mean_reward=net_mean_reward,
            n_positive=n_positive,
            n_negative=n_negative,
            n_tied=n_tied,
            challenger_rewards=challenger_rewards,
            incumbent_rewards=list(incumbent_rewards),
            run_directory=run_directory,
            parse_error=None,
        )

    # ------------------------------------------------------------------

    def _run_candidate(
        self,
        seeds: list[int],
        candidate_prompt: str,
        module: Literal["actor", "descriptor"],
        env: str,
        pipeline: str,
        prompt_variant: str,
        model: str | None,
        inference_seed: int | None,
        workers: int,
        history_window: int,
        actor_history_window: int | None,
        eval_log_dir: Path,
    ) -> tuple[list[float], Path]:
        """Write candidate prompt to a temp file, call run.py, return (rewards, run_dir)."""

        prompt_file = eval_log_dir / "candidate_prompt.txt"
        prompt_file.write_text(candidate_prompt)

        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "experiments" / "run.py"),
            "--env", env,
            "--pipeline", pipeline,
            "--seed-list", *[str(s) for s in seeds],
            "--prompt-variant", prompt_variant,
            "--log-dir", str(eval_log_dir),
            "--workers", str(workers),
            "--no-gif",
        ]
        if module == "actor":
            cmd += ["--agent-prompt-file", str(prompt_file)]
        elif module == "descriptor":
            cmd += ["--descriptor-prompt-file", str(prompt_file)]
        if pipeline == "balrog_baseline":
            cmd += ["--history-window", str(history_window)]
        if pipeline == "with_descriptor" and actor_history_window is not None:
            cmd += ["--agent-multi-turn", "--history-window", str(actor_history_window)]
        if model:
            cmd += ["--model", model]
        if inference_seed is not None:
            cmd += ["--inference-seed", str(inference_seed)]

        subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT))

        summaries = list(eval_log_dir.rglob("run_summary.json"))
        if not summaries:
            raise RuntimeError(
                f"run.py completed but no run_summary.json found under {eval_log_dir}"
            )

        summary_path = summaries[0]
        run_directory = summary_path.parent

        with open(summary_path) as f:
            summary = json.load(f)

        rewards = [ep["total_reward"] for ep in summary["episodes"]]
        return rewards, run_directory
