"""Behaviour Analyser — diagnostic component of the optimisation pipeline.

Reads compressed episode trajectories and attributes failure or success
behaviours to a specific prompt component (Descriptor or Agent).

Typical call sequence
---------------------
1. Caller selects episodes from V using seed_split logic.
2. Caller calls compress_for_ba() to preprocess the selected trajectories.
3. Caller loads current prompts via load_prompts().
4. BehaviourAnalyser.analyse() calls the LLM and returns a BAOutput.

The analyse_from_trajectory() convenience method combines steps 2–4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.llm.client import OpenAIClient
from src.optimization.preprocessing import compress_for_ba

_BA_INSTRUCTIONS_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "babyai" / "behaviour_analyser_instructions.txt"
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class BACandidate:
    module: Literal["descriptor", "actor"]
    step: str | None
    description: str
    suggested_change: str | None = None
    change_type: Literal["add", "modify", "remove"] | None = None
    location: str | None = None


@dataclass
class BAOutput:
    output_type: Literal["failure", "insight", "skip"]
    implicated_module: Literal["descriptor", "actor"] | None
    failure_step: str | None
    characterisation: str
    suggested_change: str | None
    candidate_queue: list[BACandidate] = field(default_factory=list)
    raw_reasoning: str | None = None
    raw_response: str = ""
    parse_error: str | None = None
    change_type: Literal["add", "modify", "remove"] | None = None
    location: str | None = None
    skip_reason: Literal["ambiguous_attribution", "clean_success"] | None = None
    # Token and latency stats from the LLM call
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float | None = None


# ---------------------------------------------------------------------------
# BehaviourAnalyser
# ---------------------------------------------------------------------------

class BehaviourAnalyser:
    """Diagnoses episode trajectories and attributes failures or insights
    to a specific prompt component.

    Args:
        client:        OpenAIClient instance for LLM calls. Must be configured
                       with max_tokens sufficient for staged reasoning — 32768 is
                       recommended. The default HLP_MAX_TOKENS (8092) is too low
                       for multi-episode calls and will cause parse failures.
        pipeline_mode: "with_descriptor" (two-module pipeline) or
                       "monolithic" (single agent, no Descriptor).
                       In monolithic mode, all attributions are to "actor"
                       and the Descriptor sections are omitted from the prompt.
    """

    def __init__(
        self,
        client: OpenAIClient,
        pipeline_mode: Literal["with_descriptor", "monolithic"] = "with_descriptor",
    ):
        self._client = client
        self._pipeline_mode = pipeline_mode
        self._instructions = _BA_INSTRUCTIONS_PATH.read_text()
        self.last_messages: list[dict] | None = None  # set after each analyse() call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        compressed_trajectory: str,
        agent_prompt: str,
        env_prompt: str,
        descriptor_prompt: str | None = None,
        task_prompt: str | None = None,
        hereditary_context: str | None = None,
        module_constraint: Literal["both", "actor", "descriptor"] = "both",
    ) -> BAOutput:
        """Analyse pre-compressed trajectory text.

        Args:
            compressed_trajectory: Output of compress_for_ba().
            agent_prompt:          Current Agent instructions text.
            env_prompt:            Environment layer text.
            descriptor_prompt:     Current Descriptor instructions text.
                                   Required for with_descriptor mode.
            task_prompt:           Optional task-layer text.
            hereditary_context:    Optional prior mutation history summary.
            module_constraint:     Restrict which module the BA may implicate.
                                   "both" (default) = free choice.
                                   "actor" / "descriptor" = forced attribution.

        Returns:
            BAOutput with typed diagnosis and candidate queue.
        """
        messages = self._build_messages(
            compressed_trajectory=compressed_trajectory,
            agent_prompt=agent_prompt,
            env_prompt=env_prompt,
            descriptor_prompt=descriptor_prompt,
            task_prompt=task_prompt,
            hereditary_context=hereditary_context,
            module_constraint=module_constraint,
        )
        self.last_messages = messages
        content, reasoning, usage = self._client.generate_with_reasoning(messages)
        output = self._parse_output(content or "", reasoning)
        output.prompt_tokens     = usage.get("prompt_tokens")
        output.completion_tokens = usage.get("completion_tokens")
        output.finish_reason     = usage.get("finish_reason")
        output.latency_s         = usage.get("latency_s")
        return output

    def analyse_from_trajectory(
        self,
        trajectory_path: Path,
        episode_indices: list[int] | None,
        agent_prompt: str,
        env_prompt: str,
        descriptor_prompt: str | None = None,
        task_prompt: str | None = None,
        hereditary_context: str | None = None,
        module_constraint: Literal["both", "actor", "descriptor"] = "both",
    ) -> BAOutput:
        """Preprocess trajectory file then analyse.

        Convenience method combining compress_for_ba() and analyse().

        Args:
            trajectory_path:  Path to trajectory.jsonl.
            episode_indices:  1-indexed episode numbers to include.
                              None includes all episodes.
        """
        compressed = compress_for_ba(trajectory_path, episode_indices)
        return self.analyse(
            compressed_trajectory=compressed,
            agent_prompt=agent_prompt,
            env_prompt=env_prompt,
            descriptor_prompt=descriptor_prompt,
            task_prompt=task_prompt,
            hereditary_context=hereditary_context,
            module_constraint=module_constraint,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        compressed_trajectory: str,
        agent_prompt: str,
        env_prompt: str,
        descriptor_prompt: str | None,
        task_prompt: str | None,
        hereditary_context: str | None,
        module_constraint: Literal["both", "actor", "descriptor"] = "both",
    ) -> list[dict]:
        sections = [self._instructions]

        sections.append("\n\n---\n\n## Environment\n\n" + env_prompt.strip())

        if task_prompt:
            sections.append("\n\n---\n\n## Task\n\n" + task_prompt.strip())

        if self._pipeline_mode == "with_descriptor":
            sections.append(
                "\n\n---\n\n## Pipeline — exact input/output contracts\n\n"
                "Two modules operate in sequence each step.\n\n"
                "**Descriptor**\n"
                "- Receives: (1) the mission string, (2) the raw scene text from the "
                "environment. The raw scene text includes all visible objects and their "
                "positions, and when the agent is carrying an object it also includes a "
                "carrying line (e.g. 'You carry a grey key'). Nothing else — no position, "
                "no direction, no previous action, no plan.\n"
                "- Produces: a single natural language text summary of the scene, "
                "goal-conditioned on the mission.\n"
                "- Can be responsible for: omitting or misrepresenting objects present in "
                "the raw scene text; hallucinating objects or states not in the raw scene "
                "text; failing to prioritise the mission-relevant object; unhelpful verbosity.\n"
                "- Cannot be responsible for: previous action; agent position or "
                "direction; game mechanics or physics.\n\n"
                "**Agent**\n"
                "- Receives: (1) the Descriptor's output, (2) the mission string, "
                "(3) an explicit carrying line injected by the runner directly from the "
                "environment (belt-and-suspenders, independent of the Descriptor), "
                "(4) the previous action taken, (5) the current plan from the previous step.\n"
                "- Produces: an updated plan and a single action.\n"
                "- Can be responsible for: choosing a wrong action given correct inputs; "
                "failing to update the plan when the situation changes; ignoring or "
                "misinterpreting the Descriptor's output; misusing carrying state or "
                "previous action information.\n"
                "- Cannot be responsible for: the content of the Descriptor's output "
                "(it acts on what it is given).\n\n"
                "**Not attributable to either module:** previous action value (set by the "
                "runner), agent position and direction (environment state), environment "
                "physics, step limit, or cases where the target was simply never visible "
                "during the episode. If the failure appears to stem from one of these, "
                "output Skip.\n\n"
                "**Architectural boundary — must be respected in every SUGGESTED_CHANGE:**\n"
                "The Descriptor and Agent are strictly separated by design. The Descriptor "
                "is a pure perception module: its only job is to observe what is present "
                "in the raw scene and report it faithfully. It must never reason about "
                "task state, task completion, or what the agent should do next. The Agent "
                "is the reasoning module: its job is to interpret the description it "
                "receives and decide actions. It does not parse raw scene data.\n"
                "When writing SUGGESTED_CHANGE for the Descriptor: the proposed rule must "
                "describe what to observe and report — never what to conclude. A rule that "
                "says 'state that the pick-up step is complete' or 'note that the mission "
                "goal has been reached' crosses into Agent territory and is invalid. If "
                "correcting the failure seems to require the Descriptor to reason about "
                "task progress, this is a strong signal that the failure is actually "
                "attributable to the Agent (not updating its plan given correct inputs), "
                "and attribution should be revised accordingly.\n"
                "When writing SUGGESTED_CHANGE for the Agent: the proposed rule must "
                "describe how to reason and act on received descriptions — never how to "
                "reparse or reinterpret raw scene data directly.\n\n"
                "**Verification rule — critical:** before stating that a module said or "
                "omitted something, find the exact text in the trajectory that supports "
                "your claim. Quote it. Do not paraphrase or invent text. If you cannot "
                "find the exact supporting text, do not make the attribution.\n\n"
                "**Vocabulary rule:** SUGGESTED_CHANGE must be written using only the "
                "terms and field names that appear verbatim in the implicated module's "
                "current prompt (shown above). Do NOT use pipeline-level words like "
                "'descriptor', 'agent', 'pipeline', 'module', or 'BA' unless those "
                "exact words already appear in the target module's prompt. "
                "Example: if the Agent is the implicated module, do not write "
                "'if the descriptor indicates X' — instead use whatever the Agent prompt "
                "calls its scene input, e.g. 'if the current description states X' or "
                "'if the observation shows X'. Read the current module prompt carefully "
                "before writing SUGGESTED_CHANGE to find the exact terminology in use."
            )
        else:
            sections.append(
                "\n\n---\n\n## Pipeline — exact input/output contracts\n\n"
                "Single module:\n\n"
                "**Agent**\n"
                "- Receives: (1) the raw scene text, (2) the mission string, "
                "(3) carrying state from the environment runner, (4) the previous action, "
                "(5) the current plan.\n"
                "- Produces: an updated plan and a single action.\n"
                "- Cannot be responsible for: carrying state accuracy, previous action "
                "value, environment physics, or cases where the target was never visible."
            )

        sections.append("\n\n---\n\n## Current Agent Prompt\n\n" + agent_prompt.strip())

        if descriptor_prompt and self._pipeline_mode == "with_descriptor":
            sections.append(
                "\n\n---\n\n## Current Descriptor Prompt\n\n" + descriptor_prompt.strip()
            )

        if hereditary_context:
            sections.append(
                "\n\n---\n\n## Prior Mutation History\n\n" + hereditary_context.strip()
            )

        if module_constraint == "descriptor":
            sections.append(
                "\n\n---\n\n## Module Constraint\n\n"
                "For this run you may ONLY suggest changes to the **Descriptor** module. "
                "Your IMPLICATED_MODULE output must always be `descriptor`.\n\n"
                "If the root cause of failure appears to lie in the Actor, reframe your "
                "analysis: identify what additional or corrected observational information "
                "the Descriptor could have provided that would have enabled the Actor to "
                "avoid the failure. If you genuinely cannot find a valid Descriptor-side "
                "attribution, output Skip."
            )
        elif module_constraint == "actor":
            sections.append(
                "\n\n---\n\n## Module Constraint\n\n"
                "For this run you may ONLY suggest changes to the **Actor** module. "
                "Your IMPLICATED_MODULE output must always be `actor`.\n\n"
                "If the root cause of failure appears to lie in the Descriptor, reframe "
                "your analysis: identify what reasoning or decision-making rule the Actor "
                "could have applied differently given the information it received. "
                "If you genuinely cannot find a valid Actor-side attribution, output Skip."
            )

        system_message = "".join(sections)

        user_message = (
            "Analyse the episode trajectories below following the staged reasoning "
            "process in your instructions.\n\n"
            + compressed_trajectory
            + "\n\nYou have now read all trajectories. Complete your staged reasoning, "
            "then produce exactly one ---OUTPUT--- block as specified. "
            "Do not end your response without the ---OUTPUT--- / ---END_OUTPUT--- delimiters."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, response: str, reasoning: str | None) -> BAOutput:
        # Accept block with or without closing delimiter — model occasionally omits it.
        match = re.search(r"---OUTPUT---(.*?)---END_OUTPUT---", response, re.DOTALL)
        if not match:
            match = re.search(r"---OUTPUT---(.*?)$", response, re.DOTALL)
        if not match:
            return BAOutput(
                output_type="skip",
                implicated_module=None,
                failure_step=None,
                characterisation="Output block not found in BA response.",
                suggested_change=None,
                raw_reasoning=reasoning,
                raw_response=response,
                parse_error="Missing ---OUTPUT--- / ---END_OUTPUT--- delimiters",
            )

        block = match.group(1).strip()

        try:
            raw_type = _extract_field(block, "TYPE").lower()
            if raw_type not in ("failure", "insight", "skip"):
                raise ValueError(f"Unknown TYPE value: {raw_type!r}")

            if raw_type == "skip":
                characterisation = (
                    _extract_field(block, "CHARACTERISATION", required=False)
                    or _extract_field(block, "REASON", required=False)
                    or ""
                )
                skip_reason_str = _extract_field(block, "SKIP_REASON", required=False).lower().strip()
                skip_reason: Literal["ambiguous_attribution", "clean_success"] | None = (
                    skip_reason_str  # type: ignore[assignment]
                    if skip_reason_str in ("ambiguous_attribution", "clean_success")
                    else None
                )
                return BAOutput(
                    output_type="skip",
                    implicated_module=None,
                    failure_step=None,
                    characterisation=characterisation,
                    suggested_change=None,
                    skip_reason=skip_reason,
                    raw_reasoning=reasoning,
                    raw_response=response,
                )

            module_str = _extract_field(block, "MODULE", required=False).lower()
            implicated_module: Literal["descriptor", "actor"] | None = (
                module_str if module_str in ("descriptor", "actor") else None  # type: ignore[assignment]
            )

            step_str = _extract_field(block, "STEP", required=False).strip()
            failure_step: str | None = None
            if step_str and step_str.lower() != "none":
                # Preserve full E{ep}.S{step} identifier for diagnostic use.
                # Fall back to bare integer string if model omits the episode prefix.
                m = re.match(r"(E\d+\.S\d+)", step_str)
                if m:
                    failure_step = m.group(1)
                elif re.match(r"\d+", step_str):
                    failure_step = step_str.split()[0]

            change_type_str = _extract_field(block, "CHANGE_TYPE", required=False).lower().strip()
            change_type: Literal["add", "modify", "remove"] | None = (
                change_type_str if change_type_str in ("add", "modify", "remove") else None  # type: ignore[assignment]
            )

            location: str | None = _extract_field(block, "LOCATION", required=False) or None
            if location and location.strip().lower() == "none":
                location = None

            characterisation = _extract_field(block, "CHARACTERISATION")

            suggested_change: str | None = _extract_field(block, "SUGGESTED_CHANGE", required=False) or None
            if suggested_change and suggested_change.strip().lower() == "none":
                suggested_change = None

            candidate_queue = _parse_candidates(block)

            if change_type is None and raw_type != "skip":
                import logging as _log
                _log.getLogger(__name__).warning("BA output missing CHANGE_TYPE — Mutator will infer from context")
            if location is None and raw_type != "skip":
                import logging as _log
                _log.getLogger(__name__).warning("BA output missing LOCATION — Mutator will infer from context")

            return BAOutput(
                output_type=raw_type,  # type: ignore[arg-type]
                implicated_module=implicated_module,
                failure_step=failure_step,
                change_type=change_type,
                location=location,
                characterisation=characterisation,
                suggested_change=suggested_change,
                candidate_queue=candidate_queue,
                raw_reasoning=reasoning,
                raw_response=response,
            )

        except Exception as exc:
            return BAOutput(
                output_type="skip",
                implicated_module=None,
                failure_step=None,
                characterisation=f"Output block parsing failed: {exc}",
                suggested_change=None,
                raw_reasoning=reasoning,
                raw_response=response,
                parse_error=str(exc),
            )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_field(block: str, field_name: str, required: bool = True) -> str:
    """Extract a named field from the structured output block.

    Handles both single-line values and multi-line values. A field ends
    when the next ALL_CAPS field name followed by a colon is encountered.
    """
    match = re.search(rf"^{field_name}:\s*(.*)$", block, re.MULTILINE)
    if not match:
        if required:
            raise ValueError(f"Missing required field: {field_name}")
        return ""

    first_line = match.group(1).strip()
    start_pos = match.end()

    next_field = re.search(r"^[A-Z_]+:\s*", block[start_pos:], re.MULTILINE)
    end_pos = start_pos + next_field.start() if next_field else len(block)
    continuation = block[start_pos:end_pos].strip()

    if first_line and continuation:
        return (first_line + "\n" + continuation).strip()
    return (first_line or continuation).strip()


def _parse_candidates(block: str) -> list[BACandidate]:
    """Parse the ADDITIONAL_CANDIDATES section into BACandidate objects.

    Each candidate may span two lines:
      - MODULE: ... | STEP: ... | CHARACTERISATION: <text>
        SUGGESTED_CHANGE: <text>
    """
    section = re.search(
        r"ADDITIONAL_CANDIDATES:\s*\n(.*?)(?=\n[A-Z_]+:|---END_OUTPUT---|$)",
        block, re.DOTALL,
    )
    if not section:
        return []

    candidates: list[BACandidate] = []
    raw = section.group(1).strip()

    # Split on candidate boundary lines (start with optional whitespace then "-")
    entries = re.split(r"\n\s*-\s+", raw)
    for entry in entries:
        entry = entry.strip().lstrip("-").strip()
        if not entry:
            continue

        module_m      = re.search(r"MODULE:\s*(descriptor|actor)", entry, re.IGNORECASE)
        step_m        = re.search(r"STEP:\s*(E\d+\.S\d+|\d+|none)", entry, re.IGNORECASE)
        change_type_m = re.search(r"CHANGE_TYPE:\s*(add|modify|remove)", entry, re.IGNORECASE)
        location_m    = re.search(r"LOCATION:\s*(.+?)(?=CHARACTERISATION:|SUGGESTED_CHANGE:|$)", entry, re.DOTALL | re.IGNORECASE)
        char_m        = re.search(r"CHARACTERISATION:\s*(.+?)(?=SUGGESTED_CHANGE:|LOCATION:|$)", entry, re.DOTALL | re.IGNORECASE)
        change_m      = re.search(r"SUGGESTED_CHANGE:\s*(.+?)$", entry, re.DOTALL | re.IGNORECASE)

        module: Literal["descriptor", "actor"] = (
            module_m.group(1).lower() if module_m else "actor"  # type: ignore[assignment]
        )
        step_str = step_m.group(1) if step_m else "none"
        step: str | None = None
        if step_str.lower() != "none":
            m = re.match(r"(E\d+\.S\d+)", step_str)
            if m:
                step = m.group(1)
            elif re.match(r"\d+", step_str):
                step = step_str

        change_type: Literal["add", "modify", "remove"] | None = (
            change_type_m.group(1).lower() if change_type_m else None  # type: ignore[assignment]
        )
        location = location_m.group(1).strip() if location_m else None
        description = char_m.group(1).strip() if char_m else entry.strip()
        suggested_change = change_m.group(1).strip() if change_m else None

        description = re.sub(r"\b(MODULE|STEP|CHANGE_TYPE):\s*\S+\s*\|?\s*", "", description, flags=re.IGNORECASE).strip(" |")

        candidates.append(BACandidate(
            module=module,
            step=step,
            description=description,
            suggested_change=suggested_change,
            change_type=change_type,
            location=location,
        ))

    return candidates
