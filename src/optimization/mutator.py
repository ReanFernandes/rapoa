"""Mutator — applies a Behaviour Analyser diagnosis to a module's prompt.

Receives the five actionable fields from a BAOutput or BACandidate, the
current prompt text for the implicated module, the environment layer, optional
task context, and optional hereditary log context. Produces a fully revised
prompt and a structured rationale entry for the hereditary log.

Typical call sequence
---------------------
1. Outer loop checks BAOutput.output_type != "skip" and parse_error is None.
2. Outer loop loads prompts via load_prompts() and env_prompt from environment_layer.
3. Outer loop renders the hereditary log via render_hereditary_context().
4. Mutator.mutate() is called with the five diagnosis fields + context.
5. Outer loop writes MutatorOutput.revised_prompt to a candidate file.
6. Outer loop appends a hereditary log entry with outcome="pending".
7. Evaluator runs the candidate prompt; outer loop updates outcome.

Parse failure handling
----------------------
If the LLM output cannot be parsed, parse_error is set and revised_prompt is
empty. The caller should retry (same ceiling as the BA). There is no skip
concept — the Mutator always attempts to produce a revised prompt.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.llm.client import OpenAIClient

_MUTATOR_INSTRUCTIONS_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "babyai" / "mutator_instructions.txt"
)

_MODULE_LABELS = {
    "actor":      "Actor",
    "descriptor": "Descriptor",
}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class MutatorOutput:
    revised_prompt: str            # complete revised prompt — written to candidate file
    section: str                   # exact section header edited — for hereditary log
    change: str                    # tight one-liner — for hereditary log
    principle: str                 # unconstrained — for hereditary log
    conflict_note: str | None      # set if anchor imprecise or conflict resolved
    raw_response: str = ""
    raw_reasoning: str | None = None
    parse_error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float | None = None


# ---------------------------------------------------------------------------
# Mutator class
# ---------------------------------------------------------------------------

class Mutator:

    def __init__(self, client: OpenAIClient) -> None:
        self._client = client
        self._instructions = _MUTATOR_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
        self.last_messages: list[dict] | None = None  # set after each mutate() call

    def mutate(
        self,
        module: Literal["descriptor", "actor"],
        change_type: Literal["add", "modify", "remove"],
        location: str,
        characterisation: str,
        suggested_change: str,
        current_prompt: str,
        env_prompt: str,
        task_layer: str | None = None,
        hereditary_context: str | None = None,
    ) -> MutatorOutput:
        """Apply a BA diagnosis to current_prompt and return the revised version.

        Args:
            module:           The implicated module ("actor" or "descriptor").
            change_type:      The kind of edit ("add", "modify", or "remove").
            location:         Section header + anchor text from the BA output.
            characterisation: BA explanation of what went wrong.
            suggested_change: BA proposal for what to change.
            current_prompt:   Full text of the module's current prompt.
            env_prompt:       Environment layer text (action space, physics). Read-only.
            task_layer:       Task-specific context (read-only). Optional.
            hereditary_context: Rendered prior mutation history, or None.

        Returns:
            MutatorOutput with revised_prompt and rationale fields.
            On parse failure, parse_error is set and revised_prompt is empty.
        """
        messages = self._build_messages(
            module=module,
            change_type=change_type,
            location=location,
            characterisation=characterisation,
            suggested_change=suggested_change,
            current_prompt=current_prompt,
            env_prompt=env_prompt,
            task_layer=task_layer,
            hereditary_context=hereditary_context,
        )
        self.last_messages = messages
        t0 = time.monotonic()
        content, reasoning, usage = self._client.generate_with_reasoning(messages)
        latency = time.monotonic() - t0

        output = self._parse_output(content or "", reasoning)
        output.prompt_tokens     = usage.get("prompt_tokens")
        output.completion_tokens = usage.get("completion_tokens")
        output.finish_reason     = usage.get("finish_reason")
        output.latency_s         = latency
        return output

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        module: Literal["descriptor", "actor"],
        change_type: Literal["add", "modify", "remove"],
        location: str,
        characterisation: str,
        suggested_change: str,
        current_prompt: str,
        env_prompt: str,
        task_layer: str | None,
        hereditary_context: str | None,
    ) -> list[dict]:
        label = _MODULE_LABELS.get(module, module.capitalize())

        sections = [self._instructions]
        sections.append(
            "\n\n---\n\n## Environment Layer\n\n" + env_prompt.strip()
        )
        sections.append(
            f"\n\n---\n\n## Current {label} Prompt\n\n{current_prompt.strip()}"
        )
        if task_layer:
            sections.append(
                "\n\n---\n\n## Task Context\n\n" + task_layer.strip()
            )
        if hereditary_context:
            sections.append(
                "\n\n---\n\n## Prior Mutation History\n\n" + hereditary_context.strip()
            )

        system_message = "".join(sections)

        user_message = (
            f"Apply the following diagnosis to the current {label} prompt.\n\n"
            f"MODULE: {module}\n"
            f"CHANGE_TYPE: {change_type}\n"
            f"LOCATION:\n{location.strip()}\n\n"
            f"CHARACTERISATION:\n{characterisation.strip()}\n\n"
            f"SUGGESTED_CHANGE:\n{suggested_change.strip()}\n\n"
            "Complete your staged reasoning, then produce both output blocks exactly "
            "as specified. Do not end your response without both "
            "---END_REVISED_PROMPT--- and ---END_RATIONALE--- delimiters."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, response: str, reasoning: str | None) -> MutatorOutput:
        # --- REVISED_PROMPT block ---
        revised_prompt = _extract_block(
            response,
            open_delim="---REVISED_PROMPT---",
            close_delim="---END_REVISED_PROMPT---",
        )
        if revised_prompt is None:
            return MutatorOutput(
                revised_prompt="",
                section="",
                change="",
                principle="",
                conflict_note=None,
                raw_response=response,
                raw_reasoning=reasoning,
                parse_error="Missing ---REVISED_PROMPT--- / ---END_REVISED_PROMPT--- block",
            )

        # --- RATIONALE block ---
        rationale_block = _extract_block(
            response,
            open_delim="---RATIONALE---",
            close_delim="---END_RATIONALE---",
        )
        if rationale_block is None:
            return MutatorOutput(
                revised_prompt="",
                section="",
                change="",
                principle="",
                conflict_note=None,
                raw_response=response,
                raw_reasoning=reasoning,
                parse_error="Missing ---RATIONALE--- / ---END_RATIONALE--- block",
            )

        try:
            section      = _extract_field(rationale_block, "SECTION")
            change       = _extract_field(rationale_block, "CHANGE")
            principle    = _extract_field(rationale_block, "PRINCIPLE")
            conflict_raw = _extract_field(rationale_block, "CONFLICT_NOTE", required=False)
            conflict_note: str | None = (
                None
                if not conflict_raw or conflict_raw.strip().lower() == "none"
                else conflict_raw.strip()
            )
        except ValueError as exc:
            return MutatorOutput(
                revised_prompt="",
                section="",
                change="",
                principle="",
                conflict_note=None,
                raw_response=response,
                raw_reasoning=reasoning,
                parse_error=str(exc),
            )

        return MutatorOutput(
            revised_prompt=revised_prompt.strip(),
            section=section.strip(),
            change=change.strip(),
            principle=principle.strip(),
            conflict_note=conflict_note,
            raw_response=response,
            raw_reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_block(text: str, open_delim: str, close_delim: str) -> str | None:
    """Extract content between open_delim and close_delim.

    Falls back to content from open_delim to end of string if close_delim
    is absent, stopping at the next block delimiter if one is found.
    """
    start = text.find(open_delim)
    if start == -1:
        return None
    content_start = start + len(open_delim)
    end = text.find(close_delim, content_start)
    if end == -1:
        next_block = re.search(r"\n---[A-Z_]+---", text[content_start:])
        if next_block:
            return text[content_start : content_start + next_block.start()]
        return text[content_start:]
    return text[content_start:end]


def _extract_field(block: str, field_name: str, required: bool = True) -> str:
    """Extract a named field from the RATIONALE block.

    Handles single-line and multi-line values. A field ends when the next
    ALL_CAPS field name followed by a colon is encountered, or at end of block.
    """
    match = re.search(rf"^{field_name}:\s*(.*)$", block, re.MULTILINE)
    if not match:
        if required:
            raise ValueError(f"Missing required field: {field_name}")
        return ""

    first_line = match.group(1).strip()
    start_pos  = match.end()

    next_field = re.search(r"^[A-Z_]+:\s*", block[start_pos:], re.MULTILINE)
    end_pos    = start_pos + next_field.start() if next_field else len(block)
    continuation = block[start_pos:end_pos].strip()

    if first_line and continuation:
        return (first_line + "\n" + continuation).strip()
    return (first_line or continuation).strip()
