"""Hereditary log — append-only JSONL record of optimisation cycle events.

One entry per inner loop cycle. Written by the outer loop (optimise.py)
after each BA → Mutator → Evaluation sequence.

Usage
-----
from src.optimization.hereditary import append_entry, update_outcome, render_hereditary_context

# After Mutator produces output:
append_entry(log_path, {
    "cycle": 1,
    "prompt_version": prompt_hash(current_prompt),
    "module": "actor",
    "change_type": "modify",
    "ba_characterisation": output.characterisation,
    "section": mutator_output.section,
    "change": mutator_output.change,
    "principle": mutator_output.principle,
    "conflict_note": mutator_output.conflict_note,
    "outcome": "pending",
})

# After evaluation:
update_outcome(log_path, cycle=1, outcome="accepted")

# Before calling BA or Mutator:
context = render_hereditary_context(log_path, max_entries=10)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Entry lifecycle
# ---------------------------------------------------------------------------

def append_entry(log_path: Path, entry: dict) -> None:
    """Append one entry to the JSONL log. Creates the file if it does not exist."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_outcome(
    log_path: Path,
    cycle: int,
    outcome: Literal["accepted", "rejected"],
) -> None:
    """Update the outcome field of the entry with the given cycle number.

    Rewrites the full file — safe for the ≤10 entry budget.
    Silently does nothing if the cycle is not found.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return
    entries = _load_entries(log_path)
    updated = False
    for entry in entries:
        if entry.get("cycle") == cycle:
            entry["outcome"] = outcome
            updated = True
    if updated:
        with log_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_entries(log_path: Path) -> list[dict]:
    """Return all entries from the log, oldest first."""
    return _load_entries(Path(log_path))


# ---------------------------------------------------------------------------
# Rendering for prompt injection
# ---------------------------------------------------------------------------

def render_hereditary_context(log_path: Path, max_entries: int = 10) -> str | None:
    """Render the last max_entries entries as formatted text for prompt injection.

    Returns None if the log does not exist or is empty.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    entries = _load_entries(log_path)
    if not entries:
        return None

    recent = entries[-max_entries:]
    lines: list[str] = []
    for e in recent:
        cycle      = e.get("cycle", "?")
        module     = e.get("module", "?")
        change_type = e.get("change_type", "?")
        outcome    = e.get("outcome", "pending")
        lines.append(f"Cycle {cycle} | {module} | {change_type} | outcome: {outcome}")

        ba_char = e.get("ba_characterisation", "")
        if ba_char:
            lines.append(f"  BA: {ba_char}")

        section = e.get("section", "")
        if section:
            lines.append(f"  Section: {section}")

        change = e.get("change", "")
        if change:
            lines.append(f"  Change: {change}")

        principle = e.get("principle", "")
        if principle:
            # Indent each line of a multi-line principle
            for pl in principle.strip().splitlines():
                lines.append(f"  Principle: {pl}" if not pl.startswith("  ") else pl)

        conflict = e.get("conflict_note")
        if conflict and conflict.lower() != "none":
            lines.append(f"  Conflict note: {conflict}")

        lines.append("")  # blank line between entries

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Versioning helper
# ---------------------------------------------------------------------------

def prompt_hash(prompt_text: str, length: int = 8) -> str:
    """Return a short SHA-256 hex prefix identifying this prompt version."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_entries(log_path: Path) -> list[dict]:
    entries = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries
