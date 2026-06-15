# prompts/ — Prompt Files

All LLM prompts live here, organised by environment family. The optimiser reads from and writes back to this directory as it refines prompts across optimisation cycles.

---

## Structure

```
prompts/
  babyai/
    agent_instructions_rich.txt          Actor prompt — guided variant (paper: "guided")
    agent_instructions_minimal.txt       Actor prompt — plain variant (paper: "plain")
    agent_instructions_rich_mt.txt       Actor prompt — guided variant with multi-turn history (h16)
    descriptor_instructions_rich.txt     Descriptor prompt — guided variant
    descriptor_instructions_minimal.txt  Descriptor prompt — plain variant
    balrog_instructions_rich.txt         BALROG baseline prompt — guided variant
    balrog_instructions_minimal.txt      BALROG baseline prompt — plain variant
    behaviour_analyser_instructions.txt  BA system prompt
    mutator_instructions.txt             Mutator system prompt
    environment_layer.txt                Environment physics and action space description
```

---

## Roles

**Actor** (`agent_instructions_*.txt`) — receives the mission, current inventory, descriptor output, and previous plan; outputs a `PLAN` and an `ACTION`.

**Descriptor** (`descriptor_instructions_*.txt`) — receives the mission and raw rule-based scene text from the environment; outputs a mission-focused natural language summary for the actor.

**BALROG baseline** (`balrog_instructions_*.txt`) — monolithic agent prompt used by the BALROG RobustCoTAgent. Receives the full scene text directly (no descriptor stage).

**Behaviour Analyser** (`behaviour_analyser_instructions.txt`) — receives a batch of episode trajectories and diagnoses which module (actor or descriptor) is the bottleneck; outputs a structured change proposal.

**Mutator** (`mutator_instructions.txt`) — receives the BA diagnosis and the current prompt for the targeted module; outputs a revised prompt.

**Environment layer** (`environment_layer.txt`) — describes the environment physics, action space, and movement constraints. Provided to the BA and Mutator at optimisation time. Deliberately withheld from the actor and descriptor at inference time — domain knowledge reaches the runtime agents only through the optimised prompts.

---

## Prompt Variants

| Variant | File suffix | Paper term |
|---------|-------------|------------|
| Guided (handcrafted, richer instructions) | `_rich` | `guided` |
| Plain (minimal instructions) | `_minimal` | `plain` |

---

## How the Optimiser Modifies Prompts

The optimiser never edits these source files. Instead, it writes **incumbent prompt files** to the run log directory (`optimization_runs/.../incumbent_agent_prompt.txt`, `incumbent_descriptor_prompt.txt`). These are loaded by `experiments/run.py` via `--actor-prompt-file` and `--descriptor-prompt-file` flags.

The source files here serve as the **progenitor prompts** — the starting point for optimisation cycle 0.
