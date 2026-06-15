# tools/ — Diagnostic and Monitoring Utilities

Scripts for monitoring live runs, inspecting trajectories, and debugging the BA → Mutator pipeline. All scripts are run from the repository root.

---

| Script | Description |
|--------|-------------|
| `check_setup.py` | Verify the environment after a fresh install — checks BALROG, MiniGrid, and LLM server connectivity. Run this first on a new machine. |
| `opt_progress.py` | Live optimisation progress tracker — shows cycle completion, V trajectory sparkline, and acceptance rates across all active runs. |
| `progress.py` | Live fresh eval progress tracker — shows per-campaign, per-slug, per-task completion and success rates, cross-referenced against `optimization_runs/`. |
| `analyse_runs.py` | Deeper optimisation run analysis — V start/now/max, delta from start, and per-cycle acceptance rates as a complement to `opt_progress.py`. |
| `analyse_episode.py` | Run the Behaviour Analyser on a completed trajectory and print the structured diagnosis — useful for verifying BA attributions before committing to a full run. |
| `preview_preprocessing.py` | Show the compressed trajectory text that the BA will receive for a given run, with a token estimate and episode summary. |
| `test_mutator.py` | Run the full BA → Mutator pipeline on a trajectory and save the complete trace (prompts, reasoning, diff, revised prompt) to `~/ba_mutator_traces/`. |
| `summarise_run.py` | Print a one-line-per-step summary of a trajectory — quickly see whether the agent was making progress, looping, or failing to parse. |
| `animate_episode.py` | Replay a recorded episode and save it as an animated GIF by re-running the environment with the original seed. |
| `timing_report.py` | Compute wall-clock times and LLM token counts across optimisation runs — used for the paper's compute cost reporting. |
| `generate_task_prompts.py` | LLM-assisted generation of the task-specific prompt section for a new BabyAI environment family. Useful when extending to new tasks. |
