"""BabyAI episode runner — unified entry point for all pipeline variants.

Usage:
    # With descriptor (default)
    python experiments/run.py --env BabyAI-GoToRedBall-v0 --episodes 5

    # Multi-turn agent
    python experiments/run.py --agent-multi-turn --episodes 5

    # Full flag list
    python experiments/run.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid/BabyAI envs
from PIL import Image, ImageDraw, ImageFont

import types

from balrog.agents.robust_cot import RobustCoTAgent
from balrog.environments.babyai_text import BabyAITextCleanLangWrapper
from balrog.prompt_builder.history import HistoryPromptBuilder
from src.environment.minigrid import ACTION_NAMES  # used only in _save_episode_gif fallback
from src.llm.client import OpenAIClient
from src.llm.balrog_adapter import BALROGClientAdapter
from src.llm.server_utils import find_all_gpu_servers, start_endpoint_watcher
from src.pipeline.descriptor_agent import DescriptorAgent
from src.pipeline.parsing import extract_plan_and_action, action_to_index
from src.utils.config import load_prompts, HLP_MODEL_ID
from src.logging.episode_logger import EpisodeLogger
from src.logging.run_directory import parse_env_id, create_run_directory, save_run_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run BabyAI episodes")
    parser.add_argument("--env",                    default="BabyAI-GoToRedBall-v0", help="Environment ID")
    parser.add_argument("--pipeline",               default="with_descriptor",
                        choices=["with_descriptor", "balrog_baseline"],
                        help="Pipeline variant (default: with_descriptor)")
    parser.add_argument("--episodes",               type=int, default=10,   help="Number of episodes to run")
    parser.add_argument("--max-steps-per-episode",  type=int, default=64,   help="Max steps per episode")
    parser.add_argument("--max-total-steps",        type=int, default=None, help="Global step budget (optional)")
    parser.add_argument("--render",                 default=None,           help="Render mode: 'human' or None")
    parser.add_argument("--env-seed",               type=int, default=42,   help="Base seed; episode i uses env-seed+i")
    parser.add_argument("--seed-list",              type=int, nargs="+",   default=None,
                        help="Explicit list of env seeds, one per episode. Overrides --env-seed and --episodes.")
    parser.add_argument("--inference-seed",         type=int, default=None, help="Seed for LLM sampling (required — controls inference randomness)")
    parser.add_argument("--model",                  default=None,           help="Override HLP_MODEL_ID env var")
    parser.add_argument("--agent-multi-turn",       action="store_true", default=False,
                        help="Agent accumulates conversation history across steps")
    parser.add_argument("--descriptor-multi-turn",  action="store_true", default=False,
                        help="Descriptor accumulates conversation history (with_descriptor only)")
    parser.add_argument("--history-window",         type=int, default=16,
                        help="Max turn-pairs kept in multi-turn history (default: 16)")
    parser.add_argument("--prompt-variant",          default="rich",
                        choices=["rich", "minimal"],
                        help="Prompt variant to load: rich (handcrafted) or minimal (BALROG-style)")
    parser.add_argument("--reasoning",              action="store_true", default=True,
                        help="Declare reasoning enabled on inference server — encoded in log path only")
    parser.add_argument("--agent-prompt-file",      default=None,
                        help="Path to a text file whose contents replace the default agent instructions")
    parser.add_argument("--descriptor-prompt-file", default=None,
                        help="Path to a text file whose contents replace the default descriptor instructions (with_descriptor only)")
    parser.add_argument("--log-dir",                default="logs")
    parser.add_argument("--no-gif",                action="store_true", default=False,
                        help="Skip GIF rendering; write episode_NNN.done sentinel instead. "
                             "Do not use during optimisation (opt_progress.py needs per-episode signals).")
    parser.add_argument("--workers",               type=int, default=1,
                        help="Number of parallel episode workers (default: 1 = sequential)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GIF helpers (shared between both pipelines)
# ---------------------------------------------------------------------------

def _stamp_step(frame: Image.Image, step: int) -> Image.Image:
    """Return a copy of frame with 'Step N' stamped in the top-left corner."""
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    label = f"Step {step}"
    x, y = 4, 4
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), label, fill=(0, 0, 0), font=font)
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return frame


def _make_env(env_id: str, render_mode=None):
    """Create a BabyAI env, using rejection sampling for MixedTrainLocal subtypes.

    For 'BabyAI-MixedTrainLocal-v0/<task_type>', repeatedly instantiates the base
    env until action_kinds matches the requested subtype. The task type is set at
    __init__ time and is independent of the seed passed to reset().
    """
    if "/" in env_id:
        base_id, task_type = env_id.split("/", 1)
        while True:
            env = gym.make(base_id, render_mode=render_mode)
            if env.unwrapped.action_kinds[0].replace(" ", "_") == task_type:
                return env
            env.close()
    return gym.make(env_id, render_mode=render_mode)


def _save_episode_gif(env_id: str, seed: int, actions: list[str], out_path: Path, adapter=None, frame_ms: int = 400):
    """Replay episode actions and save as an animated GIF."""
    replay_env = _make_env(env_id, render_mode="rgb_array")
    replay_env.reset(seed=seed)
    frames = [_stamp_step(Image.fromarray(replay_env.render()), 0)]
    for step, action_name in enumerate(actions, start=1):
        _names = adapter.action_names if adapter else ACTION_NAMES
        action_idx = _names.index(action_name)
        _, _, terminated, truncated, _ = replay_env.step(action_idx)
        frames.append(_stamp_step(Image.fromarray(replay_env.render()), step))
        if terminated or truncated:
            break
    replay_env.close()
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)


# ---------------------------------------------------------------------------
# Episode loop — with_descriptor
# ---------------------------------------------------------------------------

def _run_episode_with_descriptor(
    ep: int, args, env, agent: DescriptorAgent,
    run_directory: Path, trajectory_buffer: list,
) -> dict:
    episode_seed = args.episode_seeds[ep]
    obs, info = env.reset(seed=episode_seed)
    agent.reset()
    episode_t0 = time.perf_counter()

    print(f"\n{'=' * 50}")
    print(f"Episode {ep + 1} / {args.episodes}  [seed {episode_seed}]")
    print(f"{'=' * 50}")

    episode_reward = 0.0
    prev_action_name = None
    episode_actions: list[str] = []
    episode_parse_failures = 0
    episode_agent_pt = episode_agent_ct = 0
    episode_agent_lat = 0.0
    episode_desc_pt = episode_desc_ct = 0
    episode_desc_lat = 0.0

    with EpisodeLogger(run_directory, ep + 1, args.env, episode_seed) as logger:
        for step in range(args.max_steps_per_episode):
            agent_pos = tuple(env.unwrapped.agent_pos)
            info = info or {}
            info["carrying"] = args.adapter.get_inventory_text(env)

            action = agent.act(obs, info=info, agent_pos=agent_pos, prev_action=prev_action_name)

            if agent.last_parse_failed:
                prev_action_name = "go forward (default — last response could not be parsed)"
                episode_parse_failures += 1
            else:
                prev_action_name = args.adapter.action_names[action]
            episode_actions.append(args.adapter.action_names[action])

            au = agent.last_agent_usage or {}
            du = agent.last_descriptor_usage or {}
            episode_agent_pt  += au.get("prompt_tokens")     or 0
            episode_agent_ct  += au.get("completion_tokens") or 0
            episode_agent_lat += au.get("latency_s")         or 0.0
            episode_desc_pt   += du.get("prompt_tokens")     or 0
            episode_desc_ct   += du.get("completion_tokens") or 0
            episode_desc_lat  += du.get("latency_s")         or 0.0

            if args.adapter.action_names[action] == "done":
                log.info("Agent declared done at step %d — terminating as failure.", step)
                _write_trajectory(trajectory_buffer, {
                    "episode": ep + 1, "type": "summary",
                    "total_reward": episode_reward, "total_steps": step + 1,
                    "success": False, "terminated_by": "agent_done",
                })
                logger.log_episode_summary(total_reward=episode_reward, total_steps=step + 1)
                break

            new_obs, reward, terminated, truncated, info = env.step(args.adapter.action_names[action])
            episode_reward += reward

            new_agent_pos = tuple(env.unwrapped.agent_pos)
            blocked_fwd = (args.adapter.action_names[action] == "go forward" and new_agent_pos == agent_pos)
            scene_text = args.adapter.get_scene_text(obs)

            _dir_names = args.adapter.direction_names
            _write_trajectory(trajectory_buffer, {
                "episode": ep + 1, "step": step,
                "agent_pos": [int(x) for x in agent_pos],
                "direction": _dir_names[obs["direction"]] if obs["direction"] < len(_dir_names) else str(obs["direction"]),
                "mission": args.adapter.get_mission(obs),
                "scene_text": scene_text,
                "descriptor_output": agent.last_descriptor_output,
                "plan": agent.plan,
                "action": args.adapter.action_names[action],
                "parse_failed": agent.last_parse_failed,
                "reward": reward, "terminated": terminated, "truncated": truncated,
                "blocked_fwd": blocked_fwd,
                "agent_prompt_tokens":          au.get("prompt_tokens"),
                "agent_completion_tokens":      au.get("completion_tokens"),
                "agent_finish_reason":          au.get("finish_reason"),
                "agent_latency_s":              au.get("latency_s"),
                "descriptor_prompt_tokens":     du.get("prompt_tokens"),
                "descriptor_completion_tokens": du.get("completion_tokens"),
                "descriptor_finish_reason":     du.get("finish_reason"),
                "descriptor_latency_s":         du.get("latency_s"),
            })

            logger.log_step(
                step_number=step,
                raw_observation=obs,
                full_grid_encoded=env.unwrapped.grid.encode(),
                agent_position=tuple(env.unwrapped.agent_pos),
                agent_direction=env.unwrapped.agent_dir,
                agent_representation=agent.combined_log_entry(),
                llm_response=agent.last_llm_response,
                action_name=args.adapter.action_names[action],
                reward=reward, terminated=terminated, truncated=truncated,
            )
            obs = new_obs

            _print_step(step, args.adapter.action_names[action], reward, terminated, truncated)

            if terminated or truncated:
                break
            if args.max_total_steps and _check_global_budget(args, step):
                break

        steps_taken = step + 1
        episode_wall_time = round(time.perf_counter() - episode_t0, 3)
        logger.log_episode_summary(total_reward=episode_reward, total_steps=steps_taken)
        _write_trajectory(trajectory_buffer, {
            "episode": ep + 1, "type": "summary",
            "total_reward": episode_reward, "total_steps": steps_taken,
            "success": episode_reward > 0,
            "wall_time_s": episode_wall_time,
        })

    return {
        "episode": ep + 1,
        "success": episode_reward > 0,
        "total_steps": steps_taken,
        "total_reward": episode_reward,
        "wall_time_s": episode_wall_time,
        "parse_failure_count": episode_parse_failures,
        "actions": episode_actions,
        "agent_prompt_tokens": episode_agent_pt,
        "agent_completion_tokens": episode_agent_ct,
        "agent_latency_s": round(episode_agent_lat, 3),
        "descriptor_prompt_tokens": episode_desc_pt,
        "descriptor_completion_tokens": episode_desc_ct,
        "descriptor_latency_s": round(episode_desc_lat, 3),
    }


# ---------------------------------------------------------------------------
# Episode loop — balrog_baseline (RobustCoTAgent, no descriptor)
# ---------------------------------------------------------------------------

def _run_episode_balrog_baseline(
    ep: int, args, env, agent: RobustCoTAgent,
    run_directory: Path, trajectory_buffer: list,
) -> dict:
    episode_seed = args.episode_seeds[ep]
    obs, info = env.reset(seed=episode_seed)
    agent.reset()
    episode_t0 = time.perf_counter()
    balrog_prompt = (
        args.agent_prompt_override
        or load_prompts(prompt_variant=args.prompt_variant).get("balrog_instructions", "")
    )
    balrog_prompt = balrog_prompt.replace("{mission}", args.adapter.get_mission(obs))
    agent.prompt_builder.update_instruction_prompt(balrog_prompt)

    print(f"\n{'=' * 50}")
    print(f"Episode {ep + 1} / {args.episodes}  [seed {episode_seed}]")
    print(f"{'=' * 50}")

    episode_reward = 0.0
    prev_action_str: str | None = None
    episode_actions: list[str] = []
    episode_parse_failures = 0
    episode_pt = episode_ct = 0
    episode_lat = 0.0

    with EpisodeLogger(run_directory, ep + 1, args.env, episode_seed) as logger:
        for step in range(args.max_steps_per_episode):
            agent_pos = tuple(env.unwrapped.agent_pos)

            response = agent.act(obs, prev_action=prev_action_str)

            action_str = response.completion
            parse_failed = action_str not in args.adapter.action_names
            if parse_failed:
                log.warning("BALROG agent returned invalid action %r, defaulting to 'go forward'", action_str)
                action_str = "go forward"
                episode_parse_failures += 1

            episode_actions.append(action_str)
            prev_action_str = action_str

            episode_pt  += response.input_tokens or 0
            episode_ct  += response.output_tokens or 0
            episode_lat += agent.client.last_usage.get("latency_s") or 0.0

            new_obs, reward, terminated, truncated, info = env.step(action_str)
            episode_reward += reward

            new_agent_pos = tuple(env.unwrapped.agent_pos)
            blocked_fwd = (action_str == "go forward" and new_agent_pos == agent_pos)

            _write_trajectory(trajectory_buffer, {
                "episode": ep + 1, "step": step,
                "agent_pos": [int(x) for x in agent_pos],
                "direction": args.adapter.direction_names[obs["direction"]] if obs["direction"] < len(args.adapter.direction_names) else str(obs["direction"]),
                "raw_direction": int(obs["direction"]),
                "mission": args.adapter.get_mission(obs),
                "scene_text": args.adapter.get_scene_text(obs),
                "action": action_str,
                "parse_failed": parse_failed,
                "reward": reward, "terminated": terminated, "truncated": truncated,
                "blocked_fwd": blocked_fwd,
                "agent_prompt_tokens":     response.input_tokens,
                "agent_completion_tokens": response.output_tokens,
                "agent_finish_reason":     response.stop_reason,
                "agent_latency_s":         agent.client.last_usage.get("latency_s"),
            })

            sent_msgs = agent.client.last_messages
            prompt_log = "\n\n---\n\n".join(
                f"[{m['role'].upper()}]\n{m['content']}" for m in sent_msgs
            ) if sent_msgs else "(prompt not captured)"

            lm_reasoning = agent.client.last_reasoning
            full_response = response.reasoning  # full CoT text; .completion is extracted action only
            response_log = (
                (f"[ REASONING ]\n\n{lm_reasoning}\n\n[ RESPONSE ]\n\n{full_response}")
                if lm_reasoning else full_response
            )

            logger.log_step(
                step_number=step,
                raw_observation=obs,
                full_grid_encoded=env.unwrapped.grid.encode(),
                agent_position=tuple(env.unwrapped.agent_pos),
                agent_direction=env.unwrapped.agent_dir,
                agent_representation=prompt_log,
                llm_response=response_log,
                action_name=action_str,
                reward=reward, terminated=terminated, truncated=truncated,
            )
            obs = new_obs

            _print_step(step, action_str, reward, terminated, truncated)

            if terminated or truncated:
                break

        steps_taken = step + 1
        episode_wall_time = round(time.perf_counter() - episode_t0, 3)
        logger.log_episode_summary(total_reward=episode_reward, total_steps=steps_taken)
        _write_trajectory(trajectory_buffer, {
            "episode": ep + 1, "type": "summary",
            "total_reward": episode_reward, "total_steps": steps_taken,
            "success": episode_reward > 0,
            "wall_time_s": episode_wall_time,
        })

    return {
        "episode": ep + 1,
        "success": episode_reward > 0,
        "total_steps": steps_taken,
        "total_reward": episode_reward,
        "wall_time_s": episode_wall_time,
        "parse_failure_count": episode_parse_failures,
        "actions": episode_actions,
        "agent_prompt_tokens": episode_pt,
        "agent_completion_tokens": episode_ct,
        "agent_latency_s": round(episode_lat, 3),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_trajectory(buffer: list, record: dict) -> None:
    """Append a trajectory record to the per-episode in-memory buffer."""
    buffer.append(record)


def _flush_trajectory(path: Path, records: list[dict]) -> None:
    """Write a completed episode's buffered records to disk (main thread only)."""
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _print_step(step: int, action: str, reward: float, terminated: bool, truncated: bool) -> None:
    print(f"  step {step:3d}: {action:>10s}  reward={reward:.3f}", end="")
    if terminated:
        print("  [TERMINATED]", end="")
    if truncated:
        print("  [TRUNCATED]", end="")
    print()


def _check_global_budget(args, step: int) -> bool:
    """Returns True if the global step budget has been reached."""
    return bool(args.max_total_steps and step + 1 >= args.max_total_steps)


def _write_run_summary(run_directory: Path, args, model_name: str, conversation_mode: str,
                       pipeline: str, episodes_summary: list[dict]) -> None:
    n = len(episodes_summary)
    successes = [e for e in episodes_summary if e["success"]]
    summary: dict = {
        "success_rate":              round(len(successes) / n, 3) if n else 0,
        "mean_steps":                round(sum(e["total_steps"] for e in episodes_summary) / n, 2) if n else 0,
        "mean_steps_successful":     round(sum(e["total_steps"] for e in successes) / len(successes), 2) if successes else None,
        "mean_reward":               round(sum(e["total_reward"] for e in episodes_summary) / n, 4) if n else 0,
        "total_parse_failures":      sum(e["parse_failure_count"] for e in episodes_summary),
        "total_agent_prompt_tokens": sum(e["agent_prompt_tokens"] for e in episodes_summary),
        "total_agent_completion_tokens": sum(e["agent_completion_tokens"] for e in episodes_summary),
        "total_agent_latency_s":     round(sum(e["agent_latency_s"] for e in episodes_summary), 3),
    }
    if pipeline == "with_descriptor":
        summary["total_descriptor_prompt_tokens"] = sum(e.get("descriptor_prompt_tokens", 0) for e in episodes_summary)
        summary["total_descriptor_completion_tokens"] = sum(e.get("descriptor_completion_tokens", 0) for e in episodes_summary)
        summary["total_descriptor_latency_s"] = round(sum(e.get("descriptor_latency_s", 0.0) for e in episodes_summary), 3)

    with open(run_directory / "run_summary.json", "w") as f:
        json.dump({
            "env": args.env,
            "model": model_name,
            "pipeline": pipeline,
            "conversation_mode": conversation_mode,
            "reasoning": args.reasoning,
            "n_episodes": n,
            "summary": summary,
            "episodes": episodes_summary,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    if args.seed_list is not None:
        args.episode_seeds = args.seed_list
        args.episodes = len(args.seed_list)
    else:
        args.episode_seeds = [args.env_seed + ep for ep in range(args.episodes)]

    args.agent_prompt_override = (
        Path(args.agent_prompt_file).read_text() if args.agent_prompt_file else None
    )
    args.descriptor_prompt_override = (
        Path(args.descriptor_prompt_file).read_text() if args.descriptor_prompt_file else None
    )

    minigrid.register_minigrid_envs()

    # Resolve LLM endpoints — cluster GPU nodes take priority, HLP_API_BASE as fallback
    cluster_endpoints = find_all_gpu_servers()
    if cluster_endpoints:
        endpoints = cluster_endpoints
        model_name = args.model or "gpt-oss-20b"
        print(f"Cluster GPU nodes found ({len(endpoints)}): {endpoints}")
    else:
        endpoints = None  # OpenAIClient falls back to HLP_API_BASE
        model_name = args.model or HLP_MODEL_ID
        print("No cluster nodes reachable — using HLP_API_BASE")

    if cluster_endpoints:
        def _on_endpoints_change(new_endpoints: list[str]) -> None:
            if not new_endpoints:
                return
            OpenAIClient.update_endpoints(new_endpoints)

        start_endpoint_watcher(_on_endpoints_change)
        print("Endpoint watcher started — edit ~/hlp_ports.txt to add/remove nodes mid-run")

    def _make_client(**extra) -> OpenAIClient:
        kwargs = {"endpoints": endpoints} if endpoints else {}
        if args.model:
            kwargs["model"] = args.model
        elif cluster_endpoints:
            kwargs["model"] = "gpt-oss-20b"
        if args.inference_seed is not None:
            kwargs["inference_seed"] = args.inference_seed
        kwargs.update(extra)
        return OpenAIClient(**kwargs)

    env_family, task = parse_env_id(args.env)

    # Load env config and instantiate adapter dynamically
    import importlib
    from omegaconf import OmegaConf as _OC
    _CONF_ROOT = Path(__file__).resolve().parents[1] / "conf"
    _env_cfg_path = _CONF_ROOT / f"env/{env_family.lower()}.yaml"
    if _env_cfg_path.exists():
        _env_cfg = _OC.load(_env_cfg_path)
        _mod_path, _cls_name = _env_cfg.adapter_class.rsplit(".", 1)
        _adapter_cls = getattr(importlib.import_module(_mod_path), _cls_name)
    else:
        from src.environment.minigrid import MiniGridAdapter as _adapter_cls
    args.adapter = _adapter_cls()

    if args.pipeline == "balrog_baseline":
        #conversation_mode = "multi_turn"
        conversation_mode = f"history_{args.history_window}step"

    elif args.agent_multi_turn and args.pipeline == "with_descriptor" and args.descriptor_multi_turn:
        conversation_mode = "multi_turn"
    elif args.agent_multi_turn:
        conversation_mode = "agent_multi_turn"
    elif args.pipeline == "with_descriptor" and args.descriptor_multi_turn:
        conversation_mode = "desc_multi_turn"
    else:
        conversation_mode = "single_turn"

    run_directory = create_run_directory(
        base_log_directory=Path(args.log_dir),
        env_family=env_family,
        task=task,
        model=model_name,
        pipeline=args.pipeline,
        prompt_variant=args.prompt_variant,
        conversation_mode=conversation_mode,
        reasoning=args.reasoning,
        inference_seed=args.inference_seed,
    )
    save_run_config(run_directory, {
        **vars(args),
        "descriptor": args.pipeline == "with_descriptor",
        "conversation_mode": conversation_mode,
        "llm_endpoints": endpoints or ["HLP_API_BASE"],
    })

    trajectory_file = run_directory / "trajectory.jsonl"
    episodes_summary: list[dict] = []

    def _make_worker_env():
        return BabyAITextCleanLangWrapper(_make_env(args.env, render_mode=args.render or None))

    def _make_worker_agent():
        if args.pipeline == "with_descriptor":
            return DescriptorAgent(
                agent_client_factory=_make_client,
                agent_multi_turn=args.agent_multi_turn,
                descriptor_multi_turn=args.descriptor_multi_turn,
                history_window=args.history_window,
                prompt_variant=args.prompt_variant,
                agent_prompt_override=args.agent_prompt_override,
                descriptor_prompt_override=args.descriptor_prompt_override,
            )
        elif args.pipeline == "balrog_baseline":
            _balrog_client = BALROGClientAdapter(_make_client())
            _balrog_prompt_builder = HistoryPromptBuilder(
                max_text_history=args.history_window,
                max_image_history=0,
                max_cot_history=1,
            )
            _balrog_config = types.SimpleNamespace(
                agent=types.SimpleNamespace(remember_cot=True)
            )
            return RobustCoTAgent(
                client_factory=lambda: _balrog_client,
                prompt_builder=_balrog_prompt_builder,
                config=_balrog_config,
            )
        else:
            return _make_client()

    def _run_episode_worker(ep: int) -> tuple[int, list[dict], dict]:
        """Run one episode in a worker thread. Returns (ep, trajectory_records, summary)."""
        worker_env = _make_worker_env()
        worker_agent = _make_worker_agent()
        trajectory_buffer: list[dict] = []

        try:
            if args.pipeline == "with_descriptor":
                ep_summary = _run_episode_with_descriptor(
                    ep, args, worker_env, worker_agent, run_directory, trajectory_buffer,
                )
            else:
                ep_summary = _run_episode_balrog_baseline(
                    ep, args, worker_env, worker_agent, run_directory, trajectory_buffer,
                )
        finally:
            worker_env.close()

        return ep, trajectory_buffer, ep_summary

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_episode_worker, ep): ep for ep in range(args.episodes)}
        ep_results: dict[int, tuple[list[dict], dict]] = {}

        for future in as_completed(futures):
            ep, traj_buffer, ep_summary = future.result()
            ep_results[ep] = (traj_buffer, ep_summary)

            status = "SUCCESS" if ep_summary["success"] else "FAILED"
            print(f"\nEpisode {ep + 1} result: {status}  (reward = {ep_summary['total_reward']:.4f})")

            if args.no_gif:
                sentinel = run_directory / f"episode_{ep + 1:03d}.done"
                sentinel.touch()
                ep_summary.pop("actions", [])
            else:
                gif_path = run_directory / f"episode_{ep + 1:03d}.gif"
                _save_episode_gif(args.env, args.episode_seeds[ep], ep_summary.pop("actions", []), gif_path, adapter=args.adapter)
                print(f"GIF saved:      {gif_path}")

            if args.workers == 1:
                # Single worker: episodes always complete in order — flush immediately
                # so completed episodes are durable on disk if the run crashes later.
                _flush_trajectory(trajectory_file, traj_buffer)

    # Multi-worker: episodes complete out of order, so flush in episode number order
    # after all workers are done. Each episode block is still internally ordered.
    cumulative_reward = 0.0
    total_steps_taken = 0
    for ep in range(len(ep_results)):
        traj_buffer, ep_summary = ep_results[ep]
        if args.workers > 1:
            _flush_trajectory(trajectory_file, traj_buffer)

        total_steps_taken += ep_summary["total_steps"]
        cumulative_reward += ep_summary["total_reward"]
        episodes_summary.append(ep_summary)

    _write_run_summary(run_directory, args, model_name, conversation_mode, args.pipeline, episodes_summary)

    print(f"\nAverage reward: {cumulative_reward / max(len(episodes_summary), 1):.4f}")
    print(f"Logs saved to:  {run_directory}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(parse_args())
