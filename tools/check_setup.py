"""Environment and dependency setup checker.

Run this after cloning or setting up on a new machine to verify that
BALROG, MiniGrid, and the LLM server are all reachable and working.

Usage:
    python tools/check_setup.py
    python tools/check_setup.py --skip-llm   # skip LLM server check
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import traceback


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str, detail: str | None = None) -> None:
    print(f"  [FAIL] {msg}")
    if detail:
        print(f"         {detail}")


def _skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


# ---------------------------------------------------------------------------
# Check 1: core imports
# ---------------------------------------------------------------------------

def check_imports() -> bool:
    print("\n[1] Core imports")
    ok = True

    for module, install_hint in [
        ("gymnasium",                       "uv pip install gymnasium"),
        ("minigrid",                        "uv pip install minigrid"),
        ("openai",                          "uv pip install openai"),
        ("PIL",                             "uv pip install pillow"),
        ("balrog",                          "uv pip install -e external/BALROG"),
        ("balrog.environments.babyai_text", "uv pip install -e external/BALROG"),
        ("balrog.agents.robust_cot",        "uv pip install -e external/BALROG"),
    ]:
        try:
            __import__(module)
            _ok(f"import {module}")
        except ImportError as e:
            _fail(f"import {module}", f"Run: {install_hint}  ({e})")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Check 2: BabyAI environment
# ---------------------------------------------------------------------------

def check_environment() -> bool:
    print("\n[2] BabyAI environment")
    ok = True

    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        from balrog.environments.babyai_text import BabyAITextCleanLangWrapper

        minigrid.register_minigrid_envs()
        env_id = "BabyAI-GoToRedBall-v0"
        env = BabyAITextCleanLangWrapper(gym.make(env_id))
        _ok(f"created {env_id} with BabyAITextCleanLangWrapper")

        obs, _ = env.reset(seed=42)
        _ok("env.reset() succeeded")

        # Check expected observation keys
        for key in ["mission", "direction", "text"]:
            if key in obs:
                _ok(f"obs['{key}'] present")
            else:
                _fail(f"obs['{key}'] missing — wrapper may not be applying correctly")
                ok = False

        if "long_term_context" in obs.get("text", {}):
            _ok("obs['text']['long_term_context'] present")
        else:
            _fail("obs['text']['long_term_context'] missing")
            ok = False

        # Run 5 random steps using the wrapper's expected action strings
        import random
        from balrog.environments.babyai_text.clean_lang_wrapper import BABYAI_ACTION_SPACE
        for i in range(5):
            action = random.choice(list(BABYAI_ACTION_SPACE))
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=42 + i)
        _ok("5 random steps completed without error")

        env.close()

    except Exception as e:
        _fail("environment check failed", traceback.format_exc().strip().split("\n")[-1])
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Check 3: src package imports
# ---------------------------------------------------------------------------

def check_src_imports() -> bool:
    print("\n[3] src package imports")
    ok = True

    for module in [
        "src.llm.client",
        "src.llm.balrog_adapter",
        "src.pipeline.descriptor",
        "src.pipeline.descriptor_agent",
        "src.pipeline.parsing",
        "src.environment.minigrid",
        "src.logging.episode_logger",
        "src.logging.run_directory",
        "src.utils.config",
    ]:
        try:
            __import__(module)
            _ok(f"import {module}")
        except ImportError as e:
            _fail(f"import {module}", str(e))
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Check 4: LLM server reachability — one probe per expected node
# ---------------------------------------------------------------------------

def check_llm_server() -> bool:
    import os
    import socket as _socket

    from src.llm.client import OpenAIClient

    ports_file = pathlib.Path.home() / "hlp_ports.txt"
    if ports_file.exists():
        raw = ports_file.read_text().strip()
        source = f"~/hlp_ports.txt"
    else:
        raw = os.environ.get("HLP_TUNNEL_PORTS", "7347")
        source = "HLP_TUNNEL_PORTS env var"
    ports = [int(p.strip()) for p in raw.replace("\n", ",").split(",") if p.strip()]

    print(f"\n[4] LLM servers  ({source}={raw}, {len(ports)} node(s) expected)")

    n_ready = 0
    for port in ports:
        # Step 1: is the tunnel up?
        try:
            with _socket.create_connection(("localhost", port), timeout=2):
                pass
        except (_socket.error, OSError):
            _fail(f"localhost:{port} — tunnel not reachable (is the SSH tunnel running?)")
            continue

        # Step 2: does the server respond to an LLM call?
        endpoint = f"http://localhost:{port}/v1"
        try:
            client = OpenAIClient(base_url=endpoint, model="gpt-oss-20b", max_retries=1)
            response, _, usage = client.generate_with_reasoning([
                {"role": "user", "content": "Reply with the single word: hello"}
            ])
            if response:
                tokens = usage.get("completion_tokens", "?")
                _ok(f"localhost:{port} — responded ({tokens} tokens): {response[:60].strip()}")
                n_ready += 1
            else:
                _fail(f"localhost:{port} — tunnel up but LLM returned empty response")
        except Exception as e:
            _fail(f"localhost:{port} — tunnel up but LLM call failed: {e}")

    n_expected = len(ports)
    if n_ready == n_expected:
        _ok(f"{n_ready}/{n_expected} nodes ready")
        return True
    else:
        _fail(f"{n_ready}/{n_expected} nodes ready — check tunnels and server logs on missing nodes")
        return False


# ---------------------------------------------------------------------------
# Check 5: parallel workers
# ---------------------------------------------------------------------------

def check_parallel_workers() -> bool:
    print("\n[5] Parallel workers")

    try:
        from concurrent.futures import ThreadPoolExecutor

        results = []
        def worker(i):
            return i * 2

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, i) for i in range(4)]
            results = [f.result() for f in futures]

        if sorted(results) == [0, 2, 4, 6]:
            _ok("ThreadPoolExecutor with 4 workers functioning correctly")
        else:
            _fail("ThreadPoolExecutor returned unexpected results")
            return False

    except Exception as e:
        _fail("Parallel worker check failed", str(e))
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check environment and dependency setup")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM server reachability check")
    args = parser.parse_args()

    print("=" * 50)
    print("  RAPOA — setup checker")
    print("=" * 50)

    results = [
        check_imports(),
        check_environment(),
        check_src_imports(),
        check_parallel_workers(),
    ]

    if args.skip_llm:
        _skip("LLM server check (--skip-llm)")
        _skip("(endpoint discovery included in LLM check)")
    else:
        results.append(check_llm_server())

    print("\n" + "=" * 50)
    if all(results):
        print("  ALL SYSTEMS GO — ready to launch experiment.")
    else:
        n_failed = sum(1 for r in results if not r)
        print(f"  {n_failed} check(s) failed — NOT ready to launch. See above.")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()
