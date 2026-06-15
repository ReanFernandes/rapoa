"""Cluster GPU node discovery utilities."""

from __future__ import annotations

import logging
import os
import pathlib
import socket
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

# Edit this file at runtime to add/remove tunnel ports without restarting the run.
# Format: comma-separated port numbers, e.g. "7347,7348"
# Falls back to HLP_TUNNEL_PORTS env var if the file does not exist.
_DEFAULT_PORTS_FILE = pathlib.Path.home() / "hlp_ports.txt"


def _read_ports(ports_file: pathlib.Path) -> list[int]:
    if ports_file.exists():
        raw = ports_file.read_text().strip()
    else:
        raw = os.environ.get("HLP_TUNNEL_PORTS", "7347")
    return [int(p.strip()) for p in raw.replace("\n", ",").split(",") if p.strip()]


def _probe_ports(ports: list[int]) -> list[str]:
    available = []
    for port in ports:
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                available.append(f"http://localhost:{port}/v1")
        except (socket.error, OSError):
            pass
    return available


def find_all_gpu_servers() -> list[str]:
    """Return base URLs for all reachable GPU nodes.

    Reads HLP_TUNNEL_PORTS (comma-separated local port numbers), or
    ~/hlp_ports.txt if it exists (takes priority).
    Each port is an SSH tunnel mapping a local port on the login node to
    port 7347 on a GPU node running vLLM.

    Examples:
        HLP_TUNNEL_PORTS=7347            # single node (default)
        HLP_TUNNEL_PORTS=7347,7348       # two nodes
        HLP_TUNNEL_PORTS=7347,7348,7349  # three nodes
    """
    env_override = os.environ.get("HLP_TUNNEL_PORTS_FILE")
    ports_file = pathlib.Path(env_override) if env_override else _DEFAULT_PORTS_FILE
    ports = _read_ports(ports_file)

    available = _probe_ports(ports)
    for port in ports:
        url = f"http://localhost:{port}/v1"
        if url not in available:
            log.warning("No vLLM server reachable on localhost:%d — skipping", port)
    return available


def start_endpoint_watcher(
    on_change: Callable[[list[str]], None],
    interval: int = 30,
    ports_file: pathlib.Path | None = None,
) -> threading.Thread:
    """Start a background thread that re-probes servers every `interval` seconds.

    Calls on_change(new_endpoints) when the reachable set changes.
    Edit ~/hlp_ports.txt (or the file at HLP_TUNNEL_PORTS_FILE) mid-run to
    add or remove tunnel ports; the watcher picks up the change within `interval`
    seconds.
    """
    if ports_file is None:
        env_override = os.environ.get("HLP_TUNNEL_PORTS_FILE")
        ports_file = pathlib.Path(env_override) if env_override else _DEFAULT_PORTS_FILE

    def _watch() -> None:
        last: set[str] = set()
        while True:
            time.sleep(interval)
            try:
                ports = _read_ports(ports_file)
                available = _probe_ports(ports)
                current = set(available)
                if current != last:
                    last = current
                    log.info("Endpoint watcher: reachable nodes changed → %s", available)
                    on_change(available)
            except Exception:
                log.exception("Endpoint watcher error")

    t = threading.Thread(target=_watch, daemon=True, name="endpoint-watcher")
    t.start()
    return t
