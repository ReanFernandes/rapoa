#!/usr/bin/env bash
# Run on the login node after submitting start_vllm_array.sh.
# Opens SSH tunnels for all READY nodes in ~/vllm_nodes.txt and updates ~/hlp_ports.txt.
# Safe to re-run — skips ports that already have an open tunnel.
#
# Usage:
#   bash run_experiments/open_tunnels.sh           # one-shot
#   bash run_experiments/open_tunnels.sh --watch   # daemon: polls every 30s, auto-opens new tunnels

STATUS_FILE="$HOME/vllm_nodes.txt"
WATCH=false

[[ "${1:-}" == "--watch" ]] && WATCH=true

_apply() {
    [ ! -f "$STATUS_FILE" ] && return

    PORTS=()
    while IFS=' ' read -r port node status; do
        [ -z "$port" ] && continue
        if [ "$status" = "READY" ]; then
            if ss -tlnp 2>/dev/null | grep -qE ":${port}[ $]"; then
                :  # tunnel already open, silently include in ports list
            else
                echo "[open_tunnels] Opening tunnel: localhost:${port} → ${node}:7347"
                ssh -N -f -L "${port}:localhost:7347" "${node}"
            fi
            PORTS+=("$port")
        fi
    done < "$STATUS_FILE"

    if [ ${#PORTS[@]} -gt 0 ]; then
        PORTS_STR=$(IFS=,; echo "${PORTS[*]}")
        if [ ! -f "$HOME/hlp_ports.txt" ] || [ "$(cat "$HOME/hlp_ports.txt")" != "$PORTS_STR" ]; then
            echo "$PORTS_STR" > "$HOME/hlp_ports.txt"
            echo "[open_tunnels] ~/hlp_ports.txt updated: $PORTS_STR"
        fi
    fi
}

if [ "$WATCH" = false ]; then
    if [ ! -f "$STATUS_FILE" ]; then
        echo "No nodes file found at $STATUS_FILE. Submit start_vllm_array.sh first."
        exit 1
    fi
    echo "Current node status:"
    cat "$STATUS_FILE"
    echo ""
    _apply
    if [ ${#PORTS[@]} -eq 0 ] 2>/dev/null; then
        echo "No READY nodes found. Wait for jobs to start and re-run, or use --watch."
    else
        echo "If a run is active, the endpoint watcher picks this up within 30 seconds."
    fi
else
    echo "[open_tunnels] Watch mode started — polling ~/vllm_nodes.txt every 30s (Ctrl+C to stop)"
    LAST_HASH=""
    while true; do
        CURRENT_HASH=$(md5sum "$STATUS_FILE" 2>/dev/null | cut -d' ' -f1 || echo "")
        if [ "$CURRENT_HASH" != "$LAST_HASH" ]; then
            LAST_HASH="$CURRENT_HASH"
            _apply
        fi
        sleep 30
    done
fi
