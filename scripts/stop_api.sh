#!/usr/bin/env bash
# ==============================================================================
# KVMem Local API Server — Stop Daemon Script
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

PID_FILE="$DIR/server.pid"

STOPPED=0

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping KVMem API Server (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        STOPPED=1
    fi
    rm -f "$PID_FILE"
fi

# Also check port 8080 directly
PORT_PID=$(lsof -ti :8080 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
    echo "Releasing port 8080 (PID: $PORT_PID)..."
    kill -9 $PORT_PID 2>/dev/null || true
    STOPPED=1
fi

if [ $STOPPED -eq 1 ]; then
    echo "✅ KVMem API Server stopped cleanly."
else
    echo "ℹ️  No running KVMem API server found on port 8080."
fi
