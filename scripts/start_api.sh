#!/usr/bin/env bash
# ==============================================================================
# KVMem Local API Server — Start Daemon Script
# Runs the OpenAI-compatible local AI API server in the background.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

PID_FILE="$DIR/server.pid"
LOG_FILE="$DIR/server.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  KVMem API Server is already running (PID: $PID) at http://127.0.0.1:8080"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Kill any stray process holding port 8080
STRAY_PID=$(lsof -ti :8080 2>/dev/null || true)
if [ -n "$STRAY_PID" ]; then
    echo "Cleaning up port 8080 (PID: $STRAY_PID)..."
    kill -9 $STRAY_PID 2>/dev/null || true
    sleep 1
fi

echo "🚀 Starting KVMem Local OpenAI-Compatible API Server in background..."
nohup .venv/bin/python web/server.py > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

# Wait up to 5 seconds to verify startup
for i in {1..10}; do
    if curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
        echo "✅ Server started successfully!"
        echo "   • Base URL : http://127.0.0.1:8080/v1"
        echo "   • Models   : qwen3.8, qwen3.8-27b, qwen3.8-flash-next, gpt-4o"
        echo "   • PID      : $NEW_PID"
        echo "   • Logs     : $LOG_FILE"
        exit 0
    fi
    sleep 0.5
done

echo "⚠️ Server launched (PID: $NEW_PID). Check log with: tail -f server.log"
