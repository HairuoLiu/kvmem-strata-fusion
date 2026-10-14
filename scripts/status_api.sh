#!/usr/bin/env bash
# ==============================================================================
# KVMem Local API Server — Status Check Script
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

PORT_PID=$(lsof -ti :8080 2>/dev/null || true)

if [ -n "$PORT_PID" ]; then
    echo "🟢 KVMem API Server is RUNNING (PID: $PORT_PID) on port 8080"
    echo ""
    echo "• Health & Telemetry:"
    curl -s http://127.0.0.1:8080/api/monitor | .venv/bin/python -m json.tool 2>/dev/null || curl -s http://127.0.0.1:8080/api/monitor
    echo ""
    echo "• Available Models:"
    curl -s http://127.0.0.1:8080/v1/models | .venv/bin/python -c "import sys, json; data = json.load(sys.stdin); print('\n'.join(['  - ' + m['id'] for m in data['data']]))" 2>/dev/null || true
else
    echo "🔴 KVMem API Server is STOPPED (Port 8080 is idle)"
fi
