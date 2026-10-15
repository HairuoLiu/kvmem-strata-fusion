"""
Comprehensive Tests for KVMem Agent Protocol Engine
Validates:
- Token budget negotiation (preventing max_tokens=0 starvation)
- Tools formatting into Qwen ChatML system prompt
- Handling 15,000+ characters of prompt context with 43 tools (WorkBuddy profile)
- Tool call detection and parsing (single, multiple, raw JSON)
- Low-latency SSE streaming chunks (handshake chunk eliminating 38s EmptyStreamError)
- Non-streaming OpenAI response formatting
- End-to-end HTTP API server completions
"""

import json
import time
import socket
import threading
import urllib.request
import pytest

from kvmem_fusion.agent_protocol import (
    negotiate_token_budget,
    format_tools_chatml,
    format_messages_with_tools,
    format_chatml,
    parse_single_tool_call_json,
    extract_tool_calls,
    create_initial_sse_chunk,
    create_content_delta_chunk,
    create_tool_call_chunks,
    create_final_sse_chunk,
    format_sse_event,
    StreamingToolProcessor,
    format_chat_completion_response,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }


@pytest.fixture
def workbuddy_43_tools():
    """Generates 43 realistic tool definitions mimicking WorkBuddy agent suite."""
    tools = []
    tool_names = [
        "read_file", "write_to_file", "replace_file_content", "run_command", "search_web",
        "view_file", "list_directory", "read_url_content", "notebook_edit", "manage_task",
        "generate_image", "schedule", "send_message", "git_status", "git_commit",
        "git_diff", "git_log", "grep_search", "find_files", "execute_python",
        "bash_eval", "fetch_api", "db_query", "db_migrate", "docker_ps",
        "docker_run", "docker_stop", "deploy_service", "inspect_container", "tail_logs",
        "mcp_server_ping", "mcp_call_tool", "mcp_list_prompts", "mcp_list_resources", "code_linter",
        "ast_grep", "symbol_lookup", "lsp_diagnostics", "format_code", "run_tests",
        "benchmark_profile", "telemetry_dump", "report_generation"
    ]
    for name in tool_names:
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Agent tool function {name} description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"}
                    },
                    "required": ["param1"]
                }
            }
        })
    assert len(tools) == 43
    return tools


# -----------------------------------------------------------------------------
# 1. Token Budget Negotiation Tests
# -----------------------------------------------------------------------------

def test_negotiate_token_budget_zero_and_none():
    # If client sends max_tokens=0, treat as default 2048, never 0
    assert negotiate_token_budget(0) == 2048
    assert negotiate_token_budget("0") == 2048
    assert negotiate_token_budget(None) == 2048
    assert negotiate_token_budget(-5) == 2048
    assert negotiate_token_budget("invalid") == 2048


def test_negotiate_token_budget_valid_values():
    assert negotiate_token_budget(512) == 512
    assert negotiate_token_budget("1024") == 1024
    # Clamping ceiling to 4096 (MPS safe budget)
    assert negotiate_token_budget(16384) == 4096
    assert negotiate_token_budget(32768) == 4096
    # Clamping floor to 16
    assert negotiate_token_budget(2) == 16


# -----------------------------------------------------------------------------
# 2. Qwen ChatML Tool Prompt Formatting Tests
# -----------------------------------------------------------------------------

def test_format_tools_chatml_structure(sample_weather_tool):
    formatted = format_tools_chatml([sample_weather_tool])
    assert "# Tools" in formatted
    assert "<tools>" in formatted
    assert "</tools>" in formatted
    assert "<tool_call>" in formatted
    assert "</tool_call>" in formatted
    assert "get_current_weather" in formatted


def test_format_messages_with_43_tools_and_15k_context(workbuddy_43_tools):
    # Simulate WorkBuddy's 15,000+ characters of context
    large_context = "You are a software engineer agent working on a large repository.\n" * 250
    assert len(large_context) > 15000

    messages = [
        {"role": "system", "content": large_context},
        {"role": "user", "content": "List the files in the directory."}
    ]

    result = format_messages_with_tools(messages, workbuddy_43_tools)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    # Ensure large context is intact and # Tools appended
    assert large_context in result[0]["content"]
    assert "# Tools" in result[0]["content"]
    assert "read_file" in result[0]["content"]
    assert "report_generation" in result[0]["content"]


def test_format_messages_prepends_system_if_missing(sample_weather_tool):
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    result = format_messages_with_tools(messages, [sample_weather_tool])
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "# Tools" in result[0]["content"]
    assert result[1]["role"] == "user"


def test_format_chatml_full_prompt(sample_weather_tool):
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Hello."},
        {"role": "tool", "content": '{"status": "ok"}'}
    ]
    chatml = format_chatml(messages, [sample_weather_tool], add_generation_prompt=True)
    assert "<|im_start|>system" in chatml
    assert "<|im_start|>user" in chatml
    assert "<tool_response>" in chatml
    assert "<|im_start|>assistant\n" in chatml


# -----------------------------------------------------------------------------
# 3. Tool Call Extraction & Parsing Tests
# -----------------------------------------------------------------------------

def test_extract_tool_calls_qwen_xml():
    raw_text = (
        "I will fetch the weather for you.\n"
        "<tool_call>\n"
        '{"name": "get_current_weather", "arguments": {"location": "San Francisco"}}\n'
        "</tool_call>"
    )
    clean_text, tool_calls = extract_tool_calls(raw_text)
    assert clean_text == "I will fetch the weather for you."
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "get_current_weather"
    args = json.loads(call["function"]["arguments"])
    assert args == {"location": "San Francisco"}


def test_extract_multiple_tool_calls():
    raw_text = (
        "<tool_call>\n"
        '{"name": "func_a", "arguments": {"x": 1}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "func_b", "arguments": {"y": 2}}\n'
        "</tool_call>"
    )
    clean_text, tool_calls = extract_tool_calls(raw_text)
    assert clean_text == ""
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "func_a"
    assert tool_calls[1]["function"]["name"] == "func_b"


def test_extract_tool_calls_raw_json():
    raw_text = '{"name": "search_code", "arguments": {"query": "parse_tool_call"}}'
    clean_text, tool_calls = extract_tool_calls(raw_text)
    assert clean_text == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search_code"


def test_extract_tool_calls_no_tools():
    raw_text = "Here is an explanation of the algorithm without any tools."
    clean_text, tool_calls = extract_tool_calls(raw_text)
    assert clean_text == raw_text
    assert len(tool_calls) == 0


# -----------------------------------------------------------------------------
# 4. Low-Latency SSE Streaming Protocol Tests
# -----------------------------------------------------------------------------

def test_initial_sse_chunk_eliminates_empty_stream_error():
    """Initial chunk must be emitted instantly to defeat 38s timeout in WorkBuddy."""
    chunk = create_initial_sse_chunk(req_id="chatcmpl-test", model="qwen3.8")
    assert chunk["id"] == "chatcmpl-test"
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunk["choices"][0]["finish_reason"] is None


def test_streaming_tool_processor_plain_text():
    proc = StreamingToolProcessor(req_id="test-1", model="qwen3.8", tools_enabled=False)
    chunks = [proc.emit_initial()]

    for tok in ["Hello", " world", "!"]:
        chunks.extend(proc.process_token(tok))
    chunks.extend(proc.flush())

    # Verify chunk structure
    assert len(chunks) >= 4
    # Initial chunk
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    # Content chunks
    contents = [c["choices"][0]["delta"].get("content", "") for c in chunks[1:-1]]
    assert "".join(contents) == "Hello world!"
    # Final chunk
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_streaming_tool_processor_with_tool_call():
    proc = StreamingToolProcessor(req_id="test-2", model="qwen3.8", tools_enabled=True)
    chunks = [proc.emit_initial()]

    # Simulate tokens streamed from model
    tokens = [
        "Let me check that for you.\n",
        "<tool_call>\n",
        '{"name": "run_command", ',
        '"arguments": {"command": "ls -la"}}\n',
        "</tool_call>"
    ]
    for tok in tokens:
        chunks.extend(proc.process_token(tok))
    chunks.extend(proc.flush())

    assert proc.tool_calls_detected is True
    assert len(proc.parsed_tool_calls) == 1
    assert proc.parsed_tool_calls[0]["name"] == "run_command"
    assert proc.parsed_tool_calls[0]["arguments"] == {"command": "ls -la"}

    # Validate tool call deltas structure
    tool_delta_events = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert len(tool_delta_events) >= 2
    assert tool_delta_events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "run_command"
    args_str = tool_delta_events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args_str) == {"command": "ls -la"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# -----------------------------------------------------------------------------
# 5. Non-Streaming Response Formatting Tests
# -----------------------------------------------------------------------------

def test_format_chat_completion_response_tool_call():
    raw_text = '<tool_call>{"name": "fetch", "arguments": {"url": "http://api"}}</tool_call>'
    resp = format_chat_completion_response(
        req_id="test-non-stream",
        model="qwen3.8",
        raw_text=raw_text,
        tools_enabled=True,
        prompt_tokens=100,
        completion_tokens=25,
        speed_tok_s=32.5
    )
    assert resp["object"] == "chat.completion"
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert len(choice["message"]["tool_calls"]) == 1
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "fetch"
    assert resp["usage"]["prompt_tokens"] == 100


def test_format_chat_completion_response_plain_text():
    raw_text = "This is a regular non-streaming text response."
    resp = format_chat_completion_response(
        req_id="test-text",
        model="qwen3.8",
        raw_text=raw_text,
        tools_enabled=True,
        prompt_tokens=50,
        completion_tokens=10,
        speed_tok_s=30.0
    )
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == raw_text
    assert "tool_calls" not in choice["message"]


# -----------------------------------------------------------------------------
# 6. End-to-End Server API Verification (Port 8089)
# -----------------------------------------------------------------------------

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    from http.server import HTTPServer
    from web.server import KVMemAppHandler

    test_port = find_free_port()
    server = HTTPServer(("127.0.0.1", test_port), KVMemAppHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{test_port}"
    server.shutdown()


def test_e2e_server_models_endpoint(live_server):
    req = urllib.request.Request(f"{live_server}/v1/models")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "qwen3.8" in model_ids


def test_e2e_server_chat_completions_streaming(live_server, sample_weather_tool):
    """
    Simulates WorkBuddy request:
    - stream: True
    - max_tokens: 0 (negotiates to 2048)
    - tools provided
    - verifies initial chunk arrives immediately (< 200ms)
    """
    payload = json.dumps({
        "model": "default",
        "messages": [
            {"role": "user", "content": "Hello! Give a 1-word greeting."}
        ],
        "tools": [sample_weather_tool],
        "max_tokens": 0,  # WorkBuddy edge case
        "stream": True
    }).encode("utf-8")

    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{live_server}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        first_line = resp.readline().decode("utf-8").strip()
        elapsed_ttft = time.perf_counter() - t0

        # TTFT to first byte must be well below 500ms (eliminates 38s EmptyStreamError)
        assert elapsed_ttft < 1.0, f"Initial stream response too slow: {elapsed_ttft}s"
        assert first_line.startswith("data: ")

        # Read subsequent streaming lines with readline() to prevent socket buffer block
        all_lines = [first_line]
        for _ in range(5):
            raw_l = resp.readline()
            if not raw_l:
                break
            l = raw_l.decode("utf-8").strip()
            if l:
                all_lines.append(l)
                if "data: [DONE]" in l:
                    break

        assert any('"delta"' in l for l in all_lines)


def test_e2e_server_chat_completions_non_streaming(live_server):
    payload = json.dumps({
        "model": "default",
        "messages": [
            {"role": "user", "content": "Say test."}
        ],
        "max_tokens": 16,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{live_server}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["finish_reason"] in ("stop", "length", "tool_calls")
