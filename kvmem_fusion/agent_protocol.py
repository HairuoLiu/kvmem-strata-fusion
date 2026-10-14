"""
KVMem Strata Fusion - Agent Protocol Engine
Provides production-grade OpenAI agent protocol compatibility:
- Tools JSON Schema formatting into Qwen ChatML system prompts
- Tool calling detection and extraction (Qwen <tool_call> tags and raw JSON)
- OpenAI delta SSE streaming tool_calls formatting
- Token budget negotiation (preventing max_tokens=0 starvation)
- Low-latency SSE streaming handshake chunking (eliminating 38s EmptyStreamError)
"""

import json
import re
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# 1. Token Budget Negotiation
# -----------------------------------------------------------------------------

def negotiate_token_budget(
    raw_max_tokens: Any,
    default_tokens: int = 2048,
    min_tokens: int = 16,
    max_ceiling: int = 4096,
) -> int:
    """
    Handle token budget negotiations gracefully.
    WorkBuddy and other agent clients may send:
    - max_tokens: 0 (or null)
    - very large values (e.g. 16k, 32k)
    If client sends max_tokens <= 0 or invalid: treat as default (2048), NEVER 0.
    Clamp within [min_tokens, max_ceiling] for MPS unified memory safety.
    """
    if raw_max_tokens is None:
        return default_tokens

    try:
        val = int(raw_max_tokens)
    except (ValueError, TypeError):
        return default_tokens

    if val <= 0:
        return default_tokens

    return min(max(val, min_tokens), max_ceiling)


# -----------------------------------------------------------------------------
# 2. Qwen ChatML Tool Prompt Formatting
# -----------------------------------------------------------------------------

def format_tools_chatml(tools: List[Dict[str, Any]]) -> str:
    """
    Formats tool definitions into Qwen tool-calling ChatML system prompt format.
    Standard Qwen template:
    # Tools
    You may call one or more functions to assist with the user query.
    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {tool_spec}
    </tools>
    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>
    """
    if not tools:
        return ""

    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False))
    lines.extend([
        "</tools>",
        "",
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
    ])
    return "\n".join(lines)


def format_messages_with_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Injects tools into messages list. If a system message exists, appends to it;
    otherwise prepends a new system message.
    Preserves all user context, MCP servers, and system instructions.
    """
    if not tools:
        return [dict(m) for m in messages]

    tools_text = format_tools_chatml(tools)
    if not tools_text:
        return [dict(m) for m in messages]

    new_messages = [dict(m) for m in messages]
    system_idx = -1
    for i, m in enumerate(new_messages):
        if m.get("role") == "system":
            system_idx = i
            break

    if system_idx >= 0:
        content = new_messages[system_idx].get("content", "")
        if "# Tools" not in content:
            new_messages[system_idx]["content"] = f"{content}\n\n{tools_text}"
    else:
        new_messages.insert(0, {
            "role": "system",
            "content": f"You are a helpful assistant.\n\n{tools_text}"
        })

    return new_messages


def format_chatml(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Formats the entire conversation into a Qwen ChatML string:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    """
    msgs = format_messages_with_tools(messages, tools)
    prompt_parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            prompt_parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")
        else:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    if add_generation_prompt:
        prompt_parts.append("<|im_start|>assistant\n")

    return "\n".join(prompt_parts)


# -----------------------------------------------------------------------------
# 3. Tool Call Extraction & Parsing
# -----------------------------------------------------------------------------

def parse_single_tool_call_json(content: str) -> Optional[Dict[str, Any]]:
    """
    Parses a tool call JSON object or snippet into standard OpenAI {name, arguments}.
    Handles nested structures, stringified arguments, and slight formatting variations.
    """
    s = content.strip()
    if not s:
        return None

    # Strip markdown code blocks if wrapped
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    try:
        data = json.loads(s)
    except Exception:
        # Try extracting JSON object via curly braces regex
        match = re.search(r"(\{.*\})", s, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception:
                return None
        else:
            return None

    if not isinstance(data, dict):
        return None

    # Extract function name
    name = None
    if "name" in data and isinstance(data["name"], str):
        name = data["name"]
    elif "function" in data:
        fn = data["function"]
        if isinstance(fn, dict) and "name" in fn:
            name = fn["name"]
        elif isinstance(fn, str):
            name = fn

    if not name:
        return None

    # Extract arguments
    args_raw = None
    if "arguments" in data:
        args_raw = data["arguments"]
    elif "parameters" in data:
        args_raw = data["parameters"]
    elif "function" in data and isinstance(data["function"], dict) and "arguments" in data["function"]:
        args_raw = data["function"]["arguments"]

    if args_raw is None:
        args_str = "{}"
    elif isinstance(args_raw, (dict, list)):
        args_str = json.dumps(args_raw, ensure_ascii=False)
    elif isinstance(args_raw, str):
        args_raw_strip = args_raw.strip()
        if not args_raw_strip:
            args_str = "{}"
        else:
            # Validate if valid JSON; if not, wrap or pass as string
            try:
                json.loads(args_raw_strip)
                args_str = args_raw_strip
            except Exception:
                args_str = json.dumps(args_raw_strip, ensure_ascii=False)
    else:
        args_str = json.dumps(args_raw, ensure_ascii=False)

    return {"name": name, "arguments": args_str}


def extract_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts tool calls from model output.
    Returns:
        clean_text: Remaining text with tool call tags removed.
        tool_calls: List of OpenAI-format tool call dicts:
                    [{"id": "call_...", "type": "function", "function": {"name": ..., "arguments": ...}}]
    """
    tool_calls: List[Dict[str, Any]] = []

    # 1. Match <tool_call>...</tool_call> tags
    pattern = re.compile(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if matches:
        clean_parts = []
        last_end = 0
        for m in matches:
            clean_parts.append(text[last_end:m.start()])
            call_content = m.group(1).strip()
            parsed = parse_single_tool_call_json(call_content)
            if parsed:
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": parsed
                })
            last_end = m.end()
        clean_parts.append(text[last_end:])
        clean_text = "".join(clean_parts).strip()
        return clean_text, tool_calls

    # 2. Check for raw JSON tool call if entire text or markdown is a JSON object
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("```") and "name" in stripped):
        parsed = parse_single_tool_call_json(stripped)
        if parsed and parsed.get("name"):
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            return "", [{
                "id": call_id,
                "type": "function",
                "function": parsed
            }]

    return text, []


# -----------------------------------------------------------------------------
# 4. SSE Chunks & Streaming Protocol
# -----------------------------------------------------------------------------

def create_initial_sse_chunk(req_id: str, model: str) -> Dict[str, Any]:
    """
    Creates immediate initial SSE chunk. Flushed on connection start to eliminate
    the 38-second EmptyStreamError in WorkBuddy and Cursor.
    """
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None
        }]
    }


def create_content_delta_chunk(req_id: str, model: str, content: str) -> Dict[str, Any]:
    """Creates a standard content delta chunk."""
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None
        }]
    }


def create_tool_call_chunks(
    req_id: str,
    model: str,
    tool_idx: int,
    call_id: str,
    func_name: str,
    args_json_str: str,
) -> List[Dict[str, Any]]:
    """
    Creates standard OpenAI streaming tool_calls deltas:
    Chunk 1: id, type='function', function.name, empty arguments
    Chunk 2: function.arguments
    """
    ts = int(time.time())
    chunk1 = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": tool_idx,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": ""
                    }
                }]
            },
            "finish_reason": None
        }]
    }
    chunk2 = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": tool_idx,
                    "function": {
                        "arguments": args_json_str
                    }
                }]
            },
            "finish_reason": None
        }]
    }
    return [chunk1, chunk2]


def create_final_sse_chunk(
    req_id: str,
    model: str,
    finish_reason: str = "stop"
) -> Dict[str, Any]:
    """Creates final SSE chunk with finish_reason ('stop' or 'tool_calls')."""
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    }


def format_sse_event(data: Union[Dict[str, Any], str]) -> str:
    """Formats payload as an SSE data line with double newline."""
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# -----------------------------------------------------------------------------
# 5. StreamingToolProcessor - Low-Latency Stream Engine
# -----------------------------------------------------------------------------

class StreamingToolProcessor:
    """
    State machine that processes streaming tokens from inference engines (Native MPS or Ollama),
    intercepts tool calls, emits standard OpenAI delta chunks, and prevents stream timeouts.
    """

    def __init__(
        self,
        req_id: Optional[str] = None,
        model: str = "default",
        tools_enabled: bool = True,
    ):
        self.req_id = req_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.model = model
        self.tools_enabled = tools_enabled
        self.buffer = ""
        self.tool_calls_emitted: List[Dict[str, Any]] = []
        self.parsed_tool_calls: List[Dict[str, Any]] = []
        self.has_emitted_initial = False

    @property
    def tool_calls_detected(self) -> bool:
        return len(self.parsed_tool_calls) > 0

    def emit_initial(self) -> Dict[str, Any]:
        """Yields initial SSE chunk immediately on connection."""
        self.has_emitted_initial = True
        return create_initial_sse_chunk(self.req_id, self.model)

    def process_token(self, token: str) -> List[Dict[str, Any]]:
        """
        Processes an incoming token from the model.
        Returns list of OpenAI completion chunk dictionaries.
        """
        chunks: List[Dict[str, Any]] = []

        if not self.tools_enabled:
            # Fast path: tools not requested, emit token directly
            chunks.append(create_content_delta_chunk(self.req_id, self.model, token))
            return chunks

        # Tool calling enabled: inspect stream for <tool_call> tags
        self.buffer += token

        # If <tool_call> is present in the buffer
        if "<tool_call>" in self.buffer:
            prefix, rest = self.buffer.split("<tool_call>", 1)
            if prefix:
                chunks.append(create_content_delta_chunk(self.req_id, self.model, prefix))
                self.buffer = "<tool_call>" + rest

            # Check if closing tag </tool_call> is received
            if "</tool_call>" in self.buffer:
                tag_content, after_tag = self.buffer.split("</tool_call>", 1)
                call_json = tag_content.replace("<tool_call>", "").strip()
                parsed = parse_single_tool_call_json(call_json)
                if parsed:
                    tool_idx = len(self.tool_calls_emitted)
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                    tool_chunks = create_tool_call_chunks(
                        self.req_id, self.model, tool_idx, call_id,
                        parsed["name"], parsed["arguments"]
                    )
                    chunks.extend(tool_chunks)
                    self.tool_calls_emitted.append(parsed)
                    try:
                        args_dict = json.loads(parsed["arguments"])
                    except Exception:
                        args_dict = parsed["arguments"]
                    self.parsed_tool_calls.append({
                        "name": parsed["name"],
                        "arguments": args_dict
                    })
                self.buffer = after_tag
        else:
            # Safe forward streaming: if buffer does not contain partial '<' tag, flush safely
            if "<" not in self.buffer:
                if len(self.buffer) > 1:
                    flush_text = self.buffer[:-1]
                    self.buffer = self.buffer[-1]
                    chunks.append(create_content_delta_chunk(self.req_id, self.model, flush_text))

        return chunks

    def flush(self) -> List[Dict[str, Any]]:
        """
        Called when generation completes. Flushes remaining buffer, parses any raw JSON tool calls,
        and outputs final stop/tool_calls chunk.
        """
        chunks: List[Dict[str, Any]] = []

        if self.tools_enabled and self.buffer:
            clean_text, tools = extract_tool_calls(self.buffer)
            if tools:
                if clean_text:
                    chunks.append(create_content_delta_chunk(self.req_id, self.model, clean_text))
                for t in tools:
                    tool_idx = len(self.tool_calls_emitted)
                    call_id = t.get("id", f"call_{uuid.uuid4().hex[:12]}")
                    fn = t.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    tool_chunks = create_tool_call_chunks(
                        self.req_id, self.model, tool_idx, call_id,
                        name, raw_args
                    )
                    chunks.extend(tool_chunks)
                    self.tool_calls_emitted.append(fn)
                    try:
                        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args_dict = raw_args
                    self.parsed_tool_calls.append({
                        "name": name,
                        "arguments": args_dict
                    })
            else:
                chunks.append(create_content_delta_chunk(self.req_id, self.model, self.buffer))
            self.buffer = ""
        elif not self.tools_enabled and self.buffer:
            chunks.append(create_content_delta_chunk(self.req_id, self.model, self.buffer))
            self.buffer = ""

        # Determine finish_reason
        finish_reason = "tool_calls" if len(self.tool_calls_emitted) > 0 else "stop"
        chunks.append(create_final_sse_chunk(self.req_id, self.model, finish_reason))
        return chunks

    def finalize(self) -> List[Dict[str, Any]]:
        return self.flush()


# -----------------------------------------------------------------------------
# 6. Non-Streaming Response Formatter
# -----------------------------------------------------------------------------

def format_chat_completion_response(
    req_id: str,
    model: str,
    raw_text: str,
    tools_enabled: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    speed_tok_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Builds standard OpenAI non-streaming chat.completion response dictionary.
    """
    ts = int(time.time())

    if tools_enabled:
        clean_text, tool_calls = extract_tool_calls(raw_text)
        if tool_calls:
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": ts,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": clean_text if clean_text else None,
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "speed_tok_s": round(speed_tok_s, 1)
            }

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": ts,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": raw_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        },
        "speed_tok_s": round(speed_tok_s, 1)
    }
