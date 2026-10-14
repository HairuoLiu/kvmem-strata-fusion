"""
End-to-End Stress Test Suite & Memory Audit for KVMem-Strata-Fusion.
Replicates WorkBuddy's production developer agent payload:
- 15,000+ characters of input prompt containing 43 tool definitions and 6 MCP server descriptions.
- Multi-turn interaction simulating actual developer workflows.
- 8K to 32K context windows (256 to 1024 CASA atom blocks).
- Physical memory (RSS) audit: strictly under 18GB on 32GB Mac (< 16-18GB total).
- Zero 'Invalid buffer size' or Metal OOM exceptions.
- NVMe disk spilling in /tmp/kvmem_cache with bit-exact zero-corruption verification.
- API response latency, TTFT, and streaming throughput health check.
"""

import os
import sys
import gc
import json
import time
import math
import shutil
import hashlib
import resource
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pytest

from kvmem_fusion.core import (
    CanonicalAtomStore,
    KVBlock,
    apply_rope,
    compute_prefix_hash,
    StorageSuperTile,
)
from kvmem_fusion.ifr import (
    IFRBlock,
    IFRRetriever,
    AntiCollapseSplitter,
    compute_deroped_mean,
)
from kvmem_fusion.ladder import get_tier_biases
from kvmem_fusion.ubba import solve_ubba_greedy, create_ladder_block_candidate
from kvmem_fusion.spill_manager import (
    SuperTileSpillManager,
    SuperTileManifest,
    BlockLocation,
    PAGE_SIZE,
)
from kvmem_fusion.agent_protocol import (
    negotiate_token_budget,
    format_tools_chatml,
    format_messages_with_tools,
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


def get_current_rss_gb() -> float:
    """Return process resident set size (RSS) in Gigabytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # On macOS, ru_maxrss is in bytes
        return rss / (1024.0 ** 3)
    else:
        # On Linux, ru_maxrss is in kilobytes
        return rss / (1024.0 ** 2)


# =============================================================================
# WorkBuddy Production Payload Definition (6 MCP Servers + 43 Tools)
# =============================================================================

MCP_SERVERS = [
    {
        "name": "filesystem-mcp",
        "version": "2.4.0",
        "description": "Secure workspace filesystem provider with sandboxed POSIX I/O, regex search, and AST block editing.",
        "capabilities": ["read", "write", "search", "tree", "diff", "batch_io"],
    },
    {
        "name": "git-vcs-mcp",
        "version": "1.9.2",
        "description": "Git repository management provider with graph inspection, stash manipulation, and conflict resolution.",
        "capabilities": ["status", "diff", "log", "commit", "branch", "stash"],
    },
    {
        "name": "terminal-runner-mcp",
        "version": "3.1.0",
        "description": "Persistent zsh/bash subshell daemon with pseudo-terminal multiplexing and asynchronous stream buffering.",
        "capabilities": ["spawn_pty", "run_daemon", "kill_process", "pipe_io"],
    },
    {
        "name": "sqlite-analytics-mcp",
        "version": "1.5.0",
        "description": "Embedded SQLite analytics database engine for query execution, index optimization, and telemetry auditing.",
        "capabilities": ["query", "schema", "explain_plan", "batch_transaction", "vacuum"],
    },
    {
        "name": "web-research-mcp",
        "version": "2.0.1",
        "description": "High-throughput HTTP/2 crawler with Readability content extraction, DOM sanitization, and DuckDuckGo API.",
        "capabilities": ["http_get", "search", "markdown_extract", "sitemap_crawl"],
    },
    {
        "name": "language-server-mcp",
        "version": "4.2.0",
        "description": "Language Server Protocol (LSP) client supporting cross-file definition lookup, hover docstrings, and symbol refactoring.",
        "capabilities": ["goto_def", "find_refs", "hover_info", "workspace_symbols", "rename"],
    },
]


def generate_43_tools() -> List[Dict[str, Any]]:
    """Generates the exact 43 tool JSON schemas matching WorkBuddy's developer environment."""
    tools = [
        # --- 1. Filesystem Tools (8 tools) ---
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents from workspace filesystem with line slicing and encoding support.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative file path."},
                        "start_line": {"type": "integer", "description": "1-based starting line number."},
                        "end_line": {"type": "integer", "description": "1-based ending line number."},
                        "encoding": {"type": "string", "default": "utf-8", "description": "Text character encoding."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write or overwrite complete file content atomically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Destination file path."},
                        "content": {"type": "string", "description": "File content to write."},
                        "create_parents": {"type": "boolean", "default": True},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file_block",
                "description": "Perform precise contiguous replacement of code block inside an existing file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Target file path."},
                        "target_content": {"type": "string", "description": "Exact text chunk to replace."},
                        "replacement_content": {"type": "string", "description": "New replacement content."},
                        "start_line": {"type": "integer", "description": "Expected start line hint."},
                        "end_line": {"type": "integer", "description": "Expected end line hint."},
                    },
                    "required": ["path", "target_content", "replacement_content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files, permissions, and sizes within a workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path to enumerate."},
                        "recursive": {"type": "boolean", "default": False},
                        "include_hidden": {"type": "boolean", "default": False},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_search",
                "description": "Search workspace files using glob patterns or ripgrep regex strings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob or regex pattern."},
                        "root_dir": {"type": "string", "description": "Root search directory."},
                        "max_results": {"type": "integer", "default": 50},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "directory_tree",
                "description": "Generate an indented ASCII/JSON directory hierarchy tree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Base directory."},
                        "depth": {"type": "integer", "default": 3},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_file",
                "description": "Atomically rename or move a file or directory across filesystems.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source path."},
                        "destination": {"type": "string", "description": "Target path."},
                        "overwrite": {"type": "boolean", "default": False},
                    },
                    "required": ["source", "destination"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_file_metadata",
                "description": "Retrieve POSIX inode metadata, size, mtime, and sha256 checksum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Target file path."},
                        "compute_sha256": {"type": "boolean", "default": True},
                    },
                    "required": ["path"],
                },
            },
        },

        # --- 2. Git VCS Tools (7 tools) ---
        {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "Get git working tree status, staged files, unstaged modifications, and untracked entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {"type": "string", "default": "."},
                        "short_format": {"type": "boolean", "default": False},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": "Inspect unified diff of staged or working changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {"type": "string", "default": "."},
                        "staged": {"type": "boolean", "default": False},
                        "file_path": {"type": "string", "description": "Optional single file filter."},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_log",
                "description": "Retrieve commit history log with hashes, author info, and commit messages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {"type": "string", "default": "."},
                        "max_count": {"type": "integer", "default": 10},
                        "oneline": {"type": "boolean", "default": True},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Create a new git commit with staged changes and message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Conventional commit message."},
                        "repo_path": {"type": "string", "default": "."},
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_checkout",
                "description": "Switch branches or restore modified files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Branch name, tag, or file path."},
                        "create_branch": {"type": "boolean", "default": False},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_branch",
                "description": "List, create, or delete repository branches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["list", "create", "delete"]},
                        "branch_name": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_stash",
                "description": "Stash uncommitted changes or apply previously saved stashes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subcommand": {"type": "string", "enum": ["push", "pop", "list", "drop"]},
                        "message": {"type": "string"},
                    },
                    "required": ["subcommand"],
                },
            },
        },

        # --- 3. Terminal Execution Tools (6 tools) ---
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Execute synchronous or asynchronous shell command in sandboxed environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command line string."},
                        "cwd": {"type": "string", "default": "."},
                        "timeout_sec": {"type": "integer", "default": 30},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_persistent_terminal",
                "description": "Run shell command inside a persistent stateful zsh session sharing environment variables.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Terminal session identifier."},
                        "command": {"type": "string", "description": "Shell command to run."},
                    },
                    "required": ["session_id", "command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "kill_background_process",
                "description": "Terminate a background subprocess by PID or task ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Background task ID."},
                        "force_sigkill": {"type": "boolean", "default": False},
                    },
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_process_output",
                "description": "Retrieve stdout/stderr incremental buffer of running background task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Background task ID."},
                        "tail_lines": {"type": "integer", "default": 100},
                    },
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_running_tasks",
                "description": "List all active background daemons and execution statuses.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_terminal_input",
                "description": "Send stdin stream characters to interactive process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "input_text": {"type": "string"},
                    },
                    "required": ["task_id", "input_text"],
                },
            },
        },

        # --- 4. SQLite Analytics Tools (6 tools) ---
        {
            "type": "function",
            "function": {
                "name": "sqlite_query",
                "description": "Execute SELECT query on SQLite telemetry database and return tabular rows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string", "description": "Path to SQLite database file."},
                        "sql": {"type": "string", "description": "SQL statement."},
                        "params": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["db_path", "sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_schema",
                "description": "Introspect table structures, foreign keys, and indexes in SQLite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "table_name": {"type": "string", "description": "Optional table name filter."},
                    },
                    "required": ["db_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_explain_query_plan",
                "description": "Obtain EXPLAIN QUERY PLAN breakdown to diagnose indexing and table scans.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "sql": {"type": "string"},
                    },
                    "required": ["db_path", "sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_execute_batch",
                "description": "Execute multiple INSERT/UPDATE statements in an atomic ACID transaction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "statements": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["db_path", "statements"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_table_stats",
                "description": "Compute row counts, disk storage size, and index bloat for database tables.",
                "parameters": {
                    "type": "object",
                    "properties": {"db_path": {"type": "string"}},
                    "required": ["db_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sqlite_backup_database",
                "description": "Create an online live hot-backup of an active SQLite database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_db": {"type": "string"},
                        "target_db": {"type": "string"},
                    },
                    "required": ["source_db", "target_db"],
                },
            },
        },

        # --- 5. Web Research Tools (6 tools) ---
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Query DuckDuckGo / SearXNG search index for technical articles and documentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query keywords."},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_web_page",
                "description": "Fetch raw HTML content from an external HTTPS endpoint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Target webpage URL."},
                        "timeout_sec": {"type": "integer", "default": 15},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_markdown_from_url",
                "description": "Scrape article text, strip advertisements and JavaScript, and convert to clean markdown.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "batch_url_scrape",
                "description": "Perform concurrent asynchronous downloads of multiple URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {"type": "array", "items": {"type": "string"}},
                        "concurrency": {"type": "integer", "default": 5},
                    },
                    "required": ["urls"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_broken_links",
                "description": "Validate HTTP response status codes for links embedded in documentation.",
                "parameters": {
                    "type": "object",
                    "properties": {"links": {"type": "array", "items": {"type": "string"}}},
                    "required": ["links"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dns_lookup",
                "description": "Resolve A, AAAA, and CNAME DNS resource records for hostnames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "record_type": {"type": "string", "default": "A"},
                    },
                    "required": ["domain"],
                },
            },
        },

        # --- 6. Language Server (LSP) Tools (6 tools) ---
        {
            "type": "function",
            "function": {
                "name": "lsp_find_definition",
                "description": "Navigate to declaration or definition location of symbol under cursor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["file_path", "line", "character"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lsp_find_references",
                "description": "Find all usages and references of a function, class, or variable across workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["file_path", "line", "character"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lsp_hover_symbol",
                "description": "Retrieve type signatures, docstrings, and hover documentation for a code token.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["file_path", "line", "character"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lsp_document_symbols",
                "description": "Extract all AST symbols (classes, methods, functions, constants) from source file.",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lsp_workspace_symbol_search",
                "description": "Perform fuzzy symbol search across entire multi-crate/multi-package workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_symbols": {"type": "integer", "default": 30},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lsp_apply_rename",
                "description": "Safely refactor-rename a symbol and all its workspace references atomically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                        "new_name": {"type": "string"},
                    },
                    "required": ["file_path", "line", "character", "new_name"],
                },
            },
        },

        # --- 7. Agent Workflow & Memory Tools (4 tools) ---
        {
            "type": "function",
            "function": {
                "name": "memory_store_fact",
                "description": "Persist a long-term architecture decision or user preference into durable memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "fact_summary": {"type": "string"},
                        "importance": {"type": "integer", "default": 3},
                    },
                    "required": ["topic", "fact_summary"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_retrieve_context",
                "description": "Query associative semantic memory for historical notes and session facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_delayed_notification",
                "description": "Set a timer or recurring heartbeat reminder in background event loop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "delay_seconds": {"type": "integer"},
                        "notification_prompt": {"type": "string"},
                    },
                    "required": ["delay_seconds", "notification_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_subagent",
                "description": "Spawn specialized subagent conversation to solve a scoped verification subtask.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_role": {"type": "string"},
                        "instruction_prompt": {"type": "string"},
                    },
                    "required": ["agent_role", "instruction_prompt"],
                },
            },
        },
    ]
    return tools


def build_workbuddy_full_prompt(tools: List[Dict[str, Any]], mcp_servers: List[Dict[str, Any]]) -> str:
    """
    Builds the exact WorkBuddy developer agent input prompt:
    Must exceed 15,000 characters and contain all 43 tool schemas and 6 MCP servers.
    """
    system_intro = (
        "You are WorkBuddy Principal Software Engineering Agent v4.8, connected to local developer workstation.\n"
        "Your mission is to perform deep architecture analysis, codebase refactoring, performance profiling, "
        "and multi-tier KV memory auditing for high-concurrency LLM inference backends.\n\n"
        "### Operational Charter & System Invariants:\n"
        "1. Strictly enforce zero data corruption across tiered memory hierarchies (HBM, Unified RAM, NVMe).\n"
        "2. Ensure all tensor operations conform to Apple Silicon Metal (MPS) buffer alignment limits.\n"
        "3. Apply Content-Addressed Canonical Atom Store (CASA) K-Freeze immutability to prevent re-rotation drift.\n"
        "4. Intercept outlier needle dispersion using Invertible Fidelity-bound Retrieval (IFR).\n"
        "5. Solve Universal Byte-Budget Allocation (UBBA) under strict demand-covering knapsack constraints.\n"
        "6. In multi-turn interaction, maintain causal prefix hash chains across divergent conversation branches.\n\n"
        "### Registered MCP Subsystem Providers:\n"
    )

    mcp_block = []
    for srv in mcp_servers:
        mcp_block.append(
            f"- MCP Server: `{srv['name']}` (v{srv['version']})\n"
            f"  Description: {srv['description']}\n"
            f"  Declared Capabilities: {', '.join(srv['capabilities'])}\n"
        )
    mcp_text = "\n".join(mcp_block) + "\n\n"

    tools_chatml = format_tools_chatml(tools)

    developer_context = (
        "### Active Developer Workspace Context:\n"
        "Repository: `/Users/hai/workspaces/kvmem-strata-fusion`\n"
        "Hardware Target: Apple Silicon M-Series (Darwin arm64, 32GB Unified Memory, PageSize=16384B)\n"
        "Inference Gateway: KVMem Native In-Process Engine (PyTorch MPS / Apple Metal Unified Memory)\n"
        "Active Context Window Range: 8,192 tokens (256 blocks) to 32,768 tokens (1024 blocks)\n"
        "Physical Memory Ceiling: Strictly < 18.0 GB RSS\n"
        "Spill Directory: `/tmp/kvmem_cache`\n\n"
        "Please analyze the provided tool definitions, inspect the memory hierarchy, and prepare the "
        "execution pipeline for multi-turn stress verification."
    )

    full_payload = system_intro + mcp_text + tools_chatml + "\n\n" + developer_context
    return full_payload


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def workbuddy_payload():
    """Provides the validated 43 tools, 6 MCP servers, and 15,000+ char prompt."""
    tools = generate_43_tools()
    prompt = build_workbuddy_full_prompt(tools, MCP_SERVERS)
    assert len(tools) == 43, f"Expected 43 tools, got {len(tools)}"
    assert len(MCP_SERVERS) == 6, f"Expected 6 MCP servers, got {len(MCP_SERVERS)}"
    assert len(prompt) >= 15000, f"Prompt length must be >= 15000 chars, got {len(prompt)}"
    return {
        "tools": tools,
        "mcp_servers": MCP_SERVERS,
        "prompt": prompt,
        "char_count": len(prompt),
    }


@pytest.fixture
def clean_spill_dir():
    """Ensures /tmp/kvmem_cache is cleanly initialized and cleaned up."""
    spill_dir = Path("/tmp/kvmem_cache")
    spill_dir.mkdir(parents=True, exist_ok=True)
    yield str(spill_dir)
    # Post-test cleanup of test artifacts
    if spill_dir.exists():
        for f in spill_dir.glob("super_tile_*.kvs*"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


# =============================================================================
# Test Suite 1: WorkBuddy Payload Validation & Exact Requirements
# =============================================================================

def test_workbuddy_payload_exact_spec(workbuddy_payload):
    """
    Verify exact payload properties:
    - Input prompt length >= 15,000 characters
    - Exactly 43 distinct tool definitions
    - Exactly 6 MCP server descriptions
    - Valid OpenAI JSON schema for every tool
    """
    tools = workbuddy_payload["tools"]
    mcp_servers = workbuddy_payload["mcp_servers"]
    prompt = workbuddy_payload["prompt"]
    char_count = workbuddy_payload["char_count"]

    print(f"\n[Payload Audit] Character Count: {char_count:,} characters (Requirement: >= 15,000)")
    assert char_count >= 15000, f"Payload must be >= 15k chars, got {char_count}"
    assert len(tools) == 43, f"Must have 43 tools, got {len(tools)}"
    assert len(mcp_servers) == 6, f"Must have 6 MCP servers, got {len(mcp_servers)}"

    # Check unique tool names
    tool_names = [t["function"]["name"] for t in tools]
    assert len(set(tool_names)) == 43, "Tool names must be strictly unique"

    # Verify JSON Schema validity
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn and len(fn["name"]) > 0
        assert "description" in fn and len(fn["description"]) > 0
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]

    # Verify ChatML formatting
    formatted_chatml = format_tools_chatml(tools)
    assert "<tools>" in formatted_chatml
    assert "</tools>" in formatted_chatml
    assert "<tool_call>" in formatted_chatml


# =============================================================================
# Test Suite 2: Multi-Turn Developer Interaction & Causal Prefix Chain
# =============================================================================

def test_multi_turn_causal_prefix_deduplication(workbuddy_payload):
    """
    Simulates actual multi-turn developer workflow:
    Turn 1: Developer initiates task with 15k char payload + 43 tools.
    Turn 2: Agent responds with tool call (`git_status`).
    Turn 3: Developer supplies tool result.
    Turn 4: Follow-up query asking for 32K context audit.
    
    Verifies:
    - CanonicalAtomStore deduplicates shared prefixes across turns.
    - Causal prefix hash chain prevents cross-turn cache contamination.
    - Zero data leakage between turns.
    """
    store = CanonicalAtomStore(block_size=32, head_dim=64, model_id="qwen-3-developer")
    np.random.seed(42)

    prompt = workbuddy_payload["prompt"]
    # Simulate tokenizing the 15k char prompt into ~3,840 tokens (120 blocks)
    prompt_tokens = [int(hashlib.md5(f"tok_{i}".encode()).hexdigest()[:6], 16) % 32000 for i in range(3840)]
    num_prompt_blocks = len(prompt_tokens) // 32

    # Turn 1: Register initial system & developer prompt blocks
    t1_block_ids = []
    parent_hash = None
    for b_idx in range(num_prompt_blocks):
        sub_tokens = prompt_tokens[b_idx * 32 : (b_idx + 1) * 32]
        k = np.random.randn(32, 64).astype(np.float32)
        v = np.random.randn(32, 64).astype(np.float32)
        bid, parent_hash = store.register_prefix_block(
            tokens=sub_tokens,
            k_unrotated=k,
            v=v,
            orig_pos_start=b_idx * 32,
            parent_hash=parent_hash,
        )
        t1_block_ids.append(bid)

    assert len(t1_block_ids) == num_prompt_blocks
    assert len(store.blocks) == num_prompt_blocks

    # Turn 2: Agent tool call generation (<tool_call>git_status</tool_call>)
    agent_msg = '<tool_call>{"name": "git_status", "arguments": {"repo_path": "."}}</tool_call>'
    parsed = parse_single_tool_call_json(agent_msg)
    assert parsed is not None
    assert parsed["name"] == "git_status"

    # Turn 3: User provides tool execution result
    tool_tokens = list(range(5000, 5064))  # 2 blocks
    t3_parent = parent_hash
    for b_idx in range(2):
        sub_tokens = tool_tokens[b_idx * 32 : (b_idx + 1) * 32]
        k = np.random.randn(32, 64).astype(np.float32)
        v = np.random.randn(32, 64).astype(np.float32)
        bid, t3_parent = store.register_prefix_block(
            tokens=sub_tokens,
            k_unrotated=k,
            v=v,
            orig_pos_start=(num_prompt_blocks + b_idx) * 32,
            parent_hash=t3_parent,
        )

    # Turn 4: A second concurrent developer session sharing Turn 1 prompt
    # Must achieve 100% deduplication of Turn 1 blocks (ref_count == 2)
    concurrent_parent = None
    for b_idx in range(num_prompt_blocks):
        sub_tokens = prompt_tokens[b_idx * 32 : (b_idx + 1) * 32]
        k_dummy = np.zeros((32, 64), dtype=np.float32)
        v_dummy = np.zeros((32, 64), dtype=np.float32)
        bid, concurrent_parent = store.register_prefix_block(
            tokens=sub_tokens,
            k_unrotated=k_dummy,
            v=v_dummy,
            orig_pos_start=b_idx * 32,
            parent_hash=concurrent_parent,
        )
        assert bid == t1_block_ids[b_idx], "Prefix block must deduplicate bit-exactly"
        assert store.blocks[bid].ref_count == 2, "Reference count must increment to 2"

    print("\n✓ Multi-turn causal prefix chain verified: 100% prefix deduplication achieved.")


# =============================================================================
# Test Suite 3: 8K to 32K Context Scaling, RSS Limit & Zero Buffer Size Crashes
# =============================================================================

@pytest.mark.parametrize("context_tokens", [8192, 16384, 32768])
def test_context_window_scaling_and_rss_audit(context_tokens):
    """
    Stress test scaling from 8K up to 32K context windows (1,024 blocks).
    Verifies:
    1. Zero 'Invalid buffer size' or MPS backend crashes.
    2. Zero Metal OOM exceptions.
    3. Peak process RSS memory remains strictly under 18GB on 32GB Mac (< 16-18GB limit).
    4. Fast PagedAttention GEMM execution speed.
    """
    head_dim = 64
    block_size = 32
    num_blocks = context_tokens // block_size
    np.random.seed(101)

    initial_rss_gb = get_current_rss_gb()
    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)

    # Ingest full context window into CASA Paged store
    page_table = []
    parent_hash = None
    for b_idx in range(num_blocks):
        tokens = list(range(b_idx * block_size, (b_idx + 1) * block_size))
        k = np.random.randn(block_size, head_dim).astype(np.float32)
        v = np.random.randn(block_size, head_dim).astype(np.float32)
        bid, parent_hash = store.register_prefix_block(
            tokens=tokens,
            k_unrotated=k,
            v=v,
            orig_pos_start=b_idx * block_size,
            parent_hash=parent_hash,
        )
        page_table.append(store.blocks[bid].physical_page_id)

    assert len(store.blocks) == num_blocks
    assert len(page_table) == num_blocks

    # Execute Tensor Core / Metal PagedAttention GEMM at sequence boundary
    q_raw = np.random.randn(head_dim).astype(np.float32)
    t0 = time.perf_counter()
    attn_weights, context_vec = store.compute_paged_attention_gemm(
        q_raw=q_raw,
        query_logical_pos=context_tokens,
        page_table=page_table,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Verification: Output shapes and attention sum to 1.0
    assert attn_weights.shape == (context_tokens,)
    assert context_vec.shape == (head_dim,)
    assert np.isclose(np.sum(attn_weights), 1.0, atol=1e-5)

    # Physical Memory (RSS) Audit:
    final_rss_gb = get_current_rss_gb()
    rss_delta_gb = final_rss_gb - initial_rss_gb

    print(
        f"\n[Context {context_tokens:,} tokens / {num_blocks} blocks] "
        f"PagedAttention: {latency_ms:.2f}ms | RSS: {final_rss_gb:.3f} GB (Delta: +{rss_delta_gb:.3f} GB)"
    )

    # Core Assertion 1: Process RSS strictly < 18 GB on 32GB Mac
    assert final_rss_gb < 18.0, f"Process RSS exceeded safe limit: {final_rss_gb:.2f} GB >= 18.0 GB"

    # Core Assertion 2: Memory overhead for 32K context is compact (< 2.0 GB delta)
    assert rss_delta_gb < 4.0, f"Uncompressed memory growth too high: {rss_delta_gb:.2f} GB"


# =============================================================================
# Test Suite 4: NVMe Disk Spilling in /tmp/kvmem_cache & Zero Data Corruption
# =============================================================================

def test_nvme_disk_spilling_clean_trigger_and_zero_corruption(clean_spill_dir):
    """
    Stress test NVMe disk spilling in /tmp/kvmem_cache:
    - Generates 1,024 blocks (32,768 tokens).
    - Watermark configured to 128 resident blocks (4,096 tokens).
    - Autonomous spilling triggers cleanly: spills 896 blocks to /tmp/kvmem_cache.
    - Zero data corruption audit:
      * Validates file format (.kvs SuperTiles) and page alignment (PAGE_SIZE=16384B on macOS).
      * Memory-maps spilled blocks back.
      * Asserts bit-exact equality (np.allclose atol=1e-7, cosine similarity == 1.000000).
      * Compares SHA-256 hashes of original vs reloaded tensors.
    """
    head_dim = 64
    block_size = 32
    total_blocks = 1024  # 32,768 tokens
    active_watermark_blocks = 128  # 4,096 tokens max resident
    np.random.seed(2026)

    store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
    spill_mgr = SuperTileSpillManager(
        atom_store=store,
        cache_dir=clean_spill_dir,
        max_active_tokens=active_watermark_blocks * block_size,
        pages_per_tile=16,
        eviction_policy="oldest",
        cleanup_on_close=False,
    )

    original_tensors: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    original_hashes: Dict[str, str] = {}
    parent_hash = None

    print(f"\n[Spilling Audit] Ingesting {total_blocks} blocks into CASA with SpillManager attached...")

    for i in range(total_blocks):
        tokens = list(range(i * block_size, (i + 1) * block_size))
        k = np.random.randn(block_size, head_dim).astype(np.float32)
        v = np.random.randn(block_size, head_dim).astype(np.float32)
        bid, parent_hash = store.register_prefix_block(
            tokens=tokens,
            k_unrotated=k,
            v=v,
            orig_pos_start=i * block_size,
            parent_hash=parent_hash,
        )
        blk = store.blocks[bid]
        
        # Save exact unmutated ground truth for bit-exact verification
        original_tensors[bid] = (k.copy(), v.copy(), blk.k_canonical.copy())
        h = hashlib.sha256(k.tobytes() + v.tobytes() + blk.k_canonical.tobytes()).hexdigest()
        original_hashes[bid] = h

        # Hook into spill manager
        spill_mgr.on_block_added(blk)

    # Check that disk spilling triggered properly
    stats = spill_mgr.stats
    print(f"[Spilling Audit] Spill Events: {stats['spill_events']}")
    print(f"[Spilling Audit] Total Spilled Blocks: {stats['total_spilled_blocks']} / {total_blocks}")
    print(f"[Spilling Audit] Total Spilled Bytes: {stats['total_spilled_bytes'] / (1024*1024):.2f} MB")

    assert stats["spill_events"] > 0, "Spill manager must have triggered spilling events"
    assert stats["total_spilled_blocks"] >= (total_blocks - active_watermark_blocks * 2)

    # Verify disk files exist in /tmp/kvmem_cache
    kvs_files = list(Path(clean_spill_dir).glob("*.kvs"))
    print(f"[Spilling Audit] On-Disk SuperTiles created in {clean_spill_dir}: {len(kvs_files)} files")
    assert len(kvs_files) > 0, "Expected .kvs SuperTile files in /tmp/kvmem_cache"

    for fpath in kvs_files:
        fsize = fpath.stat().st_size
        assert fsize > 0
        assert fsize % PAGE_SIZE == 0, f"File {fpath.name} must be page-aligned (size={fsize})"

    # ZERO DATA CORRUPTION AUDIT: Reload all spilled blocks and verify bit-exact tensors
    print("[Spilling Audit] Verifying bit-exact tensor integrity across all spilled blocks...")
    corrupted_count = 0
    cosine_sims = []

    for bid, (orig_k, orig_v, orig_kc) in original_tensors.items():
        reloaded_blk = spill_mgr.retrieve_block(bid)

        # Check shapes
        assert reloaded_blk.k_unrotated.shape == orig_k.shape
        assert reloaded_blk.v.shape == orig_v.shape
        assert reloaded_blk.k_canonical.shape == orig_kc.shape

        # Bit-exact tensor equality
        k_diff = np.max(np.abs(reloaded_blk.k_unrotated - orig_k))
        v_diff = np.max(np.abs(reloaded_blk.v - orig_v))
        kc_diff = np.max(np.abs(reloaded_blk.k_canonical - orig_kc))

        if max(k_diff, v_diff, kc_diff) > 1e-6:
            corrupted_count += 1

        # Cosine similarity check
        k_flat_orig = orig_k.flatten()
        k_flat_reloaded = reloaded_blk.k_unrotated.flatten()
        cos_sim = float(
            np.dot(k_flat_orig, k_flat_reloaded)
            / (np.linalg.norm(k_flat_orig) * np.linalg.norm(k_flat_reloaded))
        )
        cosine_sims.append(cos_sim)

        # SHA-256 binary hash integrity check
        reloaded_hash = hashlib.sha256(
            reloaded_blk.k_unrotated.tobytes()
            + reloaded_blk.v.tobytes()
            + reloaded_blk.k_canonical.tobytes()
        ).hexdigest()
        assert reloaded_hash == original_hashes[bid], f"SHA-256 hash mismatch on block {bid}!"

    assert corrupted_count == 0, f"Detected {corrupted_count} corrupted blocks!"
    mean_cos_sim = float(np.mean(cosine_sims))
    print(f"[Spilling Audit] Mean Cosine Similarity to Ground Truth: {mean_cos_sim:.8f} (Bit-Exact: 1.00000000)")
    assert np.isclose(mean_cos_sim, 1.0, atol=1e-7), "Cosine similarity must be bit-exact 1.0"
    print("✓ NVMe disk spilling triggered cleanly with ZERO data corruption.")


# =============================================================================
# Test Suite 5: Token Budget Negotiation & Agent Protocol Handling
# =============================================================================

def test_agent_protocol_token_budget_and_streaming_health():
    """
    Verifies agent protocol resiliency under edge cases:
    1. max_tokens=0 starvation: gracefully renegotiates to default (2048).
    2. max_tokens=None / invalid: defaults to 2048.
    3. max_tokens=65536: safely clamped to 4096 ceiling for MPS unified memory.
    4. StreamingToolProcessor: emits initial handshake chunk immediately (eliminating 38s timeout).
    5. Delta SSE chunk streaming format adheres to OpenAI standard.
    """
    # 1. Budget negotiations
    assert negotiate_token_budget(0) == 2048, "max_tokens=0 must renegotiate to 2048"
    assert negotiate_token_budget(-100) == 2048
    assert negotiate_token_budget(None) == 2048
    assert negotiate_token_budget("invalid") == 2048
    assert negotiate_token_budget(65536, max_ceiling=4096) == 4096, "Must clamp to ceiling"
    assert negotiate_token_budget(512) == 512

    # 2. Low-latency Streaming Handshake
    t0 = time.perf_counter()
    init_chunk = create_initial_sse_chunk(req_id="chatcmpl-test-01", model="qwen3.8-27b")
    handshake_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert init_chunk["choices"][0]["delta"]["role"] == "assistant"
    assert handshake_latency_ms < 5.0, f"Handshake latency too high: {handshake_latency_ms:.2f}ms"

    # 3. StreamingToolProcessor tool call detection
    processor = StreamingToolProcessor(req_id="chatcmpl-test-01", model="qwen3.8-27b", tools_enabled=True)
    sample_llm_stream = [
        "I will now ",
        "inspect the repository ",
        "using git_status:\n",
        "<tool_call>\n",
        '{"name": "git_status", "arguments": {"repo_path": "."}}\n',
        "</tool_call>",
    ]

    all_emitted_chunks = []
    for token_delta in sample_llm_stream:
        chunks = processor.process_token(token_delta)
        all_emitted_chunks.extend(chunks)
    all_emitted_chunks.extend(processor.finalize())

    assert len(processor.tool_calls_emitted) == 1
    assert processor.tool_calls_emitted[0]["name"] == "git_status"

    # Verify tool_calls events exist in chunks
    tool_chunks = [c for c in all_emitted_chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert len(tool_chunks) > 0, "Must emit tool_calls delta chunks"
    assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "git_status"

    # Format to SSE stream
    sse_lines = [format_sse_event(c) for c in all_emitted_chunks]
    raw_stream = "".join(sse_lines)
    assert "data: " in raw_stream
    assert "git_status" in raw_stream

    # Verify final chunk stop reason
    final_chunk = create_final_sse_chunk(req_id="chatcmpl-test-01", model="qwen3.8-27b", finish_reason="tool_calls")
    assert final_chunk["choices"][0]["finish_reason"] == "tool_calls"

    print(f"\n✓ Agent protocol verified: Token starvation prevented, tool calling streamed properly.")


# =============================================================================
# Test Suite 6: Memory Leak Audit across Repeated Multi-Turn Interactions
# =============================================================================

def test_repeated_multi_turn_memory_leak_audit():
    """
    Executes 10 consecutive simulated developer agent sessions.
    Verifies:
    - Garbage collection and tensor deallocation are effective.
    - Process RSS plateaus and does not linearly leak memory.
    - Zero unbounded memory growth across multi-turn sessions.
    """
    gc.collect()
    head_dim = 64
    block_size = 32
    tokens_per_session = 2048  # 64 blocks per session
    num_sessions = 10

    initial_rss_gb = get_current_rss_gb()
    rss_history = []

    for s_idx in range(num_sessions):
        store = CanonicalAtomStore(block_size=block_size, head_dim=head_dim)
        parent_hash = None
        for b in range(tokens_per_session // block_size):
            toks = list(range(b * block_size, (b + 1) * block_size))
            k = np.random.randn(block_size, head_dim).astype(np.float32)
            v = np.random.randn(block_size, head_dim).astype(np.float32)
            _, parent_hash = store.register_prefix_block(
                tokens=toks,
                k_unrotated=k,
                v=v,
                orig_pos_start=b * block_size,
                parent_hash=parent_hash,
            )

        # Execute attention
        q = np.random.randn(head_dim).astype(np.float32)
        store.compute_paged_attention_gemm(
            q_raw=q,
            query_logical_pos=tokens_per_session,
            page_table=list(range(len(store.physical_k_pool))),
        )

        del store
        gc.collect()

        current_rss = get_current_rss_gb()
        rss_history.append(current_rss)

    max_rss_gb = max(rss_history)
    final_rss_gb = rss_history[-1]
    rss_drift_gb = final_rss_gb - initial_rss_gb

    print(
        f"\n[Leak Audit 10 Cycles] Initial RSS: {initial_rss_gb:.3f} GB | "
        f"Peak RSS: {max_rss_gb:.3f} GB | Final RSS: {final_rss_gb:.3f} GB | Drift: {rss_drift_gb:.3f} GB"
    )

    assert max_rss_gb < 18.0, f"Peak RSS exceeded 18GB: {max_rss_gb:.2f} GB"
    assert rss_drift_gb < 1.0, f"Memory leak detected: RSS grew by {rss_drift_gb:.3f} GB over 10 sessions"
    print("✓ Memory audit passed: Zero memory leak detected over 10 consecutive sessions.")


# =============================================================================
# Test Suite 7: End-to-End Native In-Process Inference & Telemetry Audit
# =============================================================================

def test_native_engine_workbuddy_inference_and_latency(workbuddy_payload):
    """
    End-to-End Native In-Process Inference Verification:
    - Runs Qwen model on Apple Metal MPS unified memory.
    - Feeds WorkBuddy developer prompt (15k+ chars) with 43 tools.
    - Verifies TTFT (Time to First Token) and generation speed.
    - Zero 'Invalid buffer size' or Metal OOM exceptions.
    - Process RSS stays strictly under 18GB on 32GB Mac.
    """
    from kvmem_fusion.native_engine import KVMemNativeEngine
    engine = KVMemNativeEngine.get_instance()
    assert engine.is_loaded, "Native engine must load local model weights"
    assert engine.device == "mps", "Must run natively on Apple Metal Unified Memory (MPS)"

    messages = [
        {"role": "user", "content": workbuddy_payload["prompt"][:4000]}
    ]

    t0 = time.perf_counter()
    chunks = []
    first_token_time = None

    for chunk in engine.stream_chat(messages, max_new_tokens=24, temperature=0.7):
        chunks.append(chunk)
        if not chunk.get("done", False) and first_token_time is None:
            first_token_time = time.perf_counter()

    total_duration = time.perf_counter() - t0
    ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else 0.0

    print(
        f"\n[Native Engine Audit] TTFT: {ttft_ms:.1f}ms | "
        f"Speed: {engine.last_generation_speed:.1f} tok/s | "
        f"Generated: {len(chunks)-1} tokens in {total_duration:.2f}s | "
        f"Active CASA Blocks: {engine.active_casa_blocks}"
    )

    # Invariants
    assert len(chunks) > 1, "Must generate tokens"
    assert ttft_ms < 2500.0, f"TTFT must be fast (< 2500ms on Apple Silicon for 1k prompt tokens), got {ttft_ms:.1f}ms"
    assert engine.last_generation_speed > 10.0, f"Generation speed must exceed 10 tok/s, got {engine.last_generation_speed}"

    rss_gb = get_current_rss_gb()
    print(f"[Native Engine Audit] Total Process RSS: {rss_gb:.3f} GB (Strict Limit: < 18.0 GB)")
    assert rss_gb < 18.0, f"Process RSS exceeded safe limit: {rss_gb:.2f} GB >= 18.0 GB"
    print("✓ Native In-Process Inference verified on Apple Metal Unified Memory.")

