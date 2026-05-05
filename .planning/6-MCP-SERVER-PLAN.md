# Phase 6: MCP Server — Implementation Plan

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 6 |
| **Name** | MCP Server |
| **Spec** | 6-MCP-SERVER-SPEC.md |

## Task Waves

### Wave 1: MCP SDK Setup & Tool Definitions (Independent)

#### Task 6.1: Add MCP Optional Dependency
**File:** `setup.py` or `pyproject.toml`
**Description:** Add `[mcp]` extras_require with `mcp>=1.0.0` and `anyio`.
**Acceptance:**
- `pip install mnemosyne-memory[mcp]` installs mcp SDK
- `pip install mnemosyne-memory` does NOT install mcp
- Import guard: `try/except ImportError` around mcp imports
**Estimated:** 15 minutes

#### Task 6.2: Create MCP Tool Definitions
**File:** `mnemosyne/mcp_tools.py`
**Description:** Define all 6 MCP tool schemas using `mcp.types.Tool`.
**Tools:**
1. `mnemosyne_remember` — content, source, importance, metadata, extract_entities, extract, bank
2. `mnemosyne_recall` — query, top_k, bank, temporal_weight, query_time, vec_weight, fts_weight, importance_weight
3. `mnemosyne_sleep` — dry_run, bank
4. `mnemosyne_scratchpad_read` — bank
5. `mnemosyne_scratchpad_write` — content, bank
6. `mnemosyne_get_stats` — bank
**Acceptance:**
- All tools have valid JSON schemas
- Schemas include descriptions for LLM consumption
- No destructive tools (forget, invalidate, export/import)
**Estimated:** 45 minutes

#### Task 6.3: Create Tool Handlers
**File:** `mnemosyne/mcp_tools.py` (continued)
**Description:** Implement handler functions for each tool that call Mnemosyne API.
**Acceptance:**
- `handle_remember()` calls `Mnemosyne.remember()` with correct params
- `handle_recall()` calls `Mnemosyne.recall()` with all Phase 1-5 features
- `handle_sleep()` calls `Mnemosyne.sleep()`
- `handle_scratchpad_read/write()` call appropriate methods
- `handle_get_stats()` returns JSON-serializable stats
- All handlers return MCP-compliant `CallToolResult`
**Estimated:** 1 hour

### Wave 2: MCP Server Implementation (Depends on Wave 1)

#### Task 6.4: Stdio Transport Server
**File:** `mnemosyne/mcp_server.py`
**Description:** Implement `run_stdio_server()` using `mcp.server.Server` and `stdio_server`.
**Acceptance:**
- Server starts and listens on stdin
- Responds to `initialize` request with capabilities
- Routes tool calls to correct handlers
- Graceful shutdown on EOF
**Estimated:** 1 hour

#### Task 6.5: SSE Transport Server
**File:** `mnemosyne/mcp_server.py` (continued)
**Description:** Implement `run_sse_server()` using `mcp.server.sse.SseServerTransport`.
**Acceptance:**
- HTTP server on configurable port (default 8080)
- `/sse` endpoint for SSE connection
- `/messages` endpoint for client-to-server messages
- CORS headers for web client compatibility
**Estimated:** 1 hour

#### Task 6.6: CLI Integration
**File:** `mnemosyne/cli.py`
**Description:** Add `mcp` subcommand with `--transport` and `--port` args.
**Acceptance:**
- `mnemosyne mcp` starts stdio server
- `mnemosyne mcp --transport sse --port 9000` starts SSE server
- `mnemosyne mcp --bank project_a` sets default bank
- Help text explains transports
**Estimated:** 30 minutes

### Wave 3: Testing (Depends on Wave 2)

#### Task 6.7: Unit Tests — Tool Schemas
**File:** `tests/test_mcp_server.py`
**Description:** Verify tool schemas match MCP spec and are valid JSON.
**Acceptance:**
- All 6 tool schemas parse as valid JSON
- Schemas have `type: object` and `properties`
- Required fields are marked
**Estimated:** 30 minutes

#### Task 6.8: Unit Tests — Tool Handlers
**File:** `tests/test_mcp_server.py` (continued)
**Description:** Test each handler with mocked Mnemosyne instance.
**Acceptance:**
- `handle_remember()` returns success with memory_id
- `handle_recall()` returns list of results
- `handle_sleep()` returns consolidation stats
- Error handling returns MCP-compliant error results
**Estimated:** 45 minutes

#### Task 6.9: Integration Test — Stdio Transport
**File:** `tests/test_mcp_integration.py`
**Description:** Spawn server process, send JSON-RPC over stdin, verify responses.
**Acceptance:**
- Server responds to `initialize` with correct protocol version
- Server lists tools via `tools/list`
- `tools/call` with `mnemosyne_remember` stores memory
- Server handles concurrent requests
**Estimated:** 1 hour

#### Task 6.10: Integration Test — SSE Transport
**File:** `tests/test_mcp_integration.py` (continued)
**Description:** Start SSE server, connect via HTTP, verify tool discovery and execution.
**Acceptance:**
- GET `/sse` returns event stream
- POST `/messages` routes to correct handler
- Tool results returned as SSE events
**Estimated:** 45 minutes

### Wave 4: Documentation & Polish (Depends on Wave 3)

#### Task 6.11: Import Guard in `__init__.py`
**File:** `mnemosyne/__init__.py`
**Description:** Conditionally expose MCP server if mcp package is installed.
**Acceptance:**
- `from mnemosyne import mcp_server` works if mcp installed
- No ImportError if mcp not installed
- Graceful degradation
**Estimated:** 15 minutes

#### Task 6.12: Performance Check
**Description:** Verify MCP server doesn't degrade core performance.
**Acceptance:**
- `remember()` latency unchanged with MCP server running
- `recall()` latency unchanged
- Memory usage increase <10MB with MCP server active
**Estimated:** 15 minutes

## Verification Checklist

- [ ] `pip install mnemosyne-memory[mcp]` works
- [ ] `mnemosyne mcp` starts stdio server
- [ ] `mnemosyne mcp --transport sse` starts SSE server
- [ ] MCP client can discover all 6 tools
- [ ] `mnemosyne_remember` stores memory via MCP
- [ ] `mnemosyne_recall` searches via MCP with all features
- [ ] `mnemosyne_sleep` triggers consolidation via MCP
- [ ] Stdio transport compatible with Claude Desktop
- [ ] SSE transport responds to HTTP requests
- [ ] No breaking changes to existing API
- [ ] MCP is truly optional — core works without it
- [ ] All tests pass

## Ship Criteria

1. All tasks in Waves 1-4 complete
2. All verification checks pass
3. MCP server runs without errors for 5 minutes under load
4. No performance regression in core API
5. Import guard works — clean install without mcp extra

---

*Phase 6 Implementation Plan — PlanForge*
*Date: 2026-04-29*
