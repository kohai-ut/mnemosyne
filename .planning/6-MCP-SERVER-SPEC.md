# Phase 6: MCP Server — Specification

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 6 |
| **Name** | MCP Server |
| **Requirement** | REQ-4 |
| **Goal** | Model Context Protocol server for cross-agent sharing via open standard |
| **Complexity** | High |
| **Estimated Duration** | 3-4 sessions |

## Locked Requirements

- REQ-4: MCP Server for Cross-Agent Sharing (P0)

## Context

### Current State
Mnemosyne is in-process only. Cross-agent sharing is limited to:
- Same-machine: shared SQLite file via `scope='global'`
- Cross-machine: manual `export_to_file()` / `import_from_file()` JSON transfer

There is no network API for the BEAM architecture. The old REST server (`cli.py server`) wraps the flat legacy model, not BEAM.

### What Hindsight Does
- Custom HTTP API on port 8888 (FastAPI)
- Any agent with network access can share the memory bank
- HindClaw adds multi-tenant access control

### What Mnemosyne Will Do Differently
- **MCP (Model Context Protocol)** instead of custom HTTP — open standard, interoperable
- **stdio transport** for local agents (Claude Desktop, etc.)
- **SSE transport** for remote/web clients
- **No new dependencies** — MCP is lightweight JSON-RPC over stdio/SSE
- **Tool-based exposure** — `remember`, `recall`, `sleep`, `scratchpad_read`, `scratchpad_write` as MCP tools

## Implementation Decisions

### Decision 1: MCP Library Choice
**Options:**
- A) `mcp` official SDK (Python package from Anthropic)
- B) Pure stdlib implementation (JSON-RPC over stdio/SSE)
- C) `anyio` + `httpx` for SSE transport

**Chosen: A)** — Official `mcp` SDK. It's lightweight (~50KB), well-documented, and handles stdio/SSE transports. Install as optional extra: `pip install mnemosyne-memory[mcp]`.

### Decision 2: Transport Support
**Options:**
- A) stdio only (simplest, Claude Desktop compatible)
- B) SSE only (web-friendly)
- C) Both (most flexible)

**Chosen: C)** — Both. stdio is default (local agents), SSE is opt-in (`--transport sse --port 8080`).

### Decision 3: Tool Schema Design
**Exposed Tools:**
1. `mnemosyne_remember` — Store a memory
2. `mnemosyne_recall` — Search memories
3. `mnemosyne_sleep` — Run consolidation
4. `mnemosyne_scratchpad_read` — Read scratchpad
5. `mnemosyne_scratchpad_write` — Write to scratchpad
6. `mnemosyne_get_stats` — Get memory system statistics

**Not exposed:**
- `forget` — destructive, not exposed to remote agents
- `invalidate` — destructive, not exposed
- `export_to_file` / `import_from_file` — file system access, security risk

### Decision 4: Configuration
- `MNEMOSYNE_MCP_TRANSPORT` — `stdio` (default) or `sse`
- `MNEMOSYNE_MCP_PORT` — port for SSE transport (default 8080)
- `MNEMOSYNE_MCP_BANK` — default bank for MCP operations (default "default")

## Acceptance Criteria

- [ ] `mnemosyne mcp` command starts MCP server
- [ ] Stdio transport works (for Claude Desktop, etc.)
- [ ] SSE transport works (for web clients)
- [ ] Tool schemas match MCP specification
- [ ] Any MCP client can discover and call Mnemosyne tools
- [ ] `mnemosyne_remember` stores memory with optional entity/fact extraction
- [ ] `mnemosyne_recall` searches with all Phase 1-5 features (entities, facts, temporal, banks)
- [ ] `mnemosyne_sleep` triggers consolidation
- [ ] `mnemosyne_scratchpad_read` / `write` work
- [ ] `mnemosyne_get_stats` returns memory system stats
- [ ] No breaking changes to existing API
- [ ] MCP is optional dependency — core works without it

## Verification Steps

1. **Unit tests:**
   - `test_mcp_server_init()` — server starts with correct transport
   - `test_mcp_tool_schemas()` — schemas match MCP spec
   - `test_mcp_remember()` — tool call stores memory
   - `test_mcp_recall()` — tool call returns results
   - `test_mcp_sleep()` — tool call triggers consolidation
   - `test_mcp_stdio_transport()` — stdio communication works
   - `test_mcp_sse_transport()` — SSE endpoint responds

2. **Integration tests:**
   - `test_mcp_claude_desktop_compat()` — stdio transport compatible
   - `test_mcp_cross_agent_sharing()` — two agents share same bank

3. **Manual verification:**
   - Add Mnemosyne MCP to Claude Desktop config
   - Verify tool discovery and execution

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| MCP SDK adds dependency bloat | Optional extra only, not in core package |
| SSE transport security | No auth by default — document that SSE should run behind reverse proxy |
| Tool schema drift | Unit tests verify schema against MCP spec |
| Performance degradation | MCP server is async, non-blocking to core operations |

## Files to Create/Modify

### New Files
- `mnemosyne/mcp_server.py` — MCP server implementation
- `mnemosyne/mcp_tools.py` — Tool definitions and handlers
- `tests/test_mcp_server.py` — Unit tests
- `tests/test_mcp_integration.py` — Integration tests

### Modified Files
- `mnemosyne/cli.py` — Add `mcp` subcommand
- `setup.py` / `pyproject.toml` — Add `[mcp]` optional extra
- `mnemosyne/__init__.py` — Conditionally import MCP if available

## Dependencies

- `mcp` (optional, via `pip install mnemosyne-memory[mcp]`)
- `anyio` (comes with mcp)
- `httpx` (comes with mcp)
- No new core dependencies

## Notes

- MCP is rapidly evolving standard. Pin to known working version in extras.
- Document that MCP server is for sharing, not for high-throughput — core in-process API remains fastest.
- Consider future: MCP auth, multi-bank routing, streaming recall via MCP.

---

*Phase 6 Specification — PlanForge*
*Date: 2026-04-29*
