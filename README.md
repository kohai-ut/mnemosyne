# Mnemosyne

**Local-first AI memory. No servers. No Docker. No PostgreSQL. Just Python + SQLite.**

Mnemosyne is an in-process memory layer for AI agents. You call a Python function — it stores, searches, and retrieves memories through SQLite. No HTTP, no containers, no API keys. `pip install` and you're running.

## Quick Start

```bash
# Install
pip install mnemosyne-memory

# One-time setup (downloads ~67MB embedding model)
mnemosyne-install
```

```python
from mnemosyne import remember, recall

# Store memories
remember("User prefers dark mode and uses VS Code", importance=0.8)
remember("Deployed v2.0.0 to production on Tuesday", importance=0.6)

# Search — hybrid vector + keyword retrieval
results = recall("user preferences")
print(results[0]["content"])  # "User prefers dark mode and uses VS Code"
```

Or use the CLI:

```bash
mnemosyne store "User prefers dark mode" conversation 0.8
mnemosyne recall "user preferences"
mnemosyne stats
```

## Architecture: BEAM

**Bilevel Episodic-Associative Memory** — three SQLite tables, one file:

| Tier | Purpose | Details |
|---|---|---|
| **Working memory** | Hot, recent context | FTS5-indexed. Auto-injected via Hermes `pre_llm_call` hook. TTL-based eviction (default 24h). Max 10,000 items. |
| **Episodic memory** | Long-term consolidated storage | Populated by `sleep()`. Hybrid sqlite-vec + FTS5 search. |
| **Scratchpad** | Temporary agent workspace | Not searchable, not consolidated. Max 1,000 items. |

**Memory flow:** `remember()` → working memory → `sleep()` consolidation → episodic memory with embeddings.

### Hybrid Scoring

Recall combines three signals, configurable per query:

```
score = vec_weight × vector_similarity
      + fts_weight × FTS5_rank
      + importance_weight × importance
```

Default weights: **50% vector, 30% FTS, 20% importance**. Override via parameters or environment variables (`MNEMOSYNE_VEC_WEIGHT`, `MNEMOSYNE_FTS_WEIGHT`, `MNEMOSYNE_IMPORTANCE_WEIGHT`).

Temporal recall adds an exponential decay boost:

```python
results = recall(
    "deployments",
    temporal_weight=0.5,        # Enable temporal scoring
    temporal_halflife=48.0,     # 48-hour halflife
    query_time="2026-04-29T12:00:00"  # Reference point
)
```

## Features

### Entity Extraction

Regex-based entity extraction with Levenshtein fuzzy matching. No spaCy, no PyTorch.

```python
remember(
    "Met with Abdias J about the Mnemosyne v2 release",
    extract_entities=True
)
# Extracts: "Abdias J", "Mnemosyne" — stored as triples
# Fuzzy match: querying "Abdias" finds "Abdias J" (similarity: 0.925)
```

Catches `@mentions`, `#hashtags`, `"quoted phrases"`, and capitalized sequences (2–5 words). Misses pronouns and complex coreferences — this is a deliberate trade-off for speed and zero dependencies.

### LLM-Driven Fact Extraction

Extract structured facts from raw text using an LLM, with a graceful fallback chain:

1. Remote OpenAI-compatible API (if `MNEMOSYNE_LLM_BASE_URL` is set)
2. Local ctransformers GGUF model
3. Skip — extraction fails silently, memory is still stored

```python
remember(
    "User said they prefer Python over JavaScript for backend work",
    extract=True  # Extracts 2-5 factual statements as triples
)
```

### Memory Banks

Per-bank SQLite isolation for domain separation:

```python
from mnemosyne.core.banks import BankManager
from mnemosyne import Mnemosyne

# Create isolated banks
BankManager().create_bank("work")
BankManager().create_bank("personal")

# Use them
work_mem = Mnemosyne(bank="work")
work_mem.remember("Sprint review scheduled for Friday")

# Or via module-level API
from mnemosyne import remember, set_bank
set_bank("work")
remember("Deployed to staging")  # Goes to "work" bank
```

```bash
mnemosyne bank list
mnemosyne bank create research
mnemosyne mcp --bank work  # MCP server scoped to a bank
```

### MCP Server

6 tools, 2 transports, for any MCP-compatible client:

```bash
# stdio — for Claude Desktop, etc.
mnemosyne mcp

# SSE — for web clients
mnemosyne mcp --transport sse --port 8080
```

| Tool | Description |
|---|---|
| `mnemosyne_remember` | Store a memory |
| `mnemosyne_recall` | Search with hybrid scoring |
| `mnemosyne_sleep` | Run consolidation |
| `mnemosyne_scratchpad_read` | Read scratchpad |
| `mnemosyne_scratchpad_write` | Write to scratchpad |
| `mnemosyne_get_stats` | Memory statistics |

### Hermes Integration

Native plugin for Hermes agents — 15 tools + 3 hooks (in-process, zero serialization):

- **Hooks:** `pre_llm_call` (context injection), `on_session_start`, `post_tool_call` (memory capture)
- **Tools:** `mnemosyne_remember`, `mnemosyne_recall`, `mnemosyne_stats`, `mnemosyne_sleep`, `mnemosyne_scratchpad_write`, `mnemosyne_scratchpad_read`, `mnemosyne_scratchpad_clear`, `mnemosyne_triple_add`, `mnemosyne_triple_query`, `mnemosyne_invalidate`, `mnemosyne_export`, `mnemosyne_import`, `mnemosyne_update`, `mnemosyne_forget`, `mnemosyne_diagnose`

### Streaming, Patterns, Plugins

```python
from mnemosyne import Mnemosyne

mem = Mnemosyne()

# Event streaming — push (callbacks) and pull (iterator)
mem.enable_streaming()
for event in mem.stream:
    print(event.event_type, event.memory_id)

# Pattern detection — temporal, content, sequence patterns
patterns = mem.summarize_patterns()

# Delta sync between instances
delta = mem.sync_to("peer_agent_id")
mem.sync_from("peer_agent_id", delta["delta"])

# Plugin system with lifecycle hooks
from mnemosyne.core.plugins import MnemosynePlugin

class MyPlugin(MnemosynePlugin):
    name = "audit_logger"
    def on_remember(self, memory_id, content, **kwargs):
        print(f"AUDIT: stored {memory_id}")

mem.plugins.register(MyPlugin())
```

## Performance

Benchmarks from this VPS (single-machine, in-process SQLite):

| Metric | Value |
|---|---|
| **Store latency** | ~3–18ms per memory (includes embedding computation) |
| **Recall latency** | ~2–10ms at 10K corpus |
| **Database footprint** | ~50–100MB per 10K memories |
| **Embedding model** | One-time ~67MB download (fastembed ONNX) |
| **Runtime memory** | ~10–20MB per session |
| **Default vector type** | int8 — 384 bytes per 384-dim vector |
| **Optional vector type** | bit — 48 bytes per 384-dim vector |

> **Caveat:** These latency numbers reflect the architectural advantage of in-process SQLite calls vs. HTTP round-trips to a separate service. The retrieval quality advantage of systems with cross-encoder reranking or 4-way parallel retrieval is real — Mnemosyne trades some precision for speed and simplicity.

## CLI Reference

```
mnemosyne store <content> [source] [importance]   Store a memory
mnemosyne recall <query> [top_k]                  Search memories
mnemosyne update <id> <content> [importance]      Update a memory
mnemosyne delete <id>                             Delete a memory
mnemosyne stats                                   Show statistics
mnemosyne sleep                                   Run consolidation
mnemosyne export [file.json]                      Export to JSON
mnemosyne import <file.json>                      Import from JSON
mnemosyne bank list|create|delete [name]          Manage memory banks
mnemosyne mcp [--transport sse] [--port 8080]     Start MCP server
mnemosyne diagnose                                Health check
```

## Configuration

Mnemosyne works out of the box. All configuration is optional via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Database directory |
| `MNEMOSYNE_VEC_TYPE` | `int8` | Vector type: `float32`, `int8`, or `bit` |
| `MNEMOSYNE_VEC_WEIGHT` | `0.5` | Vector similarity weight |
| `MNEMOSYNE_FTS_WEIGHT` | `0.3` | FTS5 keyword weight |
| `MNEMOSYNE_IMPORTANCE_WEIGHT` | `0.2` | Importance weight |
| `MNEMOSYNE_RECENCY_HALFLIFE` | `168` | Recency decay halflife in hours (1 week) |
| `MNEMOSYNE_WM_MAX_ITEMS` | `10000` | Max working memory items |
| `MNEMOSYNE_WM_TTL_HOURS` | `24` | Working memory TTL |
| `MNEMOSYNE_EP_LIMIT` | `50000` | Episodic memory recall limit |
| `MNEMOSYNE_LLM_BASE_URL` | — | Remote LLM endpoint for fact extraction |

## Installation Options

```bash
# Core (keyword search only — no dependencies beyond stdlib + SQLite)
pip install mnemosyne-memory

# + Semantic search (vector embeddings via fastembed ONNX)
pip install mnemosyne-memory[embeddings]

# + Local LLM for fact extraction and consolidation
pip install mnemosyne-memory[llm]

# Everything
pip install mnemosyne-memory[all]
```

**Core has zero required dependencies** — just Python stdlib and SQLite. Semantic search and LLM features are opt-in layers that degrade gracefully when unavailable.

## What Mnemosyne Is NOT

- **Not a database.** It's a memory layer built on SQLite. If you need PostgreSQL, use PostgreSQL.
- **Not a server.** It's an in-process Python library. The MCP server is for tool integration, not multi-machine sharing.
- **Not for multi-machine.** No network API. Same-machine sync via shared SQLite file or JSON export/import.
- **Not an NLP pipeline.** Entity extraction is regex + Levenshtein. No spaCy, no neural NER.
- **Not a retrieval quality benchmark winner.** Single-pass hybrid scoring. No cross-encoder reranking, no 4-way retrieval fusion.

## When to Use Mnemosyne

**Good fit:**
- Single-user, single-machine AI agents (Hermes, Claude Desktop, custom agents)
- Resource-constrained environments (VPS, CI, ephemeral VMs)
- `pip install` simplicity — no Docker, no containers
- Interactive agent loops where latency matters
- MCP-compatible tool integration

**Not a good fit:**
- Multi-machine agent clusters needing shared memory over a network
- Applications requiring automatic entity normalization or coreference resolution
- Use cases needing the highest retrieval precision (cross-encoder reranking)
- Multi-tenant SaaS with access control requirements

See [docs/comparison.md](docs/comparison.md) for a detailed, honest comparison with Hindsight self-hosted.

## Project

- **Author:** Abdias J ([AxDSan](https://github.com/AxDSan))
- **License:** MIT
- **Repository:** [github.com/AxDSan/mnemosyne](https://github.com/AxDSan/mnemosyne)
- **PyPI:** [mnemosyne-memory](https://pypi.org/project/mnemosyne-memory/)

---

*Every feature and benchmark in this README has been verified against the source code. If anything here is inaccurate, please open an issue — we'll fix it.*
