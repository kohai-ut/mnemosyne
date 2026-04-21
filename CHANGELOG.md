# Changelog

Mnemosyne uses [Simple Versioning](https://gist.github.com/jonlow/6f7610566408a8efaa4a):
given a version number **MAJOR.MINOR**, increment the:

- **MINOR** after every iteration of development, testing, and quality assurance.
- **MAJOR** for the first production release (1.0) or for significant new functionality (2.0, 3.0, etc.).

---

## 1.1

- Export / import memory for cross-machine migration
- Official bug report issue template
- Fix: `get_episodic_stats()` no longer filters by `session_id`
- Fix: episodic recall tracking now updates `recall_count` for global memories
- Fix: normalize sqlite-vec distances for int8/bit vectors (dense_score was always 0.0)
- Fix: `Mnemosyne.remember()` now accepts `valid_until` and `scope`
- Fix: added missing `Mnemosyne.invalidate()` method
- Fix: dynamic `session_id` from `HERMES_SESSION_ID` env var in tools.py

## 1.0

First major release. Production-ready.

- BEAM architecture: working_memory, episodic_memory, scratchpad
- Native vector search via sqlite-vec (HNSW-style)
- FTS5 full-text hybrid search (50% vector + 30% FTS + 20% importance)
- Dense retrieval via fastembed (bge-small-en-v1.5)
- Automatic sleep/consolidation cycle
- Temporal triples (time-aware knowledge graph)
- AAAK context compression
- Configurable vector compression: float32, int8, bit
- Cross-session global memory (`scope="global"`)
- Export/import for backup and migration
- Hermes plugin integration with CLI subcommands
- Sub-millisecond latency on CPU
