# Phase 1: Entity Sketching — Specification

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 1 |
| **Name** | Entity Sketching |
| **Requirement** | REQ-1 |
| **Goal** | Lightweight entity extraction and fuzzy matching without heavy NLP deps |
| **Complexity** | Medium |
| **Estimated Duration** | 2-3 sessions |

## Locked Requirements

- REQ-1: Entity Sketching System (P0)

## Context

### Current State
Mnemosyne stores raw text memories with no entity awareness. Two memories mentioning "Abdias" and "Abdias J" are completely unrelated. The TripleStore exists but is only used for temporal triples (`occurred_on`).

### What Hindsight Does
- spaCy + LLM entity extraction during Retain
- trigram/full resolution strategies for entity matching
- co-occurrence tracking via `entity_cooccurrences` table
- batch resolution with deadlock prevention

### What Mnemosyne Will Do Differently
- **No spaCy dependency** — regex + optional LLM only
- **No entity graph** — entity mentions are triples, not nodes
- **Fuzzy matching via Levenshtein** — pure Python, no external libs
- **Co-occurrence via TripleStore queries** — not a separate table
- **On-demand extraction** — not during every `remember()`, only when requested or when LLM is available

## Implementation Decisions

### Decision 1: Entity Extraction Strategy
**Options:**
- A) Regex-only (no LLM dependency)
- B) LLM-only (best quality, requires LLM)
- C) Tiered: regex first, LLM enhancement when available

**Chosen: C)** — Tiered approach. Regex extracts capitalized words and quoted phrases. LLM (when available) does better disambiguation. Both store results as triples.

### Decision 2: Fuzzy Matching Algorithm
**Options:**
- A) SQLite `LIKE` with wildcards (fast, imprecise)
- B) Levenshtein distance (pure Python, ~100 lines)
- C) `fuzzywuzzy`/`rapidfuzz` library (external dep)

**Chosen: B)** — Pure Python Levenshtein implementation. ~100 lines, no external dependency. Threshold: 0.8 similarity for match.

### Decision 3: Storage Schema
**Options:**
- A) New `entities` table (separate from TripleStore)
- B) Extend TripleStore with entity triples
- C) Both (entities table + triples)

**Chosen: B)** — Use existing TripleStore. Schema: `(subject=memory_id, predicate="mentions", object="entity_name", confidence=0.9)`. No new tables.

### Decision 4: API Design
**Options:**
- A) `remember(content, extract_entities=True)` — explicit flag
- B) Auto-extract on every `remember()` — always on
- C) Configurable default via env var

**Chosen: A)** — Explicit opt-in per call. Global default via `MNEMOSYNE_AUTO_EXTRACT_ENTITIES` env var.

## Acceptance Criteria

- [ ] `remember("I met Abdias yesterday", extract_entities=True)` creates triple `(memory_id, "mentions", "Abdias")`
- [ ] `recall("Abdias")` returns memories mentioning "Abdias J" via fuzzy match (Levenshtein >= 0.8)
- [ ] `triples.query(subject="*", predicate="mentions", object="Abdias")` returns all mentions
- [ ] Regex extraction works without LLM (capitalized words, quoted phrases)
- [ ] LLM enhancement works when available (better disambiguation)
- [ ] No new heavy dependencies (spaCy, nltk, etc.)
- [ ] Performance: entity extraction adds <5ms to `remember()`
- [ ] Hermes integration: `remember()` accepts `extract_entities` parameter via kwargs

## Verification Steps

1. **Unit tests:**
   - `test_entity_extraction_regex()` — capitalized words, quoted phrases
   - `test_entity_extraction_llm()` — LLM-driven extraction (mocked)
   - `test_fuzzy_matching()` — Levenshtein threshold behavior
   - `test_entity_recall()` — recall returns fuzzy-matched memories
   - `test_triple_storage()` — triples stored correctly in TripleStore

2. **Integration tests:**
   - `test_hermes_remember_with_entities()` — Hermes plugin passes `extract_entities`
   - `test_entity_extraction_performance()` — <5ms overhead measured

3. **Manual verification:**
   - Create 10 memories with entity variations
   - Verify fuzzy recall finds all variations
   - Check TripleStore contains correct triples

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Regex over-extracts (too many false entities) | Confidence scoring + LLM filtering when available |
| Regex under-extracts (misses entities) | LLM enhancement path + manual `add_entity()` API |
| Fuzzy matching too aggressive (false matches) | Configurable threshold, default 0.8 |
| Fuzzy matching too conservative (missed matches) | User can lower threshold per-query |
| Performance regression | Benchmark before/after, gate: <5ms overhead |

## Files to Create/Modify

### New Files
- `mnemosyne/core/entities.py` — Entity extraction, fuzzy matching, co-occurrence
- `tests/test_entities.py` — Unit tests
- `tests/test_entity_integration.py` — Integration tests

### Modified Files
- `mnemosyne/core/memory.py` — Add `extract_entities` parameter to `remember()`
- `mnemosyne/core/beam.py` — Add entity extraction hook in `remember()`
- `mnemosyne/core/triples.py` — Add `query_by_predicate()` helper
- `mnemosyne/hermes_plugin/tools.py` — Pass `extract_entities` from Hermes
- `mnemosyne/hermes_memory_provider/__init__.py` — Support entity extraction in provider

## Dependencies

- TripleStore (existing)
- Regex (stdlib)
- Optional: LLM (ctransformers or remote)
- No new external dependencies

## Notes

- Keep entity extraction **optional and explicit** — don't force it on users who don't need it
- Document the trade-off: regex is fast but dumb, LLM is slow but smart
- Consider future enhancement: entity aliases ("Abdias" → "Abdias J" → "AxDSan") via user-configured mapping

---

*Phase 1 Specification — PlanForge*
*Date: 2026-04-28*
