# Phase 2: Structured Memory Extract — Implementation Plan

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 2 |
| **Name** | Structured Memory Extract |
| **Spec** | 2-STRUCTURED-EXTRACT-SPEC.md |

## Task Waves

### Wave 1: Core Infrastructure (Independent tasks, parallel)

#### Task 2.1: LLM Fact Extraction Module
**File:** `mnemosyne/core/extraction.py`
**Description:** Create module with `extract_facts(text)` function that calls LLM with extraction prompt.
**Acceptance:**
- `extract_facts("I love coffee")` returns `["The user loves coffee"]` (mocked)
- Supports remote LLM via `MNEMOSYNE_LLM_BASE_URL`
- Supports local LLM via ctransformers fallback
- Returns empty list if no LLM available
- Wrapped in try/except — never raises
**Estimated:** 1 hour

#### Task 2.2: Extraction Prompt Template
**File:** `mnemosyne/core/extraction.py` (constant)
**Description:** Define default extraction prompt, configurable via env var.
**Acceptance:**
- `EXTRACTION_PROMPT` env var overrides default
- Default prompt extracts 2-5 concise factual statements
- One fact per line output format
- System prompt hardcoded, not injectable
**Estimated:** 30 minutes

#### Task 2.3: TripleStore Fact Storage
**File:** `mnemosyne/core/triples.py` (minor enhancement)
**Description:** Add `add_facts(memory_id, facts, source)` helper to batch-store fact triples.
**Acceptance:**
- `triples.add_facts("mem_123", ["fact1", "fact2"], source="conversation")`
- Stores each fact as `(memory_id, "fact", "fact_text", confidence=0.7)`
- Returns count of facts stored
**Estimated:** 30 minutes

### Wave 2: Integration (Depends on Wave 1)

#### Task 2.4: remember() with extract Parameter
**File:** `mnemosyne/core/memory.py`
**Description:** Add `extract: bool = False` parameter to `Mnemosyne.remember()` and module-level `remember()`.
**Acceptance:**
- `remember("I love coffee", extract=True)` calls extraction + stores facts
- `remember("I love coffee")` works exactly as before (backward compatible)
- Entity extraction (Phase 1) and fact extraction both work: `remember(text, extract_entities=True, extract=True)`
- Extraction is best-effort: if LLM fails, memory still stored
**Estimated:** 45 minutes

#### Task 2.5: BEAM Integration
**File:** `mnemosyne/core/beam.py`
**Description:** Add extraction hook in `BeamMemory.remember()`.
**Acceptance:**
- `beam.remember(text, extract=True)` stores facts in TripleStore
- Facts stored with `source` field for provenance
**Estimated:** 30 minutes

#### Task 2.6: Fact-Aware Recall
**File:** `mnemosyne/core/beam.py`
**Description:** Modify recall to search fact triples alongside raw memories.
**Acceptance:**
- `recall("what does the user like?")` searches both raw text AND fact triples
- Fact matches tagged with `"fact_match": True`
- Results merged with existing hybrid scoring
**Estimated:** 1.5 hours

#### Task 2.7: Hermes Plugin Integration
**File:** `mnemosyne/hermes_plugin/tools.py`
**Description:** Pass `extract` parameter through Hermes tool interface.
**Acceptance:**
- `mnemosyne_remember` tool accepts `extract` field
- Parameter forwarded to `remember()`
**Estimated:** 30 minutes

### Wave 3: Testing & Verification (Depends on Wave 2)

#### Task 2.8: Unit Tests
**File:** `tests/test_extraction.py`
**Description:** Test extraction logic with mocked LLM.
**Acceptance:**
- Mock LLM returns facts, verify triples stored
- No LLM available = graceful skip
- Prompt injection resistance
- 100% branch coverage for `extraction.py`
**Estimated:** 1 hour

#### Task 2.9: Integration Tests
**File:** `tests/test_extraction_integration.py`
**Description:** Test end-to-end: remember with extract -> recall finds facts.
**Acceptance:**
- 5 memories with facts extracted
- Recall query finds facts, not just raw text
- Verify fact triples queryable
**Estimated:** 1 hour

#### Task 2.10: Performance Benchmark
**File:** `tests/benchmark_extraction.py`
**Description:** Measure extraction latency with mock vs real LLM.
**Acceptance:**
- Mock: <10ms overhead
- Remote LLM: <500ms overhead
- Report: extraction time per memory
**Estimated:** 30 minutes

### Wave 4: Hermes Memory Provider (Depends on Wave 2)

#### Task 2.11: MemoryProvider Integration
**File:** `mnemosyne/hermes_memory_provider/__init__.py`
**Description:** Support `extract` in `_handle_remember()` and `sync_turn()`.
**Acceptance:**
- Hermes agent can pass `extract=True` via kwargs
- Auto-extract on session end if configured
**Estimated:** 30 minutes

## Verification Checklist

- [ ] All unit tests pass (`pytest tests/test_extraction.py -v`)
- [ ] All integration tests pass (`pytest tests/test_extraction_integration.py -v`)
- [ ] Performance benchmark: mock <10ms, remote <500ms
- [ ] Hermes plugin accepts `extract` parameter
- [ ] Backward compatible: `remember()` without `extract` works identically
- [ ] Works with Phase 1 entity extraction simultaneously
- [ ] No breaking changes to existing API

## Ship Criteria (Phase 2 Internal)

1. All tasks in Waves 1-4 complete
2. All verification checks pass
3. Code review: no external dependencies added
4. Performance gate: extraction overhead within limits
5. Integration gate: works with Phase 1 entity sketching

---

*Phase 2 Implementation Plan — PlanForge*
*Date: 2026-04-28*
