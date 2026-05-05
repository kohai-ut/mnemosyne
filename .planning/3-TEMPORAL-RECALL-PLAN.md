---
name: planforge-phase-3-temporal-recall-plan
title: Phase 3 — Temporal Recall Implementation Plan
---

# Phase 3: Temporal Recall — Implementation Plan

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 3 |
| **Name** | Temporal Recall |
| **Spec** | 3-TEMPORAL-RECALL-SPEC.md |
| **Requirement** | REQ-3 |

## Task Waves

### Wave 1: Core Temporal Scoring (Independent, can start immediately)

#### Task 3.1: Temporal Boost Function
**File:** `mnemosyne/core/beam.py` (new helper function)
**Description:** Implement `_temporal_boost(memory_timestamp, query_time, halflife)` that returns a boost factor based on time delta.
**Acceptance:**
- `exp(-hours_delta / halflife)` formula
- Handles invalid timestamps gracefully (returns 0.0)
- Handles future timestamps gracefully (clamps to 0 delta)
- Pure Python, no external deps
**Estimated:** 30 minutes

#### Task 3.2: Parse query_time Parameter
**File:** `mnemosyne/core/beam.py` (new helper function)
**Description:** Implement `_parse_query_time(query_time)` that accepts `None`, ISO string, or datetime object and returns a datetime.
**Acceptance:**
- `None` → `datetime.now()`
- `"2026-04-29T12:00:00"` → parsed datetime
- `datetime(2026, 4, 29)` → returned as-is
- Invalid string → raises ValueError with helpful message
**Estimated:** 20 minutes

#### Task 3.3: Integrate Temporal Boost into Scoring
**File:** `mnemosyne/core/beam.py` (modify `_score_memory()` or `recall()`)
**Description:** Add temporal scoring to the hybrid ranking pipeline.
**Acceptance:**
- `temporal_weight` parameter accepted in `recall()`
- Temporal boost applied after base hybrid score + recency decay
- Formula: `final_score = base_score * recency_factor * (1.0 + temporal_weight * temporal_boost)`
- Backward compatible: default `temporal_weight=0.0` means no change
**Estimated:** 45 minutes

### Wave 2: API Propagation (Depends on Wave 1)

#### Task 3.4: Update memory.py recall()
**File:** `mnemosyne/core/memory.py`
**Description:** Add `temporal_weight`, `query_time`, `temporal_halflife` parameters to `Mnemosyne.recall()` and module-level `recall()`.
**Acceptance:**
- All three parameters forwarded to `beam.recall()`
- Type hints correct: `Union[str, datetime, None]` for query_time
- Docstrings updated
**Estimated:** 20 minutes

#### Task 3.5: Hermes Plugin Integration
**File:** `mnemosyne/hermes_plugin/tools.py`
**Description:** Add temporal parameters to `mnemosyne_recall` tool schema and handler.
**Acceptance:**
- Tool schema includes `temporal_weight` (number, optional), `query_time` (string, optional)
- Handler passes params to `recall()`
- Default values match API defaults
**Estimated:** 20 minutes

### Wave 3: Testing & Verification (Depends on Wave 2)

#### Task 3.6: Unit Tests
**File:** `tests/test_temporal_recall.py`
**Description:** Test temporal boost, query_time parsing, backward compatibility.
**Acceptance:**
- 8+ test cases covering all acceptance criteria
- All tests pass
**Estimated:** 45 minutes

#### Task 3.7: Integration Tests
**File:** `tests/test_temporal_recall.py` (continuation)
**Description:** End-to-end: create time-spread memories, verify temporal recall ordering.
**Acceptance:**
- 5 memories with timestamps spanning 1 week
- `temporal_weight=0.3` ranks recent ones higher
- `temporal_weight=0.0` preserves old ranking
**Estimated:** 30 minutes

#### Task 3.8: Performance Benchmark
**File:** `tests/benchmark_temporal_recall.py`
**Description:** Measure temporal scoring overhead.
**Acceptance:**
- Baseline: `recall()` without temporal params
- With temporal: <1ms overhead per query
- Report: before/after times
**Estimated:** 20 minutes

### Wave 4: MemoryProvider Integration (Optional, can defer)

#### Task 3.9: Update MemoryProvider
**File:** `mnemosyne/hermes_memory_provider/__init__.py`
**Description:** Support temporal params in `_handle_recall()` and `sync_turn()`.
**Acceptance:**
- `_handle_recall()` passes temporal params through
- `sync_turn()` uses default temporal_weight=0.0
**Estimated:** 15 minutes

## Verification Checklist

- [ ] All unit tests pass (`pytest tests/test_temporal_recall.py -v`)
- [ ] All integration tests pass
- [ ] Performance benchmark shows <1ms overhead
- [ ] Hermes plugin accepts temporal parameters
- [ ] No breaking changes to existing API
- [ ] Works alongside entity extraction (Phase 1) and fact extraction (Phase 2)

## Ship Criteria

1. All tasks in Waves 1-3 complete
2. All verification checks pass
3. Code review: no external dependencies added
4. Performance gate: <1ms overhead confirmed
5. Integration with Phase 1+2 features verified

---

*Phase 3 Implementation Plan — PlanForge*
*Date: 2026-04-29*
