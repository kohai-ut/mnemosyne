---
name: planforge-phase-3-temporal-recall
title: Phase 3 — Temporal Recall Specification
---

# Phase 3: Temporal Recall — Specification

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 3 |
| **Name** | Temporal Recall |
| **Requirement** | REQ-3 |
| **Goal** | Add time-awareness to existing hybrid scoring without separate temporal retrieval pipeline |
| **Complexity** | Low |
| **Estimated Duration** | 1-2 sessions |
| **Depends On** | Phase 2 (Structured Extract) — uses same TripleStore + scoring infrastructure |

## Context

### Current State
Mnemosyne stores `timestamp` on every memory (ISO format string) and `valid_until` for temporal invalidation. The hybrid scoring formula is:

```
score = 0.5 * vec_similarity + 0.3 * fts_rank + 0.2 * importance
score *= (0.7 + 0.3 * recency_decay)
```

Recency decay uses `RECENCY_HALFLIFE_HOURS` (default 168h = 1 week). But there is **no way for the user to express temporal intent** in a query like "what happened yesterday?" or "what did I say last week?"

### What Hindsight Does
- Native date parsing during Retain (ISO format, flexible datetime)
- `occurred_start`, `occurred_end`, `mentioned_at`, `event_date` fields on memory units
- **Temporal retrieval as 4th parallel recall strategy** — dedicated temporal query path
- Temporal aggregation in consolidation: min event_date, max occurred_end

### What Mnemosyne Will Do Differently
- **No separate temporal retrieval pipeline** — temporal boost is a scoring modifier, not a parallel strategy
- **No date parsing from text** — use existing `timestamp` field, not extracted dates
- **User expresses temporal intent via parameters**, not query text analysis
- **Simple and explicit**: `recall(query, temporal_weight=0.2)` — no magic

## Implementation Decisions

### Decision 1: Temporal Boost Formula
**Options:**
- A) Exponential decay based on `|query_time - memory_timestamp|`
- B) Gaussian window centered on query time
- C) Step function (boost if within N hours, no boost otherwise)

**Chosen: A)** — Exponential decay. Consistent with existing recency decay. Formula:
```python
temporal_boost = exp(-hours_delta / TEMPORAL_HALFLIFE_HOURS)
score *= (1.0 + temporal_weight * temporal_boost)
```

### Decision 2: Query Time Source
**Options:**
- A) Always `datetime.now()` — "what's recent?"
- B) User-provided `query_time` parameter — "what happened on Tuesday?"
- C) Both — default to now, allow override

**Chosen: C)** — `recall(query, temporal_weight=0.0, query_time=None)`. Default `None` → `datetime.now()`. User can pass ISO string or datetime object.

### Decision 3: Temporal Halflife
**Options:**
- A) Hardcoded constant
- B) Configurable per-call
- C) Env var with per-call override

**Chosen: C)** — `MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS` env var (default 24), overridable per-call via `temporal_halflife` parameter.

### Decision 4: Integration with Existing Scoring
**Options:**
- A) Replace recency decay with temporal boost
- B) Multiply temporal boost on top of recency decay
- C) Make them mutually exclusive parameters

**Chosen: B)** — Temporal boost is multiplicative on top of recency decay. Both can be active. This gives maximum flexibility:
```python
score = base_score * recency_factor * (1.0 + temporal_weight * temporal_boost)
```

## Acceptance Criteria

- [ ] `recall(query, temporal_weight=0.2)` accepts temporal weight parameter
- [ ] Memories with `timestamp` closer to `query_time` get higher scores
- [ ] `query_time` parameter accepts `None` (default now), ISO string, or datetime object
- [ ] Temporal boost formula: `score *= (1.0 + temporal_weight * exp(-hours_delta / halflife))`
- [ ] Backward compatible: `recall(query)` works exactly as before (temporal_weight=0.0)
- [ ] Works with all vector types (float32, int8, bit) and FTS5 fallback
- [ ] Works alongside entity/fact extraction (Phase 1 + 2 features)
- [ ] Performance: temporal scoring adds <1ms overhead per query

## API Design

### `recall()` signature update
```python
def recall(self, query: str, top_k: int = 5,
           from_date: Optional[str] = None,
           to_date: Optional[str] = None,
           source: Optional[str] = None,
           topic: Optional[str] = None,
           temporal_weight: float = 0.0,
           query_time: Optional[Union[str, datetime]] = None,
           temporal_halflife: Optional[float] = None) -> List[Dict]:
```

### Module-level `recall()` update
```python
def recall(query: str, top_k: int = 5, *,
           from_date: Optional[str] = None,
           to_date: Optional[str] = None,
           source: Optional[str] = None,
           topic: Optional[str] = None,
           temporal_weight: float = 0.0,
           query_time: Optional[Union[str, datetime]] = None,
           temporal_halflife: Optional[float] = None) -> List[Dict]:
```

## Verification Steps

1. **Unit tests:**
   - `test_temporal_boost_recent()` — recent memory gets boost
   - `test_temporal_boost_old()` — old memory gets minimal boost
   - `test_temporal_boost_zero_weight()` — weight=0 means no effect
   - `test_temporal_query_time_string()` — ISO string parsed correctly
   - `test_temporal_query_time_datetime()` — datetime object accepted
   - `test_temporal_halflife_override()` — per-call halflife works
   - `test_temporal_with_entities()` — works alongside entity extraction
   - `test_temporal_with_facts()` — works alongside fact extraction
   - `test_temporal_backward_compat()` — default params = old behavior

2. **Integration tests:**
   - `test_temporal_recall_end_to_end()` — create memories at different times, verify ordering
   - `test_temporal_performance()` — <1ms overhead measured

3. **Manual verification:**
   - Create 5 memories with timestamps spanning 1 week
   - Query with `temporal_weight=0.3` — verify recent ones rank higher
   - Query with `temporal_weight=0.0` — verify same ranking as before

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Timestamp parsing errors | Wrap in try/except, skip temporal boost if timestamp invalid |
| `query_time` in the future | Clamp to now, no negative deltas |
| Very old memories (years) | Exponential decay naturally gives ~0 boost, no overflow |
| Performance regression | Benchmark before/after, gate: <1ms overhead |
| Confusion with `from_date`/`to_date` | `from_date`/`to_date` are hard filters; `temporal_weight` is soft scoring. Document clearly. |

## Files to Create/Modify

### Modified Files
- `mnemosyne/core/beam.py` — Add temporal scoring to `recall()` and `_score_memory()`
- `mnemosyne/core/memory.py` — Add `temporal_weight`, `query_time`, `temporal_halflife` params to `recall()`
- `mnemosyne/hermes_plugin/tools.py` — Pass temporal params through Hermes tool interface
- `mnemosyne/hermes_memory_provider/__init__.py` — Support temporal params in provider

### New Files
- `tests/test_temporal_recall.py` — Unit + integration tests
- `tests/benchmark_temporal_recall.py` — Performance benchmark

## Dependencies

- Existing BEAM scoring infrastructure
- `datetime` (stdlib)
- `math.exp` (stdlib)
- No new external dependencies

## Notes

- Keep temporal scoring **simple and explicit** — no NLP date parsing from query text
- Document the difference between `from_date`/`to_date` (hard filter) and `temporal_weight` (soft boost)
- Consider future enhancement: parse temporal intent from query text ("what happened yesterday?") — but that's v2, not this phase
- The temporal boost is intentionally subtle — default 0.0 means no change to existing behavior

---

*Phase 3 Specification — PlanForge*
*Date: 2026-04-29*
