# Phase 1: Entity Sketching — Implementation Plan

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 1 |
| **Name** | Entity Sketching |
| **Spec** | 1-ENTITY-SKETCHING-SPEC.md |

## Task Waves

### Wave 1: Core Infrastructure (Independent tasks, parallel)

#### Task 1.1: Pure Python Levenshtein Implementation
**File:** `mnemosyne/core/entities.py` (first function)
**Description:** Implement `levenshtein_distance(s1, s2)` and `similarity(s1, s2)` functions.
**Acceptance:** 
- `similarity("Abdias", "Abdias J")` >= 0.8
- `similarity("Abdias", "Abdul")` < 0.8
- Pure Python, no external deps
**Estimated:** 30 minutes

#### Task 1.2: Regex Entity Extractor
**File:** `mnemosyne/core/entities.py`
**Description:** Implement `extract_entities_regex(text)` using capitalized words and quoted phrases.
**Acceptance:**
- `extract_entities_regex("I met Abdias in New York")` -> `["Abdias", "New York"]`
- `extract_entities_regex('She said "Hello World"')` -> `["Hello World"]`
- Filters out common stop words (the, a, an, etc.)
**Estimated:** 45 minutes

#### Task 1.3: TripleStore Predicate Query Helper
**File:** `mnemosyne/core/triples.py`
**Description:** Add `query_by_predicate(predicate, object=None)` method to TripleStore.
**Acceptance:**
- `triples.query_by_predicate("mentions", "Abdias")` returns matching triples
- Works with existing TripleStore schema
**Estimated:** 30 minutes

### Wave 2: Integration (Depends on Wave 1)

#### Task 1.4: Entity Storage in remember()
**File:** `mnemosyne/core/beam.py`
**Description:** Add entity extraction hook in `BeamMemory.remember()`. When `extract_entities=True`, extract and store triples.
**Acceptance:**
- `beam.remember("I met Abdias", extract_entities=True)` creates triple `(memory_id, "mentions", "Abdias")`
- Works with both regex and LLM extraction paths
- No performance regression: <5ms overhead
**Estimated:** 1 hour

#### Task 1.5: Entity-Aware Recall
**File:** `mnemosyne/core/beam.py`
**Description:** Modify `BeamMemory.recall()` to search entity triples when query might be an entity name.
**Acceptance:**
- `recall("Abdias")` finds memories mentioning "Abdias J" via fuzzy match
- Entity results merged with existing hybrid results
- No breaking changes to recall API
**Estimated:** 1.5 hours

#### Task 1.6: Hermes Plugin Integration
**File:** `mnemosyne/hermes_plugin/tools.py`
**Description:** Pass `extract_entities` parameter through Hermes tool interface.
**Acceptance:**
- `hermes_memory_store` tool accepts `extract_entities` field
- Parameter forwarded to `remember()`
**Estimated:** 30 minutes

### Wave 3: Testing & Verification (Depends on Wave 2)

#### Task 1.7: Unit Tests
**File:** `tests/test_entities.py`
**Description:** Test regex extraction, Levenshtein matching, triple storage.
**Acceptance:**
- 100% branch coverage for `entities.py`
- All edge cases: empty text, no entities, unicode, etc.
**Estimated:** 1 hour

#### Task 1.8: Integration Tests
**File:** `tests/test_entity_integration.py`
**Description:** Test end-to-end: remember -> extract -> recall via fuzzy match.
**Acceptance:**
- 10 memories with entity variations created
- Fuzzy recall finds all variations
- Performance: <5ms extraction overhead
**Estimated:** 1 hour

#### Task 1.9: Performance Benchmark
**File:** `tests/benchmark_entity_extraction.py`
**Description:** Measure extraction overhead on 1000 memories.
**Acceptance:**
- Baseline: `remember()` without extraction
- With extraction: <5ms overhead per call
- Report: before/after times, memory usage
**Estimated:** 30 minutes

### Wave 4: Documentation (Depends on Wave 3)

#### Task 1.10: API Documentation
**File:** Update docs site
**Description:** Document `extract_entities` parameter, entity recall behavior, configuration.
**Acceptance:**
- Python SDK page updated with entity examples
- Architecture page explains entity sketching (not full resolution)
**Estimated:** 45 minutes

#### Task 1.11: Comparison Page Update
**File:** `content/comparisons/hindsight.mdx`
**Description:** Update Hindsight comparison to reflect new entity capabilities.
**Acceptance:**
- Honest assessment: "Mnemosyne has lightweight entity sketching, not full entity resolution"
- Clear distinction from Hindsight's spaCy + graph approach
**Estimated:** 30 minutes

## Verification Checklist

- [ ] All unit tests pass (`pytest tests/test_entities.py -v`)
- [ ] All integration tests pass (`pytest tests/test_entity_integration.py -v`)
- [ ] Performance benchmark shows <5ms overhead
- [ ] Hermes plugin accepts `extract_entities` parameter
- [ ] No breaking changes to existing API
- [ ] Documentation updated and accurate
- [ ] Comparison page reflects honest capabilities

## Ship Criteria

1. All tasks in Waves 1-4 complete
2. All verification checks pass
3. Code review: no external dependencies added
4. Performance gate: <5ms overhead confirmed
5. Documentation gate: no fabricated APIs

---

*Phase 1 Implementation Plan — PlanForge*
*Date: 2026-04-28*
