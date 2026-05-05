# Phase 2: Structured Memory Extract — Specification

## Phase Info

| Field | Value |
|---|---|
| **Phase** | 2 |
| **Name** | Structured Memory Extract |
| **Requirement** | REQ-2 |
| **Goal** | Optional LLM-driven fact extraction as a derived layer, not replacement for raw text |
| **Complexity** | Medium |
| **Estimated Duration** | 2-3 sessions |

## Locked Requirements

- REQ-2: Structured Memory Extract (P0)
- Depends on: REQ-1 (Entity Sketching) — completed

## Context

### Current State
Mnemosyne stores raw text memories. Entity sketching (Phase 1) extracts entity mentions as triples but does not extract structured facts. When a user asks "what does the user like?", Mnemosyne can only do keyword/semantic search on raw text — it has no structured fact layer.

### What Hindsight Does
- During Retain, LLM extracts 2-5 structured facts per content chunk
- Facts are normalized and stored with provenance (`proof_count`, `source_memory_ids`)
- Facts have types: `world`, `experience`, `observation`
- Reflect uses structured facts for higher-confidence answers

### What Mnemosyne Will Do Differently
- **Raw text remains canonical** — facts are derived views, not replacements
- **TripleStore storage** — facts stored as `(memory_id, "fact", "extracted_fact_text")` triples
- **Optional extraction** — `remember(content, extract=True)` only when LLM available
- **No fact types** — simpler, no `world`/`experience`/`observation` taxonomy
- **No provenance tracking** — facts are implicitly tied to their memory_id subject
- **Graceful degradation** — if LLM unavailable, stores empty fact set, no error

## Implementation Decisions

### Decision 1: Extraction Trigger
**Options:**
- A) Auto-extract on every `remember()` when LLM available
- B) Explicit `remember(content, extract=True)` flag
- C) Configurable default via env var

**Chosen: B + C** — Explicit per-call flag with global default via `MNEMOSYNE_AUTO_EXTRACT` env var. Default False for backward compatibility.

### Decision 2: LLM Integration
**Options:**
- A) Use existing `local_llm.py` (ctransformers + TinyLlama)
- B) Use remote LLM via `MNEMOSYNE_LLM_BASE_URL`
- C) Both with fallback chain

**Chosen: C** — Same fallback chain as `sleep()` consolidation: remote LLM → local LLM → skip extraction.

### Decision 3: Fact Format
**Options:**
- A) Free text ("The user likes coffee")
- B) Structured JSON (`{"subject": "user", "predicate": "likes", "object": "coffee"}`)
- C) Both (free text primary, structured optional)

**Chosen: A** — Free text facts stored as triples. Simpler, no schema enforcement, works with existing TripleStore. Future enhancement: structured JSON if needed.

### Decision 4: Storage Schema
**Options:**
- A) New `facts` table
- B) Extend TripleStore with `fact` predicate
- C) Both

**Chosen: B** — Use existing TripleStore. `(subject=memory_id, predicate="fact", object="extracted_fact_text", confidence=0.9)`. No new tables.

### Decision 5: Extraction Prompt
**Design:**
- System prompt: "Extract 2-5 concise factual statements from the following text. Each fact should be a complete sentence. Return one fact per line."
- No JSON schema — plain text output, one fact per line
- Parse by splitting on newlines
- Filter out empty lines

## Acceptance Criteria

- [ ] `remember("I love coffee and hate mornings", extract=True)` extracts facts and stores as triples
- [ ] `recall("what does the user like?")` searches both raw memories AND extracted facts
- [ ] Raw text remains canonical — facts are views, not replacements
- [ ] Graceful fallback: if LLM unavailable, stores empty fact set, no error
- [ ] Extraction prompt is configurable via env var
- [ ] Works with all LLM backends (remote, local ctransformers, none)
- [ ] Performance: extraction adds <500ms when LLM is remote, <5s when local
- [ ] No new external dependencies

## Verification Steps

1. **Unit tests:**
   - `test_fact_extraction_mock()` — mock LLM response, verify triples stored
   - `test_fact_extraction_no_llm()` — verify graceful skip when LLM unavailable
   - `test_fact_recall()` — recall finds facts, not just raw text
   - `test_fact_triple_storage()` — triples stored with correct predicate

2. **Integration tests:**
   - `test_end_to_end_extract_recall()` — remember with extract -> recall finds facts
   - `test_fallback_chain()` — remote fails, local works, both fail = skip

3. **Performance benchmark:**
   - Mock LLM: <10ms overhead
   - Remote LLM: <500ms overhead
   - Local LLM: <5s overhead

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| LLM hallucinates facts | Low confidence (0.7) on extracted facts; raw text remains canonical |
| LLM extraction is slow | Async/best-effort; never blocks memory write |
| Prompt injection via user content | Sanitize input; system prompt is hardcoded |
| Fact quality is poor | Configurable prompt; user can override via env var |

## Files to Create/Modify

### New Files
- `mnemosyne/core/extraction.py` — LLM fact extraction logic
- `tests/test_extraction.py` — Unit tests
- `tests/test_extraction_integration.py` — Integration tests

### Modified Files
- `mnemosyne/core/memory.py` — Add `extract` parameter to `remember()`
- `mnemosyne/core/beam.py` — Add extraction hook in `remember()`
- `mnemosyne/core/local_llm.py` — Add `extract_facts()` method
- `mnemosyne/hermes_plugin/tools.py` — Pass `extract` parameter
- `mnemosyne/hermes_memory_provider/__init__.py` — Support extraction in provider

## Dependencies

- TripleStore (existing, from Phase 1)
- LLM (optional): remote API or ctransformers
- No new external dependencies

## Notes

- Keep extraction **optional and explicit** — don't force it on users
- Document the trade-off: extraction improves recall quality but adds latency
- Consider future enhancement: fact confidence scoring based on LLM temperature
- Consider future enhancement: fact deduplication across memories

---

*Phase 2 Specification — PlanForge*
*Date: 2026-04-28*
