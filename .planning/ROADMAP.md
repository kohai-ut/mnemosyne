# Roadmap — Tiered Episodic Degradation

**Updated:** 2026-05-05

## Phase 1: Core Degradation Engine ✅ Complete

### Wave 1: Schema Migration
- Add `tier` and `degraded_at` columns to `episodic_memory`
- Update `_init_db()` with migration logic
- Backfill existing rows to tier 1

### Wave 2: Config & Constants
- Add env vars: `MNEMOSYNE_TIER2_DAYS`, `MNEMOSYNE_TIER3_DAYS`
- Add weight config: `MNEMOSYNE_TIER*_WEIGHT`
- Add `TIER_CONFIG` dict in beam.py

### Wave 3: degrade_episodic() Core
- Implement tier transition logic
- Compression pipeline (LLM summarization + text extraction fallback)
- Batch processing to limit per-sleep work

### Wave 4: Recall Weighting
- Add tier multiplier to `recall()` ranking score
- Tier 3 memories require 4x stronger semantic match

### Wave 5: Sleep Integration
- Wire `degrade_episodic()` into `sleep()` and `sleep_all_sessions()`
- Propagate `dry_run` flag
- Add degradation stats to sleep return value

### Wave 6: Tests & Verification
- 10 tests: schema, transitions, dry run, batch limit, weighting, sleep integration, end-to-end recall
- All 39 beam tests passing

## Phase 2: Smarter Compression (Planned)
- Entity-aware extraction instead of naive first-200-chars for tier 2→3
- Use structured extraction pipeline for key signal preservation
- Keep degraded content semantically searchable

## Phase 3: Memory Confidence (TBD)
- Add confidence/veracity signal to memories
- Distinguish user-stated facts from assistant inferences
- Surface potentially contaminated memories for review
