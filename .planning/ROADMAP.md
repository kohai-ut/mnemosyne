# Roadmap — Tiered Episodic Degradation

**Updated:** 2026-05-05

## Phase 1: Core Degradation Engine (Active)

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
- Unit tests for tier transitions
- Integration test: store → wait → degrade → recall
- Benchmark: recall latency before/after

## Phase 2: Dashboard Visibility (Planned)
## Phase 3: Smarter Compression (Planned)
