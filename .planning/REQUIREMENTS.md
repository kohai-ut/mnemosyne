# Requirements — Tiered Episodic Degradation

**Version:** v2.3
**Created:** 2026-05-05

## Phase 1: Core Degradation Engine

### R1.1 — Episodic Tier Schema
- Add `tier` column to `episodic_memory` (INTEGER DEFAULT 1)
- Add `degraded_at` column (TEXT, nullable)
- Migration: ALTER TABLE or schema version check in `_init_db()`
- Existing rows default to tier 1

### R1.2 — Tier Definitions
- Tier 1 (hot, 0-30 days): Full detail, 1.0x recall weight
- Tier 2 (warm, 30-180 days): Compressed, 0.5x recall weight
- Tier 3 (cold, 180+ days): Heavily compressed, 0.25x recall weight
- Tier transitions configurable via env vars
- Default thresholds: `MNEMOSYNE_TIER2_DAYS=30`, `MNEMOSYNE_TIER3_DAYS=180`

### R1.3 — Degradation Pipeline
- `degrade_episodic()` function in beam.py
- Runs automatically during `sleep()` after consolidation
- Compresses content: Tier 1→2 via LLM summarization, Tier 2→3 via text extraction
- Records `degraded_at` timestamp

### R1.4 — Tiered Recall Weighting
- `recall()` multiplies ranking score by tier weight
- Tier 3 memories require much stronger semantic match to surface
- Configurable via `MNEMOSYNE_TIER1_WEIGHT` etc.

### R1.5 — Dry Run Support
- `degrade_episodic(dry_run=True)` reports what would change
- `mnemosyne_sleep` tool's `dry_run` flag propagates to degradation

## Phase 2: Dashboard Visibility (future)
- Dashboard shows tier distribution graph
- "Memory Health" metric in overview

## Phase 3: Smarter Compression (future)
- Use importance score to bias compression
- Entity-preserving compression
