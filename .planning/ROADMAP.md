# Roadmap — Tiered Episodic Degradation

**Updated:** 2026-05-05

## Phase 1: Core Degradation Engine ✅ Complete
Tiered degradation (3 tiers), LLM compression, recall weighting, sleep integration.
6 waves, 39 tests.

## Phase 2: Smart Compression ✅ Complete
Entity-aware `_extract_key_signal()` for tier 2→3. Sentence scoring by signal density.
4 tests.

## Phase 3: Memory Confidence ✅ Complete
`veracity` field (stated/inferred/tool/imported/unknown) on working_memory and episodic_memory.
Recall weighting by veracity, veracity filter on recall(), get_contaminated() review method.
8 tests. 51 total beam tests passing.
