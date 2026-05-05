# Mnemosyne — Tiered Episodic Degradation

**Created:** 2026-05-05
**Repository:** /root/.hermes/projects/mnemosyne
**Plugin:** PlanForge

## Vision

Mnemosyne remembers what you told it a year ago — literally, not metaphorically.
Every memory persists, but old memories degrade gracefully through three tiers,
so context never bloats and queries stay fast.

## Core Value

- **Marketing truth**: "I told Mnemosyne something 6 months ago and it just recalled it."
- **Engineering reality**: Old memories are compressed summaries with low recall weight. They are present in the database, discoverable with strong semantic matches, but don't clutter everyday conversations.
- **Zero maintenance**: The system degrades itself automatically during sleep cycles. No admin pruning, no manual cleanup, no database growing out of control.

## Constraints

- Backward compatible: existing episodic entries default to tier 1 (full detail)
- No breaking changes to the recall API
- No external dependencies (pure SQLite)
- Performance: recall latency must stay sub-millisecond for typical queries
- Disk: compressed tier-3 entries should be ~10% the size of tier-1 entries

## Success Criteria

1. A memory stored today is still queryable 365 days later
2. Recall benchmark shows no regression for tier-1 hot memories
3. Database size growth is logarithmic, not linear (episodic plateaus)
4. Users never need to manually prune or clean up
