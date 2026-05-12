"""
Mnemosyne Veracity-Weighted Consolidation
=========================================
Our novel contribution: Bayesian confidence scoring + conflict resolution.

Veracity tiers:
- stated:     1.0  (user explicitly stated)
- inferred:   0.7  (inferred from context)
- tool:       0.5  (tool output, may be stale)
- imported:   0.6  (imported from external source)
- unknown:    0.8  (default, unverified)

Bayesian updating:
- confidence = 1 - (0.7^n) where n = mention count
- More mentions = higher confidence
- Contradictions detected and flagged

Conflict resolution:
- Same subject + predicate = potential conflict
- Higher confidence wins
- Lower confidence flagged for review
- Consolidation: periodic synthesis of high-confidence facts
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


# Veracity weights
VERACITY_WEIGHTS = {
    "stated": 1.0,
    "inferred": 0.7,
    "tool": 0.5,
    "imported": 0.6,
    "unknown": 0.8,
}

# Canonical allowlist for trust-boundary clamping. Anything outside this
# set bypasses the recall weighting (VERACITY_WEIGHTS.get(..., 0.8) falls
# back to the 'unknown' weight) AND pollutes the contamination filter
# downstream (which compares `veracity != 'stated'`). Callers at the
# trust boundary (LLM output, importers, MCP tool args, batch ingest)
# should clamp via clamp_veracity() so non-canonical labels don't
# persist as garbage in the row.
VERACITY_ALLOWED = frozenset(VERACITY_WEIGHTS.keys())


# Cap on the raw value included in the WARNING log. Without this, an
# importer pushing 100k items with embedded long strings as veracity
# values can flood log aggregators (cost) AND leak user content into
# operator logs (privacy). 80 chars is enough to debug typos / case
# issues without being a privacy or storage hazard.
_VERACITY_WARN_VALUE_CAP = 80


def clamp_veracity(raw, *, context: str = "veracity") -> str:
    """Normalize and clamp a veracity label to the canonical allowlist.

    Behavior:
        - None / empty / whitespace → 'unknown' silently
        - Case-and-whitespace normalize then match against VERACITY_ALLOWED
        - Anything else → 'unknown' with a WARNING log (raw value
          truncated to %d chars to bound log volume)

    `context` appears in the warning so the operator can see where
    the bad label came from (e.g. 'remember_batch.default',
    'remember_batch.per_item', 'mnemosyne_remember').
    """ % _VERACITY_WARN_VALUE_CAP
    if raw is None:
        return "unknown"
    norm = str(raw).strip().lower()
    if not norm:
        return "unknown"
    if norm in VERACITY_ALLOWED:
        return norm
    # Truncate the raw value for the log line. %r quoting prevents
    # control-character injection into log aggregators; the cap
    # prevents log-flood and content leakage from upstream typos.
    raw_str = str(raw)
    if len(raw_str) > _VERACITY_WARN_VALUE_CAP:
        raw_for_log = raw_str[:_VERACITY_WARN_VALUE_CAP] + "...[truncated]"
    else:
        raw_for_log = raw_str
    logger.warning(
        "%s received unknown veracity %r; clamping to 'unknown'",
        context, raw_for_log,
    )
    return "unknown"


@dataclass
class ConsolidatedFact:
    """A fact that has been through consolidation."""
    subject: str
    predicate: str
    object: str
    confidence: float
    mention_count: int
    first_seen: str
    last_seen: str
    sources: List[str]
    veracity: str
    superseded: bool = False


class VeracityConsolidator:
    """
    Bayesian confidence consolidation with conflict detection.
    
    Builds on:
    - Memanto's conflict resolution (arXiv:2604.22085)
    - REMem's fact preservation (arXiv:2602.13530)
    - Our novel veracity-weighted Bayesian updating
    """
    
    def __init__(self, db_path: Path = None, conn=None):
        if conn is not None:
            self.conn = conn
            self.db_path = db_path or Path(":memory:")
        else:
            self.db_path = db_path or Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            # Apply the same PRAGMA settings BeamMemory's _get_connection
            # uses (journal_mode=WAL, busy_timeout=5000ms). Without these,
            # `BEGIN IMMEDIATE` in `consolidate_fact` runs under default
            # `journal_mode=DELETE` which blocks readers too AND raises
            # `database is locked` immediately under contention instead
            # of waiting through the busy_timeout. /review (Claude
            # adversarial C3) caught the connection-setup gap.
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.Error:
                # Best-effort: some constrained environments (e.g.,
                # in-memory DBs) don't support WAL. Continue without —
                # the BEGIN IMMEDIATE path still works, just with
                # different contention semantics.
                pass
        self.conn.row_factory = sqlite3.Row
        self._owns_connection = conn is None
        self._init_tables()
    
    def _init_tables(self):
        """Initialize consolidation schema."""
        cursor = self.conn.cursor()
        
        # Consolidated facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                mention_count INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT,
                sources_json TEXT,
                veracity TEXT DEFAULT 'unknown',
                superseded_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_subject ON consolidated_facts(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_predicate ON consolidated_facts(predicate)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_object ON consolidated_facts(object)")
        
        # Conflicts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id TEXT NOT NULL,
                fact_b_id TEXT NOT NULL,
                conflict_type TEXT,
                resolution TEXT,
                resolved_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def bayesian_update(self, current_confidence: float, veracity: str) -> float:
        """
        Update confidence using Bayesian formula.
        
        Formula: new_confidence = 1 - (0.7^n) where n = mention count
        But we approximate with: new = old + (1 - old) * veracity_weight * 0.3
        
        Args:
            current_confidence: Current confidence level
            veracity: Veracity tier
            
        Returns:
            float: Updated confidence
        """
        weight = VERACITY_WEIGHTS.get(veracity, 0.8)
        increment = (1.0 - current_confidence) * weight * 0.3
        return min(current_confidence + increment, 1.0)
    
    def consolidate_fact(self, subject: str, predicate: str, object: str,
                        veracity: str = "unknown", source: str = None) -> ConsolidatedFact:
        """
        Add or update a fact in consolidation.

        Args:
            subject: Fact subject
            predicate: Fact predicate
            object: Fact object
            veracity: Veracity tier
            source: Source memory ID

        Returns:
            ConsolidatedFact: The consolidated result

        Concurrency: the SELECT-by-SPO → INSERT/UPDATE pattern is
        race-vulnerable under naive WAL execution. Two threads both
        passing the SELECT-no-match check and both attempting INSERT
        would race on the deterministic PRIMARY KEY: one INSERT
        succeeds, the other raises `IntegrityError`. Bayesian
        confidence math is path-dependent (``new = old + (1-old) *
        weight * 0.3``), so concurrent UPDATEs also race — the
        compute-then-write pattern lets a later UPDATE overwrite an
        earlier UPDATE's effect.

        Fix: wrap the read-then-write in ``BEGIN IMMEDIATE`` so the
        whole sequence is serialized at the SQLite writer-lock
        level. Acquires a RESERVED lock at BEGIN, holds it through
        SELECT + INSERT/UPDATE + conflict-recording, releases at
        COMMIT. Concurrent callers queue rather than race. Lock is
        database-wide across connections — works under BeamMemory's
        thread-local connection model.

        Nested-transaction handling: if the caller is already in a
        transaction, ``BEGIN IMMEDIATE`` would raise
        ``OperationalError: cannot start a transaction within a
        transaction``. We detect via ``conn.in_transaction`` and
        skip the BEGIN. **Caveat (Codex adversarial H3 + Claude H3):**
        a DEFERRED outer transaction (Python sqlite3's default
        implicit tx, the kind opened automatically on first
        modifying SQL) does NOT acquire the writer lock until its
        own first INSERT/UPDATE. Two threads each in their own
        DEFERRED outer tx can both pass our SELECT-no-match check
        before either writes — race window reopens. Race safety
        within a caller-owned outer tx therefore requires either
        (a) the outer tx is `BEGIN IMMEDIATE` or `BEGIN EXCLUSIVE`,
        OR (b) the caller is the only writer (e.g., E2's
        single-threaded batch enrichment loop — only one thread
        runs the loop, so there's no concurrent consolidate_fact
        call to race against). Document this assumption in the
        caller's code.

        Failure handling: if ``BEGIN IMMEDIATE`` raises
        ``OperationalError`` (e.g., ``database is locked`` after
        ``busy_timeout`` expiry), we re-raise rather than silently
        proceeding without the lock — silent fallthrough would
        reintroduce the exact race this method claims to close.
        /review (Codex structured P2 + Codex adversarial HIGH +
        Maintainability + Claude — 4-source HIGH) caught the
        original silent-fallthrough as a correctness regression.

        Other write methods on this class (``resolve_conflict``,
        ``resolve_conflict_by_facts``, ``run_consolidation_pass``)
        have similar SELECT-then-write patterns and are NOT
        wrapped by this fix. Concurrent same-conflict resolution
        from multiple writers can still be last-writer-wins. Out
        of scope for this PR; tracked separately.
        """
        cursor = self.conn.cursor()

        # Serialize concurrent consolidate_fact calls. Skip if
        # already inside a caller-supplied transaction (see caveat
        # in docstring about DEFERRED outer-tx races).
        started_tx = False
        if not self.conn.in_transaction:
            # Let OperationalError propagate. If `database is
            # locked` fires after busy_timeout, the caller's
            # retry / error-handler is the right place to decide
            # what to do — silently running the SELECT-then-write
            # without the lock would reintroduce the race we just
            # fixed.
            cursor.execute("BEGIN IMMEDIATE")
            started_tx = True

        try:
            # Check if fact already exists
            cursor.execute("""
                SELECT * FROM consolidated_facts
                WHERE subject = ? AND predicate = ? AND object = ?
            """, (subject, predicate, object))

            row = cursor.fetchone()
            now = datetime.now().isoformat()

            if row:
                # Update existing fact
                new_confidence = self.bayesian_update(row["confidence"], veracity)
                new_count = row["mention_count"] + 1

                sources = json.loads(row["sources_json"] or "[]")
                if source and source not in sources:
                    sources.append(source)

                cursor.execute("""
                    UPDATE consolidated_facts
                    SET confidence = ?, mention_count = ?, last_seen = ?,
                        sources_json = ?, veracity = ?, updated_at = ?
                    WHERE id = ?
                """, (new_confidence, new_count, now, json.dumps(sources),
                      veracity, now, row["id"]))

                if started_tx:
                    self.conn.commit()

                return ConsolidatedFact(
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    confidence=new_confidence,
                    mention_count=new_count,
                    first_seen=row["first_seen"],
                    last_seen=now,
                    sources=sources,
                    veracity=veracity
                )

            else:
                # Check for conflicts (same subject+predicate, different object)
                cursor.execute("""
                    SELECT * FROM consolidated_facts
                    WHERE subject = ? AND predicate = ? AND object != ?
                """, (subject, predicate, object))

                conflicts = cursor.fetchall()

                # Insert new fact
                fact_id = f"cf_{subject}_{predicate}_{object}".replace(" ", "_")[:100]
                base_confidence = VERACITY_WEIGHTS.get(veracity, 0.8) * 0.5

                sources = [source] if source else []

                cursor.execute("""
                    INSERT INTO consolidated_facts
                    (id, subject, predicate, object, confidence, mention_count,
                     first_seen, last_seen, sources_json, veracity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (fact_id, subject, predicate, object, base_confidence, 1,
                      now, now, json.dumps(sources), veracity))

                # Record conflicts. Pass `commit=False` so the helper
                # doesn't end our `BEGIN IMMEDIATE` transaction mid-loop.
                # /review caught the pre-fix unconditional commit as a
                # 4-source HIGH atomicity breach.
                for conflict in conflicts:
                    self._record_conflict(
                        fact_id, conflict["id"], "contradiction",
                        commit=False,
                    )

                if started_tx:
                    self.conn.commit()

                return ConsolidatedFact(
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    confidence=base_confidence,
                    mention_count=1,
                    first_seen=now,
                    last_seen=now,
                    sources=sources,
                    veracity=veracity
                )
        except Exception:
            # Roll back our own transaction on any failure; if the
            # caller owns the transaction we leave it for them to
            # handle (their except path).
            if started_tx:
                try:
                    self.conn.rollback()
                except sqlite3.Error as rb_exc:
                    # Rollback failure leaves the connection in an
                    # undefined state. Log the original + rollback
                    # errors so operators can diagnose; the original
                    # exception still propagates.
                    logger.error(
                        "consolidate_fact: rollback failed after error "
                        "(connection may be in undefined state): %s",
                        rb_exc,
                    )
            raise
    
    def _record_conflict(self, fact_a_id: str, fact_b_id: str,
                         conflict_type: str, commit: bool = True):
        """Record a conflict between two facts.

        commit (default True): whether to call self.conn.commit() after
            the INSERT. `consolidate_fact` passes `commit=False` when
            invoking this helper from within its own `BEGIN IMMEDIATE`
            scope so the fact INSERT and its conflict rows commit
            atomically (rather than _record_conflict's commit ending
            the outer transaction mid-flight). /review (4-source
            convergence: Codex structured + Codex adversarial + Claude
            adversarial + Maintainability) caught the original
            unconditional-commit as the primary atomicity breach.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type)
            VALUES (?, ?, ?)
        """, (fact_a_id, fact_b_id, conflict_type))
        if commit:
            self.conn.commit()
    
    def resolve_conflict(self, conflict_id: int, winning_fact_id: str):
        """
        Resolve a conflict by marking the losing fact as superseded.
        
        Args:
            conflict_id: Conflict to resolve
            winning_fact_id: The fact that wins
        """
        cursor = self.conn.cursor()
        
        # Get conflict details
        cursor.execute("SELECT * FROM conflicts WHERE id = ?", (conflict_id,))
        conflict = cursor.fetchone()
        
        if not conflict:
            return
        
        # Determine losing fact
        losing_id = conflict["fact_b_id"] if winning_fact_id == conflict["fact_a_id"] else conflict["fact_a_id"]
        
        # Mark as superseded
        now = datetime.now().isoformat()
        cursor.execute("""
            UPDATE consolidated_facts
            SET superseded_by = ?, updated_at = ?
            WHERE id = ?
        """, (winning_fact_id, now, losing_id))
        
        # Mark conflict as resolved
        cursor.execute("""
            UPDATE conflicts
            SET resolution = ?, resolved_at = ?
            WHERE id = ?
        """, (f"superseded_by_{winning_fact_id}", now, conflict_id))
        
        self.conn.commit()
    
    def get_conflicts(self) -> List[Dict]:
        """Get all unresolved conflicts."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM conflicts WHERE resolution IS NULL
            ORDER BY created_at DESC
        """)
        
        conflicts = []
        for row in cursor.fetchall():
            conflicts.append({
                "id": row["id"],
                "fact_a_id": row["fact_a_id"],
                "fact_b_id": row["fact_b_id"],
                "type": row["conflict_type"],
                "created_at": row["created_at"]
            })
        
        return conflicts
    
    def get_consolidated_facts(self, subject: str = None, min_confidence: float = 0.5) -> List[ConsolidatedFact]:
        """
        Get consolidated facts, optionally filtered by subject and confidence.
        
        Args:
            subject: Filter by subject
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of ConsolidatedFact
        """
        cursor = self.conn.cursor()
        
        if subject:
            cursor.execute("""
                SELECT * FROM consolidated_facts
                WHERE subject = ? AND confidence >= ? AND superseded_by IS NULL
                ORDER BY confidence DESC, mention_count DESC
            """, (subject, min_confidence))
        else:
            cursor.execute("""
                SELECT * FROM consolidated_facts
                WHERE confidence >= ? AND superseded_by IS NULL
                ORDER BY confidence DESC, mention_count DESC
            """, (min_confidence,))
        
        facts = []
        for row in cursor.fetchall():
            facts.append(ConsolidatedFact(
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                confidence=row["confidence"],
                mention_count=row["mention_count"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                sources=json.loads(row["sources_json"] or "[]"),
                veracity=row["veracity"],
                superseded=row["superseded_by"] is not None
            ))
        
        return facts
    
    def get_high_confidence_summary(self, subject: str, threshold: float = 0.8) -> str:
        """
        Generate a summary of high-confidence facts about a subject.
        
        Args:
            subject: Subject to summarize
            threshold: Confidence threshold
            
        Returns:
            str: Human-readable summary
        """
        facts = self.get_consolidated_facts(subject, min_confidence=threshold)
        
        if not facts:
            return f"No high-confidence facts about {subject}."
        
        lines = [f"High-confidence facts about {subject}:"]
        for fact in facts:
            lines.append(f"  - {fact.subject} {fact.predicate} {fact.object} "
                        f"(conf: {fact.confidence:.2f}, mentions: {fact.mention_count})")
        
        return "\n".join(lines)
    
    def run_consolidation_pass(self):
        """
        Background consolidation pass.
        
        1. Find facts with multiple mentions
        2. Boost confidence
        3. Detect conflicts
        4. Auto-resolve obvious conflicts (higher confidence wins)
        """
        cursor = self.conn.cursor()
        
        # Find facts ready for consolidation (mention_count > 2)
        cursor.execute("""
            SELECT * FROM consolidated_facts
            WHERE mention_count > 2 AND superseded_by IS NULL
            ORDER BY mention_count DESC
        """)
        
        for row in cursor.fetchall():
            subject = row["subject"]
            predicate = row["predicate"]
            
            # Find conflicts
            cursor.execute("""
                SELECT * FROM consolidated_facts
                WHERE subject = ? AND predicate = ? AND object != ?
                AND superseded_by IS NULL
            """, (subject, predicate, row["object"]))
            
            conflicts = cursor.fetchall()
            for conflict in conflicts:
                # Auto-resolve: higher confidence wins
                if row["confidence"] > conflict["confidence"]:
                    self.resolve_conflict_by_facts(row["id"], conflict["id"])
    
    def resolve_conflict_by_facts(self, winning_id: str, losing_id: str):
        """Resolve conflict by marking losing fact as superseded."""
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE consolidated_facts
            SET superseded_by = ?, updated_at = ?
            WHERE id = ?
        """, (winning_id, now, losing_id))
        
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get consolidation statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NULL")
        active_facts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NOT NULL")
        superseded_facts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM conflicts WHERE resolution IS NULL")
        unresolved_conflicts = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence) FROM consolidated_facts WHERE superseded_by IS NULL")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(mention_count) FROM consolidated_facts WHERE superseded_by IS NULL")
        avg_mentions = cursor.fetchone()[0] or 0.0
        
        return {
            "active_facts": active_facts,
            "superseded_facts": superseded_facts,
            "unresolved_conflicts": unresolved_conflicts,
            "avg_confidence": round(avg_confidence, 3),
            "avg_mentions": round(avg_mentions, 2),
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# --- Testing ---
if __name__ == "__main__":
    import tempfile
    import os
    
    print("Veracity Consolidation Tests")
    print("=" * 60)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    cons = VeracityConsolidator(db_path=Path(db_path))
    
    # Test 1: Basic consolidation
    print("\nTest 1: Basic consolidation")
    fact1 = cons.consolidate_fact("Alice", "is", "developer", "stated", "mem_001")
    print(f"  Initial: {fact1.subject} {fact1.predicate} {fact1.object} (conf: {fact1.confidence:.2f})")
    
    # Test 2: Bayesian update
    print("\nTest 2: Bayesian update")
    fact2 = cons.consolidate_fact("Alice", "is", "developer", "stated", "mem_002")
    print(f"  Updated: {fact2.subject} {fact2.predicate} {fact2.object} (conf: {fact2.confidence:.2f}, mentions: {fact2.mention_count})")
    
    # Test 3: Conflict detection
    print("\nTest 3: Conflict detection")
    fact3 = cons.consolidate_fact("Alice", "is", "manager", "inferred", "mem_003")
    print(f"  Conflict: {fact3.subject} {fact3.predicate} {fact3.object} (conf: {fact3.confidence:.2f})")
    
    conflicts = cons.get_conflicts()
    print(f"  Unresolved conflicts: {len(conflicts)}")
    
    # Test 4: Conflict resolution
    print("\nTest 4: Conflict resolution")
    if conflicts:
        cons.resolve_conflict(conflicts[0]["id"], "cf_Alice_is_developer")
        print(f"  Resolved conflict #{conflicts[0]['id']}")
    
    # Test 5: High-confidence summary
    print("\nTest 5: High-confidence summary")
    summary = cons.get_high_confidence_summary("Alice", threshold=0.5)
    print(summary)
    
    # Test 6: Stats
    print("\nTest 6: Stats")
    stats = cons.get_stats()
    print(f"  {stats}")
    
    # Cleanup
    cons.close()
    os.unlink(db_path)
    
    print("\n" + "=" * 60)
    print("Veracity consolidation tests passed!")
