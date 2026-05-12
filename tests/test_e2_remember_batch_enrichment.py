"""
Regression tests for E2 — remember_batch enrichment parity with remember().

Pre-E2: ``BeamMemory.remember_batch`` skipped the post-insert enrichment
pipeline that ``BeamMemory.remember`` runs unconditionally
(`_add_temporal_triple` + `_ingest_graph_and_veracity`). High-throughput
ingest paths — including the BEAM benchmark adapter (E1) — bypassed the
annotation / gist / fact / consolidated-fact population entirely. The
polyphonic engine's graph + fact voices then had no data to fuse for
benchmark-scale recall queries — 4-voice RRF collapsed to 2 voices.

Post-E2: ``remember_batch`` mirrors ``remember()``'s post-insert
sequence:
  - Always-on (zero-LLM, rule-based / pattern-based):
    * `_add_temporal_triple` → annotations (occurred_on, has_source)
    * `_ingest_graph_and_veracity` → gists + facts + graph_edges +
      consolidated_facts (rule-based pattern extraction)
  - Opt-in via `extract_entities=True`:
    * `_extract_and_store_entities` → annotations (mentions)
  - Opt-in via `extract=True`:
    * `_extract_and_store_facts` → LLM-extracted facts table content

These tests pin:
  - Always-on parts fire for every batch row
  - Per-row source + veracity flow correctly into annotations + facts
  - Opt-in flags are respected (default off → no entity/LLM extraction)
  - Parity with ``remember()`` for the always-on parts
  - Benchmark-scale shape: 100-row batch still enriches every row
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "mnemosyne_e2.db"


def _annotation_rows(conn: sqlite3.Connection, memory_id: str):
    return conn.execute(
        "SELECT kind, value, source, confidence "
        "FROM annotations WHERE memory_id = ? "
        "ORDER BY kind, value",
        (memory_id,),
    ).fetchall()


def _gist_count(conn: sqlite3.Connection, memory_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]


def _fact_count(conn: sqlite3.Connection, memory_id: str) -> int:
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM facts WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def _consolidated_fact_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM consolidated_facts"
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Always-on enrichment fires for every row in the batch
# ---------------------------------------------------------------------------


def test_remember_batch_writes_temporal_annotations_for_every_row(temp_db):
    """Each row should get an `occurred_on` annotation (date slice of
    the row's timestamp). Pre-fix this didn't happen — annotations
    table was empty after a batch insert."""
    beam = BeamMemory(session_id="e2-temporal", db_path=temp_db)
    ids = beam.remember_batch([
        {"content": "Alice deployed the service", "source": "convo"},
        {"content": "Bob filed a bug", "source": "convo"},
        {"content": "Carol approved the plan", "source": "convo"},
    ])
    assert len(ids) == 3
    for memory_id in ids:
        kinds = {row[0] for row in _annotation_rows(beam.conn, memory_id)}
        assert "occurred_on" in kinds, (
            f"{memory_id}: missing occurred_on annotation — "
            "_add_temporal_triple didn't fire"
        )


def test_remember_batch_writes_has_source_when_source_is_non_default(temp_db):
    """`has_source` annotation only fires for non-conversational
    sources (mirrors _add_temporal_triple's filter). Items with
    source='conversation' / 'user' / 'assistant' get only
    occurred_on; explicit non-default sources also get has_source."""
    beam = BeamMemory(session_id="e2-source", db_path=temp_db)
    ids = beam.remember_batch([
        {"content": "From a doc",  "source": "document"},
        {"content": "From convo",  "source": "conversation"},
    ])
    doc_kinds = {row[0] for row in _annotation_rows(beam.conn, ids[0])}
    convo_kinds = {row[0] for row in _annotation_rows(beam.conn, ids[1])}
    assert "has_source" in doc_kinds, (
        "non-default source should produce has_source annotation"
    )
    assert "has_source" not in convo_kinds, (
        "conversational source should NOT produce has_source annotation "
        "(matches _add_temporal_triple filter)"
    )


def test_remember_batch_extracts_gists_and_consolidated_facts(temp_db):
    """`_ingest_graph_and_veracity` should fire for every batch row,
    producing rule-based gists + facts + consolidated_facts. Content
    chosen to match the regex pattern in
    `EpisodicGraph.extract_facts` ('X is Y')."""
    beam = BeamMemory(session_id="e2-graph", db_path=temp_db)
    ids = beam.remember_batch([
        {"content": "Alice is the lead engineer", "source": "convo"},
        {"content": "Bob is a contractor",        "source": "convo"},
    ])
    # Each row should have a gist
    for memory_id in ids:
        assert _gist_count(beam.conn, memory_id) >= 1, (
            f"{memory_id}: missing gist — _ingest_graph_and_veracity "
            "didn't fire"
        )
    # The pattern extractor should have produced consolidated_facts
    # entries from the "X is Y" matches.
    assert _consolidated_fact_count(beam.conn) > 0, (
        "consolidated_facts is empty — VeracityConsolidator wasn't "
        "consulted by the batch path"
    )


# ---------------------------------------------------------------------------
# Per-row source + veracity flow through to enrichment
# ---------------------------------------------------------------------------


def test_per_row_veracity_threads_into_consolidated_facts(temp_db):
    """Per-row veracity must propagate to VeracityConsolidator so
    consolidated_facts weighting is per-row, not collapsed to the
    method-level default. /review caught the prior conditional skip
    (Claude M3 — test passed vacuously when regex didn't extract
    either subject); this version asserts extraction succeeded first
    so a future regex change can't silently neuter the contract."""
    beam = BeamMemory(session_id="e2-ver", db_path=temp_db)
    beam.remember_batch([
        {"content": "Dana is a developer", "veracity": "stated"},
        {"content": "Eric is a tester",    "veracity": "inferred"},
    ])
    rows = beam.conn.execute(
        "SELECT subject, predicate, object, confidence "
        "FROM consolidated_facts "
        "WHERE subject IN ('Dana', 'Eric') "
        "ORDER BY subject"
    ).fetchall()
    by_subject = {r[0]: r[3] for r in rows}
    # Hard-fail if the regex didn't extract either subject — the
    # test's parity claim depends on both subjects landing in
    # consolidated_facts. A vacuous skip here would let a future
    # change to extract_facts silently break veracity-threading.
    assert "Dana" in by_subject, (
        f"Regex didn't extract subject 'Dana' from 'Dana is a "
        f"developer' — consolidated_facts subjects: {list(by_subject)}"
    )
    assert "Eric" in by_subject, (
        f"Regex didn't extract subject 'Eric' from 'Eric is a tester' "
        f"— consolidated_facts subjects: {list(by_subject)}"
    )
    assert by_subject["Dana"] != by_subject["Eric"], (
        "stated and inferred veracity collapsed to same confidence — "
        "per-row veracity didn't reach VeracityConsolidator"
    )


def test_per_row_source_flows_to_has_source_annotation(temp_db):
    """`has_source` annotation value should reflect each row's own
    `source` field, not the first row's or a default."""
    beam = BeamMemory(session_id="e2-src", db_path=temp_db)
    ids = beam.remember_batch([
        {"content": "From a wiki page", "source": "wiki"},
        {"content": "From an email",    "source": "email"},
    ])
    wiki_rows = _annotation_rows(beam.conn, ids[0])
    email_rows = _annotation_rows(beam.conn, ids[1])
    wiki_has_source = {r[1] for r in wiki_rows if r[0] == "has_source"}
    email_has_source = {r[1] for r in email_rows if r[0] == "has_source"}
    assert "wiki" in wiki_has_source, (
        f"row 0 has_source = {wiki_has_source}, expected 'wiki'"
    )
    assert "email" in email_has_source, (
        f"row 1 has_source = {email_has_source}, expected 'email'"
    )


# ---------------------------------------------------------------------------
# Opt-in flags are respected (and default-off)
# ---------------------------------------------------------------------------


def test_extract_entities_off_by_default(temp_db):
    """Default `extract_entities=False`: no `mentions` annotation
    rows should appear in a fresh batch insert."""
    beam = BeamMemory(session_id="e2-no-ent", db_path=temp_db)
    ids = beam.remember_batch([
        {"content": "Alice and Bob worked on the auth refactor"},
    ])
    rows = _annotation_rows(beam.conn, ids[0])
    kinds = [r[0] for r in rows]
    assert "mentions" not in kinds, (
        "default-off entity extraction leaked a mentions annotation"
    )


def test_extract_entities_true_populates_mentions(temp_db):
    """`extract_entities=True`: regex entity scan should produce
    `mentions` annotation rows."""
    beam = BeamMemory(session_id="e2-ent-on", db_path=temp_db)
    ids = beam.remember_batch(
        [
            {"content": "Alice and Bob worked on the auth refactor"},
        ],
        extract_entities=True,
    )
    rows = _annotation_rows(beam.conn, ids[0])
    kinds = [r[0] for r in rows]
    assert "mentions" in kinds, (
        "extract_entities=True should produce mentions annotations"
    )


def test_extract_false_does_not_call_llm(temp_db):
    """Default `extract=False`: the LLM-backed
    `_extract_and_store_facts` must NOT be called. We verify by
    patching the module-level function and asserting it never fired."""
    with patch(
        "mnemosyne.core.beam._extract_and_store_facts"
    ) as mock_facts:
        beam = BeamMemory(session_id="e2-no-llm", db_path=temp_db)
        beam.remember_batch([{"content": "Some content"}])
        assert mock_facts.call_count == 0, (
            "extract=False but LLM fact extraction fired anyway"
        )


def test_extract_true_calls_llm_fact_extractor_per_row(temp_db):
    """`extract=True`: `_extract_and_store_facts` must be called
    once per batch row. We patch the module-level function so the
    test doesn't actually hit any LLM provider."""
    with patch(
        "mnemosyne.core.beam._extract_and_store_facts"
    ) as mock_facts:
        beam = BeamMemory(session_id="e2-llm-on", db_path=temp_db)
        beam.remember_batch(
            [
                {"content": "Row A"},
                {"content": "Row B"},
                {"content": "Row C"},
            ],
            extract=True,
        )
        assert mock_facts.call_count == 3, (
            f"expected 3 LLM calls (one per row), got {mock_facts.call_count}"
        )


# ---------------------------------------------------------------------------
# Parity with remember() for the always-on parts
# ---------------------------------------------------------------------------


def test_remember_batch_parity_with_remember_for_annotations(temp_db):
    """`remember_batch([single_item])` should produce the same
    annotation rows as `remember(single_item)` does for the
    always-on enrichment pipeline (excluding LLM-only paths)."""
    # Run remember() in one beam, remember_batch() in another, then
    # compare annotation shapes for the same content.
    content = "Frank is a database administrator"
    src = "wiki"

    beam_single = BeamMemory(session_id="e2-parity-a", db_path=temp_db)
    mid_single = beam_single.remember(content, source=src)

    parity_db = temp_db.parent / "parity.db"
    beam_batch = BeamMemory(session_id="e2-parity-b", db_path=parity_db)
    [mid_batch] = beam_batch.remember_batch(
        [{"content": content, "source": src}]
    )

    a_kinds = sorted({
        row[0] for row in _annotation_rows(beam_single.conn, mid_single)
    })
    b_kinds = sorted({
        row[0] for row in _annotation_rows(beam_batch.conn, mid_batch)
    })
    assert a_kinds == b_kinds, (
        f"annotation kinds diverge: remember()={a_kinds}, "
        f"remember_batch()={b_kinds}"
    )


def test_remember_batch_parity_with_remember_for_gists(temp_db):
    """Single-row remember_batch should produce at least the same
    number of gists as remember() does for identical content."""
    content = "Grace is the new VP of engineering"

    beam_single = BeamMemory(session_id="e2-gist-a", db_path=temp_db)
    mid_single = beam_single.remember(content)

    parity_db = temp_db.parent / "parity_gist.db"
    beam_batch = BeamMemory(session_id="e2-gist-b", db_path=parity_db)
    [mid_batch] = beam_batch.remember_batch([{"content": content}])

    single_count = _gist_count(beam_single.conn, mid_single)
    batch_count = _gist_count(beam_batch.conn, mid_batch)
    assert single_count == batch_count, (
        f"gist count divergence: remember()={single_count}, "
        f"remember_batch()={batch_count}"
    )


# ---------------------------------------------------------------------------
# Robustness — enrichment failures don't tear down the batch
# ---------------------------------------------------------------------------


def test_enrichment_exception_does_not_break_batch(temp_db):
    """If any single row's enrichment helper raises, the working_memory
    insert must succeed for ALL rows AND enrichment must continue for
    rows after the failure. /review caught the prior test as
    tautological (Claude M4) — the working_memory commit happens
    BEFORE the enrichment loop, so the row-count check was trivially
    true. This version asserts both contracts: rows landed AND later
    rows' enrichment still produced annotations."""
    beam = BeamMemory(session_id="e2-fault", db_path=temp_db)
    # Inject a failure into _ingest_graph_and_veracity for one specific
    # content; verify all rows still landed in working_memory AND row 3
    # got its temporal annotation.
    original = beam._ingest_graph_and_veracity
    call_count = {"n": 0}

    def faulty(memory_id, content, source, veracity):
        call_count["n"] += 1
        if "boom" in content:
            raise RuntimeError("simulated extraction failure")
        return original(memory_id, content, source, veracity)

    beam._ingest_graph_and_veracity = faulty  # type: ignore

    ids = beam.remember_batch([
        {"content": "ok row 1"},
        {"content": "row with boom inside"},
        {"content": "ok row 3"},
    ])

    # Contract A: all 3 rows landed in working_memory.
    wm_count = beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE session_id = ?",
        ("e2-fault",),
    ).fetchone()[0]
    assert wm_count == 3, (
        f"enrichment failure tore down working_memory inserts: "
        f"only {wm_count}/3 rows present"
    )

    # Contract B: enrichment ran for all 3 rows. The helper was called
    # 3 times (the assertion the prior test missed — without this the
    # loop could short-circuit on row 2's exception and never reach
    # row 3 yet the wm_count check would still pass).
    assert call_count["n"] == 3, (
        f"enrichment loop short-circuited after row-2 failure; "
        f"_ingest_graph_and_veracity called {call_count['n']} times, "
        "expected 3"
    )

    # Contract C: row 3 got its temporal annotation (proves
    # _add_temporal_triple ran for the row after the failure — the
    # whole point of the inner try/except in the helpers is so a
    # bad row doesn't blow up later rows' enrichment).
    row3_kinds = {
        r[0] for r in _annotation_rows(beam.conn, ids[2])
    }
    assert "occurred_on" in row3_kinds, (
        f"row 3 missing occurred_on annotation after row-2 failure: "
        f"row3 annotations = {row3_kinds}"
    )


# ---------------------------------------------------------------------------
# /review hardening — commit 2 regression guards
# ---------------------------------------------------------------------------


class TestReviewHardening:
    """Closes the gaps surfaced by the /review army on commit 1.

    Each test pins one of the must-fix findings:
      - per-row commit cascade → single deferred commit
      - regex backtracking via 4096-char content cap
      - MEMORY_ADDED event emission parity with remember()
      - meta_by_id dict (vs prior parallel-list + assert)
    """

    def test_enrichment_loop_uses_single_deferred_commit(
        self, temp_db, monkeypatch
    ):
        """The enrichment loop must wrap inner helper commits in
        _deferred_commits so a 250K-row batch doesn't produce
        millions of fsync round-trips. We verify by counting
        `_real_commit` invocations on the connection — the inner
        helpers all go through `commit()` which the defer flag
        short-circuits; only the deferred-commits context manager
        calls `_real_commit()` (once at the end of the scope)."""
        from mnemosyne.core.beam import _BeamConnection

        beam = BeamMemory(session_id="e2-commits", db_path=temp_db)
        assert isinstance(beam.conn, _BeamConnection), (
            "BeamMemory.conn should be a _BeamConnection (via "
            "_get_connection's factory); the deferred-commit "
            "optimization depends on it"
        )

        # Wrap _real_commit to count invocations. The inner helpers'
        # commit() calls short-circuit on _defer_commit=True and
        # never reach _real_commit, so this counter measures the
        # number of actual fsync round-trips during the batch.
        commit_count = {"n": 0}
        original_real_commit = beam.conn._real_commit

        def counting_real_commit():
            commit_count["n"] += 1
            return original_real_commit()

        monkeypatch.setattr(
            beam.conn, "_real_commit", counting_real_commit
        )

        beam.remember_batch([
            {"content": "Henry is a researcher"},
            {"content": "Ivy is a designer"},
            {"content": "Jack is a writer"},
        ])

        # Inside _deferred_commits, exactly one _real_commit fires
        # at the end of the loop. Outside the deferred scope, the
        # bulk INSERT's commit and embedding-write commits go through
        # the regular `commit()` method (no defer flag set there), so
        # _real_commit isn't called for them. Allow ≤2 to leave slack
        # for any future addition; the regression we're guarding
        # against produces 10-15 _real_commit calls per row.
        assert commit_count["n"] <= 2, (
            f"too many real-commit fsync round-trips ({commit_count['n']}) "
            "during enrichment loop — _deferred_commits not engaged "
            "(per-row commit cascade regressed)"
        )

    def test_extract_facts_caps_long_content(self, temp_db):
        """`extract_facts` (used by _ingest_graph_and_veracity) must
        cap content at 4096 chars to prevent regex backtracking on
        adversarial long inputs. We verify by passing a 10KB string
        of repeated "A is " patterns — extraction should return
        bounded results in bounded time, not stall."""
        beam = BeamMemory(session_id="e2-cap", db_path=temp_db)
        # 10KB of pattern-rich content.
        long_content = ("Anna is a developer. " * 600).strip()
        assert len(long_content) > 4096, (
            "test setup: content not long enough to exercise cap"
        )
        import time
        start = time.monotonic()
        beam.remember_batch([{"content": long_content}])
        elapsed = time.monotonic() - start
        # If the cap is broken, this could take many seconds /
        # minutes. The cap should bring it well under 5s even on a
        # slow CI machine.
        assert elapsed < 5.0, (
            f"long-content batch took {elapsed:.2f}s — content cap "
            "likely not applied (regex backtracking on full input)"
        )

    def test_remember_batch_emits_memory_added_event_per_row(self, temp_db):
        """/review caught the parity gap: `remember()` ends with
        `_emit_event("MEMORY_ADDED", ...)` but pre-fix `remember_batch`
        didn't. DeltaSync streaming + any other event consumer saw
        zero batch ingest events. Post-fix every batch row emits
        MEMORY_ADDED."""
        beam = BeamMemory(session_id="e2-events", db_path=temp_db)
        captured = []

        def collect(event):
            captured.append(event)

        beam._event_emitter = collect

        ids = beam.remember_batch([
            {"content": "Event row A", "importance": 0.3},
            {"content": "Event row B", "importance": 0.4},
            {"content": "Event row C", "importance": 0.5},
        ])
        # MemoryEvent is a dataclass-like object with attributes:
        # event_type (EventType enum) and memory_id (str). One
        # MEMORY_ADDED per row.
        added = [
            e for e in captured
            if getattr(e, "event_type", None) is not None
            and e.event_type.name == "MEMORY_ADDED"
        ]
        assert len(added) == 3, (
            f"expected 3 MEMORY_ADDED events, got {len(added)} "
            f"(captured event types: "
            f"{[getattr(e, 'event_type', None) for e in captured]})"
        )
        # Each event should carry the corresponding memory_id.
        event_ids = {e.memory_id for e in added}
        assert event_ids == set(ids), (
            f"event memory_ids {event_ids} != batch returned ids {set(ids)}"
        )

    def test_meta_by_id_dict_survives_python_o(self, temp_db):
        """The prior `assert mid_check == memory_id` parallel-list
        integrity check would have stripped under `python -O`. The
        replacement `meta_by_id: Dict[str, Tuple[str, str]]` keyed by
        memory_id eliminates the class of bug entirely — no parallel
        lists to desync, no assert to strip.

        We can't toggle -O at runtime, but we can prove the new shape
        works correctly by inducing a hypothetical desync scenario:
        force a specific ordering and verify per-row source flows to
        the correct annotation regardless of insertion order."""
        beam = BeamMemory(session_id="e2-dict", db_path=temp_db)
        ids = beam.remember_batch([
            {"content": "First from wiki", "source": "wiki"},
            {"content": "Second from email", "source": "email"},
            {"content": "Third from doc", "source": "doc"},
        ])
        # Each row's has_source annotation should match its OWN
        # source, not an adjacent row's. This is the property the
        # parallel-list pattern got wrong under refactor risk.
        for memory_id, expected_source in zip(ids, ["wiki", "email", "doc"]):
            rows = _annotation_rows(beam.conn, memory_id)
            has_source = {r[1] for r in rows if r[0] == "has_source"}
            assert expected_source in has_source, (
                f"{memory_id} has_source mismatch: got {has_source}, "
                f"expected '{expected_source}' — meta_by_id keying "
                "broken or row identification regressed"
            )

    def test_deferred_commits_rollback_on_exception(self, temp_db):
        """_deferred_commits must rollback (not commit) when the
        body raises, so partial enrichment writes don't leak into the
        DB. Verifies the exception path of the context manager."""
        from mnemosyne.core.beam import _deferred_commits

        beam = BeamMemory(session_id="e2-rollback", db_path=temp_db)
        # Pre-state: 0 annotations.
        beam.conn.execute("DELETE FROM annotations")
        beam.conn.commit()

        # Insert one row inside the deferred-commits scope, then raise.
        try:
            with _deferred_commits(beam.conn):
                beam.conn.execute(
                    "INSERT INTO annotations "
                    "(memory_id, kind, value, source, confidence, created_at) "
                    "VALUES ('test-id', 'mentions', 'Test', 'manual', 1.0, datetime('now'))"
                )
                raise RuntimeError("simulated post-write failure")
        except RuntimeError:
            pass

        # The annotation should NOT have been committed.
        count = beam.conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE memory_id = 'test-id'"
        ).fetchone()[0]
        assert count == 0, (
            f"_deferred_commits failed to rollback on exception: "
            f"{count} annotations leaked into the DB"
        )
