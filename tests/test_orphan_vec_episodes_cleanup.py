"""
Regression tests for orphan `vec_episodes` cleanup on import.

`vec_episodes` is a sqlite-vec virtual table keyed by
`episodic_memory.rowid` (an AUTOINCREMENT integer PK). When
production code DELETEs from episodic_memory and re-INSERTs, the
new row gets a fresh rowid via lastrowid — leaving the old row's
vector embedding stranded in vec_episodes pointing at a rowid that
will never be reused by AUTOINCREMENT.

Pre-fix: `import_from_dict` (beam.py:3991) DELETE+INSERTs
episodic_memory rows when `force=True` but didn't clean
vec_episodes. Operators with high import churn (regular
backup/restore cycles, multi-source imports) accumulate dead
vec_episodes entries indefinitely. Each entry is small (~768 bytes
for float32 / 48 bytes for binary quantized) but unbounded
accumulation matters at long-running-deployment scale.

Post-fix: when `import_from_dict` finds an existing episodic_memory
row and `force=True`, it deletes the corresponding vec_episodes
entry by rowid BEFORE deleting episodic_memory. The new episodic
row's embedding is re-imported in the same `import_from_dict` call
via the episodic_embeddings section.

These tests pin:
  - import_from_dict with force=True cleans vec_episodes orphans
  - import_from_dict with force=False (skip path) doesn't touch
    vec_episodes
  - import_from_dict is idempotent across multiple force=True calls
    on the same data (no accumulating orphans)
  - cleanup is best-effort (failure in DELETE FROM vec_episodes
    doesn't abort the import)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory, _vec_available, _vec_insert


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "orphan_test.db"


def _vec_count(conn) -> int:
    """Count rows in vec_episodes (skip if sqlite-vec unavailable)."""
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM vec_episodes"
        ).fetchone()[0]
    except Exception:
        return -1


def _em_count(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM episodic_memory"
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Core contract: force=True path cleans vec_episodes
# ---------------------------------------------------------------------------


def test_import_from_dict_force_cleans_vec_episodes_orphan(temp_db):
    """When import_from_dict DELETEs an episodic_memory row to
    overwrite (force=True), the corresponding vec_episodes entry
    must be deleted too. Without this, vec_episodes accumulates
    dead rowids forever."""
    beam = BeamMemory(session_id="orphan-test", db_path=temp_db)
    if not _vec_available(beam.conn):
        pytest.skip("sqlite-vec not available in this environment")

    # Seed an episodic_memory row + its vec_episodes entry.
    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, importance) "
        "VALUES ('em-1', 'original content', 'test', "
        "datetime('now'), 0.5)"
    )
    original_rowid = beam.conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", ("em-1",)
    ).fetchone()[0]
    _vec_insert(
        beam.conn, original_rowid,
        [0.1] * 384,  # dummy embedding
    )
    beam.conn.commit()

    vec_before = _vec_count(beam.conn)
    em_before = _em_count(beam.conn)
    assert vec_before == 1
    assert em_before == 1

    # Now do an import that overwrites the same id with force=True.
    payload = {
        "version": 1,
        "working_memory": [],
        "episodic_memory": [
            {
                "id": "em-1",
                "rowid": original_rowid,
                "content": "new content",
                "source": "import",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "import-session",
                "importance": 0.7,
                "metadata_json": "{}",
                "summary_of": "",
                "valid_until": None,
                "superseded_by": None,
                "scope": "session",
                "recall_count": 0,
                "last_recalled": None,
                "created_at": "2026-05-11T00:00:00",
            }
        ],
        "episodic_embeddings": [],
    }
    beam.import_from_dict(payload, force=True)

    # The episodic_memory row should still be there (new content).
    assert _em_count(beam.conn) == 1
    # vec_episodes should not have an orphan — either the old entry
    # was cleaned (count goes to 0, no new embedding imported), or
    # it was cleaned and a new one was inserted (count stays at 1).
    # We seeded no episodic_embeddings in the payload above, so we
    # expect count == 0 (the orphan was cleaned, no replacement).
    assert _vec_count(beam.conn) == 0, (
        f"orphan not cleaned — vec_episodes count is "
        f"{_vec_count(beam.conn)} after import-with-force"
    )


def test_import_from_dict_no_force_does_not_touch_vec_episodes(temp_db):
    """When force=False and the row already exists, import_from_dict
    skips. vec_episodes must NOT be touched in that path."""
    beam = BeamMemory(session_id="no-force", db_path=temp_db)
    if not _vec_available(beam.conn):
        pytest.skip("sqlite-vec not available in this environment")

    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, importance) "
        "VALUES ('em-keep', 'keep me', 'test', "
        "datetime('now'), 0.5)"
    )
    rowid = beam.conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", ("em-keep",)
    ).fetchone()[0]
    _vec_insert(beam.conn, rowid, [0.2] * 384)
    beam.conn.commit()

    vec_before = _vec_count(beam.conn)
    assert vec_before == 1

    payload = {
        "version": 1,
        "working_memory": [],
        "episodic_memory": [
            {
                "id": "em-keep",
                "rowid": rowid,
                "content": "would overwrite if force=True",
                "source": "import",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "x",
                "importance": 0.5,
                "metadata_json": "{}",
                "summary_of": "",
                "valid_until": None,
                "superseded_by": None,
                "scope": "session",
                "recall_count": 0,
                "last_recalled": None,
                "created_at": "2026-05-11T00:00:00",
            }
        ],
        "episodic_embeddings": [],
    }
    beam.import_from_dict(payload, force=False)

    # vec_episodes entry should be unchanged.
    assert _vec_count(beam.conn) == 1, (
        "vec_episodes touched on force=False skip path"
    )


# ---------------------------------------------------------------------------
# Idempotency: repeated force imports don't accumulate orphans
# ---------------------------------------------------------------------------


def test_import_from_dict_force_idempotent_no_orphan_accumulation(temp_db):
    """Running the same force=True import 5 times must not leave
    5 orphans in vec_episodes — each iteration cleans the previous
    iteration's entry."""
    beam = BeamMemory(session_id="idem", db_path=temp_db)
    if not _vec_available(beam.conn):
        pytest.skip("sqlite-vec not available in this environment")

    # Seed initial state.
    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, importance) "
        "VALUES ('em-loop', 'initial', 'test', "
        "datetime('now'), 0.5)"
    )
    initial_rowid = beam.conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", ("em-loop",)
    ).fetchone()[0]
    _vec_insert(beam.conn, initial_rowid, [0.3] * 384)
    beam.conn.commit()

    payload_template = {
        "version": 1,
        "working_memory": [],
        "episodic_memory": [
            {
                "id": "em-loop",
                "rowid": initial_rowid,
                "content": "round X",
                "source": "import",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "x",
                "importance": 0.5,
                "metadata_json": "{}",
                "summary_of": "",
                "valid_until": None,
                "superseded_by": None,
                "scope": "session",
                "recall_count": 0,
                "last_recalled": None,
                "created_at": "2026-05-11T00:00:00",
            }
        ],
        "episodic_embeddings": [],
    }

    for i in range(5):
        # Mutate the rowid to whatever the current row has — since
        # each import_from_dict reassigns via lastrowid on the new
        # INSERT, the rowid we pass in the payload only matters for
        # the old_to_new_rowid mapping used by the embeddings section.
        beam.import_from_dict(payload_template, force=True)

    # After 5 rounds, episodic_memory should have exactly 1 row.
    assert _em_count(beam.conn) == 1
    # vec_episodes should be 0 (each round cleaned, none re-imported).
    assert _vec_count(beam.conn) == 0, (
        f"orphan accumulation: vec_episodes count is "
        f"{_vec_count(beam.conn)} after 5 force-imports — cascade "
        "cleanup not idempotent"
    )


# ---------------------------------------------------------------------------
# Robustness: cleanup failure doesn't abort the import
# ---------------------------------------------------------------------------


def test_import_from_dict_cleanup_failure_is_best_effort(temp_db, monkeypatch):
    """If DELETE FROM vec_episodes raises OperationalError (e.g.,
    sqlite-vec extension unloaded mid-import, table missing
    between vec_ok check and DELETE), the import should continue
    with episodic_memory work. Data integrity over orphan cleanup.

    Exercising the actual try/except path requires a scenario
    where `_vec_available()` returns True (so the cascade runs)
    but the DELETE then raises. Codex structured /review (P2 GATE
    FAIL on commit 1) caught a prior version of this test that
    just dropped the table — that exercised the `vec_ok=False`
    skip path, not the try/except failure path.

    We force the scenario by: (1) patching _vec_available to
    return True even after we drop the table — the production
    code sees vec_ok=True, attempts the DELETE, gets
    OperationalError "no such table: vec_episodes", and the
    try/except swallows it."""
    beam = BeamMemory(session_id="best-effort", db_path=temp_db)
    if not _vec_available(beam.conn):
        pytest.skip("sqlite-vec not available in this environment")

    # Seed initial row.
    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, importance) "
        "VALUES ('em-best', 'orig', 'test', "
        "datetime('now'), 0.5)"
    )
    rowid = beam.conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", ("em-best",)
    ).fetchone()[0]
    _vec_insert(beam.conn, rowid, [0.4] * 384)
    beam.conn.commit()

    # Drop the vec_episodes table so DELETE FROM vec_episodes
    # raises. Then patch _vec_available so the production code
    # still thinks sqlite-vec is available — this routes execution
    # into the cascade-cleanup branch, where the DELETE fails and
    # the try/except is exercised.
    beam.conn.execute("DROP TABLE vec_episodes")
    beam.conn.commit()
    import mnemosyne.core.beam as beam_module
    monkeypatch.setattr(
        beam_module, "_vec_available", lambda conn: True
    )

    payload = {
        "version": 1,
        "working_memory": [],
        "episodic_memory": [
            {
                "id": "em-best",
                "rowid": rowid,
                "content": "new",
                "source": "import",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "x",
                "importance": 0.5,
                "metadata_json": "{}",
                "summary_of": "",
                "valid_until": None,
                "superseded_by": None,
                "scope": "session",
                "recall_count": 0,
                "last_recalled": None,
                "created_at": "2026-05-11T00:00:00",
            }
        ],
        "episodic_embeddings": [],
    }
    # Should NOT raise — the cleanup failure is swallowed.
    # If the try/except were removed from the production fix, this
    # call would propagate sqlite3.OperationalError.
    beam.import_from_dict(payload, force=True)

    # episodic_memory should still have the new row (despite vec
    # cleanup failure).
    assert _em_count(beam.conn) == 1


# ---------------------------------------------------------------------------
# End-to-end: full force-overwrite + new-embedding-arrives round-trip
# ---------------------------------------------------------------------------


def test_import_from_dict_force_with_new_embeddings_no_orphan(temp_db):
    """End-to-end: force-import a payload that ALSO carries
    `episodic_embeddings`. Expected final state:
      - exactly 1 episodic_memory row (the new one)
      - exactly 1 vec_episodes row (the new embedding at the new
        rowid; old orphan was cleaned)
      - vec_episodes.rowid == episodic_memory.rowid (no stale
        mapping from a payload-stated old rowid)

    /review (Claude H1) flagged this missing coverage: every other
    test uses `"episodic_embeddings": []`, so the case where the
    cascade-cleanup AND the re-import both run end-to-end was
    untested. A future regression in `old_to_new_rowid` mapping
    would slip through silently."""
    beam = BeamMemory(session_id="e2e", db_path=temp_db)
    if not _vec_available(beam.conn):
        pytest.skip("sqlite-vec not available in this environment")

    # Seed initial state.
    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, importance) "
        "VALUES ('em-e2e', 'orig', 'test', "
        "datetime('now'), 0.5)"
    )
    initial_rowid = beam.conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", ("em-e2e",)
    ).fetchone()[0]
    _vec_insert(beam.conn, initial_rowid, [0.5] * 384)
    beam.conn.commit()

    assert _em_count(beam.conn) == 1
    assert _vec_count(beam.conn) == 1

    # Import payload with the new content AND a new embedding.
    payload = {
        "version": 1,
        "working_memory": [],
        "episodic_memory": [
            {
                "id": "em-e2e",
                "rowid": initial_rowid,  # old rowid in payload
                "content": "new content",
                "source": "import",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "x",
                "importance": 0.7,
                "metadata_json": "{}",
                "summary_of": "",
                "valid_until": None,
                "superseded_by": None,
                "scope": "session",
                "recall_count": 0,
                "last_recalled": None,
                "created_at": "2026-05-11T00:00:00",
            }
        ],
        # The new embedding is keyed to the OLD rowid in the payload;
        # `import_from_dict` maps it to the NEW rowid via
        # `old_to_new_rowid` after the INSERT.
        "episodic_embeddings": [
            {
                "rowid": initial_rowid,
                "embedding": [0.9] * 384,
            }
        ],
    }
    beam.import_from_dict(payload, force=True)

    # Final state checks.
    assert _em_count(beam.conn) == 1, "duplicate episodic_memory rows"
    assert _vec_count(beam.conn) == 1, (
        f"expected 1 vec_episodes row (new embedding only); got "
        f"{_vec_count(beam.conn)} — either orphan not cleaned or "
        "new embedding not imported"
    )

    # The vec_episodes rowid should match the NEW episodic_memory rowid.
    new_em_rowid = beam.conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", ("em-e2e",)
    ).fetchone()[0]
    vec_rowid = beam.conn.execute(
        "SELECT rowid FROM vec_episodes"
    ).fetchone()[0]
    assert vec_rowid == new_em_rowid, (
        f"vec_episodes.rowid ({vec_rowid}) != "
        f"episodic_memory.rowid ({new_em_rowid}) — "
        "old_to_new_rowid mapping drifted on force path"
    )
