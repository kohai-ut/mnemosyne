"""Regression tests for C25 — DeltaSync table + column allowlist.

Pre-C25: `DeltaSync.compute_delta(peer_id, table=...)` and
`DeltaSync.apply_delta(peer_id, delta, table=...)` interpolated the
`table` kwarg directly into f-string SQL:

    cursor.execute(f"SELECT * FROM {table} WHERE ...")
    cursor.execute(f"INSERT INTO {table} ({cols}) VALUES (...)")

Plus the apply_delta path used the keys of the peer-supplied `delta`
dict to build column lists:

    cols = [k for k in mem.keys() if k not in ("rowid",)]
    cursor.execute(f"INSERT INTO {table} ({', '.join(cols)}) ...")

Two real SQL injection vectors:
  1. Caller-supplied `table` kwarg (config-file injection, plugin
     misuse, etc.)
  2. Peer-controlled column names in incoming delta dicts (a
     remote peer can send a delta that smuggles arbitrary SQL into
     a column-name slot)

Post-C25:
  - `table` is validated against `ALLOWED_DELTA_TABLES` at the public
    method boundary; anything outside raises ValueError.
  - Column names in incoming deltas are filtered against the live
    schema's column allowlist (PRAGMA-derived, per-table, cached).
    Unknown columns are silently dropped and counted in a new
    `filtered_keys` stat so operators can spot a misconfigured peer.

Maintainer note (issue #64): streaming emit was wired live by commit
`b2a7fae`, raising the practical relevance of the allowlist.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.streaming import (
    ALLOWED_DELTA_TABLES,
    DeltaSync,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def mnem(temp_db):
    """A Mnemosyne instance with a couple of working_memory rows so
    delta computation has something to return."""
    m = Mnemosyne(session_id="s1", db_path=temp_db)
    m.remember("Alice prefers Vim", source="pref", importance=0.7)
    m.remember("Bob owns the auth module", source="fact", importance=0.8)
    return m


@pytest.fixture
def sync_ckpt_dir(tmp_path):
    return tmp_path / "sync"


class TestC25TableAllowlist:

    def test_allowlist_constant_is_explicit(self):
        """The allowlist set must include both production tables AND
        be a frozenset (immutable). A test on the constant prevents
        accidental drift: someone adding `triples` or `facts` to the
        set without thinking about the schema-column allowlist
        implications would surface here."""
        assert isinstance(ALLOWED_DELTA_TABLES, frozenset)
        assert ALLOWED_DELTA_TABLES == frozenset({"working_memory", "episodic_memory"})

    def test_compute_delta_accepts_working_memory(self, mnem, sync_ckpt_dir):
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        delta = sync.compute_delta("peer-A", table="working_memory")
        assert len(delta) >= 2, "expected the seeded rows in delta"

    def test_compute_delta_accepts_episodic_memory(self, mnem, sync_ckpt_dir):
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        # No episodic rows seeded but the call should still succeed.
        delta = sync.compute_delta("peer-A", table="episodic_memory")
        assert delta == []

    def test_compute_delta_rejects_unknown_table(self, mnem, sync_ckpt_dir):
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.compute_delta("peer-A", table="some_other_table")

    def test_compute_delta_rejects_injection_attempt(self, mnem, sync_ckpt_dir):
        """The whole point of the allowlist. Pre-fix the payload
        below would have executed against the local DB."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        payload = "working_memory; DROP TABLE episodic_memory; --"
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.compute_delta("peer-A", table=payload)

        # Sanity: episodic_memory still exists.
        conn = sqlite3.connect(str(mnem.db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='episodic_memory'"
        )
        assert cursor.fetchone() is not None, (
            "injection-attempt table arg somehow affected the schema"
        )
        conn.close()

    def test_compute_delta_rejects_non_string_table(self, mnem, sync_ckpt_dir):
        """Edge case — None / int / list as `table` must error
        clearly, not silently mis-route."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        for bad in (None, 42, ["working_memory"], object()):
            with pytest.raises(ValueError):
                sync.compute_delta("peer-A", table=bad)

    def test_apply_delta_rejects_unknown_table(self, mnem, sync_ckpt_dir):
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.apply_delta("peer-A", [], table="something_else")

    def test_apply_delta_rejects_injection_attempt(self, mnem, sync_ckpt_dir):
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.apply_delta(
                "peer-A",
                [{"id": "x", "content": "y"}],
                table="working_memory; ATTACH DATABASE '/tmp/evil' AS evil; --",
            )

    def test_sync_to_inherits_validation(self, mnem, sync_ckpt_dir):
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.sync_to("peer-A", table="bogus")

    def test_sync_from_inherits_validation(self, mnem, sync_ckpt_dir):
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.sync_from("peer-A", [{"id": "x"}], table="bogus")


class TestC25ColumnAllowlist:

    def test_apply_delta_filters_unknown_column(self, mnem, sync_ckpt_dir):
        """[Attack vector] A peer sends a delta with a column name
        that doesn't exist in the schema. Pre-fix that key flowed
        straight into `INSERT INTO working_memory (col) VALUES (?)`
        and raised an OperationalError mid-batch (best case) or
        injected SQL (worst case). Post-fix it's filtered out and
        counted in `filtered_keys`; the rest of the row still applies."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        delta = [{
            "id": "new-row-1",
            "content": "legit content",
            "source": "test",
            "timestamp": "2026-05-11T00:00:00",
            "session_id": "s1",
            "importance": 0.5,
            "totally_made_up_column": "garbage",
        }]
        stats = sync.apply_delta("peer-A", delta, table="working_memory")
        assert stats["inserted"] == 1, f"expected 1 insert, got {stats}"
        assert stats["filtered_keys"] >= 1, (
            f"unknown column wasn't filtered; got {stats}"
        )

        # Row exists; the bogus column is not in the DB schema.
        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT id, content FROM working_memory WHERE id = ?",
            ("new-row-1",),
        ).fetchone()
        assert row is not None
        assert row[1] == "legit content"
        # Verify schema doesn't have the bogus column.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(working_memory)").fetchall()]
        assert "totally_made_up_column" not in cols
        conn.close()

    def test_apply_delta_filters_injection_in_column_name(
        self, mnem, sync_ckpt_dir
    ):
        """[Attack vector] A peer sends `{"foo); DROP TABLE x; --": "v"}`
        as a key. Pre-fix the malicious string would have been
        interpolated into `INSERT INTO table (foo); DROP TABLE x; --) VALUES (?)`.
        Post-fix it's filtered as not-in-schema."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        evil_col = "foo); DROP TABLE episodic_memory; --"
        delta = [{
            "id": "new-row-2",
            "content": "legit content",
            "source": "test",
            "timestamp": "2026-05-11T00:00:00",
            "session_id": "s1",
            "importance": 0.5,
            evil_col: "evil value",
        }]
        stats = sync.apply_delta("peer-A", delta, table="working_memory")
        assert stats["filtered_keys"] >= 1, (
            f"injection column wasn't filtered; got {stats}"
        )

        # Sanity: episodic_memory still exists.
        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='episodic_memory'"
        ).fetchone()
        assert row is not None, "injection in column name affected schema"
        conn.close()

    def test_apply_delta_filters_unknown_column_on_update(
        self, mnem, sync_ckpt_dir
    ):
        """Same filter applies on the UPDATE path (existing-row case)."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        # First, insert via apply_delta so the id matches the existing
        # row on the next call.
        sync.apply_delta(
            "peer-A",
            [{
                "id": "upd-row-1",
                "content": "initial content",
                "source": "test",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "s1",
                "importance": 0.5,
            }],
            table="working_memory",
        )

        # Now send an update with both a real and a fake column.
        stats = sync.apply_delta(
            "peer-A",
            [{
                "id": "upd-row-1",
                "content": "updated content",
                "made_up_column": "should be filtered",
            }],
            table="working_memory",
        )
        assert stats["updated"] == 1
        assert stats["filtered_keys"] >= 1

        # The real update landed.
        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT content FROM working_memory WHERE id = ?",
            ("upd-row-1",),
        ).fetchone()
        assert row[0] == "updated content"
        conn.close()

    def test_apply_delta_filters_reserved_columns(self, mnem, sync_ckpt_dir):
        """`rowid`, `timestamp`, `created_at` are reserved on UPDATE
        even though they're real columns in the schema. They're
        routing/metadata keys, not user-mutable fields. Pre-fix this
        was already true for UPDATE (the original code had the
        `if k not in ("id", "rowid", "timestamp", "created_at")`
        guard); C25 makes it explicit via the reserved set constant."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        # Seed a row.
        sync.apply_delta(
            "peer-A",
            [{
                "id": "reserved-row",
                "content": "original",
                "source": "test",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "s1",
                "importance": 0.5,
            }],
            table="working_memory",
        )

        # Send an update that tries to mutate timestamp + created_at.
        # Pre-fix and post-fix: both are filtered out as reserved.
        stats = sync.apply_delta(
            "peer-A",
            [{
                "id": "reserved-row",
                "content": "new content",
                "timestamp": "2099-01-01T00:00:00",  # reserved
                "created_at": "2099-01-01T00:00:00",  # reserved
            }],
            table="working_memory",
        )
        assert stats["updated"] == 1

        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT content, timestamp FROM working_memory WHERE id = ?",
            ("reserved-row",),
        ).fetchone()
        conn.close()
        assert row[0] == "new content"
        # timestamp NOT changed to 2099 (reserved on update path).
        assert "2099" not in row[1]


class TestC25ReviewHardening:
    """Findings from /review (Codex structured + Claude adversarial +
    Codex adversarial). Each test pins one of the closed bypass paths."""

    def test_str_subclass_bypass_rejected(self, mnem, sync_ckpt_dir):
        """[Claude adv #1 CRITICAL] A `str` subclass with overridden
        __eq__/__hash__ can pass `isinstance(str)` AND `in frozenset`
        while the f-string uses any payload. Strict type check
        (`type(table) is str`) closes the gate."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)

        class MyStr(str):
            def __new__(cls, allow, payload):
                inst = super().__new__(cls, payload)
                inst._allow = allow
                return inst

            def __eq__(self, other):
                return other == self._allow or super().__eq__(other)

            def __hash__(self):
                return hash(self._allow)

        evil = MyStr(
            "working_memory",
            "(SELECT data AS content, data AS id FROM sqlite_master)",
        )
        # Pre-fix this would pass isinstance + frozenset membership
        # and execute the subquery; strict type check rejects.
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.compute_delta("peer-A", table=evil)
        with pytest.raises(ValueError, match="not in the allowlist"):
            sync.apply_delta("peer-A", [], table=evil)

    def test_apply_rejects_peer_session_id_on_update(self, mnem, sync_ckpt_dir):
        """[Claude adv #2 HIGH] A peer must not be able to re-route
        a row to another session by setting session_id in an UPDATE
        delta. Pre-fix `session_id` was a schema column not in the
        reserved-update set; opt-in allowlist closes this."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        # Find an existing row.
        conn = sqlite3.connect(str(mnem.db_path))
        victim_id = conn.execute(
            "SELECT id FROM working_memory WHERE session_id = 's1' LIMIT 1"
        ).fetchone()[0]
        conn.close()

        stats = sync.apply_delta(
            "peer-attacker",
            [{
                "id": victim_id,
                "content": "redirected content",
                "session_id": "attacker-session",
            }],
            table="working_memory",
        )
        # The session_id key must have been filtered.
        assert stats["filtered_keys"] >= 1

        # Victim row's session_id is unchanged.
        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT session_id FROM working_memory WHERE id = ?",
            (victim_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "s1", (
            f"peer redirected session_id: now {row[0]!r}"
        )

    def test_apply_rejects_peer_superseded_by_on_update(self, mnem, sync_ckpt_dir):
        """[Claude adv #2 HIGH] A peer must not soft-delete any row
        by setting superseded_by via UPDATE."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        conn = sqlite3.connect(str(mnem.db_path))
        victim_id = conn.execute(
            "SELECT id FROM working_memory LIMIT 1"
        ).fetchone()[0]
        conn.close()

        stats = sync.apply_delta(
            "peer-attacker",
            [{
                "id": victim_id,
                "content": "still good",
                "superseded_by": "fake-replacement",
            }],
            table="working_memory",
        )
        assert stats["filtered_keys"] >= 1

        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT superseded_by FROM working_memory WHERE id = ?",
            (victim_id,),
        ).fetchone()
        conn.close()
        assert row[0] is None, (
            f"peer flipped superseded_by: now {row[0]!r}"
        )

    def test_insert_rejects_peer_session_id(self, mnem, sync_ckpt_dir):
        """[Claude adv #3 HIGH] On INSERT, peer cannot land a row
        directly in the destination's session via the session_id
        column. Destination's column DEFAULT applies instead."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        stats = sync.apply_delta(
            "peer-attacker",
            [{
                "id": "peer-injected-row",
                "content": "peer content",
                "source": "test",
                "timestamp": "2026-05-11T00:00:00",
                "session_id": "attacker-session-claim",
                "importance": 0.5,
            }],
            table="working_memory",
        )
        assert stats["inserted"] == 1
        # session_id was filtered → destination DEFAULT applied.
        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT session_id FROM working_memory WHERE id = ?",
            ("peer-injected-row",),
        ).fetchone()
        conn.close()
        # The schema's column DEFAULT is 'default'. The peer's claim
        # must not survive.
        assert row[0] != "attacker-session-claim", (
            f"peer landed row in claimed session: {row[0]!r}"
        )

    def test_insert_rejects_peer_created_at(self, mnem, sync_ckpt_dir):
        """[Claude adv #3 HIGH] On INSERT, peer cannot backdate
        created_at to evade retention/degradation logic."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        sync.apply_delta(
            "peer-attacker",
            [{
                "id": "backdate-attempt",
                "content": "peer content",
                "source": "test",
                "timestamp": "2026-05-11T00:00:00",
                "created_at": "1970-01-01T00:00:00",  # forged epoch
                "importance": 0.5,
            }],
            table="working_memory",
        )

        conn = sqlite3.connect(str(mnem.db_path))
        row = conn.execute(
            "SELECT created_at FROM working_memory WHERE id = ?",
            ("backdate-attempt",),
        ).fetchone()
        conn.close()
        assert row is not None
        # Destination DEFAULT (CURRENT_TIMESTAMP) — not the forged 1970.
        assert "1970" not in (row[0] or ""), (
            f"peer back-dated created_at: {row[0]!r}"
        )

    def test_apply_safe_against_temp_table_shadow(self, mnem, sync_ckpt_dir):
        """[Codex adv MEDIUM] A peer with same-connection access who
        creates a temp table `working_memory` could shadow the real
        table for unqualified SQL. Qualifying to `main.working_memory`
        defeats this. Test: create the shadow, run apply, assert the
        real table got the update (not the shadow)."""
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        # Create a temp shadow with a content column.
        mnem.conn.execute(
            "CREATE TEMP TABLE working_memory (id TEXT, content TEXT)"
        )
        try:
            stats = sync.apply_delta(
                "peer-X",
                [{
                    "id": "shadow-test-row",
                    "content": "should land in main, not temp",
                    "source": "test",
                    "timestamp": "2026-05-11T00:00:00",
                    "importance": 0.5,
                }],
                table="working_memory",
            )
            assert stats["inserted"] == 1

            # Real main.working_memory has the row.
            row = mnem.conn.execute(
                "SELECT content FROM main.working_memory WHERE id = ?",
                ("shadow-test-row",),
            ).fetchone()
            assert row is not None
            assert "should land in main" in row[0]

            # Temp shadow is empty.
            shadow = mnem.conn.execute(
                "SELECT COUNT(*) FROM temp.working_memory"
            ).fetchone()[0]
            assert shadow == 0, (
                f"row landed in temp shadow ({shadow} rows); main."
                f"qualifier didn't bind correctly"
            )
        finally:
            mnem.conn.execute("DROP TABLE temp.working_memory")

    def test_checkpoint_per_table(self, mnem, sync_ckpt_dir):
        """[Codex review P2] Checkpoints must be scoped by (peer,
        table). Pre-fix a single per-peer checkpoint stored a single
        rowid for all tables; cross-table sync with the same peer
        would skip rows whose table-local rowid was below the
        sibling-table's stored rowid."""
        from mnemosyne.core.streaming import SyncCheckpoint
        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)

        # Set distinct checkpoints for the same peer across tables.
        sync.set_checkpoint(
            "peer-X",
            SyncCheckpoint(peer_id="peer-X", last_sync_at="2026-01-01T00:00:00", last_rowid=100),
            table="working_memory",
        )
        sync.set_checkpoint(
            "peer-X",
            SyncCheckpoint(peer_id="peer-X", last_sync_at="2026-01-02T00:00:00", last_rowid=5),
            table="episodic_memory",
        )

        wm_cp = sync.get_checkpoint("peer-X", table="working_memory")
        ep_cp = sync.get_checkpoint("peer-X", table="episodic_memory")
        assert wm_cp.last_rowid == 100, "working_memory checkpoint clobbered"
        assert ep_cp.last_rowid == 5, "episodic_memory checkpoint clobbered"

        # Persistence round-trip: spawn a new DeltaSync against the
        # same checkpoint_dir and verify both load.
        sync2 = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        assert sync2.get_checkpoint("peer-X", table="working_memory").last_rowid == 100
        assert sync2.get_checkpoint("peer-X", table="episodic_memory").last_rowid == 5

    def test_legacy_checkpoint_filename_loads_as_working_memory(
        self, mnem, sync_ckpt_dir
    ):
        """Pre-/review-hardening: legacy `checkpoint_<peer>.json`
        files exist on operator disks. They must still load
        post-hardening — as the working_memory checkpoint (best-
        effort backward compat) so an upgrade doesn't lose state."""
        sync_ckpt_dir.mkdir(parents=True, exist_ok=True)
        legacy = sync_ckpt_dir / "checkpoint_legacy-peer.json"
        legacy.write_text('{"peer_id":"legacy-peer","last_sync_at":"2026-01-01T00:00:00","last_memory_id":null,"last_rowid":42}')

        sync = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        cp = sync.get_checkpoint("legacy-peer", table="working_memory")
        assert cp is not None, "legacy checkpoint file failed to load"
        assert cp.last_rowid == 42


class TestC25EndToEndRoundtrip:
    """[/regression] Normal sync flow still works post-C25.
    Filtering edges and table allowlisting must not break legitimate
    use of the public API."""

    def test_compute_then_apply_preserves_content(self, mnem, sync_ckpt_dir, tmp_path):
        """Source instance: compute_delta. Destination instance:
        apply_delta. Content + sync-relevant fields round-trip;
        lifecycle/scope/audit fields are destination-defaulted (the
        peer can't claim authorship or land in the destination's
        session)."""
        sync_src = DeltaSync(mnem, checkpoint_dir=sync_ckpt_dir)
        delta = sync_src.compute_delta("peer-B", table="working_memory")
        assert len(delta) == 2

        # Fresh destination Mnemosyne.
        dest_db = tmp_path / "dest.db"
        dest_mnem = Mnemosyne(session_id="dest-session", db_path=dest_db)
        dest_sync = DeltaSync(dest_mnem, checkpoint_dir=tmp_path / "dest_sync")
        stats = dest_sync.apply_delta("peer-B", delta, table="working_memory")

        assert stats["inserted"] == 2
        # Lifecycle/scope/audit fields get filtered on the apply side
        # — destination owns them. Counter exposes how many were
        # dropped per row (operator can monitor the noise floor).
        assert stats["filtered_keys"] > 0, (
            "filtered_keys should reflect dropped lifecycle/audit "
            "columns from the peer's record"
        )

        # Destination has the rows + content survived.
        conn = sqlite3.connect(str(dest_db))
        rows = conn.execute(
            "SELECT id, content, session_id FROM working_memory ORDER BY id"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        contents = [r[1] for r in rows]
        assert any("Alice" in c for c in contents)
        assert any("Bob" in c for c in contents)
        # session_id is DESTINATION-defaulted, not peer-controlled.
        # Source mnem used session_id='s1'; destination uses
        # 'default' (column DEFAULT) since peer can't override it.
        for row_id, row_content, row_session in rows:
            assert row_session != "s1", (
                f"row {row_id} landed with peer's session_id={row_session!r} "
                f"— C25 hardening should default to destination's column"
            )
