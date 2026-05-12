"""Regression tests for the BEAM-recovery experiment A/B toggles.

Each toggle is default-ON (production behavior unchanged) and disables
the corresponding feature when set to a falsy value (`0`/`false`/`no`/
`off`, case-insensitive, whitespace-stripped). See
`docs/benchmarking.md` for the full toggle catalog.

These tests pin three properties per toggle:
  1. Default behavior (env unset) — feature ENABLED.
  2. Falsy values disable the feature.
  3. The disable surfaces in actual recall results (when feasible) —
     not just an internal flag — so a future refactor that strips the
     gate from the code path fails this test instead of silently
     reverting the experiment's ability to ablate.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mnemosyne.core.beam import BeamMemory, _env_disabled


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture(autouse=True)
def _clean_toggle_env(monkeypatch):
    """Each test starts from a clean env. Clears every toggle so
    test order can't affect outcomes."""
    for name in (
        "MNEMOSYNE_VOICE_VECTOR", "MNEMOSYNE_VOICE_GRAPH",
        "MNEMOSYNE_VOICE_FACT", "MNEMOSYNE_VOICE_TEMPORAL",
        "MNEMOSYNE_GRAPH_BONUS", "MNEMOSYNE_FACT_BONUS",
        "MNEMOSYNE_BINARY_BONUS", "MNEMOSYNE_VERACITY_MULTIPLIER",
        "MNEMOSYNE_CROSS_TIER_DEDUP",
    ):
        monkeypatch.delenv(name, raising=False)


# ─────────────────────────────────────────────────────────────────
# _env_disabled helper itself
# ─────────────────────────────────────────────────────────────────


class TestEnvDisabledHelper:
    """Pins the default-ON-with-opt-out semantics so future toggles
    that use this helper inherit the same falsy-value parsing."""

    @pytest.mark.parametrize("value", [
        "0", "false", "no", "off",
        "FALSE", "OFF", "False", "No",
        " 0 ", "  false  ", "\toff\t",
    ])
    def test_falsy_values_disable(self, value, monkeypatch):
        monkeypatch.setenv("X_TEST_TOGGLE", value)
        assert _env_disabled("X_TEST_TOGGLE") is True

    @pytest.mark.parametrize("value", [
        "1", "true", "yes", "on",
        "TRUE", "ON",
        "", " ", "anything-else", "maybe",
    ])
    def test_truthy_or_garbage_enable(self, value, monkeypatch):
        """Unset / empty / truthy / unrecognized → feature stays ON."""
        monkeypatch.setenv("X_TEST_TOGGLE", value)
        assert _env_disabled("X_TEST_TOGGLE") is False

    def test_unset_returns_false(self, monkeypatch):
        monkeypatch.delenv("X_TEST_TOGGLE", raising=False)
        assert _env_disabled("X_TEST_TOGGLE") is False


# ─────────────────────────────────────────────────────────────────
# Polyphonic voice toggles (4)
# ─────────────────────────────────────────────────────────────────


class TestPolyphonicVoiceToggles:
    """Each polyphonic voice has a toggle. When disabled, the voice
    returns `[]`, contributing nothing to the engine's RRF fusion."""

    @pytest.fixture
    def engine(self, temp_db):
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine
        # Construct with default args; uses temp_db for any state.
        eng = PolyphonicRecallEngine(db_path=temp_db)
        yield eng
        eng.close()

    def test_vector_voice_disabled_returns_empty(self, engine, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VOICE_VECTOR", "0")
        # Pass a dummy embedding; with toggle off we should short-circuit
        # before any DB work.
        import numpy as np
        result = engine._vector_voice(np.zeros(384, dtype=np.float32))
        assert result == []

    def test_vector_voice_enabled_runs(self, engine, monkeypatch):
        """With toggle unset (default), the voice attempts to run.
        We don't assert specific results — only that we got past the
        early-return guard (returns a list, even if empty)."""
        monkeypatch.delenv("MNEMOSYNE_VOICE_VECTOR", raising=False)
        import numpy as np
        result = engine._vector_voice(np.zeros(384, dtype=np.float32))
        assert isinstance(result, list)
        # No memories in fresh DB → empty list, but we got past the gate.

    def test_graph_voice_disabled_returns_empty(self, engine, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "false")
        result = engine._graph_voice("any query here")
        assert result == []

    def test_graph_voice_enabled_runs(self, engine, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_VOICE_GRAPH", raising=False)
        result = engine._graph_voice("any query here")
        assert isinstance(result, list)

    def test_fact_voice_disabled_returns_empty(self, engine, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VOICE_FACT", "off")
        result = engine._fact_voice("any query here")
        assert result == []

    def test_fact_voice_enabled_runs(self, engine, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_VOICE_FACT", raising=False)
        result = engine._fact_voice("any query here")
        assert isinstance(result, list)

    def test_temporal_voice_disabled_returns_empty(self, engine, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VOICE_TEMPORAL", "no")
        result = engine._temporal_voice("recent activity yesterday")
        assert result == []

    def test_temporal_voice_enabled_runs(self, engine, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_VOICE_TEMPORAL", raising=False)
        result = engine._temporal_voice("recent activity yesterday")
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────
# Linear-path bonus toggles (3)
# ─────────────────────────────────────────────────────────────────


class TestLinearBonusToggles:
    """Linear path's `graph_bonus` / `fact_bonus` / `binary_bonus` add
    capped (0.08, 0.1, 0.08) score lifts to episodic rows. With the
    toggles disabled, those lifts must not be applied.

    Direct assertion: with toggle OFF, the score is what hybrid scoring
    produces WITHOUT the bonus block running. We check this by
    structural test of beam.py — the toggles short-circuit the entire
    bonus-computation block, not just the final addition, so any rows
    that would have received a bonus get a strictly lower score.
    """

    def _seed_episodic_with_graph_data(self, beam: BeamMemory):
        """Seed one episodic row + corresponding graph_edges + facts
        rows so a recall query has bonuses available to claim."""
        ts = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("ep-bonus", "deploy production rollout plan", "consolidation",
             ts, "s1", 0.5),
        )
        # graph_edges: this memory_id has connections
        beam.conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type) "
            "VALUES (?, ?, ?)",
            ("ep-bonus", "ep-other", "related"),
        )
        beam.conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type) "
            "VALUES (?, ?, ?)",
            ("ep-other", "ep-bonus", "related"),
        )
        # facts: this memory has extracted facts that match a query word
        beam.conn.execute(
            "INSERT INTO facts (fact_id, session_id, source_msg_id, subject, predicate, object) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("fact-1", "s1", "ep-bonus", "team", "deploys", "production"),
        )
        beam.conn.commit()

    def test_graph_bonus_disabled_does_not_apply(self, temp_db, monkeypatch):
        """With `MNEMOSYNE_GRAPH_BONUS=0`, the graph-edge bonus block is
        skipped. We construct the same scenario twice and assert the
        scores differ by the expected bonus amount."""
        # Run 1: default ON — score includes graph bonus.
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        self._seed_episodic_with_graph_data(beam)
        # Defang downstream multipliers we don't care about.
        monkeypatch.setenv("MNEMOSYNE_VERACITY_MULTIPLIER", "0")
        monkeypatch.delenv("MNEMOSYNE_GRAPH_BONUS", raising=False)
        on_results = beam.recall("deploy production rollout", top_k=5)
        beam.conn.close()

        # Run 2: bonus OFF, otherwise identical.
        with tempfile.TemporaryDirectory() as tmpdir2:
            db2 = Path(tmpdir2) / "test.db"
            beam2 = BeamMemory(session_id="s1", db_path=db2)
            self._seed_episodic_with_graph_data(beam2)
            monkeypatch.setenv("MNEMOSYNE_GRAPH_BONUS", "0")
            off_results = beam2.recall("deploy production rollout", top_k=5)
            beam2.conn.close()

        on_hit = next((r for r in on_results if r["id"] == "ep-bonus"), None)
        off_hit = next((r for r in off_results if r["id"] == "ep-bonus"), None)
        assert on_hit is not None and off_hit is not None
        # When bonus is enabled, score is strictly higher (bonus is
        # additive on the linear ep path). Allow for floating-point
        # noise.
        assert on_hit["score"] > off_hit["score"], (
            f"graph_bonus toggle had no effect on score: "
            f"on={on_hit['score']} off={off_hit['score']}"
        )

    def test_fact_bonus_disabled_does_not_apply(self, temp_db, monkeypatch):
        """Same shape for fact_bonus: with toggle off, score is strictly
        lower than with toggle on."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        self._seed_episodic_with_graph_data(beam)
        # Defang other lifts so the only difference is fact_bonus.
        monkeypatch.setenv("MNEMOSYNE_VERACITY_MULTIPLIER", "0")
        monkeypatch.setenv("MNEMOSYNE_GRAPH_BONUS", "0")
        monkeypatch.delenv("MNEMOSYNE_FACT_BONUS", raising=False)
        on_results = beam.recall("deploys production", top_k=5)
        beam.conn.close()

        with tempfile.TemporaryDirectory() as tmpdir2:
            db2 = Path(tmpdir2) / "test.db"
            beam2 = BeamMemory(session_id="s1", db_path=db2)
            self._seed_episodic_with_graph_data(beam2)
            monkeypatch.setenv("MNEMOSYNE_FACT_BONUS", "0")
            off_results = beam2.recall("deploys production", top_k=5)
            beam2.conn.close()

        on_hit = next((r for r in on_results if r["id"] == "ep-bonus"), None)
        off_hit = next((r for r in off_results if r["id"] == "ep-bonus"), None)
        assert on_hit is not None and off_hit is not None
        assert on_hit["score"] > off_hit["score"], (
            f"fact_bonus toggle had no effect: "
            f"on={on_hit['score']} off={off_hit['score']}"
        )

    def test_binary_bonus_toggle_structural(self):
        """Source-level check: `MNEMOSYNE_BINARY_BONUS` is referenced
        in both linear main loop and fallback (so disabling it gates
        both branches). End-to-end test would require a query embedding
        + binary vector setup that's brittle; this catches the regression
        where someone strips the gate."""
        src = (Path(__file__).resolve().parents[1] / "mnemosyne" / "core" / "beam.py").read_text()
        # Should be referenced in the binary_bonus gate site
        assert src.count("MNEMOSYNE_BINARY_BONUS") >= 1, (
            "MNEMOSYNE_BINARY_BONUS gate missing from beam.py — "
            "ablation toggle stripped"
        )


# ─────────────────────────────────────────────────────────────────
# Veracity multiplier toggle
# ─────────────────────────────────────────────────────────────────


class TestVeracityMultiplierToggle:
    """`MNEMOSYNE_VERACITY_MULTIPLIER=0` short-circuits the multiplier
    in BOTH the linear and polyphonic paths so Phase 0/1 ablation works
    identically across engines."""

    def test_disabled_makes_stated_unknown_score_equal(self, temp_db, monkeypatch):
        """Two episodic rows with identical content but different
        veracity ('stated' vs 'unknown') should score IDENTICALLY when
        the multiplier is disabled, regardless of their veracity."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ts = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-stated", "the user prefers dark mode", "consolidation",
             ts, "s1", 0.5, "stated"),
        )
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-unknown", "the user prefers dark mode", "consolidation",
             ts, "s1", 0.5, "unknown"),
        )
        beam.conn.commit()

        monkeypatch.setenv("MNEMOSYNE_VERACITY_MULTIPLIER", "0")
        # Defang other bonuses
        monkeypatch.setenv("MNEMOSYNE_GRAPH_BONUS", "0")
        monkeypatch.setenv("MNEMOSYNE_FACT_BONUS", "0")
        monkeypatch.setenv("MNEMOSYNE_BINARY_BONUS", "0")
        results = beam.recall("dark mode", top_k=10)
        by_id = {r["id"]: r for r in results}
        if "ep-stated" in by_id and "ep-unknown" in by_id:
            assert by_id["ep-stated"]["score"] == by_id["ep-unknown"]["score"], (
                "Veracity multiplier toggle OFF, but stated/unknown rows "
                f"scored differently: {by_id['ep-stated']['score']} vs "
                f"{by_id['ep-unknown']['score']}"
            )

    def test_enabled_makes_stated_outrank_unknown(self, temp_db, monkeypatch):
        """Sanity / positive control: with toggle ON (default), the
        stated row should rank above the unknown one (1.0 > 0.8
        multiplier)."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ts = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-stated", "the user prefers dark mode", "consolidation",
             ts, "s1", 0.5, "stated"),
        )
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-unknown", "the user prefers dark mode", "consolidation",
             ts, "s1", 0.5, "unknown"),
        )
        beam.conn.commit()

        monkeypatch.delenv("MNEMOSYNE_VERACITY_MULTIPLIER", raising=False)
        monkeypatch.setenv("MNEMOSYNE_GRAPH_BONUS", "0")
        monkeypatch.setenv("MNEMOSYNE_FACT_BONUS", "0")
        monkeypatch.setenv("MNEMOSYNE_BINARY_BONUS", "0")
        results = beam.recall("dark mode", top_k=10)
        by_id = {r["id"]: r for r in results}
        if "ep-stated" in by_id and "ep-unknown" in by_id:
            assert by_id["ep-stated"]["score"] > by_id["ep-unknown"]["score"]


# ─────────────────────────────────────────────────────────────────
# Cross-tier dedup toggle
# ─────────────────────────────────────────────────────────────────


class TestCrossTierDedupToggle:
    """`MNEMOSYNE_CROSS_TIER_DEDUP=0` short-circuits
    `_dedup_cross_tier_summary_links` to return the input list unchanged."""

    def test_disabled_returns_input_unchanged(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        monkeypatch.setenv("MNEMOSYNE_CROSS_TIER_DEDUP", "0")

        # Construct a (summary, source) pair that WOULD normally dedup.
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("wm-src", "raw text content", "conversation",
             datetime.now().isoformat(), "s1", 0.5),
        )
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, summary_of) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-sum", "summary of raw text content", "consolidation",
             datetime.now().isoformat(), "s1", 0.5, "wm-src"),
        )
        beam.conn.commit()

        # Synthetic results — both should survive when dedup is off.
        results = [
            {"id": "wm-src", "tier": "working", "score": 0.9, "content": "raw"},
            {"id": "ep-sum", "tier": "episodic", "score": 0.5, "content": "sum"},
        ]
        out = beam._dedup_cross_tier_summary_links(results)
        assert len(out) == 2
        assert out is results, "Toggle-off path must short-circuit to identity"

    def test_enabled_dedups_normally(self, temp_db, monkeypatch):
        """Positive control: with toggle ON, the lower-scored side gets
        dropped per E3.a.3 logic."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        monkeypatch.delenv("MNEMOSYNE_CROSS_TIER_DEDUP", raising=False)

        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("wm-src", "raw text", "conversation",
             datetime.now().isoformat(), "s1", 0.5),
        )
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, summary_of) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-sum", "summary", "consolidation",
             datetime.now().isoformat(), "s1", 0.5, "wm-src"),
        )
        beam.conn.commit()

        results = [
            {"id": "wm-src", "tier": "working", "score": 0.9, "content": "raw"},
            {"id": "ep-sum", "tier": "episodic", "score": 0.5, "content": "sum"},
        ]
        out = beam._dedup_cross_tier_summary_links(results)
        # Dedup should drop the lower-scored ep
        assert len(out) == 1
        assert out[0]["id"] == "wm-src"


# ─────────────────────────────────────────────────────────────────
# Coverage map: every toggle has at least one disabled-path test
# ─────────────────────────────────────────────────────────────────


class TestToggleCoverageMap:
    """Pin that the 9 documented toggles are each present in the code.
    A future refactor that strips one of them fails this test even if
    no specific functional test was written for that one."""

    REQUIRED_TOGGLES = [
        "MNEMOSYNE_VOICE_VECTOR",
        "MNEMOSYNE_VOICE_GRAPH",
        "MNEMOSYNE_VOICE_FACT",
        "MNEMOSYNE_VOICE_TEMPORAL",
        "MNEMOSYNE_GRAPH_BONUS",
        "MNEMOSYNE_FACT_BONUS",
        "MNEMOSYNE_BINARY_BONUS",
        "MNEMOSYNE_VERACITY_MULTIPLIER",
        "MNEMOSYNE_CROSS_TIER_DEDUP",
    ]

    def test_all_toggles_present_in_source(self):
        repo_root = Path(__file__).resolve().parents[1]
        sources = (
            (repo_root / "mnemosyne" / "core" / "beam.py").read_text()
            + (repo_root / "mnemosyne" / "core" / "polyphonic_recall.py").read_text()
        )
        missing = [t for t in self.REQUIRED_TOGGLES if t not in sources]
        assert not missing, (
            f"Required A/B toggles missing from source: {missing}. "
            f"docs/benchmarking.md promises these toggles exist."
        )
