"""Regression tests for C30 + C31 + C32 — pre-experiment cleanup follow-ups.

Three small fidelity fixes surfaced by the /review army on PRs #89 and
#90 that weren't bundled into those PRs because they're independent:

- **C30** (telemetry): `beam.py:2671` set `dense_score` for episodic
  fallback rows via `wm_vec_sims.get(row["id"], 0.0)` — but `wm_vec_sims`
  is the working-memory dict, ep ids aren't in it, so the value was
  always 0.0. Misleading provenance for post-run analysis. Fixed by
  setting `dense_score: 0.0` explicitly with a comment.
- **C31** (env parser): `MNEMOSYNE_BENCHMARK_PURE_RECALL=on` and
  `FULL_CONTEXT_MODE=on` were treated as falsy because the parser only
  accepted `1|true|yes`. Whitespace-padded values (`" 1 "`) were also
  treated as falsy. Fixed by routing through a new `_env_truthy()`
  helper that accepts `1|true|yes|on` and strips whitespace.
- **C32** (drift WARN): `MNEMOSYNE_*_WEIGHT` env vars override recall
  scoring but NOT consolidation Bayesian compounding. Fixed by emitting
  a single startup WARNING listing the overrides.
"""
from __future__ import annotations

import logging
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

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


# ─────────────────────────────────────────────────────────────────
# C30 — episodic fallback `dense_score` explicit 0.0
# ─────────────────────────────────────────────────────────────────


class TestC30EpisodicFallbackDenseScore:
    """Pre-fix: `beam.py:2671` set `dense_score` via `wm_vec_sims.get(row["id"], 0.0)`.
    Since `row["id"]` is always an episodic id and `wm_vec_sims` keys
    are working-memory ids, the lookup always returned 0.0 (the
    default). Same numeric value as the fix, but the wrong-dict
    lookup is misleading provenance: someone reading the code would
    think `dense_score` reflects a WM-vector similarity for ep rows.

    Fix: set `dense_score: 0.0` explicitly with a comment explaining
    that EM fallback rows reach this code path precisely because
    vec/FTS produced no episodic candidates, so no `sim` is computed."""

    def test_fallback_rows_have_explicit_zero_dense_score(self, temp_db, monkeypatch):
        """Seed an episodic row whose content has no vector embedding
        (so fallback is the only path that returns it), recall a query
        matching its content, assert the surviving row has dense_score=0.0."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Insert via SQL to skip embedding generation, forcing fallback path.
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("ep-no-emb", "unique-zorblax-token for fallback test",
             "consolidation", datetime.now().isoformat(), "s1", 0.5),
        )
        beam.conn.commit()

        results = beam.recall("zorblax", top_k=10)
        ep_rows = [r for r in results if r["id"] == "ep-no-emb"]
        assert ep_rows, (
            f"Expected fallback to surface seeded row; got: "
            f"{[(r['id'], r.get('tier'), r.get('content', '')[:50]) for r in results]}"
        )
        ep = ep_rows[0]
        # Field is present, explicit float 0.0.
        assert ep["dense_score"] == 0.0
        # Type is float (not None, not int) — keep downstream consumers stable.
        assert isinstance(ep["dense_score"], float)


# ─────────────────────────────────────────────────────────────────
# C31 — env-var truthy parser accepts `on` + trims whitespace
# ─────────────────────────────────────────────────────────────────


class TestC31EnvTruthyParser:
    """Pre-fix the parser was `lower() in ("1", "true", "yes")` — `on`
    and whitespace-padded values were treated as falsy. Fix: route
    through `_env_truthy()` which accepts `1|true|yes|on` (case-
    insensitive, whitespace-stripped)."""

    @pytest.fixture(autouse=True)
    def _ensure_clean_env(self, monkeypatch):
        # Clear both env vars before each test.
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        monkeypatch.delenv("FULL_CONTEXT_MODE", raising=False)

    @pytest.mark.parametrize("value", [
        "1", "true", "yes", "on",
        "TRUE", "True", "YES", "ON", "On",
        " 1 ", "  true  ", "\ton\t",  # whitespace
    ])
    def test_env_truthy_accepts_value(self, value, monkeypatch):
        from tools.evaluate_beam_end_to_end import _env_truthy

        monkeypatch.setenv("TEST_ENV_VAR", value)
        assert _env_truthy("TEST_ENV_VAR") is True

    @pytest.mark.parametrize("value", [
        "0", "false", "no", "off",
        "FALSE", "OFF",
        "", " ", "  ",
        "garbage", "maybe", "2", "y",  # non-canonical
    ])
    def test_env_truthy_rejects_value(self, value, monkeypatch):
        from tools.evaluate_beam_end_to_end import _env_truthy

        monkeypatch.setenv("TEST_ENV_VAR", value)
        assert _env_truthy("TEST_ENV_VAR") is False

    def test_env_truthy_unset_variable_is_false(self, monkeypatch):
        from tools.evaluate_beam_end_to_end import _env_truthy
        monkeypatch.delenv("TEST_ENV_VAR", raising=False)
        assert _env_truthy("TEST_ENV_VAR") is False

    def test_pure_recall_accepts_on(self, temp_db, monkeypatch):
        """End-to-end: `MNEMOSYNE_BENCHMARK_PURE_RECALL=on` now enables
        the gate. Pre-fix this was silently treated as off."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "on")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam._context_facts = {"favorite color blue": ["blue"]}
        fake_llm = MagicMock()
        fake_llm.chat = MagicMock(return_value="LLM-FALLBACK")
        from tools.evaluate_beam_end_to_end import answer_with_memory

        msgs = [{"role": "user", "content": f"row {i}"} for i in range(5)]
        answer_with_memory(
            llm=fake_llm, beam=beam,
            question="what is favorite color blue",
            conversation_messages=msgs, top_k=5, ability="IE",
        )
        # Pure-recall mode active → IE bypass disabled → LLM called.
        fake_llm.chat.assert_called_once()

    def test_pure_recall_accepts_whitespace_padded(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", " 1 ")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam._context_facts = {"favorite color blue": ["blue"]}
        fake_llm = MagicMock()
        fake_llm.chat = MagicMock(return_value="LLM-FALLBACK")
        from tools.evaluate_beam_end_to_end import answer_with_memory

        answer_with_memory(
            llm=fake_llm, beam=beam,
            question="what is favorite color blue",
            conversation_messages=[{"role": "user", "content": "x"}],
            top_k=5, ability="IE",
        )
        fake_llm.chat.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# C32 — MNEMOSYNE_*_WEIGHT env override startup WARN
# ─────────────────────────────────────────────────────────────────


class TestC32VeracityWeightOverrideWarn:
    """Pre-fix: operators setting `MNEMOSYNE_STATED_WEIGHT=0.9` (etc.)
    silently broke the 'consolidated-as-N also ranks at N' invariant
    because the consolidator's Bayesian compounding doesn't honor env
    overrides. Fix: emit a single WARNING listing the override(s).
    """

    def test_no_overrides_returns_empty_list(self, monkeypatch):
        """Sanity: when no env vars set, the helper returns empty."""
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        for name in (
            "MNEMOSYNE_STATED_WEIGHT", "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT", "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)
        assert _detect_veracity_weight_overrides() == []

    def test_single_override_returned(self, monkeypatch):
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        for name in (
            "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT", "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "0.95")
        assert _detect_veracity_weight_overrides() == ["MNEMOSYNE_STATED_WEIGHT"]

    def test_multiple_overrides_returned_in_canonical_order(self, monkeypatch):
        """Order matches the function's hard-coded canonical list so
        the warning log is deterministic across runs."""
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "1.0")
        monkeypatch.setenv("MNEMOSYNE_UNKNOWN_WEIGHT", "0.5")
        monkeypatch.setenv("MNEMOSYNE_TOOL_WEIGHT", "0.4")
        # Set some but not all; the others stay absent.
        monkeypatch.delenv("MNEMOSYNE_INFERRED_WEIGHT", raising=False)
        monkeypatch.delenv("MNEMOSYNE_IMPORTED_WEIGHT", raising=False)

        result = _detect_veracity_weight_overrides()
        # Canonical order from the function definition.
        assert result == [
            "MNEMOSYNE_STATED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ]

    def test_empty_string_value_still_counted_as_override(self, monkeypatch):
        """Operators sometimes export `VAR=` (empty) intending to
        unset; the helper treats presence as override. This is a
        defensible call — if the var is exported, the operator likely
        wanted it to affect something. Test pins the behavior so a
        future change to filter empty values is explicit."""
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "")
        for name in (
            "MNEMOSYNE_INFERRED_WEIGHT", "MNEMOSYNE_TOOL_WEIGHT",
            "MNEMOSYNE_IMPORTED_WEIGHT", "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)
        assert _detect_veracity_weight_overrides() == ["MNEMOSYNE_STATED_WEIGHT"]

    def test_warn_fires_when_overrides_present(self, monkeypatch, caplog):
        """Call `_warn_about_veracity_weight_overrides()` directly with
        env overrides set; assert WARN logged + returns True.

        This avoids `importlib.reload(beam_module)` — reloading the
        module poisons the test session because other tests imported
        constants like `STATED_WEIGHT` at session start. The helper-
        function approach exercises identical code without touching
        module-level state."""
        from mnemosyne.core.beam import _warn_about_veracity_weight_overrides

        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "0.5")
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            emitted = _warn_about_veracity_weight_overrides()

        assert emitted is True
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "Veracity weight env overrides detected" in r.message]
        assert warnings, (
            f"Expected veracity-weight WARN; got records: "
            f"{[r.message[:100] for r in caplog.records]}"
        )
        # The WARN should mention the specific env var.
        assert any("MNEMOSYNE_STATED_WEIGHT" in r.message for r in warnings)

    def test_warn_silent_when_no_overrides(self, monkeypatch, caplog):
        """Negative control: clean env → no WARN, returns False."""
        from mnemosyne.core.beam import _warn_about_veracity_weight_overrides

        for name in (
            "MNEMOSYNE_STATED_WEIGHT", "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT", "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)

        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            emitted = _warn_about_veracity_weight_overrides()

        assert emitted is False
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "Veracity weight env overrides detected" in r.message]
        assert warnings == []
