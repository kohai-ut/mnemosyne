"""Regression tests for E7/E8/E9 — `MNEMOSYNE_BENCHMARK_PURE_RECALL` gate.

`tools/evaluate_beam_end_to_end.py` historically shipped four bypass
paths that let the harness answer benchmark questions WITHOUT going
through Mnemosyne recall:

- **E7 TR oracle:** TR (Temporal Reasoning) questions extracted a
  timeline from raw `conversation_messages` and returned the LLM
  answer directly (line 1080) before any `BeamMemory.recall()`.
- **E7 CR augmentation:** CR (Contradiction Resolution) questions
  injected contradiction context built from raw messages into the
  answer prompt (line 1089).
- **E8 IE/KU side-index:** `_context_facts` (built from raw messages
  at ingest, line 418) was queried by IE/KU questions; matching
  values were returned directly at line 1291.
- **E9 RECENT CONVERSATION:** the last 12 raw messages were prepended
  to every answer prompt (line 1282) regardless of arm or recall
  quality.

For the BEAM-recovery experiment (Arms A/B/C compare recall pathways),
these bypasses mean the harness measures a harness-side oracle on
TR/CR/IE/KU and the recent-context shortcut on every question type —
NOT what the arms actually retrieve. `MNEMOSYNE_BENCHMARK_PURE_RECALL=1`
(or `--pure-recall`) disables all four.

Default behavior preserved when env unset.
"""
from __future__ import annotations

import os
import sys
import tempfile
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


@pytest.fixture
def fake_llm():
    """LLMClient stand-in that captures the messages it was called with."""
    llm = MagicMock()
    llm.chat = MagicMock(return_value="LLM-FALLBACK-ANSWER")
    return llm


def _build_msgs(n: int = 20) -> List[dict]:
    """Synthetic conversation messages — `n` user turns."""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"message-{i} payload alpha"})
    return msgs


@pytest.fixture
def beam_with_context_facts(temp_db):
    """A BeamMemory with a non-empty `_context_facts` map so we can
    exercise the IE/KU side-index path."""
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam._context_facts = {"favorite color blue": ["blue"]}
    return beam


# ─────────────────────────────────────────────────────────────────
# Default mode (env unset) — existing bypasses still fire
# ─────────────────────────────────────────────────────────────────


class TestDefaultModeBehaviorUnchanged:
    """When `MNEMOSYNE_BENCHMARK_PURE_RECALL` is unset, the existing
    bypass paths still fire (zero behavioral regression for callers
    who haven't migrated)."""

    def test_default_ie_returns_context_fact_value(self, beam_with_context_facts, fake_llm, monkeypatch):
        """IE question with a matching `_context_facts` entry returns
        the value directly — bypass is active."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(20),
            top_k=5,
            ability="IE",
        )
        assert ans == "blue"
        # LLM was NOT called — bypass returned the value directly.
        fake_llm.chat.assert_not_called()

    def test_default_recent_context_included(self, temp_db, fake_llm, monkeypatch):
        """When LLM is invoked (e.g., for ABS questions), the prompt
        includes the RECENT CONVERSATION section by default."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        msgs = _build_msgs(20)
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="some abstract reasoning question",
            conversation_messages=msgs,
            top_k=5,
            ability="ABS",
        )
        # LLM called; its prompt includes "RECENT CONVERSATION".
        fake_llm.chat.assert_called()
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        assert "RECENT CONVERSATION" in user_msg


# ─────────────────────────────────────────────────────────────────
# Pure-recall mode — all bypasses disabled
# ─────────────────────────────────────────────────────────────────


class TestPureRecallModeDisablesBypasses:
    """When `MNEMOSYNE_BENCHMARK_PURE_RECALL=1`, every bypass is
    disabled and every answer must go through the full LLM path with
    only retrieved memories in context."""

    def test_pure_recall_ie_does_not_return_context_fact_value(
        self, beam_with_context_facts, fake_llm, monkeypatch
    ):
        """IE question with matching `_context_facts` should NOT short-
        circuit in pure-recall mode — the LLM gets called instead."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        from tools.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(20),
            top_k=5,
            ability="IE",
        )
        # LLM was called — bypass disabled.
        fake_llm.chat.assert_called_once()
        # Whatever the LLM returned is the answer (our fake returns LLM-FALLBACK-ANSWER).
        assert ans == "LLM-FALLBACK-ANSWER"

    def test_pure_recall_tr_does_not_short_circuit_via_oracle(
        self, temp_db, fake_llm, monkeypatch
    ):
        """TR question should NOT take the timeline-oracle path that
        returns an LLM answer directly from extracted dates. Instead
        it falls through to the standard recall + LLM path."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        # TR-shaped question with multiple date references that would
        # otherwise trigger the oracle.
        msgs = [
            {"role": "user", "content": "I started the project on 2024-01-15"},
            {"role": "user", "content": "Then I deployed on 2024-03-22"},
            {"role": "user", "content": "Final release was 2024-06-30"},
        ]
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="how many days between project start and deployment?",
            conversation_messages=msgs,
            top_k=5,
            ability="TR",
        )
        # The TR-bypass returns the LLM answer with a date-calculator
        # system prompt; the pure-recall path uses ANSWER_SYSTEM_PROMPT.
        # Inspect the system prompt sent to the LLM to distinguish.
        sent_messages = fake_llm.chat.call_args[0][0]
        system_msg = next(m for m in sent_messages if m["role"] == "system")
        assert "date calculator" not in system_msg["content"].lower(), (
            f"TR-bypass fired despite pure-recall mode; got system prompt: "
            f"{system_msg['content'][:200]}"
        )

    def test_pure_recall_cr_does_not_inject_contradiction_context(
        self, temp_db, fake_llm, monkeypatch
    ):
        """CR question should NOT inject `_detect_contradictions`
        output into the prompt — pure recall means recall alone."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        # Contradictory statements that would trigger _detect_contradictions.
        msgs = [
            {"role": "user", "content": "I love coffee, it's my favorite drink"},
            {"role": "user", "content": "Actually, I prefer tea over coffee"},
        ]
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="what is my preferred drink?",
            conversation_messages=msgs,
            top_k=5,
            ability="CR",
        )
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        # The CR-inject prefix is "I notice you've mentioned contradictory…"
        # — assert it's NOT in the prompt.
        assert "contradictory information" not in user_msg, (
            f"CR-bypass injected contradiction context despite pure-recall mode; "
            f"prompt: {user_msg[:300]}"
        )

    def test_pure_recall_excludes_recent_conversation_section(
        self, temp_db, fake_llm, monkeypatch
    ):
        """The 'RECENT CONVERSATION' block (last 12 raw messages) is
        always included pre-fix. Pure-recall mode strips it so the LLM
        sees only what each arm's recall returned."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        msgs = _build_msgs(20)
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="some abstract reasoning question",
            conversation_messages=msgs,
            top_k=5,
            ability="ABS",
        )
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        assert "RECENT CONVERSATION" not in user_msg, (
            f"RECENT CONVERSATION section leaked into pure-recall prompt: "
            f"{user_msg[:300]}"
        )


class TestPureRecallEnvValueParsing:
    """The env var is treated as truthy on '1', 'true', 'yes' (lowercase
    or any case), falsy/unset otherwise. Locks the parsing surface."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES"])
    def test_truthy_values_enable_gate(self, value, beam_with_context_facts, fake_llm, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", value)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(5),
            top_k=5,
            ability="IE",
        )
        # IE bypass should NOT fire; LLM should be called.
        fake_llm.chat.assert_called()

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "anything-else"])
    def test_falsy_values_preserve_default(self, value, beam_with_context_facts, fake_llm, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", value)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(5),
            top_k=5,
            ability="IE",
        )
        # IE bypass SHOULD fire — LLM not called, value returned directly.
        assert ans == "blue"
        fake_llm.chat.assert_not_called()
