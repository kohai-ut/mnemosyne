"""Tests for identity memory auto-capture in sync_turn()."""

import pytest
import sqlite3
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestIdentitySignals:
    """Verify _capture_identity_signals detects identity-significant expressions."""

    @pytest.fixture
    def provider(self):
        """Create a MnemosyneMemoryProvider with a temp database."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "mnemosyne.db")
            provider = MnemosyneMemoryProvider()
            # Initialize with temp path
            provider._agent_context = "test"
            provider._session_id = "test-session"
            provider._skip_contexts = set()
            provider._turn_count = 0
            provider._auto_sleep_enabled = False

            from mnemosyne.core.beam import BeamMemory
            beam = BeamMemory(db_path=db_path)
            beam.initialize()
            provider._beam = beam

            yield provider

            # Cleanup
            try:
                os.remove(db_path)
                for ext in ("-wal", "-shm"):
                    try:
                        os.remove(db_path + ext)
                    except OSError:
                        pass
            except OSError:
                pass

    def test_captures_imposter_syndrome(self, provider):
        """Identity memory saved when user expresses imposter feelings."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "I feel like an imposter, I barely know my own codebase",
            "Response here",
        )
        # Should have called remember for identity signal
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) >= 1
        call = identity_calls[0].kwargs
        assert call["importance"] >= 0.8
        assert call["scope"] == "global"
        assert call["source"] == "identity"

    def test_captures_pride(self, provider):
        """Identity memory saved when user expresses pride."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "I'm proud of what we shipped today",
            "Great work!",
        )
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) >= 1

    def test_no_false_positive_on_neutral(self, provider):
        """No identity memory for neutral conversation."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "Can you run the build and deploy to production?",
            "Sure, running build now.",
        )
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) == 0

    def test_only_one_identity_per_turn(self, provider):
        """Only one identity memory saved per turn (break after first match)."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "I feel like a fraud and I don't even know how to fix this",
            "Let me help.",
        )
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) == 1

    def test_identity_stored_in_db(self, provider):
        """Identity memory actually persists to the database."""
        provider.sync_turn(
            "I feel like I barely know my own architecture",
            "Response",
        )
        # Query the database directly
        rows = provider._beam.recall("imposter feeling architecture")
        identity_rows = [r for r in rows if r.get("source") == "identity"]
        assert len(identity_rows) >= 1
        assert identity_rows[0]["importance"] >= 0.8
        assert "[IDENTITY]" in identity_rows[0]["content"]
