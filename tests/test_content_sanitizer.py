"""Tests for content sanitizer — binary payload extraction to blob storage."""

import base64
import os
import pytest
from pathlib import Path

from mnemosyne.core.content_sanitizer import (
    sanitize_content,
    _is_data_uri,
    _parse_data_uri,
    _shannon_entropy,
    _looks_like_base64_blob,
    _compute_sha256,
    _store_blob,
)


class TestDataUriDetection:
    def test_is_data_uri_png(self):
        assert _is_data_uri("data:image/png;base64,iVBORw0KGgo=")

    def test_is_data_uri_plain(self):
        assert _is_data_uri("data:text/plain;base64,SGVsbG8=")

    def test_is_not_data_uri(self):
        assert not _is_data_uri("Hello world")
        assert not _is_data_uri("just some text")

    def test_parse_png_uri(self):
        png_dot = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        content = f"data:image/png;base64,{png_dot}"
        result = _parse_data_uri(content)
        assert result is not None
        mime, raw = result
        assert mime == "image/png"
        assert raw == b"\x89PNG\r\n\x1a\n"

    def test_parse_no_mime(self):
        content = "data:;base64,SGVsbG8="
        result = _parse_data_uri(content)
        assert result is not None
        mime, raw = result
        assert mime == "application/octet-stream"
        assert raw == b"Hello"

    def test_parse_invalid_base64(self):
        content = "data:image/png;base64,!!!not-valid!!!"
        assert _parse_data_uri(content) is None

    def test_parse_missing_scheme(self):
        assert _parse_data_uri("just text") is None


class TestShannonEntropy:
    def test_uniform_distribution_high_entropy(self):
        # All characters equally likely → near maximum entropy
        s = "abcdefghijklmnopqrstuvwxyz0123456789+/ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 2000
        entropy = _shannon_entropy(s)
        assert entropy > 5.5  # Near log2(64) = 6.0

    def test_normal_text_low_entropy(self):
        s = "hello world this is normal english text with common letters and patterns " * 1000
        entropy = _shannon_entropy(s)
        assert entropy < 5.0  # Natural language is ~4.0-4.5

    def test_repeated_chars_very_low(self):
        s = "aaaaa" * 10000
        entropy = _shannon_entropy(s)
        assert entropy < 0.1

    def test_empty(self):
        assert _shannon_entropy("") == 0.0


class TestLooksLikeBase64Blob:
    def test_actual_base64_blob(self, tmp_path, monkeypatch):
        # Generate proper base64 of binary data
        raw = os.urandom(200_000)  # 200KB of random binary
        b64 = base64.b64encode(raw).decode()
        assert _looks_like_base64_blob(b64)

    def test_normal_english_not_blob(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5000
        assert len(text.encode("utf-8")) > 100_000
        assert not _looks_like_base64_blob(text)

    def test_code_not_blob(self):
        code = "def foo():\n    return 42\n" * 20000
        assert len(code.encode("utf-8")) > 100_000
        assert not _looks_like_base64_blob(code)


class TestBlobStorage:
    def test_store_and_hash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
        data = b"binary blob content for testing"
        sha256 = _store_blob(data)
        assert len(sha256) == 64
        blob_path = tmp_path / "blobs" / sha256[:2] / sha256[:4] / sha256
        assert blob_path.exists()
        assert blob_path.read_bytes() == data

    def test_store_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
        data = b"idempotent test"
        sha1 = _store_blob(data)
        mtime1 = (tmp_path / "blobs" / sha1[:2] / sha1[:4] / sha1).stat().st_mtime
        sha2 = _store_blob(data)
        assert sha1 == sha2
        mtime2 = (tmp_path / "blobs" / sha2[:2] / sha2[:4] / sha2).stat().st_mtime
        assert mtime1 == mtime2


class TestSanitizeContent:
    def test_normal_text_passes_through(self):
        content = "This is normal conversational text."
        result, meta = sanitize_content(content)
        assert result == content
        assert meta == {}

    def test_data_uri_extracted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
        raw = b"\x89PNG header fake binary data for test"
        b64 = base64.b64encode(raw).decode()
        content = f"data:image/png;base64,{b64}"
        result, meta = sanitize_content(content)
        assert "Binary content extracted" in result
        assert "blob://sha256/" in result
        assert meta["extraction_reason"] == "data_uri"
        assert meta["mime"] == "image/png"
        assert meta["original_size"] == len(raw)

    def test_large_content_extracted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
        content = "x" * 1_000_001
        result, meta = sanitize_content(content)
        assert "Large content extracted" in result
        assert meta["extraction_reason"] == "size_cap"

    def test_high_entropy_blob_extracted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
        # Generate high-entropy string >100KB (random-looking base64)
        raw = os.urandom(150_000)
        b64 = base64.b64encode(raw).decode()
        assert len(b64.encode("utf-8")) > 100_000
        result, meta = sanitize_content(b64)
        assert "Encoded content extracted" in result
        assert meta["extraction_reason"] == "high_entropy"
        assert meta["entropy"] > 5.0

    def test_normal_prose_not_extracted(self):
        # Build a large amount of normal English text
        text = "This is a normal paragraph of English text. It discusses various topics in a conversational tone. " * 3000
        assert len(text.encode("utf-8")) > 100_000
        result, meta = sanitize_content(text)
        assert result == text
        assert meta == {}

    def test_small_content_not_extracted(self):
        content = "Small text, under all thresholds."
        result, meta = sanitize_content(content)
        assert result == content
        assert meta == {}
