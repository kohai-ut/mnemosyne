"""
Mnemosyne Dense Retrieval
Local embedding-based memory retrieval using fastembed (ONNX, no PyTorch).
Falls back to keyword-only if fastembed is unavailable.
"""

import json
import numpy as np
from typing import List, Optional
from functools import lru_cache

# Optional dependency
try:
    from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
except Exception:
    _FASTEMBED_AVAILABLE = False
    TextEmbedding = None

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_embedding_model = None


def _get_model() -> Optional[TextEmbedding]:
    """Lazy-load the embedding model."""
    global _embedding_model
    if not _FASTEMBED_AVAILABLE:
        return None
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=_DEFAULT_MODEL)
    return _embedding_model


def available() -> bool:
    """Check if dense retrieval is available."""
    return _FASTEMBED_AVAILABLE and _get_model() is not None


@lru_cache(maxsize=512)
def embed_query(text: str) -> Optional[np.ndarray]:
    """
    Encode a single query text into a dense vector with LRU caching.
    Repeated queries (very common in agent loops) are near-instant.
    """
    model = _get_model()
    if model is None or not text:
        return None
    vectors = list(model.embed([text]))
    if not vectors:
        return None
    return vectors[0].astype(np.float32)


def embed(texts: List[str]) -> Optional[np.ndarray]:
    """
    Encode texts into dense vectors.

    Args:
        texts: List of strings to encode

    Returns:
        Numpy array of shape (n_texts, embedding_dim) or None if unavailable
    """
    if not texts:
        return None
    # Use cached single-query path for common case of 1 text
    if len(texts) == 1:
        v = embed_query(texts[0])
        if v is None:
            return None
        return np.stack([v])
    model = _get_model()
    if model is None:
        return None
    vectors = list(model.embed(texts))
    return np.stack(vectors).astype(np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and documents.

    Args:
        query_vec: shape (dim,)
        doc_vecs: shape (n_docs, dim)

    Returns:
        similarities: shape (n_docs,)
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    docs_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return docs_norm @ query_norm


def serialize(vec: np.ndarray) -> str:
    """Serialize embedding to JSON string."""
    return json.dumps(vec.tolist())


def deserialize(text: str) -> Optional[np.ndarray]:
    """Deserialize embedding from JSON string."""
    if not text:
        return None
    return np.array(json.loads(text), dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding compression / quantization
# ---------------------------------------------------------------------------

def quantize_to_int8(vec: np.ndarray) -> np.ndarray:
    """
    Quantize float32 embedding to int8.
    4x memory reduction: 384 floats -> 384 bytes.
    """
    vec = vec.astype(np.float32)
    return np.round(vec * 127.0).astype(np.int8)


def dequantize_int8(vec: np.ndarray) -> np.ndarray:
    """Dequantize int8 embedding back to float32."""
    vec = vec.astype(np.float32) / 127.0
    # Re-normalize to preserve cosine behavior
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def quantize_to_bit(vec: np.ndarray) -> np.ndarray:
    """
    Binarize embedding to 1-bit (0 or 1).
    32x memory reduction: 384 floats -> 48 bytes.
    sqlite-vec stores bit embeddings as arrays of 0/1 ints.
    """
    return (vec >= 0).astype(np.uint8)


def dequantize_bit(vec: np.ndarray) -> np.ndarray:
    """Dequantize 1-bit embedding back to float32 (-1 or +1)."""
    return vec.astype(np.float32) * 2.0 - 1.0


def quantize(vec: np.ndarray, vec_type: str) -> np.ndarray:
    """Quantize vector according to target type."""
    if vec_type == "int8":
        return quantize_to_int8(vec)
    if vec_type == "bit":
        return quantize_to_bit(vec)
    return vec.astype(np.float32)


def dequantize(vec: np.ndarray, vec_type: str) -> np.ndarray:
    """Dequantize vector according to source type."""
    if vec_type == "int8":
        return dequantize_int8(vec)
    if vec_type == "bit":
        return dequantize_bit(vec)
    return vec.astype(np.float32)
