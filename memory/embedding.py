from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import re

import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class CachingEmbeddingModel(EmbeddingModel):
    """Case-local cache backed by one reusable model instance.

    Missing texts are encoded in one batch. Scope this cache to one benchmark
    case so model weights are reused without retaining a full dataset's source
    text and vectors in memory.
    """

    def __init__(self, backend: EmbeddingModel):
        self.backend = backend
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            vectors = self.backend.encode(missing)
            if len(vectors) != len(missing):
                raise ValueError("embedding backend returned an unexpected row count")
            for text, vector in zip(missing, vectors):
                self._cache[text] = np.asarray(vector, dtype=np.float32)
        return np.stack([self._cache[text] for text in texts]).astype(np.float32, copy=False)


class HashEmbeddingModel(EmbeddingModel):
    """Dependency-light dense hashing embedder for development and CI.

    This is not the intended final competition embedding model. It gives a
    deterministic dense vector so the B2 retrieval path is executable offline.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # English/alnum words + Chinese character bigrams.
        ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        zh_bigrams = ["".join(chars[i:i+2]) for i in range(max(0, len(chars) - 1))]
        return ascii_tokens + chars + zh_bigrams

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                idx = value % self.dim
                sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
                vectors[row, idx] += sign
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


class SentenceTransformerEmbedder(EmbeddingModel):
    """Optional production-like embedding backend.

    Install manually: pip install sentence-transformers
    Example model: BAAI/bge-small-zh-v1.5 or another ARM64-compatible model.
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
