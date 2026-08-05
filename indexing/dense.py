from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

_EMBEDDER_CACHE = {}


class HashingEmbedder:
    """Deterministic offline embedder used only when no model path is configured."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = int(dimension)

    def eval(self):
        return self

    def encode(self, texts, **kwargs):
        del kwargs
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        matrix = np.zeros((len(values), self.dimension), dtype=np.float32)
        for row, text in enumerate(values):
            for token in re.findall(r"[A-Za-z0-9']+", str(text).casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix[0] if single else matrix


def build_dense_embeddings(texts: Iterable[str], config: dict):
    retriever = config.get("retriever", {}).get("kwargs", {})
    model_name = retriever.get("embedder")
    if model_name:
        from sentence_transformers import SentenceTransformer

        cache_key = (str(model_name), str(retriever.get("device", "cuda")))
        embedder = _EMBEDDER_CACHE.get(cache_key)
        if embedder is None:
            embedder = SentenceTransformer(model_name, device=cache_key[1])
            embedder.eval()
            _EMBEDDER_CACHE[cache_key] = embedder
        matrix = embedder.encode(list(texts), batch_size=16, device=retriever.get("device", "cuda"))
        return np.asarray(matrix, dtype=np.float32), {
            "backend": "sentence_transformer",
            "model": str(model_name),
            "dimension": int(matrix.shape[1]),
        }
    dimension = int(config.get("indexing", {}).get("hash_embedding_dimension", 256))
    embedder = HashingEmbedder(dimension)
    matrix = embedder.encode(list(texts))
    return matrix, {"backend": "hashing", "model": None, "dimension": dimension}
