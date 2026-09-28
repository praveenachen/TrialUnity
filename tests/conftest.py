import hashlib
import re

import numpy as np
import pytest


class DeterministicEmbeddingModel:
    """Small local test double; production still uses SentenceTransformer."""

    def encode(self, texts, normalize_embeddings):
        vectors = []
        for text in texts:
            vector = np.zeros(128, dtype=float)
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:2], "big") % len(vector)] += 1.0
            norm = np.linalg.norm(vector)
            if normalize_embeddings and norm:
                vector /= norm
            vectors.append(vector)
        return np.array(vectors)


@pytest.fixture(autouse=True)
def local_embedding_model(monkeypatch, request):
    if request.node.get_closest_marker("model_cache"):
        return
    from backend.app.services.retrieval import dense

    monkeypatch.setattr(dense, "_get_model", lambda: DeterministicEmbeddingModel())
