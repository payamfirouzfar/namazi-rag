import hashlib
import logging

import numpy as np

from app.config import Settings
from app.text import tokenize

log = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Real embedding model. Vectors are normalized, so dot product = cosine similarity."""

    def __init__(self, model_name: str, query_prefix="", passage_prefix="", batch_size=32):
        from sentence_transformers import SentenceTransformer  # slow import, so do it here

        log.info("loading embedding model %s", model_name)
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        texts = [self.passage_prefix + t for t in texts]
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # no progress bar: with many questions at once the bars crash each other
        vector = self.model.encode([self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vector, dtype=np.float32)


class HashEmbedder:
    """Tiny word-hashing embedder. Needs no download, so tests and CI can use it.
    Never use it for real answers - it only knows which words overlap."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 20) % 2 == 0 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def make_reranker(settings: Settings):
    """A second model that puts the best chunks in a better order. None means off."""
    if not settings.rerank_model:
        return None
    from sentence_transformers import CrossEncoder  # slow import, so do it here

    return CrossEncoder(settings.rerank_model, max_length=512)


def make_embedder(settings: Settings):
    if settings.embedding_model == "hash":
        log.warning("using the hash embedder - fine for tests, not for real use")
        return HashEmbedder()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.query_prefix,
        settings.passage_prefix,
        settings.embedding_batch_size,
    )
