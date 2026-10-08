"""Hybrid retrieval: BM25 (exact words) + dense vectors (meaning), merged with
Reciprocal Rank Fusion. RRF only uses ranks, so the two score scales never need matching."""
from dataclasses import dataclass

import numpy as np

from app.chunking import Chunk
from app.config import Settings
from app.index import Index


@dataclass
class RetrievalParams:
    top_k: int = 5
    oversample: int = 3
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_bm25: float = 1.0
    rerank_pool: int = 20

    @classmethod
    def from_settings(cls, s: Settings) -> "RetrievalParams":
        return cls(s.top_k, s.oversample, s.rrf_k, s.weight_dense, s.weight_bm25, s.rerank_pool)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    dense_score: float  # cosine similarity with the question


def rrf_fuse(rankings: list[list[int]], weights: list[float], k: int = 60) -> list[tuple[int, float]]:
    """Each ranking is a list of chunk numbers, best first. Returns (chunk, score), best first."""
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_no in enumerate(ranking, start=1):
            scores[chunk_no] = scores.get(chunk_no, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _top_n(scores: np.ndarray, n: int) -> list[int]:
    n = min(n, len(scores))
    if n == 0:
        return []
    best = np.argpartition(-scores, n - 1)[:n]
    return best[np.argsort(-scores[best])].tolist()


class HybridRetriever:
    def __init__(self, index: Index, embedder, params: RetrievalParams, reranker=None):
        self.index = index
        self.embedder = embedder
        self.params = params
        self.reranker = reranker  # a CrossEncoder, or None to keep the merged order

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        p = self.params
        top_k = top_k or p.top_k
        pool = top_k * p.oversample

        query_vector = self.embedder.embed_query(query)
        dense = self.index.dense_scores(query_vector)
        lexical = self.index.bm25_scores(query)

        rankings, weights = [], []
        if p.weight_dense > 0:
            rankings.append(_top_n(dense, pool))
            weights.append(p.weight_dense)
        if p.weight_bm25 > 0:
            # chunks sharing no word with the question are not a real BM25 match
            ranked = [i for i in _top_n(lexical, pool) if lexical[i] > 0]
            rankings.append(ranked)
            weights.append(p.weight_bm25)

        fused = rrf_fuse(rankings, weights, p.rrf_k)[: p.rerank_pool if self.reranker else top_k]
        hits = [Hit(self.index.chunks[i], score, float(dense[i])) for i, score in fused]
        return self.rerank(query, hits)[:top_k] if self.reranker else hits

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        scores = self.reranker.predict([(query, h.chunk.text) for h in hits], batch_size=16)
        return [h for _, h in sorted(zip(scores, hits), key=lambda pair: -pair[0])]
