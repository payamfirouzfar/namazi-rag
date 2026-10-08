"""The search index: chunks + embeddings on disk, BM25 rebuilt in memory at startup."""
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking import Chunk
from app.text import tokenize

log = logging.getLogger(__name__)


class IndexLoadError(Exception):
    pass


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict, k1=1.2, b=0.75):
        if len(chunks) != len(embeddings):
            raise IndexLoadError("chunks and embeddings do not match, rebuild the index")
        self.chunks = chunks
        self.embeddings = embeddings
        self.meta = meta
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def __len__(self):
        return len(self.chunks)

    def dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        return self.embeddings @ query_vector

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    # ---- build / save / load -------------------------------------------------

    @classmethod
    def build(cls, chunks, embedder, chunk_size, overlap, k1=1.2, b=0.75):
        embeddings = embedder.embed_passages([c.text for c in chunks])
        meta = {
            "embedding_model": embedder.name,
            "dim": int(embeddings.shape[1]) if len(embeddings) else embedder.dim,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "n_chunks": len(chunks),
            "books": sorted({c.book for c in chunks}),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return cls(chunks, embeddings, meta, k1, b)

    def save(self, folder: str) -> None:
        """Write to a temporary folder first so a crash never leaves a half-written index."""
        target = Path(folder)
        tmp = target.with_name(target.name + ".new")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        np.save(tmp / "embeddings.npy", self.embeddings)
        (tmp / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))
        # move the old index aside first, so a crash never leaves us with no index at all
        old = target.with_name(target.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            target.rename(old)
        tmp.rename(target)
        shutil.rmtree(old, ignore_errors=True)
        log.info("saved index with %d chunks to %s", len(self.chunks), target)

    @classmethod
    def load(cls, folder: str, embedding_model: str, k1=1.2, b=0.75) -> "Index":
        path = Path(folder)
        if not (path / "meta.json").exists():
            raise IndexLoadError(f"no index found in {folder}. Run: python -m scripts.ingest")
        meta = json.loads((path / "meta.json").read_text())
        # vectors from different models cannot be compared, so refuse to start
        if meta["embedding_model"] != embedding_model:
            raise IndexLoadError(
                f"index was built with '{meta['embedding_model']}' but EMBEDDING_MODEL is "
                f"'{embedding_model}'. Rebuild the index or fix the setting."
            )
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        embeddings = np.load(path / "embeddings.npy")
        return cls(chunks, embeddings, meta, k1, b)
