"""Experiment 3: the dense / BM25 weights, and a reranker on top (embedding model: bge-m3).

    python -m experiments.rerank_and_weights

The reranker re-orders the 20 best chunks of the hybrid search and the best 5 go on.
"""
import time

from app.embedder import SentenceTransformerEmbedder
from app.retriever import HybridRetriever, RetrievalParams
from experiments.common import answerable_questions, build_index, load_cached_books, make_chunks, persian_in_english, score, show

WEIGHTS = [(1, 1), (2, 1), (3, 1), (1, 0)]  # dense, BM25


def main():
    from sentence_transformers import CrossEncoder

    chunks = make_chunks(load_cached_books())
    questions = answerable_questions()
    english_of = persian_in_english(questions)
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", "", "")
    index = build_index(chunks, embedder)

    for dense, bm25 in WEIGHTS:
        retriever = HybridRetriever(index, embedder, RetrievalParams(10, 3, 60, dense, bm25))
        show(f"bge-m3 weights {dense}:{bm25}", len(chunks), score(lambda text: retriever.search(text, 10), questions, english_of))

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    retriever = HybridRetriever(index, embedder, RetrievalParams(10, 3, 60, 2, 1, 20), reranker)
    started = time.time()
    result = score(lambda text: retriever.search(text, 10), questions, english_of)
    show("bge-m3 + reranker (20 -> 5)", len(chunks), result)
    print(f"the reranker needed {time.time() - started:.0f} seconds for all the questions")


if __name__ == "__main__":
    main()
