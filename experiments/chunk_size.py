"""Experiment 4: how big should a chunk be? (bge-m3, with and without the reranker)

    python -m experiments.chunk_size

The tuner (scripts/tune.py) has no reranker. This checks the sizes the way the app really searches.
"""
from app.embedder import SentenceTransformerEmbedder
from app.retriever import HybridRetriever, RetrievalParams
from experiments.common import answerable_questions, build_index, load_cached_books, make_chunks, persian_in_english, score, show

SIZES = [(150, 30), (200, 40), (250, 50)]  # words, overlap


def main():
    from sentence_transformers import CrossEncoder

    books, questions = load_cached_books(), answerable_questions()
    english_of = persian_in_english(questions)
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", "", "")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    for size, overlap in SIZES:
        chunks = make_chunks(books, size, overlap)
        index = build_index(chunks, embedder, size, overlap)
        for label, tool in (("no reranker", None), ("+ reranker", reranker)):
            retriever = HybridRetriever(index, embedder, RetrievalParams(10, 3, 60, 2, 1, 20), tool)
            show(f"{size}/{overlap} {label}", len(chunks), score(lambda text: retriever.search(text, 10), questions, english_of))


if __name__ == "__main__":
    main()
