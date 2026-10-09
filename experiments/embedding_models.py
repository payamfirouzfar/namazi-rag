"""Experiment 2: which embedding model finds the right passage best?

    python -m experiments.embedding_models

Same chunks, same questions, hybrid search 2:1 without a reranker. The two big models are
downloaded the first time (about 2 GB each).
"""
from app.embedder import SentenceTransformerEmbedder
from app.retriever import HybridRetriever, RetrievalParams
from experiments.common import answerable_questions, build_index, load_cached_books, make_chunks, persian_in_english, score, show

MODELS = [  # label, model, query prefix, passage prefix
    ("multilingual-e5-base (first choice)", "intfloat/multilingual-e5-base", "query: ", "passage: "),
    ("multilingual-e5-large", "intfloat/multilingual-e5-large", "query: ", "passage: "),
    ("bge-m3", "BAAI/bge-m3", "", ""),
]


def main():
    chunks = make_chunks(load_cached_books())
    questions = answerable_questions()
    english_of = persian_in_english(questions)
    for label, model, query_prefix, passage_prefix in MODELS:
        embedder = SentenceTransformerEmbedder(model, query_prefix, passage_prefix)
        retriever = HybridRetriever(build_index(chunks, embedder), embedder, RetrievalParams(10, 3, 60, 2, 1))
        show(label, len(chunks), score(lambda text: retriever.search(text, 10), questions, english_of))


if __name__ == "__main__":
    main()
