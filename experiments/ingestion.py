"""Experiment 1: does cleaning the books better before chunking help the search?

    python -m experiments.ingestion

Same questions, same embedding model (multilingual-e5-base), only the chunks change.
"""
from app.embedder import SentenceTransformerEmbedder
from app.retriever import HybridRetriever, RetrievalParams
from experiments.common import answerable_questions, build_index, load_cached_books, make_chunks, persian_in_english, score, show

IDEAS = [
    ("current ingestion", {}),
    ("+ remove repeated page headers", {"clean_headers": True}),
    ("+ join words the PDF broke", {"join_words": True}),
    ("+ book title in every chunk", {"title": True}),
    ("all three", {"clean_headers": True, "join_words": True, "title": True}),
]


def main():
    books, questions = load_cached_books(), answerable_questions()
    english_of = persian_in_english(questions)
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-base", "query: ", "passage: ")
    for label, options in IDEAS:
        chunks = make_chunks(books, **options)
        retriever = HybridRetriever(build_index(chunks, embedder), embedder, RetrievalParams(10, 3, 60, 2, 1))
        result = score(lambda text: retriever.search(text, 10), questions, english_of, join_words=options.get("join_words", False))
        show(label, len(chunks), result)


if __name__ == "__main__":
    main()
