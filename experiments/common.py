"""Small helpers that the experiments share.

Reading the PDFs and translating the Persian questions are slow, so both are saved in
data/tuning/experiment_cache/ and every experiment reuses them.
"""
import collections
import json
import pickle
import re
from pathlib import Path

from app.chunking import chunk_pages
from app.config import get_settings
from app.index import Index
from app.loader import load_books
from app.rag import has_persian
from app.text import clean_text
from scripts.tune import load_eval, norm

CACHE = Path("data/tuning/experiment_cache")

# Pieces of words that the PDF text breaks in two: "bene fits" should be "benefits".
BROKEN_PIECES = "bene|de|ef|signi|identi|con|modi|insuf|speci|classi|suf|calci|dif|immunode|justi|veri|quali|ampli|pro|re|arti"


def load_cached_books() -> dict:
    """The pages of every book, read from the PDFs only the first time."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "books.pkl"
    if not path.exists():
        path.write_bytes(pickle.dumps(load_books("data/books")))
    return pickle.loads(path.read_bytes())


def answerable_questions() -> list[dict]:
    return [q for q in load_eval("data/eval/eval_questions.jsonl") if q.get("evidence")]


def persian_in_english(questions: list[dict]) -> dict:
    """The English version of every Persian question, made by the LLM once and then saved."""
    path = CACHE / "persian_in_english.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from app.llm import OpenAICompatibleLLM
    from app.rag import RagService

    rag = RagService(None, OpenAICompatibleLLM(get_settings()))
    english = {q["question"]: rag.english_for_search(q["question"]) for q in questions if has_persian(q["question"])}
    path.write_text(json.dumps(english, ensure_ascii=False), encoding="utf-8")
    return english


# ---- ideas for the ingestion ----

def join_broken_words(text: str) -> str:
    return re.sub(rf"\b({BROKEN_PIECES}) (?=(?:fi|fl|ff|ffi|ffl)[a-z])", r"\1", text)


def strip_running_lines(pages: list, share: float = 0.25) -> list:
    """Drop short lines that repeat on many pages of a book (page headers and footers)."""
    if len(pages) <= 8:
        return pages
    count = collections.Counter(line.strip() for _, text in pages for line in set(text.split("\n")) if 0 < len(line.split()) <= 12)
    repeated = {line for line, n in count.items() if n >= share * len(pages)}
    return [(number, "\n".join(line for line in text.split("\n") if line.strip() not in repeated)) for number, text in pages]


def make_chunks(books: dict, size: int = 250, overlap: int = 50, clean_headers=False, join_words=False, title=False) -> list:
    chunks = []
    for name, pages in books.items():
        if clean_headers:
            pages = strip_running_lines(pages)
        if join_words:
            pages = [(number, join_broken_words(clean_text(text))) for number, text in pages]
        for chunk in chunk_pages(name, pages, size, overlap):
            if title:
                chunk.text = f"{name}. {chunk.text}"
            chunks.append(chunk)
    return chunks


def build_index(chunks: list, embedder, size: int = 250, overlap: int = 50) -> Index:
    s = get_settings()
    return Index.build(chunks, embedder, size, overlap, s.bm25_k1, s.bm25_b)


# ---- scoring ----

def score(search, questions: list[dict], english_of: dict, join_words=False) -> dict:
    """Hit rate@5 and MRR for English, Persian as typed, and Persian translated to English.
    `search` takes a question text and returns the hits, best first."""
    ranks = collections.defaultdict(list)
    for item in questions:
        evidence = norm(item["evidence"])
        if join_words:
            evidence = join_broken_words(evidence)  # the questions were written before the words were joined
        question = item["question"]
        if has_persian(question):
            versions = [("Persian typed", question), ("Persian translated", english_of[question])]
        else:
            versions = [("English", question)]
        for group, text in versions:
            hits = search(text)[:10]
            ranks[group].append(next((n for n, hit in enumerate(hits, 1) if evidence in norm(hit.chunk.text)), None))
    return {
        group: (sum(1 for r in found if r and r <= 5) / len(found), sum(1 / r for r in found if r) / len(found))
        for group, found in ranks.items()
    }


def show(label: str, chunks: int, result: dict) -> None:
    columns = " | ".join(f"{group}: hit@5 {hit:.3f} MRR {mrr:.3f}" for group, (hit, mrr) in result.items())
    print(f"{label:34} {chunks:6} chunks | {columns}", flush=True)
