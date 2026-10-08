import re
from dataclasses import dataclass

from app.text import clean_text, split_sentences


@dataclass
class Chunk:
    id: str
    book: str
    page: int
    text: str


def slugify(name: str) -> str:
    return re.sub(r"[^\w]+", "-", name.lower(), flags=re.UNICODE).strip("-") or "book"


def chunk_pages(
    book: str,
    pages: list[tuple[int, str]],
    chunk_size: int = 250,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a book into chunks of about `chunk_size` words.

    Chunks always end on a sentence boundary. The last sentences of a chunk are
    repeated at the start of the next one (up to `overlap` words) so a fact that
    sits on the border is not cut in half. Each chunk remembers the page where it starts.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # list of (page, words) - one entry per sentence
    units: list[tuple[int, list[str]]] = []
    for page_no, text in pages:
        for sentence in split_sentences(clean_text(text)):
            words = sentence.split()
            # a "sentence" longer than a whole chunk (tables, bad OCR) gets cut by words
            for i in range(0, len(words), chunk_size):
                units.append((page_no, words[i : i + chunk_size]))

    slug = slugify(book)
    chunks: list[Chunk] = []

    def emit(current):
        chunks.append(
            Chunk(
                id=f"{slug}_{len(chunks)}",
                book=book,
                page=current[0][0],
                text=" ".join(" ".join(w) for _, w in current),
            )
        )

    current: list[tuple[int, list[str]]] = []
    count = 0
    for page, words in units:
        if current and count + len(words) > chunk_size:
            emit(current)
            current = _tail(current, overlap)
            count = sum(len(w) for _, w in current)
            if count + len(words) > chunk_size:  # overlap leaves no room, drop it
                current, count = [], 0
        current.append((page, words))
        count += len(words)

    if current:
        emit(current)
    return chunks


def _tail(units, overlap):
    """Last sentences of a chunk that fit inside the overlap budget."""
    tail, total = [], 0
    for unit in reversed(units):
        if total + len(unit[1]) > overlap:
            break
        tail.insert(0, unit)
        total += len(unit[1])
    return tail
