"""Read books from a folder. PDF, TXT and MD files are supported."""
import logging
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".txt", ".md"}


def book_name(path: Path) -> str:
    return path.stem.replace("_", " ").strip()


def read_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for one file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((number, text))
        # a scanned book has pictures of pages and no text layer
        if len(reader.pages) > 5 and len(pages) < len(reader.pages) * 0.2:
            log.warning("%s looks scanned (little text found). Run OCR on it first.", path.name)
        return pages
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(1, text)] if text.strip() else []


def load_books(folder: str) -> dict[str, list[tuple[int, str]]]:
    """Load every supported file. A broken file is logged and skipped."""
    books = {}
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        try:
            pages = read_pages(path)
        except Exception:
            log.exception("could not read %s, skipping it", path)
            continue
        if not pages:
            log.warning("%s has no text, skipping it", path.name)
            continue
        books[book_name(path)] = pages
        log.info("loaded %s (%d pages)", path.name, len(pages))
    return books
