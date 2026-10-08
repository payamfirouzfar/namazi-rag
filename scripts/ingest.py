"""Read the books, cut them into chunks, embed them and save the index.

    python -m scripts.ingest
    python -m scripts.ingest --books data/sample --chunk-size 200 --overlap 40
"""
import argparse
import logging
import sys
import time

from app.chunking import chunk_pages
from app.config import get_settings
from app.embedder import make_embedder
from app.index import Index
from app.loader import load_books
from app.logger import setup_logging

log = logging.getLogger("ingest")


def main() -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books", default=s.books_dir, help="folder with PDF/TXT/MD books")
    parser.add_argument("--index", default=s.index_dir, help="where to write the index")
    parser.add_argument("--chunk-size", type=int, default=s.chunk_size)
    parser.add_argument("--overlap", type=int, default=s.chunk_overlap)
    args = parser.parse_args()

    setup_logging(s.log_level)
    started = time.time()
    books = load_books(args.books)
    if not books:
        log.error("no readable books found in %s", args.books)
        return 1

    chunks = []
    for name, pages in books.items():
        book_chunks = chunk_pages(name, pages, args.chunk_size, args.overlap)
        log.info("%s -> %d chunks", name, len(book_chunks))
        chunks.extend(book_chunks)

    log.info("settings: chunk_size=%d overlap=%d embedding_model=%s", args.chunk_size, args.overlap, s.embedding_model)
    embedder = make_embedder(s)
    index = Index.build(chunks, embedder, args.chunk_size, args.overlap, s.bm25_k1, s.bm25_b)
    index.save(args.index)
    log.info("done: %d chunks from %d books in %d seconds", len(chunks), len(books), time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
