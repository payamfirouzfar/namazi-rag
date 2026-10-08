"""Find good chunking and retrieval settings by testing them on YOUR questions.

    python -m scripts.tune --books data/books --eval data/eval/eval_questions.jsonl

The eval file has one JSON object per line:
    {"question": "...", "evidence": "a short phrase that must appear in the right chunk"}
    {"question": "...", "evidence": null}      <- a question the books cannot answer

Tips for writing the evidence phrase: copy it from inside ONE sentence of the book (so
chunking never cuts it in half) and keep it short and distinctive.

What it does
  1. Splits the questions 70/30 into a dev set and a test set (fixed seed).
  2. Tries every chunk size, then every retrieval setting, and scores them on the dev set
     with hit rate (is the right chunk in the top k?) and MRR (how high is it ranked?).
  3. Reports the winner once on the untouched test set, so you see a fair number.
  4. Suggests MIN_DENSE_SCORE from the questions the books cannot answer.

Every chunk size re-embeds all the books, so on a full library start with --quick.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

from app.chunking import chunk_pages
from app.config import get_settings
from app.embedder import make_embedder
from app.index import Index
from app.loader import load_books
from app.logger import setup_logging
from app.retriever import HybridRetriever, RetrievalParams
from app.text import clean_text

# The defaults come first, so on a tie the default setting wins.
CHUNK_GRID = [(250, 50), (150, 30), (400, 80)]  # (words per chunk, overlap)
WEIGHT_GRID = [(1, 1), (2, 1), (1, 2), (1, 0), (0, 1)]  # (dense weight, BM25 weight)
RRF_K_GRID = [60, 20, 100]
QUICK_CHUNK_GRID = [(250, 50)]


def norm(text: str) -> str:
    return " ".join(clean_text(text).lower().split())


def load_eval(path: str) -> list[dict]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "question" not in item:
            raise ValueError(f"{path} line {number} has no 'question'")
        items.append(item)
    return items


def split_eval(items: list[dict], seed: int, dev_fraction: float = 0.7):
    items = items[:]
    random.Random(seed).shuffle(items)
    cut = max(1, int(len(items) * dev_fraction))
    return items[:cut], items[cut:]


def evaluate(retriever: HybridRetriever, items: list[dict], top_k: int) -> dict:
    ranks, answerable_scores, unanswerable_scores = [], [], []
    for item in items:
        hits = retriever.search(item["question"], top_k)
        best_dense = max((h.dense_score for h in hits), default=0.0)
        if not item.get("evidence"):
            unanswerable_scores.append(best_dense)
            continue
        answerable_scores.append(best_dense)
        evidence = norm(item["evidence"])
        ranks.append(next((n for n, h in enumerate(hits, 1) if evidence in norm(h.chunk.text)), None))
    n = len(ranks)
    return {
        "n": n,
        "hit_rate": sum(1 for r in ranks if r) / n if n else 0.0,
        "mrr": sum(1 / r for r in ranks if r) / n if n else 0.0,
        "answerable_scores": answerable_scores,
        "unanswerable_scores": unanswerable_scores,
    }


def main() -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books", default=s.books_dir)
    parser.add_argument("--eval", default="data/eval/eval_questions.jsonl")
    parser.add_argument("--top-k", type=int, default=s.top_k)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/tuning")
    parser.add_argument("--quick", action="store_true", help="keep the chunk size, tune only retrieval")
    args = parser.parse_args()

    setup_logging("WARNING")
    books = load_books(args.books)
    items = load_eval(args.eval)
    dev, test = split_eval(items, args.seed)
    if not books or not any(i.get("evidence") for i in dev):
        print("Need at least one book and one answerable dev question.")
        return 1
    print(f"{len(books)} books, {len(dev)} dev questions, {len(test)} test questions")
    total_words = sum(len(t.split()) for pages in books.values() for _, t in pages)
    if total_words < 5000:
        print("WARNING: the corpus is tiny, so almost every chunk lands in the top k and the "
              "scores will look perfect. Tune on the real books.")

    embedder = make_embedder(s)
    rows, best = [], None  # best = (row, index)

    for size, overlap in (QUICK_CHUNK_GRID if args.quick else CHUNK_GRID):
        chunks = [c for name, pages in books.items() for c in chunk_pages(name, pages, size, overlap)]
        print(f"chunk size {size}, overlap {overlap}: {len(chunks)} chunks, building the index...")
        index = Index.build(chunks, embedder, size, overlap, s.bm25_k1, s.bm25_b)
        for w_dense, w_bm25 in WEIGHT_GRID:
            # with a single retriever there is nothing to fuse, so rrf_k does not matter
            for rrf_k in RRF_K_GRID if (w_dense and w_bm25) else RRF_K_GRID[:1]:
                params = RetrievalParams(args.top_k, s.oversample, rrf_k, w_dense, w_bm25)
                result = evaluate(HybridRetriever(index, embedder, params), dev, args.top_k)
                row = {
                    "chunk_size": size, "overlap": overlap, "weight_dense": w_dense,
                    "weight_bm25": w_bm25, "rrf_k": rrf_k, "n_chunks": len(chunks),
                    "dev_hit_rate": round(result["hit_rate"], 4), "dev_mrr": round(result["mrr"], 4),
                }
                rows.append(row)
                print(f"  tried chunk={size}/{overlap} dense:bm25={w_dense}:{w_bm25} rrf_k={rrf_k}"
                      f"  hit_rate={row['dev_hit_rate']:.3f}  MRR={row['dev_mrr']:.3f}")
                if best is None or (row["dev_mrr"], row["dev_hit_rate"]) > (best[0]["dev_mrr"], best[0]["dev_hit_rate"]):
                    best = (row, index, params, result)

    row, index, params, dev_result = best

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (-r["dev_mrr"], -r["dev_hit_rate"])))
    (out / "best.json").write_text(json.dumps(row, indent=2))

    print(f"\ntried {len(rows)} settings (full table: {out / 'results.csv'})")
    print("\nbest on the dev set:")
    print(f"  chunk_size={row['chunk_size']} overlap={row['overlap']} "
          f"weights dense:bm25={row['weight_dense']}:{row['weight_bm25']} rrf_k={row['rrf_k']}")
    print(f"  dev  hit_rate@{args.top_k}={row['dev_hit_rate']:.3f}  MRR={row['dev_mrr']:.3f}")

    test_result = evaluate(HybridRetriever(index, embedder, params), test, args.top_k)
    if test_result["n"]:
        print(f"  test hit_rate@{args.top_k}={test_result['hit_rate']:.3f}  MRR={test_result['mrr']:.3f}"
              f"  ({test_result['n']} questions - the honest number)")
    else:
        print("  no answerable test questions, add more questions to the eval file")

    # --- abstention threshold ---
    answerable, unanswerable = dev_result["answerable_scores"], dev_result["unanswerable_scores"]
    print("\nanswer / no-answer threshold:")
    if answerable and unanswerable:
        threshold = round(float(np.percentile(answerable, 5)), 3)
        rejected = sum(1 for x in unanswerable if x < threshold)
        print(f"  MIN_DENSE_SCORE={threshold} lets 95% of answerable questions through "
              f"and stops {rejected} of {len(unanswerable)} unanswerable ones")
        if rejected == 0:
            print("  that does not separate them on this data: keep MIN_DENSE_SCORE=0 "
                  "and rely on the prompt rule (NO_ANSWER)")
    else:
        print("  add both answerable and unanswerable questions to calibrate it")

    print("\nput this in your .env:")
    print(f"CHUNK_SIZE={row['chunk_size']}\nCHUNK_OVERLAP={row['overlap']}\nRRF_K={row['rrf_k']}")
    print(f"WEIGHT_DENSE={row['weight_dense']}\nWEIGHT_BM25={row['weight_bm25']}")
    print("then rebuild the index: python -m scripts.ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
