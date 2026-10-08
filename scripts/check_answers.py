"""Run the evaluation questions through the whole pipeline with the real LLM.

    python -m scripts.check_answers

tune.py only checks the search. This checks the answers too:
  - for a question the books can answer: did the answer cite the passage that holds the evidence?
  - for a question the books cannot answer: did the system say it could not find it?
The answers are saved to a csv file so a doctor can read and grade them.
"""
import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.embedder import make_embedder, make_reranker
from app.index import Index
from app.llm import LLMError, OpenAICompatibleLLM
from app.logger import setup_logging
from app.rag import RagService, build_context, has_persian, user_message
from app.retriever import HybridRetriever, RetrievalParams
from scripts.tune import load_eval, norm

log = logging.getLogger("check_answers")

JUDGE_PROMPT = ("You check an answer against its sources. Reply SUPPORTED if every statement of the answer "
                "is backed by the sources, otherwise reply NOT_SUPPORTED. Reply with one word only.")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def holds_evidence(item: dict, hit) -> bool:
    """Is this the right passage? It has the evidence phrase, or it comes from one of the books the question expects."""
    if item.get("evidence"):
        return norm(item["evidence"]) in norm(hit.chunk.text)
    return hit.chunk.book in item.get("books", [])


def numbers_in(text: str) -> set[str]:
    text = re.sub(r"\[\d+\]", " ", text.translate(PERSIAN_DIGITS))  # the [1] citations are not facts
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def made_up_numbers(answer: str, sources) -> int:
    """How many numbers of the answer do not appear in any source. A wrong dose or lab value is the worst mistake."""
    source_numbers = numbers_in(" ".join(h.chunk.text for h in sources))
    answer = re.sub(r"(?m)^\s*\d+[.)]\s", " ", answer)  # the 1. 2. 3. of a list are not facts either
    return len(numbers_in(answer) - source_numbers)


def judge_supported(llm, question: str, answer: str, sources, max_chars: int) -> bool:
    context, _ = build_context(sources, max_chars)
    reply = llm.complete(JUDGE_PROMPT, user_message(context, question) + f"\n\nANSWER:\n{answer}")
    return reply.text.strip().upper().startswith("SUPPORTED")


def main() -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", default="data/eval/eval_questions.jsonl")
    parser.add_argument("--out", default="data/tuning/answer_check.csv")
    parser.add_argument("--judge", action="store_true", help="also ask the LLM if the answers are backed by the sources (slower)")
    args = parser.parse_args()

    setup_logging("WARNING")
    index = Index.load(s.index_dir, s.embedding_model, s.bm25_k1, s.bm25_b)
    retriever = HybridRetriever(index, make_embedder(s), RetrievalParams.from_settings(s), make_reranker(s))
    llm = OpenAICompatibleLLM(s)
    rag = RagService(retriever, llm, s.max_context_chars, s.min_dense_score)
    items = load_eval(args.eval)
    print(f"{len(items)} questions, LLM = {s.llm_model}\n")

    rows = []
    for number, item in enumerate(items, start=1):
        started = time.time()
        try:
            result = rag.ask(item["question"])
        except LLMError as error:
            print(f"{number:2}. LLM failed: {error}")
            rows.append({"question": item["question"], "problem": "llm failed"})
            continue
        seconds = time.time() - started

        known = bool(item.get("evidence") or item.get("books"))  # does the books hold an answer?
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", result.answer)]
        cited_hits = [result.sources[n - 1] for n in cited if 1 <= n <= len(result.sources)]
        row = {
            "question": item["question"],
            "answerable": known,
            "answered": result.answered,
            "has_citation": bool(cited_hits),
            "evidence_in_sources": known and any(holds_evidence(item, h) for h in result.sources),
            "evidence_cited": known and any(holds_evidence(item, h) for h in cited_hits),
            "made_up_numbers": made_up_numbers(result.answer, result.sources) if result.answered else 0,
            "bad_citation": any(not 1 <= n <= len(result.sources) for n in cited),
            "judge_supported": judge_supported(llm, item["question"], result.answer, result.sources, s.max_context_chars)
            if args.judge and result.answered else "",
            "seconds": round(seconds, 1),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "answer": result.answer,
        }
        rows.append(row)

        if known:
            verdict = "OK " if row["evidence_cited"] else "MISS"
        else:
            verdict = "OK " if not result.answered else "BAD"  # it should have said "not found"
        print(f"{number:2}. {verdict} {seconds:5.1f}s  {item['question'][:70]}")

    done = [r for r in rows if "answerable" in r]
    print_summary("all questions", done)
    print_summary("English questions", [r for r in done if not has_persian(r["question"])])
    print_summary("Persian questions", [r for r in done if has_persian(r["question"])])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(done[0]) if done else ["question"])
        writer.writeheader()
        writer.writerows(done)
    print(f"\nanswers saved to {out} (open it and grade the answers)")
    return 0


def print_summary(title: str, done: list[dict]) -> None:
    answerable = [r for r in done if r["answerable"]]
    unanswerable = [r for r in done if not r["answerable"]]
    print(f"\n--- summary: {title} ---")
    if answerable:
        print(f"answerable questions        : {len(answerable)}")
        print(f"  gave an answer            : {sum(r['answered'] for r in answerable)}")
        print(f"  answer has a [n] citation : {sum(r['has_citation'] for r in answerable)}")
        print(f"  right passage was in the sources : {sum(r['evidence_in_sources'] for r in answerable)}")
        print(f"  right passage was cited   : {sum(r['evidence_cited'] for r in answerable)}")
    if unanswerable:
        print(f"unanswerable questions      : {len(unanswerable)}")
        print(f"  correctly said not found  : {sum(not r['answered'] for r in unanswerable)}")
    given = [r for r in done if r["answered"]]
    if given:
        print(f"answers with a number that is in no source : {sum(r['made_up_numbers'] > 0 for r in given)} of {len(given)}")
        print(f"answers citing a source that does not exist: {sum(r['bad_citation'] for r in given)} of {len(given)}")
        judged = [r for r in given if r["judge_supported"] != ""]
        if judged:
            print(f"answers the judge found NOT backed by the sources: {sum(r['judge_supported'] is False for r in judged)} of {len(judged)}")
    if done:
        print(f"average time per question   : {sum(r['seconds'] for r in done) / len(done):.1f} s")
        print(f"tokens in all {len(done)} questions    : {sum(r['prompt_tokens'] for r in done)} prompt, "
              f"{sum(r['completion_tokens'] for r in done)} answer")


if __name__ == "__main__":
    sys.exit(main())
