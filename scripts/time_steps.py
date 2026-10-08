"""Find out where the time goes when the app answers a question.

    python -m scripts.time_steps --questions 30

Every step is timed on its own, for English and for Persian questions, and saved to docs/time_steps.csv.
(The step times of the real app are the same, this script only calls the same parts one after the other.)
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

from app.config import get_settings
from app.main import build_rag
from app.rag import SYSTEM_PROMPT, build_context, has_persian, user_message
from scripts.tune import load_eval


def lap(steps: dict, name: str, started: float) -> float:
    """Save how long a step took (in milliseconds) and return the time for the next step."""
    steps[name] = (time.perf_counter() - started) * 1000
    return time.perf_counter()


def ollama_split(llm, system: str, user: str) -> dict:
    """Ollama tells how long it took to read the prompt and to write the answer. Other servers do not, so we skip."""
    base = getattr(getattr(llm, "client", None), "base_url", None)
    if base is None:
        return {}
    body = {"model": llm.model, "stream": False, "options": {"temperature": 0, "num_predict": llm.max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    request = urllib.request.Request(str(base).split("/v1")[0] + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        answer = json.load(urllib.request.urlopen(request, timeout=300))
    except OSError:
        return {}
    return {"LLM: load model": answer["load_duration"] / 1e6,
            "LLM: read the sources": answer["prompt_eval_duration"] / 1e6,
            "LLM: write the answer": answer["eval_duration"] / 1e6}


def time_steps(rag, question: str) -> dict:
    """Run one question and return the milliseconds of every step."""
    steps, retriever = {}, rag.retriever
    t = time.perf_counter()
    english = rag.english_for_search(question)  # only a Persian question needs this LLM call
    t = lap(steps, "translate", t)
    vector = retriever.embedder.embed_query(english)
    t = lap(steps, "embed question", t)
    retriever.index.dense_scores(vector)
    t = lap(steps, "dense search", t)
    retriever.index.bm25_scores(english)
    t = lap(steps, "BM25 search", t)
    hits = retriever.search(english)  # the whole search again, what is left is the merge and the reranker
    t = lap(steps, "merge + rerank", t)
    steps["merge + rerank"] = max(0.0, steps["merge + rerank"] - steps["embed question"] - steps["dense search"] - steps["BM25 search"])
    context, _ = build_context(hits, rag.max_context_chars)
    user = user_message(context, question)
    t = lap(steps, "build prompt", t)
    split = ollama_split(rag.llm, SYSTEM_PROMPT, user)
    if split:
        steps.update(split)
    else:
        rag.llm.complete(SYSTEM_PROMPT, user)
        lap(steps, "LLM answer", t)
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=int, default=30, help="how many English and how many Persian questions")
    parser.add_argument("--eval", default="data/eval/eval_questions.jsonl")
    parser.add_argument("--out", default="docs/time_steps.csv")
    args = parser.parse_args()

    rag = build_rag(get_settings())
    questions = [i["question"] for i in load_eval(args.eval) if i.get("evidence")]
    english = [q for q in questions if not has_persian(q)][:args.questions]
    persian = [q for q in questions if has_persian(q)][:args.questions]
    time_steps(rag, english[0])  # the first question also loads models, so it does not count

    rows = []
    for language, group in (("English", english), ("Persian", persian)):
        for question in group:
            rows.append({"language": language, **time_steps(rag, question)})
    names = list(rows[0])
    for row in rows:
        row.update({name: round(row.get(name, 0.0), 1) for name in names[1:]})
    with open(Path(args.out), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)

    for language in ("English", "Persian"):
        part = [r for r in rows if r["language"] == language]
        print(f"\n{language}, median milliseconds of {len(part)} questions")
        for name in names[1:]:
            values = sorted(r[name] for r in part)
            print(f"  {name:24} {values[len(values) // 2]:8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
