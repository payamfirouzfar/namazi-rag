"""Put answer-check runs side by side (different prompts, different LLMs, before and after a change).

    python -m experiments.compare_runs base=data/tuning/prompt_base.csv short=data/tuning/prompt_short.csv

Each file is the csv that scripts/check_answers.py writes.
"""
import sys

import pandas as pd

from app.rag import has_persian


def numbers(path: str, language: str) -> dict | None:
    """The numbers of one run for one language, or None if the file has no question in that language."""
    runs = pd.read_csv(path)
    runs["answered"] = runs["answered"] & ~runs["answer"].str.contains("NO_ANSWER", case=False, na=False)
    runs["persian"] = runs["question"].map(has_persian)
    if language != "All":
        runs = runs[runs["persian"] == (language == "Persian")]
    if runs.empty:
        return None
    good, bad, given = runs[runs["answerable"]], runs[~runs["answerable"]], runs[runs["answered"]]
    result = {
        "answerable questions": len(good),
        "...got an answer": int(good["answered"].sum()),
        "...right passage in the sources": int(good["evidence_in_sources"].sum()),
        "...right passage cited": int(good["evidence_cited"].sum()),
        "unanswerable questions": len(bad),
        "...still got an answer": int(bad["answered"].sum()),
        "answers with a number in no source": int((given["made_up_numbers"] > 0).sum()),
        "median answer tokens": int(given["completion_tokens"].median()),
        "median seconds": round(runs["seconds"].median(), 1),
    }
    if language != "English":  # a model that ignores the language of the question
        result["Persian questions answered in English"] = int((~given[given["persian"]]["answer"].map(has_persian)).sum())
    return result


def main():
    runs = [argument.split("=", 1) for argument in sys.argv[1:]]
    for language in ("All", "English", "Persian"):
        columns = {name: numbers(path, language) for name, path in runs}
        columns = {name: column for name, column in columns.items() if column}
        if columns:
            print(f"\n{language} questions")
            print(pd.DataFrame(columns).to_string())


if __name__ == "__main__":
    main()
