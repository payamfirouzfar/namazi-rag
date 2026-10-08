"""Make a sheet for doctors to read: some doubtful answers and some random ones.

    python -m scripts.review_sample

It reads the patient answer checks in docs/ and writes docs/doctor_review.csv. A doctor fills in the three empty
columns. The last column shows what the LLM judge thought, so doctors should not look at it before they decide.
"""
import csv
import json
import random
import sys

DOUBTFUL, RANDOM = 15, 35  # per language


def sample(rows: list[dict], seed: int = 1) -> list[dict]:
    """The doubtful answers first, then random ones from the rest."""
    doubtful = [r for r in rows if r["judge_supported"] == "False"]
    rest = [r for r in rows if r["judge_supported"] != "False"]
    chosen = random.Random(seed).sample(doubtful, min(DOUBTFUL, len(doubtful)))
    return chosen + random.Random(seed).sample(rest, min(RANDOM, len(rest)))


def answered_rows(language: str) -> list[dict]:
    items = [json.loads(line) for line in open(f"data/eval/patient_questions_{language}.jsonl", encoding="utf-8")]
    info = {i["question"]: i for i in items}
    rows = csv.DictReader(open(f"docs/patient_check_{language}.csv", encoding="utf-8-sig"))
    return [{**r, "id": info[r["question"]]["id"], "source": "; ".join(info[r["question"]]["books"]), "language": language}
            for r in rows if r["answered"] == "True" and "NO_ANSWER" not in r["answer"].upper()]


def main() -> int:
    columns = ["id", "language", "question", "answer", "source", "Correct? (yes / partly / no)",
               "Safe for a patient? (yes / no)", "Comment", "judge_supported"]
    with open("docs/doctor_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for language in ("en", "fa"):
            writer.writerows(sample(answered_rows(language)))
    print("wrote docs/doctor_review.csv (100 answers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
