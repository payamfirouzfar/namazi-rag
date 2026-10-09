"""Experiment 5: does a different system prompt give better answers?

    python -m experiments.prompts short
    python -m experiments.prompts partial --eval data/eval/eval_questions.jsonl

The first word is the prompt: base (the first prompt), short (the one the app uses now), partial or both.
Anything after it goes to scripts/check_answers.py. The answers are saved in data/tuning/prompt_<name>.csv
and experiments/compare_runs.py puts the runs side by side.
"""
import sys

import app.rag as rag
from scripts import check_answers

FIRST_RULE_6 = "6. Keep the answer short and clear."
SHORT_RULE_6 = "6. Start with the direct answer. Keep it to 2 to 4 short sentences in simple words."
STRICT_RULE_3 = "3. If the sources do not contain the answer, reply with exactly NO_ANSWER and nothing else."
PARTIAL_RULE_3 = ("3. If the sources answer only part of the question, answer that part and say what is missing. "
                  "If they do not contain the answer at all, reply with exactly NO_ANSWER and nothing else.")


def prompts() -> dict:
    short = rag.SYSTEM_PROMPT.replace(FIRST_RULE_6, SHORT_RULE_6)  # works whichever of the two rules the app has now
    base = short.replace(SHORT_RULE_6, FIRST_RULE_6)
    for text, rule in ((base, FIRST_RULE_6), (short, SHORT_RULE_6), (base, STRICT_RULE_3)):
        assert rule in text, f"the prompt in app/rag.py has changed, update this script: {rule}"
    return {"base": base, "short": short, "partial": base.replace(STRICT_RULE_3, PARTIAL_RULE_3),
            "both": short.replace(STRICT_RULE_3, PARTIAL_RULE_3)}


def main() -> int:
    name, extra = sys.argv[1], sys.argv[2:]
    rag.SYSTEM_PROMPT = prompts()[name]
    sys.argv = ["check_answers", "--out", f"data/tuning/prompt_{name}.csv", *extra]
    return check_answers.main()


if __name__ == "__main__":
    sys.exit(main())
