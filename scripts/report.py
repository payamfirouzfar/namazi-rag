"""Print a short usage report from the database, so the hospital can see how the tool is doing.

    python -m scripts.report
"""
import sqlite3
import sys

from app.config import get_settings


def build_report(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT answered, latency_ms, prompt_tokens, completion_tokens FROM conversations").fetchall()
    if not rows:
        return "No questions have been asked yet."

    total = len(rows)
    answered = sum(r[0] for r in rows)
    times = sorted(r[1] for r in rows)
    slowest_normal = times[int(total * 0.95) - 1] if total >= 20 else times[-1]
    thumbs = dict(conn.execute("SELECT value, COUNT(*) FROM feedback GROUP BY value").fetchall())

    lines = [
        f"questions asked        : {total}",
        f"answered from the books: {answered} ({answered / total:.0%})",
        f"not found in the books : {total - answered}",
        f"thumbs up / down       : {thumbs.get(1, 0)} / {thumbs.get(-1, 0)}",
        f"average answer time    : {sum(times) / total / 1000:.1f} s",
        f"95% of answers within  : {slowest_normal / 1000:.1f} s",
        f"tokens used            : {sum(r[2] for r in rows)} prompt, {sum(r[3] for r in rows)} answer",
    ]

    # questions the books could not answer show which book is missing (empty if LOG_QUESTIONS=false)
    missing = conn.execute(
        "SELECT question FROM conversations WHERE answered = 0 AND question != '' ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    if missing:
        lines.append("\nlast questions with no answer:")
        lines += [f"  - {q[0][:100]}" for q in missing]

    disliked = conn.execute(
        "SELECT c.question FROM feedback f JOIN conversations c ON c.id = f.conversation_id "
        "WHERE f.value = -1 AND c.question != '' ORDER BY f.created_at DESC LIMIT 10"
    ).fetchall()
    if disliked:
        lines.append("\nlast questions with a thumbs down:")
        lines += [f"  - {q[0][:100]}" for q in disliked]
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_report(get_settings().db_path))
    sys.exit(0)
