"""Small SQLite store for conversations and thumbs up/down feedback.
Every query is parameterized (no string formatting into SQL)."""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    answer TEXT NOT NULL,
    answered INTEGER NOT NULL,
    sources TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id),
    value INTEGER NOT NULL CHECK (value IN (-1, 1)),
    comment TEXT,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str, log_questions: bool = True):
        self.path = path
        self.log_questions = log_questions
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # a shared in-memory db needs one connection that stays open
        self._keep = sqlite3.connect(path, check_same_thread=False) if path == ":memory:" else None
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = self._keep or sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        if not self._keep:
            conn.execute("PRAGMA journal_mode = WAL")
        return _Conn(conn, owned=self._keep is None)

    def save_conversation(self, conv_id, question, answer, answered, sources, latency_ms, prompt_tokens, completion_tokens):
        q_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        # when LOG_QUESTIONS=false nothing a patient could be identified by is stored
        stored_question = question if self.log_questions else ""
        stored_answer = answer if self.log_questions else ""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (conv_id, now(), stored_question, q_hash, stored_answer, int(answered),
                 json.dumps(sources), latency_ms, prompt_tokens, completion_tokens),
            )

    def save_feedback(self, conv_id: str, value: int, comment: str | None = None) -> bool:
        """Returns False if the conversation does not exist."""
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not exists:
                return False
            conn.execute(
                "INSERT INTO feedback VALUES (?,?,?,?) ON CONFLICT(conversation_id) "
                "DO UPDATE SET value = excluded.value, comment = excluded.comment, created_at = excluded.created_at",
                (conv_id, value, comment, now()),
            )
            return True

    def get_conversation(self, conv_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            return dict(row) if row else None

    def get_feedback(self, conv_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM feedback WHERE conversation_id = ?", (conv_id,)).fetchone()
            return dict(row) if row else None


class _Conn:
    """Commits on success, rolls back on error, closes real connections."""

    def __init__(self, conn, owned):
        self.conn, self.owned = conn, owned
        conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, *_):
        self.conn.rollback() if exc_type else self.conn.commit()
        if self.owned:
            self.conn.close()
