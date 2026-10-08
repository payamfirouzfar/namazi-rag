from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 2000

DISCLAIMER = (
    "Reference information from textbooks, not a diagnosis. "
    "A clinician must verify it before any decision about a patient. "
    "In an emergency call your local emergency number."
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class Source(BaseModel):
    id: str
    book: str
    page: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    conversation_id: str
    answer: str
    answered: bool
    sources: list[Source]
    disclaimer: str = DISCLAIMER
    latency_ms: int


class FeedbackRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    feedback: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=1000)
