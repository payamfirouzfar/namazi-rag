import pytest
from pydantic import ValidationError

from app.db import Database
from app.security import RateLimiter, check_api_key
from tests.conftest import make_settings


def test_defaults_are_valid():
    s = make_settings()
    assert s.chunk_overlap < s.chunk_size and s.top_k == 5


def test_production_requires_api_keys():
    with pytest.raises(ValidationError):
        make_settings(env="production", api_keys="")
    assert make_settings(env="production", api_keys="abc").is_production


def test_bad_chunk_settings_rejected():
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_bad_weights_rejected():
    with pytest.raises(ValidationError):
        make_settings(weight_dense=0, weight_bm25=0)
    with pytest.raises(ValidationError):
        make_settings(weight_dense=-1)


def test_key_and_origin_lists():
    s = make_settings(api_keys=" a , b,, ", cors_origins="http://x.com, http://y.com")
    assert s.key_list == ["a", "b"]
    assert s.origin_list == ["http://x.com", "http://y.com"]


def test_check_api_key():
    assert check_api_key("good", ["bad", "good"])
    assert not check_api_key("nope", ["good"])
    assert not check_api_key(None, ["good"])
    assert not check_api_key("", ["good"])


def test_rate_limiter_window(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("app.security.time.monotonic", lambda: clock[0])
    limiter = RateLimiter(2)
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")  # other callers are independent
    clock[0] += 61
    assert limiter.allow("a")  # window has passed


def test_rate_limiter_zero_means_off():
    limiter = RateLimiter(0)
    assert all(limiter.allow("a") for _ in range(100))


def test_database_rejects_sql_injection_in_ids():
    db = Database(":memory:")
    assert db.save_feedback("x'; DROP TABLE conversations;--", 1) is False
    db.save_conversation("c1", "q", "a", True, [], 5, 1, 1)  # table still exists
    assert db.get_conversation("c1")["answer"] == "a"


def test_database_works_with_a_file(tmp_path):
    db = Database(str(tmp_path / "sub" / "app.db"))
    db.save_conversation("c1", "question", "answer", False, [{"id": "x"}], 10, 0, 0)
    assert db.save_feedback("c1", -1)
    assert db.get_feedback("c1")["value"] == -1


def test_env_example_parses_cleanly():
    """The shipped .env.example must work as-is (a comment once leaked into API_KEYS)."""
    from pathlib import Path

    from app.config import Settings

    s = Settings(_env_file=str(Path(__file__).resolve().parent.parent / ".env.example"))
    assert s.api_keys == "" and s.llm_base_url is None and s.cors_origins == ""
    assert s.query_prefix == "query: "
