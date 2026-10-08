import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import create_app
from tests.conftest import make_settings

QUESTION = {"question": "What is the INR target for patients on warfarin?"}


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["chunks"] > 0


def test_ready_reports_missing_index(tmp_path, db):
    settings = make_settings(index_dir=str(tmp_path / "missing"))
    with TestClient(create_app(settings, db=db)) as c:
        assert c.get("/health").status_code == 200  # process is alive
        response = c.get("/ready")
        assert response.status_code == 503 and "scripts.ingest" in response.json()["detail"]
        assert c.post("/v1/ask", json=QUESTION).status_code == 503


def test_ask_returns_answer_with_sources(client):
    response = client.post("/v1/ask", json=QUESTION)
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["answer"]
    assert body["sources"][0]["book"] == "warfarin"
    assert {"id", "book", "page", "score", "snippet"} <= body["sources"][0].keys()
    assert "not a diagnosis" in body["disclaimer"]
    assert body["conversation_id"] and body["latency_ms"] >= 0


def test_ask_is_saved_to_the_database(client, db):
    body = client.post("/v1/ask", json=QUESTION).json()
    saved = db.get_conversation(body["conversation_id"])
    assert saved["question"] == QUESTION["question"]
    assert saved["prompt_tokens"] == 100


def test_unanswerable_question(client, llm):
    llm.text = "NO_ANSWER"
    body = client.post("/v1/ask", json=QUESTION).json()
    assert body["answered"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "  "},
        {"question": "hi"},
        {"question": "x" * 2001},
        {"question": "valid question here", "top_k": 0},
        {"question": "valid question here", "top_k": 11},
        {"question": 123456},
    ],
)
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_top_k_changes_number_of_sources(client):
    body = client.post("/v1/ask", json={**QUESTION, "top_k": 2}).json()
    assert len(body["sources"]) == 2


# ---- security -------------------------------------------------------------

def secured_client(rag, db, **overrides):
    app = create_app(make_settings(api_keys="key-one, key-two", **overrides), rag=rag, db=db)
    return TestClient(app)


def test_api_key_required(rag, db):
    with secured_client(rag, db) as c:
        assert c.post("/v1/ask", json=QUESTION).status_code == 401
        assert c.post("/v1/ask", json=QUESTION, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/v1/ask", json=QUESTION, headers={"X-API-Key": "key-two"}).status_code == 200
        assert c.get("/health").status_code == 200  # health checks need no key


def test_feedback_also_needs_key(rag, db):
    with secured_client(rag, db) as c:
        payload = {"conversation_id": "x", "feedback": 1}
        assert c.post("/v1/feedback", json=payload).status_code == 401


def test_rate_limit(rag, db):
    app = create_app(make_settings(rate_limit_per_minute=2), rag=rag, db=db)
    with TestClient(app) as c:
        codes = [c.post("/v1/ask", json=QUESTION).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_docs_hidden_in_production(rag, db):
    app = create_app(make_settings(env="production", api_keys="k"), rag=rag, db=db)
    with TestClient(app) as c:
        assert c.get("/docs").status_code == 404
        assert c.get("/openapi.json").status_code == 404


def test_docs_available_in_development(client):
    assert client.get("/docs").status_code == 200


# ---- privacy --------------------------------------------------------------

def test_question_text_not_stored_when_logging_is_off(rag):
    from app.db import Database

    database = Database(":memory:", log_questions=False)
    with TestClient(create_app(make_settings(log_questions=False), rag=rag, db=database)) as c:
        body = c.post("/v1/ask", json=QUESTION).json()
    saved = database.get_conversation(body["conversation_id"])
    assert saved["question"] == "" and saved["answer"] == ""
    assert len(saved["question_hash"]) == 64


# ---- feedback ---------------------------------------------------------------

def test_feedback_saved_and_updatable(client, db):
    conv = client.post("/v1/ask", json=QUESTION).json()["conversation_id"]
    assert client.post("/v1/feedback", json={"conversation_id": conv, "feedback": 1}).status_code == 200
    assert db.get_feedback(conv)["value"] == 1
    client.post("/v1/feedback", json={"conversation_id": conv, "feedback": -1, "comment": "wrong dose"})
    assert db.get_feedback(conv)["value"] == -1
    assert db.get_feedback(conv)["comment"] == "wrong dose"


def test_feedback_validation(client):
    conv = client.post("/v1/ask", json=QUESTION).json()["conversation_id"]
    for bad in (0, 2, "up", None):
        assert client.post("/v1/feedback", json={"conversation_id": conv, "feedback": bad}).status_code == 422


def test_feedback_for_unknown_conversation(client):
    response = client.post("/v1/feedback", json={"conversation_id": "nope", "feedback": 1})
    assert response.status_code == 404


# ---- failures ---------------------------------------------------------------

def test_llm_failure_gives_503_without_internals(client, llm):
    llm.error = LLMError("connection refused to http://10.0.0.5:8000")
    response = client.post("/v1/ask", json=QUESTION)
    assert response.status_code == 503
    assert "10.0.0.5" not in response.text


def test_unexpected_error_gives_clean_500(rag, db, llm):
    llm.error = RuntimeError("secret internal detail")
    app = create_app(make_settings(), rag=rag, db=db)
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post("/v1/ask", json=QUESTION)
    assert response.status_code == 500
    assert "secret" not in response.text


def test_database_failure_does_not_break_the_answer(rag, monkeypatch, db):
    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(db, "save_conversation", boom)
    with TestClient(create_app(make_settings(), rag=rag, db=db)) as c:
        assert c.post("/v1/ask", json=QUESTION).status_code == 200


# ---- request id ---------------------------------------------------------------

def test_request_id_is_returned(client):
    assert client.get("/health").headers["x-request-id"]


def test_valid_request_id_is_kept_and_bad_one_replaced(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    bad = client.get("/health", headers={"X-Request-ID": "bad id with spaces!"}).headers["x-request-id"]
    assert bad != "bad id with spaces!"


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Namazi" in response.text
