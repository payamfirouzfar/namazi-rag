from app.metrics import Metrics
from tests.test_api import QUESTION


def test_counts_questions_tokens_and_not_found():
    m = Metrics()
    m.add_question(0.4, True, 100, 10)
    m.add_question(3.0, False, 200, 20)
    m.add_error()
    text = m.text()
    assert "namazi_questions_total 2" in text
    assert "namazi_not_found_total 1" in text
    assert "namazi_errors_total 1" in text
    assert "namazi_prompt_tokens_total 300" in text
    assert 'namazi_answer_seconds_bucket{le="0.5"} 1' in text  # only the fast one
    assert 'namazi_answer_seconds_bucket{le="5"} 2' in text
    assert 'namazi_answer_seconds_bucket{le="+Inf"} 2' in text


def test_metrics_endpoint_follows_the_questions(client):
    client.post("/v1/ask", json=QUESTION)
    text = client.get("/metrics").text
    assert "namazi_questions_total 1" in text
    assert "namazi_answer_seconds_count 1" in text


def test_a_very_slow_answer_is_counted_in_the_long_buckets():
    m = Metrics()
    m.add_question(200, True, 100, 10)  # 200 seconds, like a long queue
    text = m.text()
    assert 'namazi_answer_seconds_bucket{le="120"} 0' in text
    assert 'namazi_answer_seconds_bucket{le="300"} 1' in text
