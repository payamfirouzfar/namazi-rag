import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from tenacity import wait_none

from app.llm import LLMError, OpenAICompatibleLLM
from tests.conftest import make_settings


@pytest.fixture
def fake_server():
    """A tiny local server that speaks enough of the OpenAI chat API."""
    state = {"calls": 0, "script": []}  # script: list of (status, content) per call

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            status, content = state["script"][min(state["calls"], len(state["script"]) - 1)]
            state["calls"] += 1
            body = json.dumps(
                {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                 "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}
                if status == 200 else {"error": {"message": "boom"}}
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()


def make_llm(url, retries=3):
    return OpenAICompatibleLLM(make_settings(llm_base_url=url, llm_retries=retries, llm_timeout_s=5), wait=wait_none())


def test_successful_call(fake_server):
    fake_server["script"] = [(200, "  The answer [1]  ")]
    result = make_llm(fake_server["url"]).complete("system", "user")
    assert result.text == "The answer [1]"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)


def test_retries_server_errors_then_succeeds(fake_server):
    fake_server["script"] = [(500, ""), (500, ""), (200, "ok")]
    assert make_llm(fake_server["url"]).complete("s", "u").text == "ok"
    assert fake_server["calls"] == 3


def test_gives_up_after_retries(fake_server):
    fake_server["script"] = [(500, "")]
    with pytest.raises(LLMError):
        make_llm(fake_server["url"], retries=2).complete("s", "u")
    assert fake_server["calls"] == 2


def test_client_errors_are_not_retried(fake_server):
    fake_server["script"] = [(400, "")]
    with pytest.raises(LLMError):
        make_llm(fake_server["url"]).complete("s", "u")
    assert fake_server["calls"] == 1


def test_empty_answer_is_an_error(fake_server):
    fake_server["script"] = [(200, "   ")]
    with pytest.raises(LLMError):
        make_llm(fake_server["url"]).complete("s", "u")


def test_unreachable_server_is_an_error():
    with pytest.raises(LLMError):
        make_llm("http://127.0.0.1:9/v1", retries=2).complete("s", "u")
