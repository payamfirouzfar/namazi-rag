"""A fake LLM server for load tests. It always gives the same short answer after a small wait.

    python -m scripts.fake_llm --delay 1.0

Then start the app with LLM_BASE_URL=http://127.0.0.1:9301/v1. This tests the app itself
(search, database, API), not the speed of a real model.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = {"role": "assistant", "content": "This is a fake answer from the sources [1]."}


class Handler(BaseHTTPRequestHandler):
    delay = 1.0

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(self.delay)  # a real model also takes a while
        body = json.dumps({
            "id": "fake", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "message": ANSWER, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 20, "total_tokens": 2020},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the terminal quiet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=9301)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait before every answer")
    args = parser.parse_args()
    Handler.delay = args.delay
    print(f"fake LLM on http://127.0.0.1:{args.port}/v1 (waits {args.delay} s per answer)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
