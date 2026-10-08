import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


def check_api_key(provided: str | None, allowed: list[str]) -> bool:
    if not provided:
        return False
    # compare_digest so response time does not leak how much of a key was right
    return any(hmac.compare_digest(provided.encode(), key.encode()) for key in allowed)


class RateLimiter:
    """Sliding window limiter kept in memory.
    Note: each worker process counts on its own. For strict limits across workers,
    also set a limit in the reverse proxy (nginx etc.)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, caller: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            window = self.hits[caller]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True


def caller_id(request: Request) -> str:
    key = request.headers.get("x-api-key")
    if key:
        return "key:" + key[:8]
    return "ip:" + (request.client.host if request.client else "unknown")


def make_auth_dependency(allowed_keys: list[str], limiter: RateLimiter):
    """Dependency that checks the API key (if keys are configured) and the rate limit."""

    def dependency(request: Request):
        if allowed_keys and not check_api_key(request.headers.get("x-api-key"), allowed_keys):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        if not limiter.allow(caller_id(request)):
            raise HTTPException(status_code=429, detail="Too many requests, slow down")

    return dependency
