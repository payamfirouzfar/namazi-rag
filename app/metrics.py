"""A few numbers for Prometheus, written as plain text. Counted per worker, like the rate limiter."""
import threading

SLOW_LIMITS = [0.5, 1, 2, 5, 10, 30, 60]  # answer time buckets, in seconds


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.questions = self.not_found = self.errors = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0
        self.under = [0] * len(SLOW_LIMITS)  # questions answered within each limit

    def add_question(self, seconds: float, answered: bool, prompt_tokens: int, completion_tokens: int):
        with self.lock:
            self.questions += 1
            self.not_found += 0 if answered else 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.seconds += seconds
            for n, limit in enumerate(SLOW_LIMITS):
                if seconds <= limit:
                    self.under[n] += 1

    def add_error(self):
        with self.lock:
            self.errors += 1

    def text(self) -> str:
        with self.lock:
            lines = [
                f"namazi_questions_total {self.questions}",
                f"namazi_not_found_total {self.not_found}",
                f"namazi_errors_total {self.errors}",
                f"namazi_prompt_tokens_total {self.prompt_tokens}",
                f"namazi_completion_tokens_total {self.completion_tokens}",
                f"namazi_answer_seconds_sum {self.seconds:.3f}",
                f"namazi_answer_seconds_count {self.questions}",
            ]
            for limit, count in zip(SLOW_LIMITS, self.under):
                lines.append(f'namazi_answer_seconds_bucket{{le="{limit}"}} {count}')
            lines.append(f'namazi_answer_seconds_bucket{{le="+Inf"}} {self.questions}')
        return "\n".join(lines) + "\n"
