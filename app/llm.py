import logging
from dataclasses import dataclass

import openai
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """The language model could not be reached or gave no answer."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _worth_retrying(exc: BaseException) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class OpenAICompatibleLLM:
    def __init__(self, s: Settings, wait=None):
        self.model = s.llm_model
        self.temperature = s.llm_temperature
        self.max_tokens = s.llm_max_tokens
        self.client = openai.OpenAI(
            api_key=s.llm_api_key or "not-needed",  # local servers ignore the key
            base_url=s.llm_base_url,
            timeout=s.llm_timeout_s,
            max_retries=0,  # we do our own retrying below
        )
        self._call = retry(
            retry=retry_if_exception(_worth_retrying),
            stop=stop_after_attempt(s.llm_retries),
            wait=wait or wait_exponential(multiplier=1, max=10),
            reraise=True,
            before_sleep=lambda state: log.warning("LLM call failed, retrying: %s", state.outcome.exception()),
        )(self._request)

    def _request(self, system: str, user: str):
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    def complete(self, system: str, user: str) -> LLMResult:
        try:
            response = self._call(system, user)
        except openai.OpenAIError as exc:
            raise LLMError(f"LLM request failed: {exc.__class__.__name__}") from exc
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("LLM returned an empty answer")
        usage = response.usage
        if usage:
            log.info("LLM %s used %d prompt tokens and %d answer tokens",
                     self.model, usage.prompt_tokens, usage.completion_tokens)
        return LLMResult(
            text,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
