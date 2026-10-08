from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings come from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"  # "development" or "production"
    log_level: str = "INFO"

    # paths
    books_dir: str = "data/books"
    index_dir: str = "data/index"
    db_path: str = "data/app.db"

    # chunking (sizes are in words)
    chunk_size: int = 250
    chunk_overlap: int = 50

    # embeddings. multilingual-e5 handles Persian questions against English books.
    # Use "hash" only for tests / offline development.
    embedding_model: str = "intfloat/multilingual-e5-base"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    embedding_batch_size: int = 32

    # retrieval
    top_k: int = 5
    oversample: int = 3  # candidates per retriever = top_k * oversample
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_bm25: float = 1.0
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    min_dense_score: float = 0.0  # 0 = off. Calibrate with scripts/tune.py
    max_context_chars: int = 12000

    # LLM: any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, ...)
    llm_base_url: str | None = None
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700
    llm_timeout_s: float = 60.0
    llm_retries: int = 3

    # API
    api_keys: str = ""  # comma separated list of allowed keys
    cors_origins: str = ""  # comma separated, empty = no CORS
    rate_limit_per_minute: int = 30
    log_questions: bool = True  # False = store only a hash of the question

    @field_validator("llm_base_url")
    @classmethod
    def empty_url_means_openai(cls, v):
        return (v or "").strip() or None  # "LLM_BASE_URL=" in .env gives "" not None

    @model_validator(mode="after")
    def check_values(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.weight_dense < 0 or self.weight_bm25 < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.weight_dense == 0 and self.weight_bm25 == 0:
            raise ValueError("at least one retrieval weight must be above 0")
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1")
        if self.env == "production" and not self.key_list:
            raise ValueError("API_KEYS must be set when ENV=production")
        return self

    @property
    def key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
