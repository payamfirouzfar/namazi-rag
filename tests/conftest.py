from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chunking import chunk_pages
from app.config import Settings
from app.db import Database
from app.embedder import HashEmbedder
from app.index import Index
from app.llm import LLMResult
from app.loader import load_books
from app.main import create_app
from app.rag import RagService
from app.retriever import HybridRetriever, RetrievalParams

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


class FakeLLM:
    """Stands in for the real model so tests are fast, free and repeatable."""

    def __init__(self, text="Answer taken from the sources [1]."):
        self.text = text
        self.calls = []
        self.error = None

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return LLMResult(self.text, 100, 20)


def make_settings(**overrides) -> Settings:
    values = dict(
        env="development",
        api_keys="",
        db_path=":memory:",
        embedding_model="hash",
        rerank_model="",  # the real model is a big download, tests use a fake one
        rate_limit_per_minute=1000,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)  # _env_file=None: ignore any real .env


@pytest.fixture(scope="session")
def sample_books():
    return load_books(str(SAMPLE_DIR))


@pytest.fixture(scope="session")
def index(sample_books):
    chunks = [c for name, pages in sample_books.items() for c in chunk_pages(name, pages, 100, 20)]
    return Index.build(chunks, HashEmbedder(), 100, 20)


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def rag(index, llm):
    return RagService(HybridRetriever(index, HashEmbedder(), RetrievalParams()), llm)


@pytest.fixture
def db():
    return Database(":memory:")


@pytest.fixture
def client(rag, db):
    app = create_app(make_settings(), rag=rag, db=db)
    with TestClient(app) as c:
        yield c
