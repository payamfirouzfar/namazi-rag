import numpy as np

from app.embedder import SentenceTransformerEmbedder


class FakeModel:
    def __init__(self):
        self.options = {}

    def encode(self, texts, **options):
        self.options = options
        return np.ones((len(texts), 3))


def test_query_embedding_has_no_progress_bar():
    # many questions at once crashed the progress bars (found by the load test)
    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder.model, embedder.query_prefix = FakeModel(), "query: "
    embedder.embed_query("What is INR?")
    assert embedder.model.options["show_progress_bar"] is False
