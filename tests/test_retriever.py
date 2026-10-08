import pytest

from app.embedder import HashEmbedder
from app.retriever import HybridRetriever, RetrievalParams, rrf_fuse


def test_rrf_scores():
    fused = dict(rrf_fuse([[1, 2, 3], [3, 2, 1]], [1, 1], k=60))
    assert fused[2] == pytest.approx(2 / 62)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[1] == pytest.approx(fused[3])


def test_rrf_weights_change_the_winner():
    rankings = [[1, 2], [2, 1]]
    assert rrf_fuse(rankings, [3, 1])[0][0] == 1
    assert rrf_fuse(rankings, [1, 3])[0][0] == 2


def test_rrf_empty():
    assert rrf_fuse([], []) == []
    assert rrf_fuse([[]], [1]) == []


def search(index, query, **params):
    retriever = HybridRetriever(index, HashEmbedder(), RetrievalParams(**params))
    return retriever.search(query)


def test_finds_the_right_book(index):
    hits = search(index, "What is the INR target for patients on warfarin?")
    assert hits[0].chunk.book == "warfarin"


def test_finds_persian_text(index):
    hits = search(index, "فشار خون بالا چگونه تشخیص داده می‌شود؟")
    assert hits[0].chunk.book == "hypertension fa"


def test_abbreviation_query_matches_full_term(index):
    hits = search(index, "DKA treatment", weight_dense=0, weight_bm25=1)
    assert hits[0].chunk.book == "diabetes"


def test_top_k_and_ordering(index):
    hits = search(index, "blood pressure drugs", top_k=3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_dense_only_and_bm25_only_both_work(index):
    assert search(index, "warfarin INR", weight_bm25=0)
    assert search(index, "warfarin INR", weight_dense=0)


def test_bm25_ignores_chunks_with_no_matching_word(index):
    assert search(index, "zzzzqqqq xxxxyyyy", weight_dense=0) == []


def test_hit_has_dense_score(index):
    hit = search(index, "warfarin INR")[0]
    assert -1.0 <= hit.dense_score <= 1.0
