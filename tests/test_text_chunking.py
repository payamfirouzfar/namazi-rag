import pytest

from app.chunking import chunk_pages
from app.text import clean_text, split_sentences, tokenize


def test_clean_text_fixes_pdf_damage():
    assert clean_text("hyper-\ntension") == "hypertension"
    assert clean_text("ﬁbrosis") == "fibrosis"  # ligature
    assert "ی" in clean_text("فشار خون بالاي")  # Arabic yeh becomes Persian yeh
    assert clean_text("می‌شود") == "می شود"  # zero-width non-joiner


def test_split_sentences_handles_persian_question_mark():
    parts = split_sentences("This is one. This is two! آیا خوب است؟ بله")
    assert len(parts) == 4


def test_tokenize_drops_stopwords_and_expands_abbreviations():
    tokens = tokenize("What is the INR target in HTN?")
    assert "the" not in tokens and "what" not in tokens
    assert "inr" in tokens and "international" in tokens
    assert "htn" in tokens and "hypertension" in tokens


def test_tokenize_persian():
    tokens = tokenize("فشار خون بالا در بارداری")
    assert "فشار" in tokens and "در" not in tokens


def make_pages():
    sentence = "Word " * 19 + "end."  # 20 words
    page1 = " ".join([sentence] * 6)
    page2 = " ".join([sentence] * 6)
    return [(1, page1), (2, page2)]


def test_chunks_never_exceed_size():
    chunks = chunk_pages("Book", make_pages(), chunk_size=50, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 50 for c in chunks)


def test_chunks_overlap():
    chunks = chunk_pages("Book", make_pages(), chunk_size=50, overlap=20)
    tail = chunks[0].text.split()[-20:]
    assert " ".join(tail) in chunks[1].text


def test_no_overlap_when_zero():
    chunks = chunk_pages("Book", make_pages(), chunk_size=40, overlap=0)
    assert sum(len(c.text.split()) for c in chunks) == 240  # nothing repeated


def test_page_number_is_where_chunk_starts():
    chunks = chunk_pages("Book", make_pages(), chunk_size=50, overlap=0)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2


def test_ids_are_unique_and_use_book_name():
    chunks = chunk_pages("Harrison Internal Medicine", make_pages(), 50, 10)
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids[0].startswith("harrison-internal-medicine_")


def test_very_long_sentence_is_split_by_words():
    chunks = chunk_pages("Book", [(1, "word " * 500)], chunk_size=100, overlap=10)
    assert len(chunks) >= 5
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_empty_input_gives_no_chunks():
    assert chunk_pages("Book", [], 100, 10) == []
    assert chunk_pages("Book", [(1, "   \n  ")], 100, 10) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages("Book", make_pages(), chunk_size=50, overlap=50)
