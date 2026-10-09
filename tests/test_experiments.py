import pandas as pd

from app.embedder import HashEmbedder
from app.retriever import HybridRetriever, RetrievalParams
from experiments.common import build_index, join_broken_words, make_chunks, score, strip_running_lines
from experiments.compare_runs import numbers
from experiments.prompts import prompts


def test_words_that_the_pdf_broke_are_joined_but_normal_words_are_not():
    assert join_broken_words("the bene fits are signi ficant and speci fic") == "the benefits are significant and specific"
    assert join_broken_words("the first final findings of atrial fibrillation") == "the first final findings of atrial fibrillation"


def test_lines_that_repeat_on_many_pages_are_dropped():
    pages = [(n, f"www.journal.org\nA real sentence about the kidney number {n}.") for n in range(1, 11)]
    kept = strip_running_lines(pages)
    assert all("www.journal.org" not in text for _, text in kept)
    assert all("real sentence" in text for _, text in kept)
    assert strip_running_lines(pages[:3]) == pages[:3]  # a short book has no repeated headers to find


def test_a_title_can_be_added_to_every_chunk():
    books = {"Kidney Book": [(1, "The kidney filters blood. It also makes urine and hormones.")]}
    chunks = make_chunks(books, size=50, overlap=10, title=True)
    assert chunks and all(chunk.text.startswith("Kidney Book. ") for chunk in chunks)


def test_score_gives_hit_rate_and_mrr_for_each_language(sample_books):
    chunks = make_chunks(sample_books, size=100, overlap=20)
    embedder = HashEmbedder()
    retriever = HybridRetriever(build_index(chunks, embedder, 100, 20), embedder, RetrievalParams())
    questions = [{"question": "Which drug raises bradykinin and causes a cough?", "evidence": "raise bradykinin levels"}]
    result = score(lambda text: retriever.search(text, 10), questions, {})
    assert list(result) == ["English"]
    hit_rate, mrr = result["English"]
    assert hit_rate == 1.0 and 0 < mrr <= 1


def test_compare_runs_counts_answers_and_the_language(tmp_path):
    rows = [
        {"question": "What is INR?", "answerable": True, "answered": True, "evidence_in_sources": True, "evidence_cited": True,
         "made_up_numbers": 0, "completion_tokens": 40, "seconds": 3.0, "answer": "A blood test [1]."},
        {"question": "INR چیست؟", "answerable": True, "answered": True, "evidence_in_sources": True, "evidence_cited": False,
         "made_up_numbers": 1, "completion_tokens": 60, "seconds": 5.0, "answer": "It is a blood test."},
        {"question": "How is Graves treated?", "answerable": False, "answered": False, "evidence_in_sources": False, "evidence_cited": False,
         "made_up_numbers": 0, "completion_tokens": 5, "seconds": 2.0, "answer": "I could not find the answer."},
    ]
    path = tmp_path / "run.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = numbers(str(path), "All")
    assert result["answerable questions"] == 2 and result["...right passage cited"] == 1
    assert result["unanswerable questions"] == 1 and result["...still got an answer"] == 0
    assert result["answers with a number in no source"] == 1
    assert result["Persian questions answered in English"] == 1  # the Persian question got an English answer


def test_the_four_prompts_differ_in_the_rules_we_test():
    texts = prompts()
    assert set(texts) == {"base", "short", "partial", "both"}
    assert "Keep the answer short and clear." in texts["base"] and "2 to 4 short sentences" in texts["short"]
    assert "say what is missing" in texts["partial"] and "say what is missing" in texts["both"]


def test_compare_runs_skips_a_language_that_has_no_questions(tmp_path):
    rows = [{"question": "INR چیست؟", "answerable": True, "answered": True, "evidence_in_sources": True, "evidence_cited": True,
             "made_up_numbers": 0, "completion_tokens": 60, "seconds": 5.0, "answer": "آزمایش خون [1]"}]
    path = tmp_path / "persian_only.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert numbers(str(path), "English") is None
    assert numbers(str(path), "Persian")["...right passage cited"] == 1
