from scripts.check_answers import judge_supported, made_up_numbers, print_summary


def row(answerable, answered, cited):
    return {"answerable": answerable, "answered": answered, "has_citation": answered,
            "evidence_in_sources": answerable, "evidence_cited": cited, "made_up_numbers": 0,
            "bad_citation": False, "judge_supported": "", "seconds": 2.0,
            "prompt_tokens": 100, "completion_tokens": 10}


def test_summary_counts_answerable_and_unanswerable(capsys):
    rows = [row(True, True, True), row(True, True, False), row(False, False, False), row(False, True, False)]
    print_summary("English questions", rows)
    out = capsys.readouterr().out
    assert "summary: English questions" in out
    assert "answerable questions        : 2" in out
    assert "right passage was cited   : 1" in out
    assert "unanswerable questions      : 2" in out
    assert "correctly said not found  : 1" in out


def test_summary_with_no_questions_does_not_crash(capsys):
    print_summary("Persian questions", [])
    assert "summary: Persian questions" in capsys.readouterr().out


class Chunk:
    def __init__(self, text):
        self.text, self.book, self.page = text, "Book", 1


class Hit:
    def __init__(self, text):
        self.chunk = Chunk(text)


def test_made_up_numbers_finds_numbers_that_are_in_no_source():
    sources = [Hit("The target INR is 2.0 to 3.0 for most patients.")]
    assert made_up_numbers("The target INR is 2.0 to 3.0 [1].", sources) == 0  # [1] is a citation, not a number
    assert made_up_numbers("The target INR is 2.5 to 3.0 [1].", sources) == 1  # 2.5 is invented


def test_persian_digits_count_as_numbers():
    sources = [Hit("Give 5 mg once a day.")]
    assert made_up_numbers("۵ میلی‌گرم روزانه [1]", sources) == 0
    assert made_up_numbers("۷ میلی‌گرم روزانه [1]", sources) == 1


def test_judge_reads_the_first_word_of_the_reply():
    from tests.conftest import FakeLLM

    sources = [Hit("Dexamethasone is preferred.")]
    assert judge_supported(FakeLLM("SUPPORTED"), "q", "a", sources, 1000) is True
    assert judge_supported(FakeLLM("NOT_SUPPORTED because..."), "q", "a", sources, 1000) is False


def test_a_question_can_name_the_books_that_should_be_cited():
    from scripts.check_answers import holds_evidence

    hit = Hit("Some text.")
    hit.chunk.book = "NHS Depression"
    assert holds_evidence({"books": ["NHS Depression", "WHO Depression"]}, hit) is True
    assert holds_evidence({"books": ["MedlinePlus Asthma"]}, hit) is False
    assert holds_evidence({"evidence": "some text"}, hit) is True  # the evidence phrase still works


def test_the_numbers_of_a_list_are_not_facts():
    sources = [Hit("Check your blood sugar every day.")]
    answer = "Questions to ask:\n1. How often should I check my blood sugar? [1]\n2. What is my target? [1]"
    assert made_up_numbers(answer, sources) == 0
