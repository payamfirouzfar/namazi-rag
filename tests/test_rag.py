from app.llm import LLMError, LLMResult
from app.rag import (NO_ANSWER, NOT_FOUND_MESSAGE, TRANSLATE_PROMPT, URGENT_MESSAGE, RagService, build_context,
                     has_persian, looks_urgent, user_message)
from app.retriever import Hit
from app.chunking import Chunk


def hit(text, number=1, dense=0.9):
    return Hit(Chunk(f"b_{number}", "Book", number, text), 0.1, dense)


def test_answer_uses_sources_and_returns_them(rag, llm):
    result = rag.ask("What is the INR target for warfarin?")
    assert result.answered
    assert result.answer == llm.text
    assert result.sources[0].chunk.book == "warfarin"
    system, user = llm.calls[0]
    assert "[1]" in user and "QUESTION: What is the INR target" in user
    assert "ONLY the numbered sources" in system
    assert result.prompt_tokens == 100


def test_no_answer_from_llm_becomes_polite_message(rag, llm):
    for text in (NO_ANSWER, "NO_ANSWER.", " no_answer "):
        llm.text = text
        result = rag.ask("What is the INR target for warfarin?")
        assert not result.answered
        assert result.answer == NOT_FOUND_MESSAGE


def test_no_answer_followed_by_more_text_is_still_no_answer(rag, llm):
    llm.text = "NO_ANSWER. The sources only talk about diabetes."
    result = rag.ask("What is the INR target for warfarin?")
    assert not result.answered
    assert result.answer == NOT_FOUND_MESSAGE


def test_no_answer_at_the_end_after_an_explanation_is_still_no_answer(rag, llm):
    # found by the patient questions: the model explained first and wrote NO_ANSWER last, and a patient saw that word
    llm.text = "The sources provided do not explicitly state this. Therefore, the answer is:\n\nNO_ANSWER"
    result = rag.ask("What is the INR target for warfarin?")
    assert not result.answered
    assert result.answer == NOT_FOUND_MESSAGE


def test_normal_answer_is_not_mistaken_for_no_answer(rag, llm):
    llm.text = "The target INR is 2.0 to 3.0 [1]."
    assert rag.ask("What is the INR target for warfarin?").answered


def test_ask_writes_log_lines(rag, caplog):
    with caplog.at_level("INFO"):
        rag.ask("What is the INR target for warfarin?")
    text = caplog.text
    assert "search found" in text and "sending" in text and "answered with" in text


def test_low_similarity_skips_the_llm(index, llm):
    from app.embedder import HashEmbedder
    from app.retriever import HybridRetriever, RetrievalParams

    strict = RagService(HybridRetriever(index, HashEmbedder(), RetrievalParams()), llm, min_dense_score=0.99)
    result = strict.ask("What is the INR target for warfarin?")
    assert not result.answered
    assert llm.calls == []


def test_no_hits_skips_the_llm(llm):
    class EmptyRetriever:
        def search(self, question, top_k=None):
            return []

    result = RagService(EmptyRetriever(), llm).ask("anything")
    assert not result.answered and llm.calls == []


def test_build_context_numbers_the_sources():
    context, used = build_context([hit("first text", 1), hit("second text", 2)], 10_000)
    assert "[1] (Book, page 1)\nfirst text" in context
    assert "[2] (Book, page 2)\nsecond text" in context
    assert len(used) == 2


def test_build_context_respects_budget_but_keeps_first():
    hits = [hit("a" * 500, 1), hit("b" * 500, 2), hit("c" * 500, 3)]
    context, used = build_context(hits, 700)
    assert len(used) == 1
    context, used = build_context(hits, 100)  # even a tiny budget keeps the best source
    assert len(used) == 1 and len(context) <= 100


PERSIAN_QUESTION = "هدف INR برای بیمار تحت درمان با وارفارین چیست؟"


def test_persian_question_is_translated_for_the_search(rag, llm):
    llm.text = "What is the INR target for warfarin?"
    result = rag.ask(PERSIAN_QUESTION)
    assert result.answered
    assert llm.calls[0][0] == TRANSLATE_PROMPT
    assert PERSIAN_QUESTION in llm.calls[1][1]  # the answer step still sees the original question


def test_english_question_is_not_translated(rag, llm):
    rag.ask("What is the INR target for warfarin?")
    assert len(llm.calls) == 1


def test_failed_translation_falls_back_to_the_original_question(index):
    from app.embedder import HashEmbedder
    from app.retriever import HybridRetriever, RetrievalParams

    class BrokenTranslator:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user):
            self.calls += 1
            if system == TRANSLATE_PROMPT:
                raise LLMError("down")
            return LLMResult("Answer [1].", 10, 5)

    llm = BrokenTranslator()
    rag = RagService(HybridRetriever(index, HashEmbedder(), RetrievalParams()), llm)
    assert rag.ask(PERSIAN_QUESTION).answered
    assert llm.calls == 2


def test_has_persian():
    assert has_persian(PERSIAN_QUESTION)
    assert not has_persian("What is the INR target?")


def test_an_urgent_question_starts_with_the_emergency_line(rag, llm):
    result = rag.ask("I have chest pain and the INR target for warfarin is what?")
    assert result.answer.startswith(URGENT_MESSAGE)
    assert "[1]" in result.answer  # the normal answer is still there


def test_urgent_words_work_in_persian_with_a_half_space():
    assert looks_urgent("درد قفسه سینه دارم، چه کنم؟")
    assert looks_urgent("نمی\u200cتوانم نفس بکشم")
    assert not looks_urgent("What is the INR target for warfarin?")


def test_a_normal_question_has_no_emergency_line(rag, llm):
    assert URGENT_MESSAGE not in rag.ask("What is the INR target for warfarin?").answer


def test_a_persian_answer_with_broken_letters_is_not_shown(rag, llm):
    llm.text = "وارفارین 你好 [1]"
    result = rag.ask("هدف INR برای وارفارین چیست؟")
    assert not result.answered
    assert result.answer == NOT_FOUND_MESSAGE


def test_the_message_tells_the_model_which_language_to_answer_in():
    assert user_message("some sources", "What is INR?").endswith("Write the answer in English.")
    assert user_message("some sources", "INR چیست؟").endswith("Write the answer in Persian.")
