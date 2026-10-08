from scripts.time_steps import time_steps

STEPS = {"translate", "embed question", "dense search", "BM25 search", "merge", "build prompt", "LLM answer"}


def test_every_step_is_timed(rag):
    steps = time_steps(rag, "What is the INR target for patients on warfarin?")
    assert set(steps) == STEPS  # the fake LLM is not Ollama, so the LLM is one step
    assert all(ms >= 0 for ms in steps.values())


def test_only_a_persian_question_is_translated(rag):
    english = time_steps(rag, "What is the INR target for patients on warfarin?")
    persian = time_steps(rag, "هدف INR برای بیماران تحت درمان با وارفارین چیست؟")
    assert english["translate"] < 1  # nothing to translate
    assert persian["translate"] >= 0
    assert len(rag.llm.calls) >= 2 + 1  # two answers and one translation
