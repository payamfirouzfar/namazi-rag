from app.db import Database
from scripts.report import build_report


def test_report_on_an_empty_database(tmp_path):
    path = str(tmp_path / "app.db")
    Database(path)
    assert "No questions" in build_report(path)


def test_report_counts_answers_and_feedback(tmp_path):
    path = str(tmp_path / "app.db")
    db = Database(path)
    db.save_conversation("a", "What is the INR target?", "2 to 3 [1]", True, [], 1000, 100, 20)
    db.save_conversation("b", "Treatment of psoriasis?", "not found", False, [], 3000, 80, 5)
    db.save_feedback("b", -1)
    report = build_report(path)
    assert "questions asked        : 2" in report
    assert "answered from the books: 1 (50%)" in report
    assert "thumbs up / down       : 0 / 1" in report
    assert "Treatment of psoriasis?" in report
