from scripts.fetch_sources import page_text, read_questions

HEADER = ('"Question ID","Topic (English)","موضوع (فارسی)","Question intent","Question (English)","پرسش (فارسی)",'
          '"Source IDs","Source URLs","Provenance"\n')
ROWS = (
    '"Q001","Type 2 diabetes","دیابت نوع ۲","Basics","What is Type 2 diabetes in simple terms?","دیابت نوع ۲ به زبان ساده چیست؟",'
    '"S01; S02","Paraphrased patient-style prompt","https://medlineplus.gov/diabetestype2.html ; https://www.who.int/news-room/fact-sheets/detail/diabetes"\n'
    '"Q141","Depression","افسردگی","Basics","What is Depression in simple terms?","افسردگی به زبان ساده چیست؟",'
    '"S11; S13","Paraphrased patient-style prompt","https://www.nhs.uk/mental-health/conditions/depression-in-adults/ ; https://www.mayoclinic.org/diseases-conditions/depression/"\n'
    '"Q501","Pregnancy","بارداری","Timing","Can pregnancy be possible late in the cycle?","آیا بارداری ممکن است؟",'
    '"IR01","Anonymized paraphrase of a public Iranian patient Q&A question","https://www.darmankade.com/forum/questions/"\n'
)


def test_page_text_keeps_sentences_and_drops_menus():
    html = ("<html><nav>Home About Contact us today please</nav><main><h1>Type 2 diabetes</h1>"
            "<p>Type 2 diabetes is a long-term condition that affects how the body uses sugar.</p>"
            "<a>Menu</a><script>var secret = 'one two three four five six';</script>"
            "<p>Symptoms can be mild, so many people <b>do not notice</b> them at first.</p></main></html>")
    text = page_text(html)
    assert "long-term condition" in text
    assert "do not notice them at first" in text  # a word in bold does not split the sentence
    assert "Home About" not in text and "secret" not in text and "Menu" not in text


def test_sources_come_from_the_column_that_holds_the_links(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(HEADER + ROWS, encoding="utf-8")
    rows, sources = read_questions(str(path))
    assert len(rows) == 3
    assert sources["S01"] == ("MedlinePlus Type 2 diabetes", "https://medlineplus.gov/diabetestype2.html")
    assert sources["S02"][0] == "WHO Type 2 diabetes"
    assert sources["S11"][0] == "NHS Depression"
    assert "S13" not in sources  # Mayo Clinic is not downloaded
    assert "IR01" not in sources  # a forum, not a book
