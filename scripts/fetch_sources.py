"""Download the web pages that the patient questions name as their sources.

    python -m scripts.fetch_sources data/eval/patient_questions_english_persian_525.csv

It writes one text file per page into data/patient_books/ (not committed: check the licence of every site
before you use the pages for real) and the questions into data/eval/:
  patient_questions_en.jsonl and patient_questions_fa.jsonl  questions with the books that should be cited
  patient_questions_open.jsonl                               questions with no usable web page (nothing is expected)
Then: BOOKS_DIR=data/patient_books INDEX_DIR=data/index_patient python -m scripts.ingest
"""
import csv
import json
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

# Only these sites are downloaded. Others (for example Mayo Clinic) do not allow copying their pages.
SITES = {"medlineplus.gov": "MedlinePlus", "www.nhs.uk": "NHS", "www.who.int": "WHO", "www.cdc.gov": "CDC",
         "www.nimh.nih.gov": "NIMH", "www.niddk.nih.gov": "NIDDK"}
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"}
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "br", "div", "tr", "section"}
MIN_WORDS = 200  # a page with less text is a menu page, the real text is on other pages


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip:
            self.skip -= 1
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def page_text(html: str) -> str:
    """The readable text of a page. Menus and buttons are short, so lines under 5 words are dropped."""
    start = html.find("<main")
    parser = PageText()
    parser.feed(html[start:] if start >= 0 else html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if len(line.split()) >= 5)


def download(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (namazi-rag research test)"})
    return urllib.request.urlopen(request, timeout=30).read().decode("utf-8", errors="ignore")


def read_questions(path: str):
    """Returns the rows and the sources {source id: (book name, url)}."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    sources = {}
    for row in rows:
        ids = [i.strip() for i in row["Source IDs"].split(";")]
        # the url column is not always under its own name, so take the field that holds the links
        urls = next((v for v in row.values() if v.startswith("http")), "").split(" ; ")
        for source_id, url in zip(ids, urls):
            site = SITES.get(urlparse(url.strip()).netloc)
            if site and source_id not in sources:
                sources[source_id] = (f"{site} {row['Topic (English)'].split(' (')[0]}", url.strip())
    return rows, sources


def main() -> int:
    rows, sources = read_questions(sys.argv[1])
    folder = Path("data/patient_books")
    folder.mkdir(parents=True, exist_ok=True)

    got = {}  # source id -> book name, for the pages that downloaded
    for source_id, (book, url) in sources.items():
        try:
            text = page_text(download(url))
        except OSError as error:  # HTTPError and URLError are OSErrors too
            print(f"{source_id} {book}: could not download ({error})")
            continue
        words = len(text.split())
        if words < MIN_WORDS:
            print(f"{source_id} {book}: only {words} words, skipped")
            continue
        print(f"{source_id} {book}: {words} words")
        (folder / f"{book}.txt").write_text(text, encoding="utf-8")
        got[source_id] = book
        time.sleep(1)  # be polite to the sites

    files = {"en": [], "fa": [], "open": []}
    for row in rows:
        books = [got[i.strip()] for i in row["Source IDs"].split(";") if i.strip() in got]
        for language, column in (("en", "Question (English)"), ("fa", "پرسش (فارسی)")):
            item = {"id": row["Question ID"], "question": row[column], "books": books}
            files[language if books else "open"].append(item)

    for name, items in files.items():
        out = Path(f"data/eval/patient_questions_{name}.jsonl")
        out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        print(f"{out}: {len(items)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
