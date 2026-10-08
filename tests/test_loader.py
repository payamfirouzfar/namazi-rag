import pytest

from app.chunking import chunk_pages
from app.loader import load_books


def make_pdf(path, pages):
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path))
    for text in pages:
        pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()


def test_loads_txt_and_md_and_ignores_other_files(tmp_path):
    (tmp_path / "Book_One.txt").write_text("Hello there. This is book one.")
    (tmp_path / "notes.md").write_text("# Notes\nSome notes here.")
    (tmp_path / "image.png").write_bytes(b"not text")
    books = load_books(str(tmp_path))
    assert set(books) == {"Book One", "notes"}  # underscores become spaces


def test_loads_pdf_pages_with_numbers(tmp_path):
    pytest.importorskip("reportlab")
    make_pdf(tmp_path / "Pharmacology_Basics.pdf", ["Metformin is first line.", "", "Warfarin needs INR checks."])
    books = load_books(str(tmp_path))
    pages = books["Pharmacology Basics"]
    assert [number for number, _ in pages] == [1, 3]  # the empty page 2 is skipped
    assert "Warfarin" in pages[1][1]
    chunks = chunk_pages("Pharmacology Basics", pages, 50, 10)
    assert chunks[0].page == 1


def test_broken_pdf_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    (tmp_path / "good.txt").write_text("A good book with some text.")
    assert list(load_books(str(tmp_path))) == ["good"]


def test_empty_file_is_skipped(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n ")
    assert load_books(str(tmp_path)) == {}


def test_nested_folders_are_searched(tmp_path):
    (tmp_path / "internal").mkdir()
    (tmp_path / "internal" / "Harrison.txt").write_text("Text about internal medicine.")
    assert "Harrison" in load_books(str(tmp_path))


def test_loads_aes_encrypted_pdf(tmp_path):
    pytest.importorskip("reportlab")
    from pypdf import PdfReader, PdfWriter

    make_pdf(tmp_path / "plain.pdf", ["Metformin is first line."])
    writer = PdfWriter(clone_from=PdfReader(str(tmp_path / "plain.pdf")))
    writer.encrypt("", owner_password="owner", algorithm="AES-256")  # like the GOLD guide
    folder = tmp_path / "books"
    folder.mkdir()
    with open(folder / "Locked_Guide.pdf", "wb") as f:
        writer.write(f)
    books = load_books(str(folder))
    assert "Metformin" in books["Locked Guide"][0][1]
