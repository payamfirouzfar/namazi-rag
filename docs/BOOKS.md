# Which books to put in the knowledge base

The system can only answer from what you give it, so the book list matters more than any
parameter. I cannot ship the books (they are copyrighted) and I cannot download them. Put
**licensed** digital copies in `data/books/` (ask the hospital library about institutional
access) and run `python -m scripts.ingest`.

## Start with these (tier 1)

These cover most questions clinicians ask. Start here, measure, then add more.

| Area | Book |
|---|---|
| Internal medicine | Harrison's Principles of Internal Medicine |
| Quick clinical reference | Current Medical Diagnosis & Treatment (CMDT) |
| Drugs | Katzung's Basic & Clinical Pharmacology (or Goodman & Gilman's) |
| Paediatrics | Nelson Textbook of Pediatrics |
| Obstetrics | Williams Obstetrics |
| Emergency medicine | Tintinalli's Emergency Medicine |
| Surgery | Schwartz's Principles of Surgery (or Bailey & Love) |

## Add next (tier 2)

Goldman-Cecil Medicine or Davidson's (second opinion on internal medicine) · Braunwald's Heart
Disease · Mandell, Douglas and Bennett's Principles and Practice of Infectious Diseases ·
Novak's Gynecology · Adams and Victor's Principles of Neurology · Kaplan & Sadock's
Psychiatry / DSM-5-TR · Marino's The ICU Book · Robbins & Cotran Pathologic Basis of Disease.

## Basic science (tier 3, mainly for students and teaching)

Guyton & Hall Physiology · Gray's Anatomy for Students · Lippincott Biochemistry ·
Janeway's Immunobiology · Ross Histology. MedRAG's 18-textbook corpus is made of books like
these (plus Harrison's, First Aid, Pathoma, ...), so results on it are comparable to that paper.

## Guidelines (free, and usually more current than textbooks)

ADA Standards of Care in Diabetes · ACC/AHA hypertension and heart-failure guidelines · KDIGO
(kidney) · Surviving Sepsis Campaign · GOLD (COPD) and GINA (asthma) · WHO guidelines and the
WHO Essential Medicines List.

**The most valuable files are the hospital's own:** local protocols, the formulary, the
antibiogram, and the national guidelines of the Iranian Ministry of Health and Medical
Education that Namazi already follows. When a textbook and a local protocol disagree, the
protocol is the right answer, so make sure it is in the library.

## Rules for the files

- **Use current editions.** Old editions contain outdated doses and thresholds. This is the
  biggest clinical risk in the whole system.
- **One file per book.** The file name becomes the name shown in citations, so name it well:
  `Harrison_Principles_of_Internal_Medicine.pdf` is shown as "Harrison Principles of Internal Medicine".
- **Text PDFs only.** A scanned book (pictures of pages) has no text. The loader warns when it
  finds one; run OCR on it first (for example with `ocrmypdf`).
- **Language.** The books are English. Persian questions work because the embedding model is
  multilingual, but the keyword half (BM25) cannot match Persian words to English text, so
  Persian-to-English search leans on the embeddings only. Add Persian questions to your eval
  file (`data/eval/`) and watch the numbers. Persian documents (protocols) work in both halves.
- **Re-ingest after changes.** Add or remove a book, then run `python -m scripts.ingest`
  (and restart the API).

## Books used in the demo run (free to download)

These 20 files were used to test the system. They are free to read and download, but they are
not stored in this repository, and their own licences apply (WHO documents are mostly
CC BY-NC-SA 3.0 IGO; KDIGO and GOLD keep their copyright). Check the terms before reusing them.

- GOLD COPD Pocket Guide 2024 (goldcopd.org)
- KDIGO guidelines (kdigo.org): CKD 2024 · Diabetes in CKD 2022 · Blood Pressure in CKD 2012 ·
  CKD Evaluation and Management 2012 · Anemia in CKD 2012 and 2026 · Acute Kidney Injury 2012 ·
  Lipid Management 2013 · CKD-MBD 2017 · Hepatitis C in CKD 2018 · Glomerular Diseases 2021 ·
  Kidney Transplant Candidate 2020 · Kidney Transplant Recipient Care 2009
- WHO: Meningitis Guidelines 2025 · Clinical Management of COVID-19 (2023) · Systematic Screening
  for Tuberculosis · Universal Access to Malaria Diagnostic Testing · Child Health Recommendations ·
  mhGAP Intervention Guide v2.0

The GINA asthma report was left out because every page says "do not copy or distribute".
Some PDFs turn symbols such as >= into odd letters when the text is extracted, which can hurt
searches for exact numbers.
