# Parameter tuning

Every number below is a setting in `.env`. The defaults are **starting points taken from the
four reference projects, not results measured on Namazi's books.** To get real values, run the
tuning script on the real library (see the bottom).

## What each parameter does

| Setting | Default | What it changes | Reference projects used |
|---|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 250 / 50 words | Small chunks = precise search but little context for the LLM. Big chunks = more context but a diluted match. Overlap stops a fact on a boundary from being cut in half. Needs a re-ingest. | medical-rag-qa: 512 tokens / 100 overlap (about 380 / 75 words). MedRAG: 1000 chars / 200 overlap (about 150 / 30 words). |
| `RRF_K` | 60 | How strongly the top ranks are favoured when the two result lists are merged. Smaller = the top result of either list wins more often. | 60 in medical_assistant_rag and medical-rag-qa, 100 in MedRAG. |
| `WEIGHT_DENSE` / `WEIGHT_BM25` | 1 / 1 | How much each retriever counts. BM25 is best for exact drug names, numbers and abbreviations; embeddings are best for meaning and for Persian-to-English. | medical-rag-qa tuned these on a validation set and found BM25 deserved more weight on drug and lab-value questions. |
| `TOP_K` | 5 | Chunks sent to the LLM. More = better recall, more tokens, more noise. | medical_assistant_rag: 5. MedRAG: 32 (needs a large-context model). |
| `OVERSAMPLE` | 3 | Each retriever fetches `TOP_K x 3` candidates before merging. | medical-rag-qa: 3x. |
| `BM25_K1` / `BM25_B` | 1.2 / 0.75 | Standard BM25 curve. Rarely worth touching. | medical-rag-qa: 1.2 / 0.75. |
| `MIN_DENSE_SCORE` | 0 (off) | If even the best chunk is less similar than this, skip the LLM and say "not found". Depends on the embedding model, so it must be calibrated. | medical-rag-qa: explicit `NO_ANSWER` behaviour. |
| `MAX_CONTEXT_CHARS` | 12000 | Hard cap on the text sent to the LLM. Lower it for a small-context local model. | MedRAG cut the context to the model's limit. |
| `LLM_TEMPERATURE` | 0 | 0 gives the same answer to the same question, which is what you want here. | MedRAG: 0. |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | The biggest quality lever. Needs a re-ingest. | See the list below. |

## Embedding models to compare

Run the tuning script once per model (set `EMBEDDING_MODEL`, and the two prefixes if the model
needs them) and compare the test numbers.

| Model | Prefixes | Notes |
|---|---|---|
| `intfloat/multilingual-e5-base` (default) | `query: ` / `passage: ` | Good balance, handles Persian. |
| `intfloat/multilingual-e5-large` | `query: ` / `passage: ` | Better and slower. |
| `BAAI/bge-m3` | none | Strong multilingual, larger. |
| `paraphrase-multilingual-MiniLM-L12-v2` | none | Small and fast. Used by DoctorRAG. |
| `pritamdeka/S-PubMedBert-MS-MARCO` | none | English biomedical only. Used by medical-rag-qa. |

I could not download any of these in my build environment, so none has been run on real data
yet. MedCPT (used by MedRAG) needs two different encoders for questions and passages, which this
code does not support, to keep it simple.

## Lessons from the reference projects

- **Hybrid is not automatically better.** In medical-rag-qa a fixed-weight hybrid scored below
  plain BM25 on drug and number questions. That is why the grid also tries dense-only `(1,0)`
  and BM25-only `(0,1)`.
- **Gains from fine tuning can be small.** medical_assistant_rag reported MRR 90.60% before and
  90.81% after tuning its boosts. Chunking and book quality usually matter more.
- **Tune on one set, report on another.** medical-rag-qa selected weights on a validation set
  only and kept the test set untouched. `scripts/tune.py` does the same (70% dev / 30% test).

## How to tune (about an hour of work, mostly writing questions)

1. Put the books in `data/books/`.
2. Write 50 to 200 real questions in `data/eval/eval_questions.jsonl`, one JSON per line:
   ```json
   {"question": "What INR is targeted for atrial fibrillation?", "evidence": "the target inr is 2.0 to 3.0"}
   {"question": "What is the capital of France?", "evidence": null}
   ```
   `evidence` is a short phrase copied from inside **one sentence** of the book, so chunking can
   never split it. Use `null` for questions the books cannot answer (include about 20% of these,
   they are used to set `MIN_DENSE_SCORE`). Include Persian questions if clinicians will ask them.
   Ask clinicians for the questions, not the developers.
3. Run `python -m scripts.tune`. Use `--quick` first if the library is big (every chunk size
   re-embeds all the books).
4. Copy the printed lines into `.env`, then `python -m scripts.ingest` and restart.

The script prints the dev score of the best setting, then the test score of that one setting.
Trust the test score. It also writes every setting it tried to `data/tuning/results.csv`.

## Your results (fill in after running the script on the real books)

| Date | Books | Embedding model | Chunk / overlap | Weights dense:bm25 | RRF k | Test hit rate | Test MRR |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
