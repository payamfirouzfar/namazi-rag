# Namazi Hospital Medical RAG

A question-answering app for medical staff and patients. You ask a question in English or Persian, it searches 20 public medical guidelines, and a local LLM writes a short answer **using only the passages it found**, with book and page citations. If the books do not contain the answer, it says "I don't know" instead of guessing.

Everything runs on our own server (Ollama), so no question leaves the network. This README shows what we built, how we tested it, and the numbers we measured.

## At a glance

- **Knowledge:** 20 public guidelines (WHO, KDIGO, GOLD), 7,112 chunks. For the patient questions, 25 public web pages from MedlinePlus, NHS, WHO, NIMH and NIDDK.
- **Search:** hybrid search (BM25 + bge-m3 embeddings, merged with Reciprocal Rank Fusion) and a reranker. The right passage is in the top 5 for **89%** of our 141 test questions (English 93%, Persian typed as is 84%).
- **Answers:** English cites the right passage for 52 of 58 questions, Persian for 41 of 49. All 34 questions the books cannot answer got "I don't know".
- **Patient questions:** 525 questions in English and Persian. About 7 in 10 get an answer that cites the right page, the rest get "I don't know".
- **Speed:** an English answer takes about 3.4 s on one A100. The search itself takes about 40 ms.
- **Many users:** the app handled 10,000 users in one test with no errors, about 20 questions per second. One GPU makes about 0.4 answers per second, so the LLM is the limit.
- **Cost:** no token price with the local model. The same 10,000 questions on a paid API would cost roughly USD 6 to 48.
- 121 tests, a GitHub Actions test run, Prometheus metrics, alerts and a Grafana dashboard.

## How it works

```
 books (PDF/TXT) --ingest--> chunks --> embeddings + BM25 index   (data/index)

 question --> embed --+--> dense search (meaning)  --+
 (Persian is          |                              +--> RRF merge --> best 20 --> reranker --> top 5 chunks
  translated first)   +--> BM25 search (exact words)-+                                                 |
                                                                                                       v
                      answer + citations  <--  LLM (answers only from the chunks)  <-- prompt
```

The main ideas: hybrid BM25 + vector search, sentence-aware chunks with overlap, a reranker for the best 20 chunks, a `NO_ANSWER` rule against made-up answers, an emergency line for urgent words, thumbs up/down feedback, conversation logging, and evaluation on questions we wrote ourselves, with the tuning set kept apart from the test set.

## Quick start (about 5 minutes, no downloads)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest                                   # 121 tests, a few seconds

# try it on the 4 tiny sample files with the offline test embedder
export EMBEDDING_MODEL=hash RERANK_MODEL=
python -m scripts.ingest --books data/sample
uvicorn app.main:app_factory --factory --port 8000  # needs an LLM, see LLM_* in .env.example
```

`walkthrough.ipynb` shows every step with its results and plots: the books, the chunks, dense / BM25 / hybrid search, the reranker, the evaluations, the answer checks, the timing and the load tests. The saved outputs are already inside. To run it yourself: `pip install -r requirements-dev.txt` and open it in VS Code or Jupyter (it needs the index from `python -m scripts.ingest`).

## Setup with your own books

```bash
cp .env.example .env            # set LLM_* and API_KEYS
# put the books in data/books/  (see docs/BOOKS.md)
python -m scripts.ingest        # downloads the embedding model (about 2 GB) on first run, the reranker on the first question
python -m scripts.tune          # find good settings on your own questions (docs/TUNING.md)
uvicorn app.main:app_factory --factory --host 127.0.0.1 --port 8000
```

Step by step setup, backups, systemd services, Prometheus / Grafana and nginx notes are in [docs/DEPLOY.md](docs/DEPLOY.md). The Docker files are included, but we did not run them on our test server (the Docker service was not running there).

## API

```bash
curl -X POST localhost:8000/v1/ask \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is the target INR for a patient on warfarin?"}'
```

```json
{
  "conversation_id": "0c52b878b7df4ebe88f61e0804cc547c",
  "answer": "The target INR is 2.0 to 3.0 for most indications [1].",
  "answered": true,
  "sources": [{"id": "warfarin_0", "book": "Harrison ...", "page": 123, "score": 0.032, "snippet": "..."}],
  "disclaimer": "Reference information from textbooks, not a diagnosis. ...",
  "latency_ms": 840
}
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | `{question, top_k?}` returns an answer and sources. `answered: false` means not found in the books. |
| `POST /v1/feedback` | `{conversation_id, feedback: 1 or -1, comment?}` thumbs up / down |
| `GET /` | a simple question page (works for Persian, with thumbs up/down) |
| `GET /health` | the process is alive |
| `GET /ready` | the index is loaded and questions can be answered (503 otherwise) |
| `GET /metrics` | numbers for Prometheus |

Errors: `401` bad API key, `422` bad input, `429` rate limit, `503` LLM or index unavailable.

## What is built in

| Concern | What the code does |
|---|---|
| Secrets | everything from environment / `.env`, nothing hard-coded (`.env` is git-ignored) |
| Authentication | `X-API-Key` header, constant-time comparison; the app refuses to start in secure mode without keys |
| Abuse | per-caller rate limit, question length limit, `top_k` limit |
| Privacy | `LOG_QUESTIONS=false` stores only a hash, no question or answer text |
| Made-up answers | answer only from numbered sources, `NO_ANSWER` rule (found anywhere in the reply), disclaimer in every response |
| Emergency | an urgent word (chest pain, trouble breathing, suicide ...) puts an emergency line first, in English and Persian |
| Broken text | a Persian answer with Chinese, Japanese or Korean letters is replaced by "I don't know" |
| Prompt injection | book text is marked as data, not instructions |
| Reliability | LLM timeout and retry with backoff, clean 503 when the LLM is down, a failed log write never fails an answer |
| Safe index | index written to a temp folder then swapped in; refuses to load an index built with a different embedding model |
| Observability | request id on every log line, timing, token counts, `/health`, `/ready`, `/metrics`, alerts |
| Data safety | all SQL parameterized, SQLite in WAL mode |
| Errors | no stack traces or internal addresses sent to clients |
| Tests | 121 tests: chunking, retrieval, index, RAG logic, every endpoint, auth, rate limit, LLM retry (against a local fake server), PDF loading, metrics, the evaluation scripts. A search-quality check on the sample questions fails the build if the hit rate drops. |

## How we tested it

We tested it three ways: our own question set on the guidelines, patient questions on public web pages, and load tests. All numbers below come from runs we made; the answers and tables are saved in `docs/`.

### 1. The guideline questions (141 questions)

141 questions we wrote ourselves in `data/eval/eval_questions.jsonl`: 76 English and 65 Persian, 107 answerable and 34 not answerable by the books. 36 of the Persian questions ask the same thing as an English one, so the two languages can be compared fact by fact. Every question has an evidence phrase, copied from one sentence of the books and checked against `data/index/chunks.jsonl`.

**How we measure it**

| What | Measure | Why |
|---|---|---|
| Search | hit@k (is the right passage in the top k?), MRR, nDCG@10 | the usual retrieval numbers. One right passage per question, so recall@k equals hit@k |
| Answers | right passage in the sources, right passage cited, share with a citation | did the answer use the right text? |
| Refusing | share of unanswerable questions that got an answer | the main number against made-up answers |
| Made-up facts | numbers in the answer that are in no source, citations to sources that do not exist, an LLM judge ("backed by the sources?") | cheap checks that need no person |
| Speed | search time, answer time (median, 95%), questions per second, tokens | latency and cost |

**Search** (98 tuning questions, 250-word chunks; Persian questions are searched as typed, without the translation step):

| Setting | Hit rate@5 | MRR |
|---|---|---|
| BM25 only (exact words) | 0.467 | 0.347 |
| dense only (meaning) | 0.747 | 0.595 |
| hybrid 1:1 | 0.747 | 0.594 |
| hybrid 2:1 | 0.760 | 0.598 |
| **hybrid 2:1 + reranker (used)** | **0.880** | **0.791** |

Over all 141 questions: 0.449 (BM25), 0.729 (dense), 0.766 (hybrid 2:1) and **0.888 (with the reranker)**. Our first embedding model (multilingual-e5-base) reached only 0.746 with the same hybrid search. The held-out test split (43 questions) gives 0.906 with the reranker.

**Search by language** (the app's search, all answerable questions):

| | hit@1 | hit@3 | hit@5 | hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| English (58) | 0.74 | 0.88 | **0.93** | 0.97 | 0.82 | 0.86 |
| Persian as typed (49) | 0.65 | 0.80 | **0.84** | 0.86 | 0.73 | 0.76 |
| Persian after the translation step (49) | | | **0.92** (45 of 49) | | | |

**Answers**, from the real LLM (`qwen2.5:32b-instruct` in Ollama, context 8192), every answer in [docs/answer_check.csv](docs/answer_check.csv). "First version" is our first setup (e5 embeddings, no reranker, longer prompt).

| What was checked | English, first version | English, final | Persian, first version | Persian, final |
|---|---|---|---|---|
| answerable questions | 58 | 58 | 49 | 49 |
| ...that got an answer | 57 | 58 | 41 | 48 |
| ...the right passage was among the sources sent to the LLM | 50 | **56** | 40 | **45** |
| ...the answer cited the right passage | 46 | **52** (90%) | 34 | **41** (84%) |
| unanswerable questions | 18 | 18 | 16 | 16 |
| ...correctly answered "I don't know" | 18 | **18** | 14 | **16** |
| answers with a number that is in no source | 0 | 0 | 0 | 0 |
| answers citing a source that does not exist | 0 | 0 | 0 | 0 |
| answers the LLM judge found not backed by the sources | 0 | 0 | 0 | 0 |
| median / average time per question | 2.8 s / 3.1 s | 3.4 s / 3.7 s | 4.0 s / 4.5 s | 4.8 s / 4.9 s |

On the 38 facts asked in both languages, every fact got an answer in both, and the right passage was cited 32 times in English and 31 times in Persian.

### 2. The patient questions (525 questions)

A second test with questions a patient would ask, in English and in Persian (`data/eval/patient_questions_english_persian_525.csv`): 500 questions about 25 common conditions (diabetes, asthma, high blood pressure, COPD, depression, kidney disease, migraine and others, 20 questions about each), and 25 questions paraphrased from questions asked on public Iranian patient Q&A sites (pregnancy and rheumatic diseases). They are answered from public web pages (MedlinePlus, NHS, WHO, NIMH, NIDDK) that `python -m scripts.fetch_sources` downloads into a **separate** knowledge base of 25 pages.

The csv names the web pages each question belongs to, but has no reference answers, so we measure whether the app answers or says "I don't know", whether it cites the named page, whether the numbers are in the sources, and what the LLM judge thinks. Flu, COVID and stroke pages could not be downloaded (the CDC site blocks scripts) and the NHS breast cancer page has no text, so those topics have no page. Their questions and the 50 forum-style questions are "open" questions: for most of them "I don't know" is the right answer. The page licences are not checked and the pages are not committed to git. The csv in this repo was rebuilt from the text of the original file (two forum links are shortened).

| 420 questions per language | English | Persian |
|---|---|---|
| answered | 297 (71%) | 314 (75%) |
| said "I don't know" | 123 (29%) | 106 (25%) |
| answers citing the expected page | 293 of 297 (99%) | 305 of 314 (97%) |
| answers with a number that is in no source | 1 | 2 |
| answers the LLM judge doubts | 20 (7%) | 84 (27%) |
| median time per question (three runs shared the GPU) | 5.1 s | 5.8 s |

- The 210 open questions got "I don't know" 187 times. The 23 answers came from nearby pages (for example the asthma page does talk about flu).
- The pages explain a disease well, but say little about follow-up (95% "I don't know"), a missed dose (86%) or how to ask about a test result (81%).
- For Persian the search is fine (the expected page reached the LLM for 415 of 420 questions); what is weaker is how the LLM writes Persian. The judge is the same model that wrote the answers and it reads Persian worse than English, so its 27% is a warning, not an exact number.

### 3. Speed and cost

`python -m scripts.time_steps` times every step of an answer (30 English and 30 Persian questions, one at a time, `docs/time_steps.csv`):

![One answer, step by step](docs/time_steps.png)

For an English question (3.4 s) reading the sources takes 49% of the time (about 1.65 s), writing the answer 33% (1.1 s) and the reranker 17% (0.6 s). Embedding, dense search and BM25 together take about 40 ms. A Persian question (4.9 s) also needs a translation call (0.9 s, 18%). The reranker costs 0.6 s but lifted the right passage in the top 5 from 77% to 89%.

Cost on a paid API for 10,000 questions, with 2,146 prompt and 52 answer tokens per question (prices from summary websites on 2026-10-08, sources in [docs/api_prices.csv](docs/api_prices.csv), check the official pages):

| Model | USD per 1,000 questions | USD per 10,000 questions |
|---|---|---|
| OpenAI GPT-5 mini | 0.64 | 6.40 |
| Claude Haiku 4.5 | 2.40 | 24.03 |
| OpenAI GPT-5 | 3.20 | 31.98 |
| Claude Sonnet 5 | 4.81 | 48.07 |

The local model has no token price. It costs a GPU server, about 7 hours of one A100 for 10,000 questions, and the questions never leave our network.

### 4. Load test: 10,000 users, and what happens when users grow

`python -m scripts.load_test` sends many questions at once and prints speed, answer times and failures. All runs hit a copy of the app on one A100 (rate limit off, its own database).

**The app alone** (a fake LLM that waits 1 s, `scripts/fake_llm.py`), 2,000 users each time:

| Users at the same moment | Answered | Questions per second | Answer time (median / 95%) |
|---|---|---|---|
| 10 | 2,000 | 9.5 | 1.1 s / 1.1 s |
| 50 | 2,000 | 22.2 | 2.2 s / 2.7 s |
| 100 | 2,000 | 21.5 | 4.5 s / 5.5 s |
| 200 | 1,995 | 19.8 | 9.4 s / 17.7 s |
| 500 | 1,991 | 19.4 | 25.5 s / 32.9 s |
| 1,000 | 1,999 | 19.3 | 49.9 s / 53.8 s |
| **10,000 users in one run** (100 at a time) | **10,000** | 23.4 | 4.1 s / 5.5 s |

The app does about 20 questions per second. More users do not make it faster, they only wait longer (the median wait is about users ÷ 20). From 200 users at the same moment, 1 to 9 of 2,000 requests were dropped with a `ReadError` (the app logged no error; probably a connection closed while the client reused it, not proven). The first 10,000-user run also found a real bug (a progress bar crashed when many questions were embedded at once); it is fixed and has a test.

**With the real model** (qwen2.5:32b, 4 questions in parallel in Ollama):

| Users at the same moment | Users | Answered | Questions per second | Answer time (median / 95%) |
|---|---|---|---|---|
| 8 | 400 | 400 | 0.4 | 17.8 s / 31.6 s |
| 32 | 128 | 128 | 0.4 | 75 s / 91 s |
| 128 | 256 | 256 | 0.4 | 226 s / 442 s |

One A100 makes about 0.4 answers per second (about 1,400 per hour), so the waiting time is roughly **users at the same moment ÷ 0.4 seconds**. Nothing failed, but with 128 users at once the slowest waited 7 minutes. 10,000 real questions would take about 7 hours (an estimate from these runs; we did not run all 10,000 with the real model). More users need more GPUs, a smaller model, or a faster model server such as vLLM. These runs were made before the search upgrade; the answer is now a bit slower (the reranker) but shorter.

## What we tuned and what we tried

We tried each idea on the same questions and kept only what helped.

| Try | Result | Kept? |
|---|---|---|
| embedding model multilingual-e5-base (first choice) | right passage in the top 5: English 0.85, Persian typed 0.31 | no |
| embedding model multilingual-e5-large | 0.79 / 0.55: worse for English | no |
| **embedding model bge-m3** | **0.90 / 0.61**, MRR also better | **yes** |
| dense:BM25 weight 1:1, 2:1, 3:1 with bge-m3 | all within a few points | 2:1 |
| **reranker (bge-reranker-v2-m3, best 20 chunks)** | **0.93 / 0.84**, MRR 0.82 / 0.73 | **yes** |
| chunks of 150, 200 or 250 words, with the reranker | all within noise | 250 |
| remove repeated page headers / join words the PDF broke / book title in every chunk | no clear gain | no |
| a bigger LLM (3B to 32B) | right passage cited 13 to 16 of 21 on the first 27 questions | yes |
| **prompt: "start with the direct answer, 2 to 4 short sentences, simple words"** | same quality, answers about 27% shorter | **yes** |
| prompt that answers part of a question and says what is missing | right page cited less often (86 against 92 of 107) | no |
| a line "write the answer in Persian/English" at the end of the message | needed by other models; Qwen already did it | yes |
| Gemma 3 12B instead of Qwen 32B | 1.4 times faster, same Persian citations, but it made up 3 of 18 English unanswerable answers | no |
| a similarity cutoff (`MIN_DENSE_SCORE`) | answerable and unanswerable questions overlap (0.44 to 0.81 against 0.45 to 0.61) | no |

The tuner (`python -m scripts.tune`, 33 settings, table in [docs/tuning_results.csv](docs/tuning_results.csv)) prefers 150-word chunks without the reranker, but with the reranker the sizes are within noise, so we kept 250. Every setting is explained in [docs/TUNING.md](docs/TUNING.md).

Problems the tests found, and what we did:
- **"NO_ANSWER" shown to the user.** In 10 English answers the model explained first and wrote the word at the end, and the app only looked at the start of the reply. Now a reply that contains it anywhere becomes "I don't know" (with a test).
- **Wrong "numbers in no source" count.** The check counted the 1. 2. 3. of numbered lists as facts. Fixed, with a test.
- **Broken Persian.** 13 answers had Chinese, Japanese or Korean letters. The app now hides such answers behind "I don't know".
- **Emergency questions.** Urgent words now put an emergency line first (115 in Iran). The word list is short on purpose and should be reviewed by a clinician.
- **A column mix-up** in the patient csv (the links sit under "Provenance"); the script handles it. **A wrong label** in our first question set: the books do give "INR 2-3" for warfarin (KDIGO 2024, page 129, for patients with atrial fibrillation), so that question is now answerable.

## Monitoring

The app shows its numbers at `/metrics`. Prometheus collects them and Grafana draws them ([docs/prometheus.yml](docs/prometheus.yml), [docs/grafana_dashboard.json](docs/grafana_dashboard.json)): questions per second, median and 95% answer time, errors, share of "not found", tokens per minute, questions in progress, and what the tokens would cost on a paid API. [docs/alerts.yml](docs/alerts.yml) has four alerts (app down, errors on the rise, slow answers, too many "not found"). Both tools listen on 127.0.0.1 only.

![Prometheus data of the load tests](docs/grafana_results.png)

The picture is drawn with matplotlib from the same Prometheus data and queries as the Grafana panels (the server has no browser for a screenshot). Top row: the app alone with the fake LLM, from 10 up to 1,000 users at the same moment. Bottom row: the real model with 32 users at the same moment.

## What we know it does not cover yet

- **No clinician has read the answers.** `docs/doctor_review.csv` (`python -m scripts.review_sample`) holds 100 answers with empty columns for "correct?" and "safe for a patient?". Until it is filled in, we only know that the answers follow the sources, not that a doctor agrees.
- **Persian wording** is weaker than English: the judge doubts 27% of the Persian patient answers (English 7%), and the Persian text can be awkward. Read a Persian answer next to its English source.
- **The test sets are small and ours.** We wrote the 141 guideline questions from the book sentences, which helps exact-word search. "No unanswerable question answered" comes from 34 questions and a judge that is the same model as the answerer.
- **The knowledge base has gaps:** flu, COVID, stroke and breast cancer pages, and little on follow-up, missed doses and test results.
- **Capacity:** one GPU makes about 0.4 answers per second, so 20 users asking at the same moment wait about 50 seconds.
- **Not set up or not checked:** HTTPS (nginx steps are in docs/DEPLOY.md, untested), user accounts (there are only API keys), a privacy policy for patient questions, the licence of every web page, and the rules for medical software.
- **Answers are reference text from books, not medical advice.**

## Project layout

```
app/        main.py (API) · rag.py (prompt + answer logic) · retriever.py (hybrid search + RRF + reranker)
            index.py · chunking.py · text.py · loader.py · embedder.py · llm.py · metrics.py
            db.py · security.py · schemas.py · config.py · logger.py
scripts/    ingest.py (build the index) · tune.py (find good settings)
            check_answers.py (test the answers with the real LLM) · report.py (usage report)
            load_test.py (many users at once) · fake_llm.py (a fake model for load tests)
            time_steps.py (time every step of an answer) · fetch_sources.py (download the pages for the patient questions)
            review_sample.py (a sheet for doctors)
tests/      pytest suite (no downloads, no network)
walkthrough.ipynb   step by step run of everything with the results
data/       books/ · sample/ · eval/ (questions) · index/ (generated)
docs/       BOOKS.md (which books) · TUNING.md (every setting explained) · DEPLOY.md (setup)
            prometheus.yml · alerts.yml · grafana_dashboard.json · grafana_results.png · time_steps.png
            answer_check.csv · patient_check_*.csv · tuning_results.csv · time_steps.csv · load_test_results.csv · api_prices.csv · doctor_review.csv
```
