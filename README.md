# Namazi Hospital Medical RAG

A question-answering backend for medical staff. You ask a question (English or Persian), it
searches the hospital's medical books, and an LLM writes a short answer **using only the
passages it found**, with book and page citations. If the books do not contain the answer, it
says so instead of guessing.

## Status: what is real and what is not

I built this for a hospital pilot. It has **not** been used in a hospital: no staff, no patients and no real logs exist, and
I could not have shared them anyway. What is real: the code, the 121 tests, the evaluation on 20 public guidelines (results below),
the load tests, the monitoring and the deployment steps I ran myself on a GPU server. What is not: any real-world usage.
The load tests use made-up traffic (the 76 English eval questions, asked again and again). Answers are reference text from books,
not medical advice.

## At a glance

- Answers questions from 20 public medical guidelines, in English and Persian, with book and page citations.
- **Search:** the embedding model bge-m3 plus a reranker put the right passage in the top 5 for **89%** of my 141 questions (English 93%, Persian as typed 84%). The first version had 75%.
- **Answers:** English cites the right passage for 52 of 58 questions, Persian for 41 of 49. All 34 questions the books cannot answer got "I don't know".
- **Speed:** an English answer takes about 3.4 s on one A100 (reading the sources 49%, writing the answer 33%, the reranker 17%). The search itself takes about 40 ms.
- **Many users:** the app alone took 10,000 users with no errors (about 20 questions per second). The real limit is the model: one GPU makes about 0.4 answers per second.
- **Patient questions:** 525 questions in English and Persian, answered from public web pages. About 7 in 10 get an answer that cites the right page, the rest get an honest "I don't know". Persian is weaker (the judge doubts 27% of its answers, English 7%).
- **Cost:** no token price with the local model. The same 10,000 questions on a paid API would cost roughly USD 6 to 48.

## How it works

```
 books (PDF/TXT) --ingest--> chunks --> embeddings + BM25 index   (data/index)

 question --> embed --+--> dense search (meaning)  --+
                      |                              +--> RRF merge --> top 5 chunks
                      +--> BM25 search (exact words)-+                      |
                                                                            v
        answer + citations  <--  LLM (answers only from the chunks)  <-- prompt
```

The main ideas: hybrid BM25 + vector search merged with Reciprocal Rank Fusion, sentence-aware chunks with overlap,
a `NO_ANSWER` rule against made-up answers, thumbs up/down feedback, conversation logging, hit rate / MRR evaluation,
and tuning on a dev set that is kept apart from the test set.

## Quick start (about 5 minutes, no downloads)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest                                   # 121 tests, a few seconds

# try it on the 4 tiny sample files with the offline test embedder
export EMBEDDING_MODEL=hash
python -m scripts.ingest --books data/sample
uvicorn app.main:app_factory --factory --port 8000  # needs an LLM, see LLM_* in .env.example
```

## See every step with its results

`walkthrough.ipynb` loads the books, shows the chunks, compares dense / BM25 / hybrid search,
runs the evaluation, shows where it fails, tries the answer step with a fake LLM and runs the
tests. The saved outputs are already inside the notebook. To run it yourself:
`pip install -r requirements-dev.txt` and open it in VS Code or Jupyter (it needs the index from
`python -m scripts.ingest`).

## Real use

```bash
cp .env.example .env            # set LLM_*, API_KEYS, ENV=production
# put licensed books in data/books/  (see docs/BOOKS.md)
python -m scripts.ingest        # downloads the embedding model (about 2 GB) on first run; the reranker downloads on the first question
python -m scripts.tune          # find good settings on your own questions (docs/TUNING.md)
docker compose up -d --build    # not tested, see the notes at the end. Or: gunicorn "app.main:app_factory()" -k uvicorn.workers.UvicornWorker
```

Ingest inside Docker: `docker compose run --rm api python -m scripts.ingest`
(the `data/` folder must be writable by uid 1000: `sudo chown -R 1000:1000 data`).

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
| `GET /health` | process is alive (for the load balancer) |
| `GET /ready` | index loaded, ready to answer (503 otherwise) |
| `GET /docs` | interactive docs, only when `ENV` is not `production` |

Errors: `401` bad API key, `422` bad input, `429` rate limit, `503` LLM or index unavailable.

## Results on 20 free medical books

I ran the whole pipeline on 20 freely downloadable guidelines (WHO, KDIGO, GOLD; list in `docs/BOOKS.md`): 7,112 chunks of 250 words,
`bge-m3` embeddings, a reranker, and 141 hand-written questions in `data/eval/eval_questions.jsonl`:
76 English and 65 Persian, 107 answerable and 34 not answerable by the books. 36 of the Persian questions ask the same thing as an English one,
so the languages can be compared fact by fact. Every evidence phrase is copied from one sentence of the books and was checked against `data/index/chunks.jsonl`.
Everything below is also in `walkthrough.ipynb` with plots (sections 11 to 20).

**How I measure it**

| What | Measure | Why |
|---|---|---|
| Search | hit@k (is the right passage in the top k?), MRR, nDCG@10 | the usual retrieval numbers. One right passage per question, so recall@k equals hit@k |
| Answers | right passage in the sources, right passage cited, share with a citation | did the answer use the right text? |
| Refusing | share of unanswerable questions that got an answer | the main hallucination number |
| Made-up facts | numbers in the answer that are in no source, citations to sources that do not exist, an LLM judge ("backed by the sources?") | the three cheap checks I could run without a person |
| Speed | search time, answer time (median, 95%), questions per second, tokens | latency and cost |

**Search only** (the 98 dev questions, 250-word chunks, RRF constant 60; the Persian questions are searched as typed, without the translation step):

| Setting | Dev hit rate@5 | Dev MRR |
|---|---|---|
| BM25 only (exact words) | 0.467 | 0.347 |
| dense only (meaning) | 0.747 | 0.595 |
| hybrid 1:1 | 0.747 | 0.594 |
| hybrid 2:1 | 0.760 | 0.598 |
| **hybrid 2:1 + reranker (used)** | **0.880** | **0.791** |

Over all 141 questions the numbers are 0.449 (BM25), 0.729 (dense), 0.766 (hybrid 2:1) and **0.888 (with the reranker)**. With the first embedding model (multilingual-e5-base) hybrid 2:1 reached only 0.746.

**What I tried to make the search better, on the 107 answerable staff questions** (right passage in the top 5, English / Persian as typed / Persian translated):

| Try | Result | Kept? |
|---|---|---|
| embedding model multilingual-e5-base (first choice) | 0.85 / 0.31 / 0.82 | no |
| embedding model multilingual-e5-large | 0.79 / 0.55 / 0.88 | no, worse for English |
| **embedding model bge-m3** | **0.90 / 0.61 / 0.82** (MRR also better) | **yes** |
| dense:BM25 weight 1:1, 2:1, 3:1 with bge-m3 | all within a few points | 2:1 stays |
| **+ reranker (bge-reranker-v2-m3, best 20 chunks)** | **0.93 / 0.84 / 0.90**, MRR 0.82 / 0.73 / 0.78 | **yes**, costs about 0.6 s per question |
| chunks of 150, 200 or 250 words, with the reranker | all within noise (English 0.90, 0.97, 0.91) | 250 stays |
| remove repeated page headers / join words that the PDF broke / book title in every chunk | no clear gain | no |

The tuner (`scripts.tune`, no reranker) now prefers 150-word chunks (dev hit rate 0.81 against 0.76), but once the reranker is on the three sizes are within noise, so I kept 250 and the answers I measured.
Held-out test split (43 questions, 32 answerable): hit rate@5 0.906 with the reranker. It is small, so read it as a rough number.

**Search by language** (the app's search: hybrid 2:1 + reranker; all answerable questions):

| | hit@1 | hit@3 | hit@5 | hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| English (58) | 0.74 | 0.88 | **0.93** | 0.97 | 0.82 | 0.86 |
| Persian as typed (49) | 0.65 | 0.80 | **0.84** | 0.86 | 0.73 | 0.76 |
| Persian after the translation step (49) | | | **0.92** (45 of 49) | | | |

- Many KDIGO guidelines repeat the same recommendation in a summary and in a chapter, so the right topic is found but not always the exact passage I marked.
- The translation step (the LLM translates a Persian question to English first) still helps: 84% as typed, 92% translated.
- The score cutoff `MIN_DENSE_SCORE` does not separate answerable from unanswerable questions (answerable go down to 0.44, unanswerable up to 0.61), so it stays off and the `NO_ANSWER` prompt rule does that job.
- 141 questions is still small, and I wrote them myself from book sentences, which favours exact-word search. A real evaluation needs a few hundred, written by clinicians.

## What is in here for production

| Concern | What the code does |
|---|---|
| Secrets | everything from environment / `.env`, nothing hard-coded (`.env` is git-ignored) |
| Authentication | `X-API-Key` header, constant-time comparison; the app refuses to start in production without keys |
| Abuse | per-caller rate limit, question length limit, `top_k` limit |
| Privacy | `LOG_QUESTIONS=false` stores only a hash, no question or answer text |
| Hallucination | answer only from numbered sources, `NO_ANSWER` rule, optional similarity threshold, disclaimer in every response |
| Prompt injection | book text is marked as data, not instructions |
| Reliability | LLM timeout and retry with backoff (5xx, timeouts, rate limits only), clean 503 when the LLM is down, a failed log write never fails an answer |
| Safe index | index written to a temp folder then swapped in; refuses to load an index built with a different embedding model |
| Observability | request id on every log line and response header, timing, token counts, `/health` and `/ready` |
| Data safety | all SQL parameterized, SQLite in WAL mode |
| Errors | no stack traces or internal addresses sent to clients |
| Deployment | non-root Docker user, healthcheck, `/docs` hidden in production |
| Tests | 121 tests: chunking, retrieval maths, index, RAG logic, every endpoint, auth, rate limit, LLM retry (against a local fake server), PDF loading |

## Made for a hospital without cloud

The language model runs inside the hospital (Ollama), so no question or patient detail leaves the network.
Extras for real use: a simple question page at `/` (works for Persian, with thumbs up/down),
Persian questions are translated to English for the search (the LLM does this, so the answer check below includes it), `python -m scripts.report` shows
usage and unanswered questions, and `python -m scripts.check_answers` runs the questions through the
real LLM and checks the answers, not only the search. Step by step setup, including offline
installation and backups: [docs/DEPLOY.md](docs/DEPLOY.md).

### Answer check with a real local model

`qwen2.5:32b-instruct` (Ollama, context 8192) on one A100 40 GB, all 141 questions, every answer in [docs/answer_check.csv](docs/answer_check.csv)
(`python -m scripts.check_answers --judge`), with the final search and prompt. Two runs with the same settings are not identical (about 3 in 4 answers are word for word the same,
the count of right citations moved by 1 or 2 between my runs), so ignore differences of one or two questions. "Before" is the first version (e5 embeddings, no reranker, longer prompt).

| What was checked | English before | English now | Persian before | Persian now |
|---|---|---|---|---|
| answerable questions | 58 | 58 | 49 | 49 |
| ...that got an answer | 57 | 58 | 41 | 48 |
| ...the right passage was among the sources sent to the LLM | 50 | **56** | 40 | **45** |
| ...the answer cited the right passage | 46 | **52** (90%) | 34 | **41** (84%) |
| unanswerable questions | 18 | 18 | 16 | 16 |
| ...correctly answered "not found" | 18 | **18** | 14 | **16** |
| answers with a number that is in no source | 0 | 0 | 0 | 0 |
| answers citing a source that does not exist | 0 | 0 | 0 | 0 |
| answers the LLM judge found not backed by the sources | 0 | 0 | 0 | 0 |
| median / average time per question | 2.8 s / 3.1 s | 3.4 s / 3.7 s | 4.0 s / 4.5 s | 4.8 s / 4.9 s |

On the 38 facts that were asked in both languages every fact got an answer in both languages, and the right passage was cited 32 times in English and 31 times in Persian (before: 31 and 26).

**What is still not good, as it is:**
- **The judge and the cheap checks are weak instruments.** All of them found nothing, but a made-up answer can repeat a real text about another topic and pass them (that is what the two invented Persian answers of the first version did: eczema from a scabies passage, snake bite with broken words). The judge is the same model as the answerer. I checked it with made-up wrong answers (it flagged 2 of 2), but it is a rough signal. 34 unanswerable questions is a small test, so "0 made up" is not a guarantee.
- **Persian wording** can be awkward, and the model sometimes mixes in odd words (the first version even wrote Chinese or Korean letters, which the app now hides). A Persian answer must be read next to its English source.
- The right passage was cited for 93 of 107 answerable questions (87%); 101 of 107 had it in the sources. For the others the model used a different passage that often says nearly the same, but a doctor has to confirm that.
- The old README said the model made up an INR answer for warfarin. That label was wrong: the books give "INR 2-3" for warfarin in a KDIGO 2024 table for CKD patients with atrial fibrillation (page 129) and the answer cited it. I changed the evidence of that question. The answer is right for that patient group only, and it does not say so.

**What I tried on the prompt and the model, on the same questions (cited the right passage / made-up answers):**

| Try | Result | Kept? |
|---|---|---|
| (a) bigger model, 3B to 32B | better citations (13 to 16 of 21 on the first 27 questions) | yes |
| (b) a stricter sentence about "a different disease or patient group" | no change on the first 93 questions | no |
| (c) `MIN_DENSE_SCORE` | not applied: the similarities of answerable and unanswerable questions overlap | no |
| a prompt that answers part of a question and says what is missing | cited the right page less often (86 against 92 of 107) | no |
| **"start with the direct answer, 2 to 4 short sentences, simple words"** | same quality (92 and 1 made up), answers about 27% shorter | **yes** |
| "write the answer in Persian/English" at the end of the message | needed by other models; Qwen already did it | yes, harmless |
| Gemma 3 12B instead of Qwen 32B (Persian questions) | 41 cited (Qwen 41) and about 1.4 times faster (3.6 s against 4.9 s per Persian question), but on the English questions it made up 3 of 18 unanswerable answers | no |

### Patient questions

A second test, closer to a real product: 525 questions that patients might ask, in English and in Persian (`data/eval/patient_questions_english_persian_525.csv`),
answered from public web pages (MedlinePlus, NHS, WHO, NIMH, NIDDK) that I downloaded with `python -m scripts.fetch_sources` into a **separate** knowledge base of 25 pages.
The csv has the questions and the pages they belong to, but **no reference answers**, so I can't say "this answer is right". I measure whether it answers or says "I don't know",
whether it cites the named page, whether the numbers are in the sources, and what the LLM judge thinks. Q001 to Q500 are 25 diseases times the same 20 question types, so this tests topic
coverage, not 500 different patients. Flu, COVID and stroke pages could not be downloaded (CDC blocks scripts) and the NHS breast cancer page has no text, so those topics have no page.
Their questions and 50 forum-style questions are the "open" questions, where "I don't know" is mostly the right answer. The page licences are not checked and the pages are not committed.
The csv in the repo was rebuilt from the text of the original file (the two forum links are shortened); replace it with the original if you have it.

| 420 questions per language | English | Persian |
|---|---|---|
| answered | 297 (71%) | 314 (75%) |
| said "I don't know" | 123 (29%) | 106 (25%) |
| answers citing the expected page | 293 of 297 (99%) | 305 of 314 (97%) |
| answers with a number that is in no source | 1 | 2 |
| answers the LLM judge doubts | 20 (7%) | **84 (27%)** |
| median time per question (three runs shared the GPU) | 5.1 s | 5.8 s |

The 210 open questions got "I don't know" 187 times; the 23 answers came from nearby pages, for example the asthma page does talk about flu.
The web pages explain a disease well but say little about follow-up (95% "I don't know"), a missed dose (86%) or how to ask about a test result (81%). That is honest but thin.
Persian is the weak part: the expected page reached the LLM for 415 of 420 questions, so the search is fine, but the LLM's Persian is not, and the judge (the same model, reading Persian worse than English) doubts 27%.
Before the search and prompt upgrade the English numbers were 68% answered and 9% doubted, the Persian 76% answered and 23% doubted: the upgrade helped English and made no difference for Persian.

**What I fixed because of this test:** 10 English answers ended with the literal word `NO_ANSWER` (the model explained first, and the app only looked at the start of the reply). Now a reply that contains it anywhere
becomes "I don't know". I also added an emergency line for urgent words (chest pain, trouble breathing, suicide ...), a guard that shows "I don't know" instead of a Persian answer with Chinese, Japanese or Korean letters,
Prometheus alerts (`docs/alerts.yml`) and a sheet for doctors (`python -m scripts.review_sample`, 100 answers, `docs/doctor_review.csv`).

### Where the time goes

`python -m scripts.time_steps` times every step of an answer (30 English and 30 Persian questions, one at a time, `docs/time_steps.csv`, plots in the notebook):

![One answer, step by step](docs/time_steps.png)

For an English question (3.4 s) reading the sources takes 49% of the time (about 1.65 s), writing the answer 33% (about 1.1 s) and the reranker 17% (about 0.6 s).
Embedding, dense search and BM25 together take about 40 ms, around 1%. A Persian question (4.9 s) also needs a translation call (0.9 s, 18%).
**So the LLM is still what to optimize**, in this order: shorter chunks to shorten the prompt (the tuner prefers 150 words, and with the reranker the search found the right passage almost as often, but I did not test the answers),
a faster or smaller model or server (writing), a smaller reranker or 10 candidates instead of 20 (about half of the 0.6 s, not tested), a cheaper translation.
The reranker costs 0.6 s but lifted the right passage in the top 5 from 77% to 89%, which is worth it.

### Speed and cost

- **Search alone:** embedding, dense search and BM25 take about 40 ms; with the merge and the reranker the whole search takes about 0.6 s. **A whole answer:** about 3.4 s (English median). The LLM is most of it.
- **Tokens:** about 2,146 prompt and 52 answer tokens per question.
- **Cost on a paid API for 10,000 questions** (prices from summary websites on 2026-10-08, listed with their sources in [docs/api_prices.csv](docs/api_prices.csv), please check the official pages):

| Model | USD per 1,000 questions | USD per 10,000 questions |
|---|---|---|
| OpenAI GPT-5 mini | 0.64 | 6.40 |
| Claude Haiku 4.5 | 2.40 | 24.03 |
| OpenAI GPT-5 | 3.20 | 31.98 |
| Claude Sonnet 5 | 4.81 | 48.07 |

  The local model has no token price. It costs a GPU server, about 7 hours of one A100 for 10,000 questions. A paid API would send the questions out of the hospital network, which this project was built to avoid.
  Persian questions need one more small call for the translation, and the judge calls are not counted.

### Load test: 10,000 users, and what happens when users grow

`python -m scripts.load_test` sends many questions at once and prints speed, answer times and failures; `--save` adds the result to [docs/load_test_results.csv](docs/load_test_results.csv).
All runs hit a copy of the app on one A100 (rate limit off, its own database). They were made before the search and prompt upgrade; the answer is now a bit slower (the reranker) but shorter, so the 0.4 answers per second is a good estimate.

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

The app does about 20 questions per second. More users do not make it faster, they only wait longer (median wait is about users ÷ 20). From 200 users at the same moment, 1 to 9 of 2,000 requests
were dropped with a `ReadError` (the app logged no error; probably a connection closed while the client reused it, not proven). This is the test of the app, not of the model.

**With the real model** (qwen2.5:32b, 4 questions in parallel in Ollama):

| Users at the same moment | Users | Answered | Questions per second | Answer time (median / 95%) |
|---|---|---|---|---|
| 8 | 400 | 400 | 0.4 | 17.8 s / 31.6 s |
| 32 | 128 | 128 | 0.4 | 75 s / 91 s |
| 128 | 256 | 256 | 0.4 | 226 s / 442 s |

One A100 makes about 0.4 answers per second (about 1,400 per hour), so the waiting time is roughly **users at the same moment ÷ 0.4 seconds**. Nothing failed, but with 128 users at once the slowest waited 7 minutes.
10,000 real questions would take about 7 hours (an estimate from these runs, I did not run all 10,000 with the real model). To serve more users: more GPUs (the plot in the notebook calculates 2 and 4), a smaller model,
or a faster model server such as vLLM. The first run of the 10,000 test found a real bug (a progress bar crashed when many questions were embedded at once; 2 of 10,000 requests failed). It is fixed and has a test.

## Monitoring (Prometheus and Grafana)

The app shows its numbers at `/metrics`: questions, "not found", errors, answer time, tokens. Prometheus collects them and Grafana draws them
([docs/prometheus.yml](docs/prometheus.yml) and [docs/grafana_dashboard.json](docs/grafana_dashboard.json)). The dashboard has seven panels:
questions per second, median and 95% answer time, errors, share of "not found", tokens per minute, questions in progress, and what the tokens would cost on a paid API.
Setup is in docs/DEPLOY.md. Both tools listen on 127.0.0.1 only.

![Prometheus data of the load tests](docs/grafana_results.png)

I could not take a real screenshot because the server has no browser. This picture is drawn with matplotlib from the same Prometheus data and the same queries that the Grafana panels use.
Top row: the app alone with the fake LLM, from 10 up to 1,000 users at the same moment. The answer time grows with the number of users, and the right panel shows the users who are waiting inside the app.
Bottom row: the real model with 32 users at the same moment, about 0.4 questions per second the whole time. The answer-time lines are rounded to the app's time buckets (one bucket covers 60 to 120 s),
so they are approximate; the exact numbers are in the load test tables below. The app times a question from the moment the request arrives, so waiting for a free worker counts too.

The last dashboard panel multiplies the tokens by two prices you type in. The prices in it are examples.

## Tests on every push

`.github/workflows/tests.yml` installs the requirements and runs `pytest` on every push and pull request. The first run on GitHub passed (56 seconds).
The tests check the code. One of them also checks the search quality: it runs the sample questions with the offline embedder and fails if the hit rate drops,
so a change that makes the search worse turns the build red. The real evaluation (141 questions, real model) needs a GPU and the books, so I run it by hand.

## Is it production ready?

No. The code side is in good shape, the rest needs people:

| Part | State |
|---|---|
| Tests, CI, metrics, dashboard, alerts, backups, restart after a crash | done and tested |
| "I don't know" and citations | done: no unanswerable staff question was answered, 99% (English) and 97% (Persian) of the patient answers cite the expected page |
| Search quality | much better: right passage in the top 5 for 89% of the questions (was 75%) |
| Emergency line, broken-text guard | done, a simple word list that a clinician must review |
| A doctor reads the answers | **not done**: `docs/doctor_review.csv` is ready, someone has to fill it in |
| Persian quality | **still weaker**: the judge doubts 27% of the Persian patient answers (English 7%); the LLM's Persian is the weak part |
| Missing pages (flu, COVID, stroke, breast cancer), thin topics (follow-up, missed dose) | **not done** |
| Page licences | **not checked** |
| HTTPS, user accounts, privacy policy for patient questions | **not done**: nginx steps are in docs/DEPLOY.md, untested |
| Capacity | one GPU gives about 0.4 answers per second, 20 users at the same moment wait about 50 s |
| Medical-device rules | **not checked**: health software can be regulated, ask a lawyer |

## Before going live (things code cannot do for you)

1. **Clinical validation.** Have clinicians review a few hundred real answers. Use the thumbs-down data.
2. **HTTPS.** Put nginx or similar in front. This service speaks plain HTTP.
3. **Patient data policy.** Decide with the hospital whether questions may contain patient details. If the LLM is an external service, they leave the hospital network. A local OpenAI-compatible server (Ollama, vLLM) avoids that.
4. **Licences.** Make sure the hospital may index the books it uses.
5. **Backups.** Back up `data/app.db` (feedback and logs). The index can always be rebuilt.
6. **Docker was not tested.** On the pilot server there was no running Docker daemon, so `docker compose up` was never run. The app and the LLM were run and tested as `systemd --user` services instead (docs/DEPLOY.md).
7. **Limits to know about.** The rate limiter and SQLite are per machine; for several servers use a gateway limit and Postgres. There is no user login, only API keys.

## Project layout

```
app/        main.py (API) · rag.py (prompt + answer logic) · retriever.py (hybrid search + RRF)
            index.py · chunking.py · text.py · loader.py · embedder.py · llm.py · metrics.py
            db.py · security.py · schemas.py · config.py · logger.py
scripts/    ingest.py (build the index) · tune.py (find good settings)
            check_answers.py (test the answers with the real LLM) · report.py (usage report)
            load_test.py (many users at once) · fake_llm.py (a fake model for load tests)
            time_steps.py (time every step of an answer) · fetch_sources.py (download the pages for the patient questions)
            review_sample.py (a sheet for doctors)
tests/      pytest suite (no downloads, no network)
walkthrough.ipynb   step by step run of everything with the results (books, chunks, search, evaluation, tests)
data/       books/ · sample/ · eval/ · index/ (generated)
docs/       BOOKS.md (which books) · TUNING.md (every parameter explained)
            prometheus.yml · grafana_dashboard.json · grafana_results.png (monitoring)
            alerts.yml · api_prices.csv · load_test_results.csv · tuning_results.csv · answer_check.csv
            patient_check_*.csv · time_steps.csv · doctor_review.csv
```
