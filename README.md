# Namazi Hospital Medical RAG

A question-answering backend for medical staff. You ask a question (English or Persian), it
searches the hospital's medical books, and an LLM writes a short answer **using only the
passages it found**, with book and page citations. If the books do not contain the answer, it
says so instead of guessing.

## Status: what is real and what is not

I built this for a hospital pilot. It has **not** been used in a hospital: no staff, no patients and no real logs exist, and
I could not have shared them anyway. What is real: the code, the 118 tests, the evaluation on 20 public guidelines (results below),
the load tests, the monitoring and the deployment steps I ran myself on a GPU server. What is not: any real-world usage.
The load tests use made-up traffic (the 76 English eval questions, asked again and again). Answers are reference text from books,
not medical advice.

## At a glance

- Answers questions from 20 public medical guidelines, in English and Persian, with book and page citations.
- **English:** the right passage is in the top 5 for 86% of the questions, and the answer cites it for 79%. **Persian:** 82% reach the model (after a translation step) and 69% are cited.
- It says "not found" for 32 of the 34 questions the books cannot answer. The 2 misses are both Persian, and they are made-up answers. More below.
- **Speed:** search takes about 21 ms, a whole answer about 3 s on one A100.
- **Many users:** the app alone took 10,000 users with no errors (about 20 questions per second). The real limit is the model: one GPU makes about 0.4 answers per second.
- **Patient questions:** 525 questions in English and Persian, answered from public web pages. Roughly 7 in 10 get an answer that cites the right page, the rest get an honest "I don't know". Persian is weaker. Details below.
- **Cost:** no token price with the local model. The same 10,000 questions on a paid API would cost roughly USD 6 to 47.

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
python -m pytest                                   # 118 tests, a few seconds

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
python -m scripts.ingest        # downloads the embedding model on first run
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

I ran the whole pipeline on 20 freely downloadable guidelines (WHO, KDIGO, GOLD; list in `docs/BOOKS.md`): 7,112 chunks,
`multilingual-e5-base` embeddings and 141 hand-written questions in `data/eval/eval_questions.jsonl`:
76 English and 65 Persian, 107 answerable and 34 not answerable by the books. 36 of the Persian questions ask the same thing as an English one,
so the languages can be compared fact by fact. Every evidence phrase is copied from one sentence of the books and was checked against `data/index/chunks.jsonl`.
Everything below is also in `walkthrough.ipynb` with plots (sections 11 to 16).

**How I measure it**

| What | Measure | Why |
|---|---|---|
| Search | hit@k (is the right passage in the top k?), MRR, nDCG@10 | the usual retrieval numbers. One right passage per question, so recall@k equals hit@k |
| Answers | right passage in the sources, right passage cited, share with a citation | did the answer use the right text? |
| Refusing | share of unanswerable questions that got an answer | the main hallucination number |
| Made-up facts | numbers in the answer that are in no source, citations to sources that do not exist, an LLM judge ("backed by the sources?") | the three cheap checks I could run without a person |
| Speed | search time, answer time (median, 95%), questions per second, tokens | latency and cost |

**Search only** (the 98 dev questions, 250-word chunks, RRF constant 60). The Persian questions are searched as typed here, without the translation step, which is why the numbers are low:

| Setting | Dev hit rate@5 | Dev MRR |
|---|---|---|
| dense only (meaning) | 0.547 | 0.392 |
| BM25 only (exact words) | 0.467 | 0.347 |
| hybrid 1:1 | 0.600 | 0.456 |
| **hybrid 2:1 (used)** | **0.613** | **0.465** |

I tried 33 settings (3 chunk sizes, 5 dense/BM25 mixes, 3 RRF constants, table in [docs/tuning_results.csv](docs/tuning_results.csv)). Hybrid is clearly better than either search alone.
On the 141 questions the tuner picks exactly the setting I use (250 words, 2:1). On the first 93 questions it had preferred 400-word chunks with a 1:2 mix; I built that index and
ran the full answer check on it twice. It was not better (right passage cited 57 and 55 of 71 against 55, and worse for English), so I had kept 250 words. The larger question set now agrees.
Held-out test split (43 questions, 32 answerable): hit rate@5 0.594, MRR 0.329. It is small, so read it as a rough number.

**Search by language** (all answerable questions, the real settings):

| | hit@1 | hit@3 | hit@5 | hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| English (58) | 0.50 | 0.72 | **0.86** | 0.91 | 0.64 | 0.70 |
| Persian as typed (49) | 0.12 | 0.18 | 0.31 | 0.39 | 0.20 | 0.23 |
| Persian after the translation step (49) | | | **0.82** (40 of 49) | | | |

- Many KDIGO guidelines repeat the same recommendation in a summary and in a chapter, so the right topic is found but not always the exact passage I marked.
- Persian questions do not match English books directly (BM25 cannot cross languages), so the app translates them with the LLM first. Without it Persian would not work.
- The score cutoff `MIN_DENSE_SCORE` does not separate answerable from unanswerable questions (see below), so it stays off and the `NO_ANSWER` prompt rule does that job.
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
| Tests | 118 tests: chunking, retrieval maths, index, RAG logic, every endpoint, auth, rate limit, LLM retry (against a local fake server), PDF loading |

## Made for a hospital without cloud

The language model runs inside the hospital (Ollama), so no question or patient detail leaves the network.
Extras for real use: a simple question page at `/` (works for Persian, with thumbs up/down),
Persian questions are translated to English for the search (the LLM does this, so the answer check below includes it), `python -m scripts.report` shows
usage and unanswered questions, and `python -m scripts.check_answers` runs the questions through the
real LLM and checks the answers, not only the search. Step by step setup, including offline
installation and backups: [docs/DEPLOY.md](docs/DEPLOY.md).

### Answer check with a real local model

`qwen2.5:32b-instruct` (Ollama, context 8192) on one A100 40 GB, all 141 questions, every answer in [docs/answer_check.csv](docs/answer_check.csv)
(`python -m scripts.check_answers --judge`). Two runs with the same settings are not identical (about 3 in 4 answers are word for word the same,
the count of right citations moved by 1 between my runs), so ignore differences of one or two questions.

| What was checked | English | Persian |
|---|---|---|
| answerable questions | 58 | 49 |
| ...that got an answer | 57 | 41 |
| ...answer has a [n] citation | 57 | 41 |
| ...the right passage was among the sources sent to the LLM | 50 (86%) | 40 (82%) |
| ...the answer cited the right passage | 46 (79%) | 34 (69%) |
| unanswerable questions | 18 | 16 |
| ...correctly answered "not found" | **18** | **14** |
| ...**got an answer anyway (made up)** | 0 | **2** |
| answers with a number that is in no source | 0 of 57 | 0 of 43 |
| answers citing a source that does not exist | 0 of 57 | 0 of 43 |
| answers the LLM judge found not backed by the sources | 0 of 57 | 0 of 43 |
| median / average time per question | 2.8 s / 3.1 s | 4.0 s / 4.5 s |

On the 38 facts that were asked in both languages, the right passage was cited 31 times in English and 26 times in Persian.

**What went wrong, as it is:**
- **Two answers are made up, both in Persian.** The question about eczema in children got a treatment list built from the permethrin (scabies) text of the child-health book.
  The question about first aid for a snake bite got a list of adrenaline, oxygen and fluid amounts taken from unrelated texts, with broken words in it. 32 of 34 unanswerable questions were refused correctly.
- **The three cheap checks did not catch them.** Both answers are faithful to some real text of the books, only to the wrong topic. The numbers in the snake bite answer
  exist somewhere in the sources, and the judge accepted both answers. So "0 made-up numbers" must not be read as "no hallucination": the share of unanswerable questions that get an answer is the number to watch.
  The judge is the same model as the answerer. I checked it with made-up wrong answers (it flagged 2 of 2), but it is a rough signal.
- **Persian answers contain broken words** (a stray Korean character, "adrenaline" mixed with Persian letters). Qwen2.5 writes poor Persian. A Persian answer must be read next to its English source.
- **Persian refuses too often:** 8 of 49 answerable Persian questions got "not found" (English: 1 of 58). That is the safer mistake, but it loses useful answers.
- The right passage was cited for 80 of 107 answerable questions (75%); 90 of 107 had it in the sources. For the others the model used a different passage that often says nearly the same, but a doctor has to confirm that.
- The old README said the model made up an INR answer for warfarin. That label was wrong: the books give "INR 2-3" for warfarin in a KDIGO 2024 table for CKD patients with atrial fibrillation (page 129) and the answer cited it. I changed the evidence of that question. The answer is right for that patient group only, and it does not say so.

**Fixes I tried for "made up answers", one at a time, re-running the check after each:**

| Try | Result | Kept? |
|---|---|---|
| (a) bigger model, 3B to 32B | better citations (13 to 16 of 21 on the first 27 questions); INR still answered, but that answer was grounded | yes |
| (b) one stricter sentence in the prompt ("...or only mention the topic for a different disease or patient group") | no change on the first 93 questions: eczema still answered, right citations 54 vs 54 | no, removed |
| (c) `MIN_DENSE_SCORE` | not applied. Best similarity is 0.75 to 0.92 for answerable and 0.77 to 0.85 for unanswerable questions, so they overlap. The tuner's cutoff that lets 95% of answerable questions through stops only 8 of 23 unanswerable dev questions | no |
| other index (400 words, 1:2 mix) | not better, see above | no |

So the invented answers are still open. Doctors still need to read the answers in the csv file.

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
| answered | 287 (68%) | 317 (76%) |
| said "I don't know" | 133 (32%) | 102 (24%) |
| answers citing the expected page | 284 of 287 (99%) | 310 of 317 (98%) |
| answers with a number that is in no source | 11 (4%) | 26 (8%) |
| answers the LLM judge doubts | 27 (9%) | **74 (23%)** |
| answers with broken letters | 0 | **13** |
| median time per question (three runs shared the GPU) | 4.4 s | 6.5 s |

(Persian: 419 of 420, one question made the model server fail.) The 210 open questions got "I don't know" 188 times; the 22 answers came from nearby pages, for example the asthma page does talk about flu.
The web pages explain a disease well but say little about follow-up (95% "I don't know"), a missed dose (86%) or how to ask about a test result (81%). That is honest but thin.

**What I fixed because of this test:** 10 English answers ended with the literal word `NO_ANSWER` (the model explained first, and the app only looked at the start of the reply). Now a reply that contains it anywhere
becomes "I don't know". I also added an emergency line for urgent words (chest pain, trouble breathing, suicide ...), a guard that shows "I don't know" instead of a Persian answer with Chinese, Japanese or Korean letters,
Prometheus alerts (`docs/alerts.yml`) and a sheet for doctors (`python -m scripts.review_sample`, 100 answers, `docs/doctor_review.csv`).
The judge is the same model as the answerer and reads Persian worse than English, so the Persian doubts are a warning, not an exact number.

### Where the time goes

`python -m scripts.time_steps` times every step of an answer (30 English and 30 Persian questions, one at a time, `docs/time_steps.csv`, plots in the notebook):

![One answer, step by step](docs/time_steps.png)

For an English question, writing the answer takes 62% of the time (about 2.8 s) and reading the sources 37% (about 1.6 s). The whole search (embedding, dense, BM25, merge) takes about 36 ms, under 1%.
A Persian question also needs a translation call (1.1 s, 17%). **So the LLM is what to optimize**, in this order: a faster or smaller model or server (writing), fewer chunks in the prompt (reading, but the right passage is found less often:
86% in the top 5, 72% in the top 3), a cheaper translation. The search is not worth touching. These steps add up to a little more than the 3.1 s and 4.9 s medians of the staff answer check; I don't know the exact reason.

### Speed and cost

- **Search alone:** about 21 ms (median, 95% under 32 ms). **A whole answer:** about 3 s (English median 2.8 s). The LLM is nearly all of it. The very first question took 14 s because the model had to load into the GPU.
- **Tokens:** about 2,051 prompt and 61 answer tokens per question.
- **Cost on a paid API for 10,000 questions** (prices from summary websites on 2026-10-08, listed with their sources in [docs/api_prices.csv](docs/api_prices.csv), please check the official pages):

| Model | USD per 1,000 questions | USD per 10,000 questions |
|---|---|---|
| OpenAI GPT-5 mini | 0.64 | 6.35 |
| Claude Haiku 4.5 | 2.36 | 23.57 |
| OpenAI GPT-5 | 3.18 | 31.75 |
| Claude Sonnet 5 | 4.71 | 47.14 |

  The local model has no token price. It costs a GPU server, about 7 hours of one A100 for 10,000 questions. A paid API would send the questions out of the hospital network, which this project was built to avoid.
  Persian questions need one more small call for the translation, and the judge calls are not counted.

### Load test: 10,000 users, and what happens when users grow

`python -m scripts.load_test` sends many questions at once and prints speed, answer times and failures; `--save` adds the result to [docs/load_test_results.csv](docs/load_test_results.csv).
All runs hit a copy of the app on one A100 (rate limit off, its own database).

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
| "I don't know" and citations | done |
| Emergency line, broken-text guard | done, a simple word list that a clinician must review |
| A doctor reads the answers | **not done**: `docs/doctor_review.csv` is ready, someone has to fill it in |
| Persian quality | **weak**: the judge doubts 23% of the answers |
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
