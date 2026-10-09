# Setup and running

The app runs on one machine: the API, the search models and the LLM (through Ollama). No question leaves it. This page has the steps and the commands I used.

## What you need
- A Linux machine with 16 GB RAM. A GPU makes answers much faster (a 40 GB card runs the 32B model, 6 GB is enough for a small one).
- The books as PDF, TXT or MD files (see [BOOKS.md](BOOKS.md) for the ones I used).
- Python 3.10 or newer and [Ollama](https://ollama.com).

## 1. Get the models
If the machine has no direct internet access, download on another computer and copy the folders over.
1. The language model: `ollama pull qwen2.5:32b-instruct` (or a smaller one, see "Small computers" below) and copy the Ollama models folder.
2. The embedding model and the reranker: `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` (about 2 GB each).
   In `.env` set `EMBEDDING_MODEL` and `RERANK_MODEL` to the folder paths. The index remembers the name of the embedding model, so use the same one for ingest and for the API.

## 2. Settings
Copy `.env.example` to `.env`. The ones you will touch:
- `API_KEYS`: one long random key per group of users (the comments in `.env.example` explain when the app insists on keys).
- `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_S`: where the LLM is and how long to wait for it. `LLM_TIMEOUT_S=180` or more for a big local model.
- `LOG_QUESTIONS=false` stores only a hash of each question, not its text.

## 3. Books and index
Put the files in `data/books/`, then run `python -m scripts.ingest`. Every book should show a number of chunks in the log. A book with no text is a scan and needs OCR first.

## 4. Start
`uvicorn app.main:app_factory --factory --host 127.0.0.1 --port 8000`, then open `http://127.0.0.1:8000/` for the question page. `/ready` says when the index is loaded.

## Every day
- Back up `data/app.db` every night (feedback and logs). The index can always be rebuilt from the books. Don't copy the file with `cp` while the app runs, because SQLite keeps part of the data in a second file. Use the SQLite backup command, for example from `crontab -e`:

      0 2 * * * cd /path/to/namazi-rag && mkdir -p backups && sqlite3 data/app.db ".backup 'backups/app-$(date +\%F).db'" && find backups -name 'app-*.db' -mtime +30 -delete

  It keeps 30 days. I checked that the command makes a copy that passes `pragma integrity_check`.
- `python -m scripts.report` shows the questions, the ones with no answer, thumbs up and down, and the answer time. A question with no answer shows which book is missing, and thumbs down show answers to review.
- New or changed book: copy it into `data/books/`, run ingest again, restart the app.

## Keeping the app and the LLM running
Two `systemd --user` services restart them if they crash. Save these files in `~/.config/systemd/user/` and change the paths to yours:

    # ollama.service
    [Unit]
    Description=Ollama (local LLM)
    [Service]
    ExecStart=%h/ollama/bin/ollama serve
    Environment=OLLAMA_HOST=127.0.0.1:11435
    Environment=OLLAMA_MODELS=%h/ollama/models
    # pick a free GPU if several people share the machine
    Environment=CUDA_VISIBLE_DEVICES=3
    # answer 4 questions at the same time (needs about 8 GB more GPU memory)
    Environment=OLLAMA_NUM_PARALLEL=4
    Restart=on-failure
    [Install]
    WantedBy=default.target

    # namazi-rag.service
    [Unit]
    Description=Namazi RAG API
    After=ollama.service
    [Service]
    WorkingDirectory=%h/namazi-rag
    ExecStart=%h/namazi-rag/.venv/bin/uvicorn app.main:app_factory --factory --host 127.0.0.1 --port 8000
    Environment=CUDA_VISIBLE_DEVICES=3
    Restart=on-failure
    [Install]
    WantedBy=default.target

Then `systemctl --user daemon-reload && systemctl --user enable --now ollama namazi-rag`, and `journalctl --user -u namazi-rag -f` for the log.
Both listen on 127.0.0.1 only. User services stop when you log out, unless an administrator runs `loginctl enable-linger <user>` once.
I started both, asked a question, killed the app, and it was back after about 30 seconds (loading the embedding model takes that long).

By default Ollama unloads the model after 5 idle minutes, and the next question then waits about 12 s for it to load again. `OLLAMA_KEEP_ALIVE` keeps it in memory.

## Watching it: Prometheus and Grafana
The app shows its numbers at `http://127.0.0.1:8000/metrics`. To see them as graphs:
1. Download Prometheus and Grafana (OSS, Linux tarballs) and unpack them in your home folder. No root needed.
2. Start Prometheus with `prometheus.yml` (`--config.file=... --web.listen-address=127.0.0.1:9091`).
3. Start Grafana on `127.0.0.1:3001` and set the admin password with `GF_SECURITY_ADMIN_PASSWORD` (keep it in a file outside the project).
4. In Grafana add a Prometheus data source with uid `prom` and url `http://127.0.0.1:9091`, then import `grafana_dashboard.json`. (Or copy both as provisioning files and they load by themselves.)

The last panel shows what the tokens would cost on a paid API. Type your own two prices (USD per million tokens) at the top of the dashboard.
I checked the data through the Grafana API, but I never looked at the graphs in a browser.

[alerts.yml](alerts.yml) has four alerts (app down, errors on the rise, slow answers, too many "not found"). They show up in Prometheus under "Alerts". For an email or a message you need Alertmanager, which isn't set up here. Check the files with `promtool check config prometheus.yml`.

## Load test
Run a test copy of the app (own port and database, rate limit off), not the one people use:

    python -m scripts.fake_llm --delay 1.0 &        # a fake model, tests only the app
    LLM_BASE_URL=http://127.0.0.1:9301/v1 LLM_MODEL=fake RATE_LIMIT_PER_MINUTE=0 DB_PATH=data/loadtest.db \
        uvicorn app.main:app_factory --factory --host 127.0.0.1 --port 8001 &
    python -m scripts.load_test --users 10000 --concurrency 100

For the real model, skip the fake server and the `LLM_*` settings, so the test copy reads your `.env`. That run is slow (about 0.4 questions per second on one A100). Delete `data/loadtest.db` afterwards. The results are in the README.

## Testing with the patient questions
`python -m scripts.fetch_sources data/eval/patient_questions_english_persian_525.csv` downloads the pages, then
`BOOKS_DIR=data/patient_books INDEX_DIR=data/index_patient python -m scripts.ingest` builds their index, and
`python -m scripts.check_answers --judge --eval data/eval/patient_questions_en.jsonl` runs the questions, with `INDEX_DIR=data/index_patient` (the same for `_fa` and `_open`).
All 1,050 questions take about 2 hours on one GPU, because every question needs 2 or 3 LLM calls. Check the licence of every site before you use its pages. `python -m scripts.review_sample` makes a sheet of answers for a doctor to read.

## Limits
- One machine and one SQLite file. For several machines you would need Postgres and a shared rate limit.
- A small local model is slower and less careful than the big online ones. Run `scripts/check_answers.py` and read the answers before you trust a model.
- There are no user accounts, only API keys.

## Small computers
A model that is too big for the GPU crashes with an out-of-memory error. On my 6 GB laptop the 7B model did not start and the 3B model (`qwen2.5:3b-instruct`) did.
The answers also need a context window of at least 8192 tokens, because the sources are long (Ollama silently cuts anything longer). Make a model with that setting:

    # file Modelfile
    FROM qwen2.5:7b-instruct
    PARAMETER num_ctx 8192

    ollama create namazi-qwen -f Modelfile      # then set LLM_MODEL=namazi-qwen

Keep at least 16 GB of free memory for the API and the LLM together, and don't run them on a disk that is nearly full.

## Big GPU
On an A100 40 GB, `qwen2.5:32b-instruct` (19 GB download, about 21 GB in memory with `num_ctx 8192`) runs 100% on one card. A 72B model would need about 47 GB, so it doesn't fit.
If the GPU driver is older than your PyTorch build, `torch.cuda.is_available()` says False and the embeddings run on the CPU. Install the build that matches the driver
(`nvidia-smi` shows the CUDA version; mine was 12.8, so `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`).
Some PDFs (the GOLD guide) are AES encrypted and need the `cryptography` package, otherwise ingest skips them and only logs an error, so check that every book shows a number of chunks.
