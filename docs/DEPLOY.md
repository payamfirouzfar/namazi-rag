# Putting it in the hospital

This is the plan for a hospital with no cloud and limited internet. Everything runs on one
server inside the hospital network, and no question leaves it.

## What you need
- One server (Linux, Docker). 16 GB RAM is enough. A GPU with 6 GB or more makes answers much faster.
- The books as PDF files (current editions, with permission to use them).
- A person from IT who can set up HTTPS and the backup.

## Step 1: get the downloads on a computer with internet
Docker Hub and Hugging Face can be slow or blocked, so download once and copy by USB or the internal network.
1. Docker images: `docker pull python:3.11-slim ollama/ollama`, then `docker save -o images.tar ...` and `docker load -i images.tar` on the server.
2. The language model: run `ollama pull qwen2.5:7b-instruct` and copy the Ollama models folder to the server.
3. The embedding model and the reranker: download `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` (about 2 GB each) and copy the folders to the server.
   In `.env` set `EMBEDDING_MODEL` and `RERANK_MODEL` to those folder paths (the index remembers the name of the embedding model, so use the same one for ingest and for the API).

## Step 2: settings
Copy `.env.example` to `.env` and set:
- `ENV=production` and `API_KEYS=` one long random key per department (the app will not start without keys)
- `LLM_BASE_URL=http://ollama:11434/v1`, `LLM_MODEL=qwen2.5:7b-instruct`, `LLM_TIMEOUT_S=180`
- `LOG_QUESTIONS=false` if the hospital does not allow storing question text (only a hash is kept)

## Step 3: books and index
Put the PDFs in `data/books/`, then run `docker compose run --rm api python -m scripts.ingest`.
Check the log: every book should show a number of chunks. A book with no text is a scan and needs OCR first.

## Step 4: start
`docker compose up -d --build`. Open `http://server:8000/` for the question page. `/ready` should say ready.

## Step 5: before the doctors use it
1. Test with real questions from the doctors. Write 50 to 100 into `data/eval/eval_questions.jsonl` (with a short evidence phrase)
   and run `python -m scripts.tune` and `python -m scripts.check_answers`. Have a doctor read `answer_check.csv`.
2. HTTPS: put nginx (or any proxy the hospital already uses) in front. This app speaks plain HTTP.
3. Tell the doctors: it is a reference tool, not a diagnosis. The page already shows this.

## Every day
- Backup `data/app.db` every night. The index can always be rebuilt from the books. Do not copy the file with `cp`
  while the app runs (SQLite keeps part of the data in a second file). Use the SQLite backup command, for example from `crontab -e`:

      0 2 * * * cd /path/to/namazi-rag && mkdir -p backups && sqlite3 data/app.db ".backup 'backups/app-$(date +\%F).db'" && find backups -name 'app-*.db' -mtime +30 -delete

  It keeps 30 days. Copy the `backups` folder to another machine too. Tested: the command makes a copy that passes `pragma integrity_check`.
- `docker compose run --rm api python -m scripts.report` shows questions, answers not found, thumbs up/down and answer time.
  Questions with no answer show which book is missing. Thumbs down show wrong answers to review.
- New or changed book: copy it into `data/books/`, run ingest again, restart the api.

## Keeping it running (without Docker)
On a server without Docker, two `systemd --user` services restart the LLM and the app if they crash.
Save these two files in `~/.config/systemd/user/` (change the paths and the model folder to yours):

    # ollama.service
    [Unit]
    Description=Ollama (local LLM)
    [Service]
    ExecStart=%h/ollama/bin/ollama serve
    Environment=OLLAMA_HOST=127.0.0.1:11435
    Environment=OLLAMA_MODELS=%h/ollama/models
    # only on a shared GPU server: pick a free GPU
    Environment=CUDA_VISIBLE_DEVICES=3
    # answer 4 questions at the same time (needs about 8 GB more GPU memory)
    Environment=OLLAMA_NUM_PARALLEL=4
    Restart=on-failure
    [Install]
    WantedBy=default.target

    # namazi-rag.service
    [Unit]
    Description=Namazi hospital RAG API
    After=ollama.service
    [Service]
    WorkingDirectory=%h/namazi-rag
    ExecStart=%h/namazi-rag/.venv/bin/uvicorn app.main:app_factory --factory --host 127.0.0.1 --port 8000
    Environment=CUDA_VISIBLE_DEVICES=3
    Restart=on-failure
    [Install]
    WantedBy=default.target

Then: `systemctl --user daemon-reload && systemctl --user enable --now ollama namazi-rag`.
Look at the log with `journalctl --user -u namazi-rag -f`. Both listen on 127.0.0.1 only: reach the page with an SSH tunnel
(`ssh -L 8000:127.0.0.1:8000 user@server`) or put nginx in front.
User services stop when you log out unless the administrator runs `loginctl enable-linger <user>` once.
Tested on the pilot server: both services started, answered a question, and the app came back about 30 seconds after
I killed it (loading the embedding model takes that long). Not tested: starting at boot (needs `enable-linger`).

## Not tested
`docker compose up` was **not tested**. The pilot server has the Docker client but no running Docker daemon and no
`docker compose` plugin, and I may not install system packages there. Everything above was run without Docker.
Also, `docker-compose.yml` publishes port 8000 on all network interfaces: change it to `127.0.0.1:8000:8000`
if nginx runs on the same machine.

## Watching it: Prometheus and Grafana
The app shows its numbers at `http://127.0.0.1:8000/metrics`. To see them as graphs:
1. Download Prometheus and Grafana (OSS, Linux tarballs) and unpack them in your home folder, no root needed.
2. Start Prometheus with `docs/prometheus.yml` (`--config.file=... --web.listen-address=127.0.0.1:9091`).
3. Start Grafana on `127.0.0.1:3001` and set the admin password with `GF_SECURITY_ADMIN_PASSWORD` (keep it in a file outside the project, never commit it).
4. In Grafana add a Prometheus data source with uid `prom` and url `http://127.0.0.1:9091`, then import `docs/grafana_dashboard.json`.
   (Or copy both as "provisioning" files, then they load by themselves.)
The last panel shows what the tokens would cost on a paid API. Type your own two prices (USD per million tokens) at the top of the dashboard.
I ran exactly this setup on the pilot server: Prometheus saw the app, Grafana loaded the data source and the dashboard.
I checked the data through the Grafana API, I did not look at the graphs in a browser.

## Load test
To see how the app copes with many users, run a test copy (own port and database, rate limit off), never the live one:

    python -m scripts.fake_llm --delay 1.0 &        # a fake model, tests only the app
    LLM_BASE_URL=http://127.0.0.1:9301/v1 LLM_MODEL=fake RATE_LIMIT_PER_MINUTE=0 DB_PATH=data/loadtest.db \
        uvicorn app.main:app_factory --factory --host 127.0.0.1 --port 8001 &
    python -m scripts.load_test --users 10000 --concurrency 100

For the real model, skip the fake server and the `LLM_*` settings, so the test copy reads your `.env` and uses your Ollama. A real run is slow
(about 0.4 questions per second on one A100). Delete `data/loadtest.db` afterwards. Results are in the README.

## HTTPS with nginx (not tested)
The app speaks plain HTTP. Put nginx in front and let it handle the certificate. This is the usual setup, but I did not run it on the pilot server (no root there):

    server {
        listen 443 ssl;
        server_name hospital.example;
        ssl_certificate     /etc/ssl/certs/hospital.pem;       # your certificate
        ssl_certificate_key /etc/ssl/private/hospital.key;
        location / {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Forwarded-For $remote_addr;
            proxy_read_timeout 300s;                           # a slow answer can take a while
            limit_req zone=ask burst=10;                       # add: limit_req_zone $binary_remote_addr zone=ask:10m rate=30r/m; (in http{})
        }
    }

Keep `ENV=production` and the `API_KEYS` in the `.env`. Do not open port 8000 to the network, only 443.

## Alerts
`docs/alerts.yml` has four alerts (app down, errors on the rise, slow answers, too many "not found"). `docs/prometheus.yml` loads them and they show up in Prometheus under "Alerts".
To get an email or a message you need Alertmanager, which is not set up here. Check the file with `promtool check config docs/prometheus.yml`.

## Patient questions
To test with the patient question file: `python -m scripts.fetch_sources data/eval/patient_questions_english_persian_525.csv`, then
`BOOKS_DIR=data/patient_books INDEX_DIR=data/index_patient python -m scripts.ingest`, then run `python -m scripts.check_answers --judge --eval data/eval/patient_questions_en.jsonl`
with `INDEX_DIR=data/index_patient` (the same for `_fa` and `_open`). It takes about 2 hours for all 1,050 questions on one GPU, because every question needs 2 or 3 LLM calls.
Check the licence of every site before you use the pages. Use `python -m scripts.review_sample` to make the sheet for doctors.

## Limits to tell the hospital
- One server, one SQLite file. For many servers use Postgres and a shared rate limit.
- The small local model is slower and less careful than the big online ones. Check the answers (step 5) before trusting it.
- There are no user accounts, only department API keys.

## Small computers
A model that is too big for the GPU crashes with an out-of-memory error. On my 6 GB laptop the 7B model
did not start and the 3B model (`qwen2.5:3b-instruct`) did. The answers also need a context window of at least 8192 tokens,
because the sources are long (Ollama silently cuts anything longer). Make a model with that setting:

    # file Modelfile
    FROM qwen2.5:7b-instruct
    PARAMETER num_ctx 8192

    ollama create namazi-qwen -f Modelfile      # then set LLM_MODEL=namazi-qwen

Use at least 16 GB of free memory for the API and the LLM together, and do not run them on a disk that is nearly full.

## Big GPU (what the pilot server uses)
The pilot server has A100 40 GB cards shared with other people. `qwen2.5:32b-instruct` (19 GB download, about 21 GB in memory with
`num_ctx 8192`) runs 100% on one card. A 72B model would need about 47 GB, so it does not fit. If the GPU driver is older than your
PyTorch build, `torch.cuda.is_available()` says False and the embeddings run on the CPU: install the build that matches the driver
(`nvidia-smi` shows the CUDA version; here 12.8, so `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`).
Some PDFs (the GOLD guide) are AES encrypted and need the `cryptography` package, otherwise ingest skips them and only logs an error:
check that every book shows a number of chunks.
