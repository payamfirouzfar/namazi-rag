# Experiments

Before I settled on the search and the prompt, I tried a lot of ideas. This folder holds the scripts, so anyone can run them again, and the results I got. Nothing here is needed to run the app.

The rule I followed: change **one thing at a time**, use the **same questions**, and keep an idea only if it clearly helped. Most ideas did not.

## The questions

The search experiments use the 107 answerable questions of `data/eval/eval_questions.jsonl` (58 English, 49 Persian). The right chunk is the one that contains the evidence phrase of the question. I report two numbers:

- **hit@5**: is the right chunk among the 5 best results? (the app sends 5 chunks to the LLM)
- **MRR**: how high is the right chunk? 1.0 means always first.

Persian questions are measured two ways, as typed and **translated** into English by the LLM (the app translates them too). The translations are made once and saved in `data/tuning/experiment_cache/`, together with the pages of the books, so the next experiment starts faster. That folder is git-ignored.

## What I learned, in short

| Experiment | Question | Answer |
|---|---|---|
| 1. Ingestion | Does cleaning the books better help? | No. Nothing helped clearly |
| 2. Embedding model | Which model finds the right chunk best? | **bge-m3**, especially for Persian |
| 3. Weights and reranker | Dense vs BM25 weights, and a reranker? | Weights barely matter. **The reranker helps a lot** |
| 4. Chunk size | How big should a chunk be? | With the reranker 150, 200 and 250 words are about the same |
| 5. Prompt | Does another system prompt give better answers? | A **shorter** answer is as good and a bit shorter. Answering "part" of a question hurts |
| 6. LLM | Is Gemma 3 12B better than Qwen 2.5 32B? | Faster, but it made up English answers. I kept Qwen |

## How to run them

Run everything from the project folder with the virtual environment on. Use a free GPU, and `CUDA_VISIBLE_DEVICES=3` (for example) to pick it.

```bash
python -m experiments.ingestion              # about 6 minutes
python -m experiments.embedding_models       # about 10 minutes, downloads two models (about 2 GB each)
python -m experiments.rerank_and_weights     # about 8 minutes
python -m experiments.chunk_size             # about 25 minutes

# the prompt and the LLM need the answer check and a running LLM (Ollama), 141 questions take about 10 minutes
python -m experiments.prompts short          # also: base, partial, both. The csv goes to data/tuning/prompt_<name>.csv
python -m experiments.compare_runs base=data/tuning/prompt_base.csv short=data/tuning/prompt_short.csv
```

My own results are in `results/`: one text file per search experiment, and the answer csvs of the prompt and LLM experiments, with the comparison tables as text files.

## 1. Ingestion: cleaning the books

`python -m experiments.ingestion`. Same embedding model (multilingual-e5-base), only the chunks change. I looked at the books and found three problems, and tried to fix each one:

- the same page header on many pages (for example `www.kidney-international.org` is in 15% of the chunks),
- words that the PDF broke in two (`bene fits`, `signi ficant`),
- the chunk does not say which book it comes from.

| Idea | Chunks | English hit@5 / MRR | Persian typed | Persian translated |
|---|---|---|---|---|
| current ingestion | 7112 | 0.845 / 0.630 | 0.306 / 0.187 | 0.816 / 0.597 |
| remove repeated page headers | 7087 | 0.862 / 0.625 | 0.265 / 0.181 | 0.796 / 0.571 |
| join the words the PDF broke | 7101 | 0.845 / 0.622 | 0.306 / 0.179 | 0.816 / 0.595 |
| book title in every chunk | 7112 | 0.828 / 0.608 | 0.245 / 0.148 | 0.816 / 0.634 |
| all three | 7076 | 0.862 / 0.627 | 0.204 / 0.142 | 0.796 / 0.571 |

I kept the current ingestion. A few numbers go up and a few go down by one or two questions, which is noise. The extra code would not pay for itself.

One thing I got wrong at first: my first try at joining the broken words did nothing, because the PDF text still has the ligature glyph (`ﬁ`) and the cleaning that turns it into `fi` runs later. The script now joins the words after the cleaning.

## 2. Embedding models

`python -m experiments.embedding_models`. Same chunks, hybrid search 2:1, no reranker.

| Model | English hit@5 / MRR | Persian typed | Persian translated |
|---|---|---|---|
| multilingual-e5-base (my first choice) | 0.845 / 0.630 | 0.306 / 0.187 | 0.816 / 0.597 |
| multilingual-e5-large | 0.793 / 0.585 | 0.551 / 0.373 | 0.878 / 0.611 |
| **bge-m3** | **0.897 / 0.639** | **0.612 / 0.480** | 0.816 / **0.680** |

I switched to bge-m3. It is the best for English and doubles the Persian result when the question is typed in Persian. The bigger e5 model helped Persian but was worse for English. It is only a setting (`EMBEDDING_MODEL`, no prefixes), no new code.

## 3. Weights and the reranker

`python -m experiments.rerank_and_weights`. Embedding model: bge-m3. The reranker (`BAAI/bge-reranker-v2-m3`) is a second model that reads the question together with each of the 20 best chunks and puts them in a better order. The best 5 are sent on.

| Setting | English hit@5 / MRR | Persian typed | Persian translated |
|---|---|---|---|
| dense 1 : BM25 1 | 0.879 / 0.661 | 0.571 / 0.459 | 0.837 / 0.664 |
| dense 2 : BM25 1 (used) | 0.897 / 0.639 | 0.612 / 0.480 | 0.816 / 0.680 |
| dense 3 : BM25 1 | 0.914 / 0.643 | 0.612 / 0.484 | 0.837 / 0.668 |
| dense only | 0.810 / 0.596 | 0.633 / 0.520 | 0.755 / 0.599 |
| **dense 2 : BM25 1 + reranker (20 to 5)** | **0.914 / 0.800** | **0.837 / 0.727** | **0.878 / 0.759** |

I kept 2:1 (the weights are within a few questions of each other) and added the reranker. The biggest change is in the MRR: the right chunk moves from "somewhere in the top 5" to "first or second". For Persian typed, the hit rate goes from 0.61 to 0.84. It costs about 0.6 s per question (the whole experiment took 125 seconds for 156 searches). It is a setting too (`RERANK_MODEL`, empty means off) and a few lines in `app/retriever.py`.

The notebook shows 0.93 for English instead of 0.91 here. The notebook looks at a deeper list of candidates before it reranks, which moves one question. Both are right, and over all the questions the two give the same 0.89.

## 4. Chunk size

`python -m experiments.chunk_size`. The tuner (`scripts/tune.py`) has no reranker and, with bge-m3, prefers small chunks of 150 words. I checked the sizes the way the app really searches, with the reranker.

| Chunk (words / overlap) | Chunks | English hit@5 / MRR | Persian typed | Persian translated |
|---|---|---|---|---|
| 150 / 30, no reranker | 11891 | 0.879 / 0.667 | 0.592 / 0.416 | 0.755 / 0.605 |
| 150 / 30 + reranker | 11891 | 0.897 / 0.797 | 0.816 / 0.704 | 0.837 / 0.693 |
| 200 / 40, no reranker | 8913 | 0.862 / 0.693 | 0.633 / 0.430 | 0.755 / 0.625 |
| 200 / 40 + reranker | 8913 | **0.966 / 0.818** | 0.837 / 0.740 | 0.898 / 0.736 |
| 250 / 50, no reranker | 7112 | 0.897 / 0.639 | 0.612 / 0.480 | 0.816 / 0.680 |
| 250 / 50 + reranker (used) | 7112 | 0.914 / 0.800 | 0.837 / 0.727 | 0.878 / 0.759 |

I kept 250. With the reranker the three sizes are within noise (200 words is 3 English questions better, and slightly worse on the Persian MRR). Smaller chunks would make the prompt shorter and the LLM faster, so this is worth testing again on the answers, not only on the search.

## 5. The prompt

`python -m experiments.prompts <name>`, then `python -m experiments.compare_runs`. Every variant answers the same 141 questions (the staff questions) with the real LLM. I changed one or two rules of the system prompt:

- **base**: my first prompt ("6. Keep the answer short and clear.").
- **short**: rule 6 becomes "Start with the direct answer. Keep it to 2 to 4 short sentences in simple words."
- **partial**: rule 3 lets the model answer the part of a question that the sources cover and say what is missing.
- **both**: short and partial together.

| All 141 questions | base | **short** | partial | both |
|---|---|---|---|---|
| answerable questions that got an answer (of 107) | 100 | 100 | 95 | 97 |
| right passage cited (of 107) | 92 | **92** | 86 | 88 |
| unanswerable questions that got an answer (of 34) | 1 | 1 | 1 | 0 |
| answers with a number in no source | 0 | 0 | 0 | 0 |
| median answer length in tokens | 58 | **52** | 63 | 57 |

Right passage cited, English / Persian: base 53 / 39, short 53 / 39, partial 53 / 33, both 52 / 36.

The short prompt wins. The quality is the same and the answers are about 10% shorter overall (English 47 to 42 tokens; the Persian answers did not get shorter), so they are a little faster to write and easier to read. The "partial" idea sounded good, but the model then refused more questions (95 answers instead of 100) and cited the right page less often. The times in the csvs (about 10 seconds) are not comparable, because four runs shared the GPU.

## 6. The LLM: Qwen 2.5 32B or Gemma 3 12B?

Qwen writes awkward Persian, so I tried Gemma 3 12B, a smaller model with a good reputation for many languages. I ran it with `LLM_MODEL=namazi-gemma12 python -m scripts.check_answers` (a model made from `gemma3:12b` with a context of 8192). Both runs below use the final prompt.

| All 141 questions | Qwen 32B | Gemma 12B |
|---|---|---|
| answerable questions that got an answer (of 107) | 106 | 107 |
| right passage cited (of 107) | 93 | 92 |
| **unanswerable questions that got an answer (of 34)** | **0** | **3** (all English) |
| answers with a number in no source | 0 | 1 |
| median seconds per question | 3.8 | 3.2 |

On the 65 Persian questions alone (the answers are in `results/persian_*.csv`):

| Persian questions | Qwen 32B | Gemma 12B | Gemma 12B, no language line |
|---|---|---|---|
| answerable questions that got an answer (of 49) | 48 | 49 | 47 |
| right passage cited | 41 | 41 | 38 |
| unanswerable questions that got an answer (of 16) | 2 | 0 | 0 |
| median seconds per question | 4.9 | 3.6 | 3.2 |
| **Persian questions answered in English** | 0 | 0 | **41** |

Gemma first answered almost every Persian question **in English**. A last line in the message to the model ("Write the answer in Persian.") fixed it, and I kept that line in the app, because it costs nothing and protects any model.

I kept Qwen. Gemma is about 1.4 times faster and as good for Persian, but it made up answers for 3 of the 18 English questions the books cannot answer, and for a medical app refusing correctly matters more than speed. A note on noise: in this Persian-only run Qwen made up 2 answers, while in my main run it made up none. A difference of one or two answers between two runs is noise, and the 34 unanswerable questions are not many.

## Ideas I have not tried yet

- A Persian-capable LLM only for Persian questions, or Persian health pages as sources, so that Persian is not a translation.
- Searching with both the typed and the translated Persian question and merging the results.
- A faster model server (llama.cpp server, vLLM, SGLang) for many users at the same time.
- A cache for questions that people ask again and again.
- A second check, "does the answer match the question?", to catch off-topic answers.
- A different model as the judge, so that the answering model does not judge its own answers.
