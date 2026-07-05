# RAG Knowledge Base

A production-grade RAG (Retrieval-Augmented Generation) document Q&A system with
hybrid retrieval (BM25 + vector), reranking, citation-verified generation, and a
repeatable evaluation harness.

See `BLUEPRINT.md` for the full design rationale and trade-off discussion this code
implements.

## Quick start (local, Docker)

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   # edit .env: OPENAI_API_KEY, COHERE_API_KEY (or swap for your chosen providers)
   ```

2. Bring up the stack:
   ```bash
   docker-compose up --build
   ```
   This starts:
   - `postgres` — metadata, chunk text, BM25 index, logs, feedback
   - `qdrant` — vector index
   - `api` — FastAPI backend on http://localhost:8000
   - `frontend` — Streamlit UI on http://localhost:8501

3. On first boot, the API automatically runs `scripts/init_db.sql` to create tables.

4. Open http://localhost:8501, upload a PDF, and start asking questions.

## Running without Docker (local Python)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# You still need Postgres and Qdrant running somewhere (Docker is easiest for just these two):
docker-compose up postgres qdrant -d

export $(cat .env | xargs)   # or use direnv / your own env loader
uvicorn app.main:app --reload --port 8000

# In a second terminal:
streamlit run frontend/streamlit_app.py
```

## Running the evaluation harness

```bash
python -m app.eval.run_eval
```

This sends every question in `app/eval/eval_set.json` through the real `/query`
endpoint, scores retrieval (Recall@5, MRR) and generation (faithfulness, answer
relevancy, context relevancy via Ragas, plus a custom citation-correctness check),
and writes a timestamped report to `eval_results/`.

To compare two runs (e.g. before/after a chunking change):

```bash
python -m app.eval.compare_runs eval_results/<run_a>.json eval_results/<run_b>.json
```

## Running tests

```bash
pytest tests/unit -v
pytest tests/integration -v     # requires the Docker stack running
```

## Project layout

```
app/
  main.py              FastAPI app: /upload, /query, /eval, /feedback
  config.py            All tunables as env vars
  db.py                Postgres connection + schema helpers
  models.py            Pydantic request/response models
  logging_utils.py      Structured JSON logging
  ingestion/
    parser.py          PyMuPDF + pdfplumber PDF parsing
    chunker.py         Metadata-aware recursive chunking
    dedupe.py          Content-hash duplicate detection
    indexer.py         Writes chunks to Qdrant + Postgres FTS
  retrieval/
    bm25.py            Postgres full-text search
    vector.py          Qdrant similarity search
    fusion.py          Reciprocal Rank Fusion
    reranker.py        Cohere / cross-encoder reranking
    pipeline.py        Orchestrates the full retrieval flow
  generation/
    prompts.py         System prompt templates
    generator.py       LLM call + fallback logic
    citation_check.py  Citation-ID verification + self-check call
  eval/
    eval_set.json       30-question labeled eval set (answerable/partial/unanswerable)
    metrics.py           Recall@5, MRR, Ragas wrappers, citation correctness
    run_eval.py          Eval loop entrypoint
    compare_runs.py      Diff two eval runs
frontend/
  streamlit_app.py      Upload + chat UI
tests/
  unit/                 Fast, no external services required
  integration/           Requires the Docker stack
scripts/
  init_db.sql            Postgres schema
.github/workflows/ci.yml  Test + eval-gate CI pipeline
```

## What's stubbed vs real

Everything in this scaffold is real, runnable code against the exact stack chosen in
the blueprint — nothing here is a mock. The two things you must supply yourself:

1. **API keys** in `.env` (OpenAI/Anthropic for embeddings+generation, Cohere for
   reranking) — these cost a small amount of real money per call.
2. **Your own eval questions** — `app/eval/eval_set.json` ships with 10 example
   questions structured correctly; expand to 30–50 against your actual corpus before
   you trust the metrics (see `BLUEPRINT.md` §8).

## Build order

Follow the order in `BLUEPRINT.md`'s closing section: ingestion → basic hybrid
retrieval → generation → eval baseline → add reranker (measure delta) → add
citation-verification + self-check (measure delta) → UI/Docker/deploy → CI gate.
The code here is already wired end-to-end, so you can also just run it as-is and
then rip pieces out to rebuild them yourself if you want the "build it from scratch"
learning experience for interviews.
