# RAG Knowledge Base — Production Build Blueprint

**Assumptions stated up front (per your request to flag assumptions):**
- Solo builder, no team, no budget for enterprise infra.
- You have API access to at least one hosted LLM (OpenAI/Anthropic) and can spend a small amount ($10–30) on API calls during eval runs.
- "Production-grade" here means *disciplined engineering practices at small scale* — proper evaluation, proper logging, proper failure handling — not Kubernetes clusters or multi-region deployments. That's the honest positioning for an early-career portfolio piece, and it's also what interviewers actually respect, because it shows you know what production means without over-building.
- Document corpus size target: 50–500 documents, 1k–50k chunks. Large enough to require real retrieval engineering, small enough to run on a laptop + one cheap cloud VM.

---

## 1. Project Summary

**RAG Knowledge Base** is a document question-answering system: a user uploads PDFs (policy docs, manuals, internal wikis, contracts), the system ingests and indexes them, and the user can then ask natural-language questions and get back an answer that is grounded in the actual document text, with page/chunk-level citations, a relevance/confidence signal, and an honest "I don't know" when the documents don't support an answer.

**Real-world problem it solves:** Every company with more than a handful of internal documents (HR policies, compliance manuals, engineering runbooks, vendor contracts) has the same problem — employees either can't find the answer or find a stale/wrong one, and search-by-keyword tools don't understand phrasing variation. This system replaces "ctrl+F across 40 PDFs" with an answer plus proof of where it came from.

**Why it's better than a basic chatbot:** A basic chatbot (or naive "stuff everything into one prompt" RAG demo) will confidently answer questions it has no grounding for, can't tell you *where* an answer came from, and degrades silently as the document set grows. This system is differentiated by three engineering decisions that most portfolio RAG projects skip: (1) hybrid retrieval instead of vector-only, because keyword/BM25 catches exact terms (policy numbers, acronyms, names) that embeddings blur; (2) a citation-verification step so the model can't claim a source it didn't use; (3) a repeatable evaluation harness, so you can prove — not just claim — that a change improved the system. Those three things are what separate "I called an embeddings API" from "I built a retrieval system."

---

## 2. Feature List

### MVP (must-have)
| Feature | Must/Optional |
|---|---|
| PDF/text upload + parsing | Must |
| Metadata-aware chunking (page, section, doc id) | Must |
| Hybrid retrieval (BM25 + vector) | Must |
| Reranking of top candidates | Must |
| Grounded answer generation with citations | Must |
| "Not supported by documents" fallback | Must |
| Basic web UI (Streamlit) to upload + query | Must |
| Evaluation set + scoring script | Must |
| Request/response logging | Must |

### Advanced (optional, in priority order)
| Feature | Must/Optional |
|---|---|
| Table-aware parsing (structured extraction from tables) | Optional — high value |
| Multi-doc comparison answers ("how does policy A differ from B") | Optional |
| Query rewriting / decomposition for multi-hop questions | Optional |
| Conversation memory (follow-up questions) | Optional |
| User feedback capture (thumbs up/down → eval set growth) | Optional — cheap, high resume value |
| Admin dashboard for corpus stats + eval history | Optional |
| Streaming responses | Optional — nice UX, low technical depth signal |
| Access control / per-user document scoping | Optional — skip unless you want to demonstrate multi-tenant design |

I'd build feedback capture and table-aware parsing before anything else on this list — they're cheap and they directly strengthen the evaluation story, which is your biggest interview lever.

---

## 3. Architecture

```
                          ┌─────────────────────────┐
                          │   Streamlit Frontend     │
                          │  (upload, chat, sources) │
                          └───────────┬─────────────┘
                                      │ HTTP
                          ┌───────────▼─────────────┐
                          │   FastAPI Backend        │
                          │  /upload  /query  /eval  │
                          └───────────┬─────────────┘
                    ┌─────────────────┼───────────────────┐
                    │                 │                   │
        ┌───────────▼───────┐ ┌───────▼────────┐ ┌────────▼────────┐
        │ Ingestion Pipeline │ │ Retrieval       │ │ Generation      │
        │                    │ │ Pipeline        │ │ Pipeline        │
        │ parse→chunk→embed  │ │ BM25 + vector → │ │ prompt build →  │
        │ →index             │ │ merge → rerank  │ │ LLM call →      │
        │                    │ │                 │ │ citation check  │
        └───────────┬────────┘ └───────┬─────────┘ └────────┬────────┘
                    │                  │                    │
        ┌───────────▼──────────────────▼────────────────────▼────────┐
        │  Storage Layer                                              │
        │  - Postgres: doc metadata, chunk metadata, feedback, logs   │
        │  - Qdrant: chunk vectors                                     │
        │  - Local/S3: raw PDF files                                   │
        │  - SQLite/Postgres FTS or Elasticsearch: BM25 index          │
        └───────────────────────────┬──────────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Evaluation Pipeline    │
                          │ (offline script, run   │
                          │  against eval set)     │
                          └────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Logging / Monitoring   │
                          │ (structured logs +     │
                          │  simple metrics dash)  │
                          └────────────────────────┘
```

### Data flow, step by step

**Ingestion (offline, triggered by upload):**
1. User uploads PDF via Streamlit → FastAPI `/upload`.
2. Backend saves raw file to disk/S3, creates a `documents` row (id, filename, upload_ts, page_count, checksum).
3. Parser extracts text per page (and tables separately — see §5).
4. Chunker splits text into overlapping chunks, attaching metadata to each (doc_id, page_number, section_heading, chunk_index).
5. Each chunk is embedded and written to Qdrant; the same chunk text is written to the BM25/FTS index and to a `chunks` Postgres table (source of truth for text + metadata).
6. Duplicate-content check (see §5) runs before indexing to avoid redundant chunks.

**Query (online, per user question):**
1. User submits question via Streamlit → FastAPI `/query`.
2. Retrieval pipeline runs BM25 search and vector search in parallel, merges results (reciprocal rank fusion), sends top ~20 to a reranker, keeps top 5.
3. Generation pipeline builds a prompt with the 5 chunks (each tagged with a citation ID), calls the LLM.
4. Post-processing step verifies every citation the model used actually maps to a retrieved chunk (see §7) and computes a relevance/confidence signal from reranker scores.
5. If top reranker score is below threshold, short-circuit to fallback response instead of calling the LLM (saves cost, avoids hallucination on empty retrieval).
6. Response (answer + citations + confidence) returned to frontend; full request/response/scores logged.

**Evaluation (offline, run manually or in CI):**
1. Script loads fixed eval set (JSON of question, expected answer type, expected source chunks).
2. Runs each question through the same `/query` path (not a separate code path — this matters, see §8).
3. Scores retrieval and generation, writes a report (JSON + printed table) with run timestamp and git commit hash, so you can diff across changes.

---

## 4. Tech Stack (single recommended stack, justified)

| Component | Choice | Why |
|---|---|---|
| Backend | **FastAPI** | You already know it; async support matters for parallel BM25+vector calls. |
| Frontend | **Streamlit** | You already know it; fastest path to a usable demo UI, and demo-ability matters more than a custom React frontend for this project's ROI. |
| PDF parsing | **PyMuPDF (fitz)** + **pdfplumber** for tables | PyMuPDF is fast and gives reliable page-level text + layout info; pdfplumber is specifically better at table cell extraction. Using both for their strengths is a defensible, specific choice — don't reach for a heavier framework (e.g. Unstructured.io) unless you want to spend time explaining a black box in interviews. |
| Chunking | **Custom recursive chunker** (LangChain's `RecursiveCharacterTextSplitter` as a base, with your own metadata wrapper) | Don't hand the whole pipeline to a framework — write the chunker yourself so you can explain every decision in an interview. Using LangChain only for the splitter primitive is fine. |
| Embeddings | **OpenAI `text-embedding-3-small`** | Cheap (~$0.02/1M tokens), strong enough at this corpus scale, no self-hosting overhead. If you want a fully open-source stack for cost/story reasons, swap for `BAAI/bge-small-en-v1.5` via `sentence-transformers` — mention both, pick one and commit. |
| Vector DB | **Qdrant** (self-hosted via Docker, or Qdrant Cloud free tier) | Easier local Docker story than Milvus, more production-credible than FAISS-in-memory (FAISS has no persistence/filtering story out of the box), and has native hybrid-search support if you want to consolidate later. |
| Keyword/BM25 search | **Postgres full-text search (tsvector)** | You already need Postgres for metadata; avoid running a second heavy service (Elasticsearch) just for BM25 at this corpus size. This is a real, defensible trade-off to state in interviews: "Elasticsearch would be the standard hybrid-search partner, but at this corpus size Postgres FTS gives 90% of the value with one fewer service to operate." |
| Reranker | **Cohere Rerank API** (`rerank-english-v3.0`) or **`cross-encoder/ms-marco-MiniLM-L-6-v2`** self-hosted | Cohere for simplicity/cost at low volume; the self-hosted cross-encoder if you want to show you can run inference yourself. Pick Cohere for MVP, mention the self-hosted swap as a documented trade-off. |
| LLM for generation | **Claude Sonnet or GPT-4o-mini** | Sonnet-class models are noticeably better at instruction-following for "only answer from context" prompts than smaller models; GPT-4o-mini is the cheap fallback. Use whichever you have API credits for — the prompt design in §7 matters more than the model choice. |
| Evaluation framework | **Custom scripts + Ragas library** for the standard RAG metrics (faithfulness, answer relevancy, context relevancy) | Ragas gives you defensible, named metrics instead of ad hoc scoring, which reads much better on a resume and in interviews than "I eyeballed the answers." |
| Storage | **Postgres** (metadata, chunks, logs, feedback) + local disk or **S3** (raw files) | Postgres is your single source of truth; keep the vector DB and BM25 index as *derived* stores you can always rebuild from Postgres. This is an important design point to be able to explain: never let Qdrant be the only place chunk text lives. |
| Deployment | **Docker Compose** locally, **single VM (Render/Railway/Fly.io or a $6 DigitalOcean droplet)** for the demo | Realistic for a solo builder; skip Kubernetes entirely and say so explicitly if asked — "I chose not to use Kubernetes because a single-service, low-traffic portfolio app doesn't need orchestration, and I can explain what I'd change at higher scale" is a strong interview answer. |

---

## 5. Document Ingestion Pipeline

### Parsing
- Use PyMuPDF to extract text **per page**, preserving page numbers as metadata immediately — never lose the page number after this step.
- Use PyMuPDF's block/font-size info to detect headings heuristically: a line whose font size is materially larger than the page's median font size, or that's bold and short (<80 chars), is treated as a section heading candidate. Store the most recent heading seen as `section_heading` metadata for all subsequent chunks until the next heading.
- Run pdfplumber separately on each page to detect tables (`page.extract_tables()`). If a page has tables, extract them as **markdown tables** and store them as their own chunks tagged `content_type: table`, separate from the surrounding prose chunk for that page (don't interleave table text into paragraph chunking — it corrupts both).

### Chunking strategy
- Chunk size: **800 characters** (~150-200 tokens) with **150-character overlap**. This size balances retrieval precision (small enough that a chunk is about one idea) against generation quality (large enough to contain a full sentence/answer with surrounding context).
- Chunk on paragraph boundaries first (split on `\n\n`), then apply the 800-char recursive splitter within any paragraph that's still too long. Never split mid-sentence if avoidable — use the recursive splitter's separator hierarchy (`\n\n`, `\n`, `. `, ` `).
- Table chunks are **not** split by character count — each table is one chunk (or one chunk per table if a table is huge, split by row groups, never by column).

### Metadata per chunk (store all of this — every field is used somewhere downstream)
```json
{
  "chunk_id": "uuid",
  "doc_id": "uuid",
  "doc_title": "Employee Handbook 2025",
  "page_number": 14,
  "section_heading": "3.2 Remote Work Policy",
  "content_type": "prose | table",
  "chunk_index": 42,
  "char_count": 780,
  "content_hash": "sha256 of normalized text",
  "upload_ts": "..."
}
```

### Handling duplicate content
- Before indexing a chunk, compute a normalized content hash (lowercase, strip whitespace, strip punctuation) and check it against a `content_hash` index in Postgres.
- If a hash collision is found **within the same document** (common with repeated headers/footers, boilerplate, table of contents entries), skip indexing that chunk.
- If a hash collision is found **across different documents** (e.g., the same policy paragraph appears in two handbooks), keep both chunks but store a `duplicate_of` pointer — this lets you, at query time, deduplicate near-identical citations shown to the user so the answer doesn't cite the same sentence twice under two document names.
- This is worth explicitly building — it's a real production RAG problem (corpora accumulate near-duplicate documents) and it's a good interview talking point.

### Indexing strategy
- Vector index: one Qdrant collection, payload includes all metadata above so you can filter (e.g., "search only doc_type=policy") without a second lookup.
- BM25 index: a Postgres `tsvector` column on the `chunks` table, GIN-indexed, kept in sync via a trigger or an explicit write in the ingestion code (explicit write is more debuggable — prefer that).
- Re-indexing: if a document is re-uploaded (checksum match on the raw file), skip full ingestion; if the file changed, delete old chunks by `doc_id` from both indexes before re-inserting — never let stale chunks accumulate.

---

## 6. Retrieval Strategy

### Why hybrid, precisely
Vector search is good at *semantic* matches ("time off policy" matching a chunk that says "leave entitlement") but is weak on exact tokens — policy numbers, product codes, acronyms, proper nouns — because embeddings compress those into generic-ish vectors. BM25 is the reverse: exact token overlap wins, semantic paraphrase loses. Real user questions mix both needs in the same query, so you run both and merge.

### The mechanics
1. Run BM25 (Postgres FTS `ts_rank`) → top 20 candidates.
2. Run vector search (Qdrant cosine similarity) → top 20 candidates.
3. Merge with **Reciprocal Rank Fusion (RRF)**: `score(chunk) = Σ 1/(k + rank_i)` across both lists, `k=60` (standard RRF constant). This avoids needing to normalize BM25 and cosine scores onto the same scale, which is a real problem if you try to just average raw scores.
4. Take the top ~20 fused candidates, send to the **reranker** (Cohere or cross-encoder), which scores each candidate against the raw query with a model built specifically for query-document relevance (much more accurate than either retrieval method alone, but too slow/expensive to run over the whole corpus, hence the two-stage design).
5. Keep the reranker's top 5 as final context.

### Top-k choices, stated explicitly
- BM25 top-20, vector top-20 (retrieval stage): wide enough to catch true positives that rank low in either individual method.
- Reranker top-5 (final context): narrow enough to keep the prompt focused and citations traceable; wider than 5 tends to dilute the generation model's grounding and increases hallucination risk from irrelevant chunks sitting in context.

### Fallback logic
- If the reranker's top score is below a threshold (start at **0.3** for Cohere's 0–1 relevance score, tune against your eval set), skip generation entirely and return: *"I couldn't find information about this in the uploaded documents."* This is your hallucination firewall at the retrieval layer, before the LLM is even involved.
- If BM25 returns zero results and vector search returns low-similarity results, that's a strong unanswerable signal — surface it distinctly in logs so you can tell "no relevant docs exist" apart from "docs exist but the reranker was unsure."

### Preserving citation quality
- Every chunk sent to the LLM keeps its `chunk_id`, `doc_title`, `page_number` attached in the prompt (e.g., `[Source 3 — Employee Handbook 2025, p.14]`).
- The LLM is instructed to cite using these exact source tags (see §7 prompt).
- After generation, a citation-verification step (regex-extract `[Source N]` tags from the answer, confirm N is a valid index into the chunks that were actually sent) rejects/flags any citation to a source that wasn't in context — this is a cheap, high-value hallucination check most portfolio projects skip.

---

## 7. Generation Strategy

### System prompt for the answer generation step
```
You are a document question-answering assistant. You answer ONLY using the
numbered source excerpts provided below. You do not use outside knowledge,
even if you know the answer from general training.

Rules:
1. Every factual claim in your answer must be followed by a citation tag
   in the form [Source N], where N matches the source excerpt number.
2. If the answer is fully supported by the sources, answer directly and
   completely, with citations.
3. If the sources only partially answer the question, answer what is
   supported, cite it, and explicitly state what part of the question is
   NOT answered by the provided documents.
4. If none of the sources are relevant to the question, respond exactly
   with: "I don't have enough information in the uploaded documents to
   answer this." Do not guess.
5. Never combine information across sources to infer something that is
   not explicitly stated in at least one of them.
6. Be concise. Do not restate the question. Do not add disclaimers beyond
   what these rules require.

Source excerpts:
{numbered_chunks_with_metadata}

Question: {user_question}
```

### Forcing grounded answers
- The prompt structure above (numbered sources, mandatory citation format) is the primary lever — models are markedly better at "cite what you use" than "don't say anything false," because the former is a checkable format instruction and the latter is an unverifiable request.
- Set `temperature=0` for generation — this isn't a creative task, and determinism makes your eval runs comparable across changes.
- Add a **second, cheap LLM call** (or a simple heuristic) as a self-check: send the generated answer plus the sources back with the prompt *"Does every claim in this answer appear in the sources? Answer only YES or NO."* If NO, discard the answer and return the fallback message. This costs one extra small call per query but meaningfully reduces hallucination — and it's a specific, explainable technique for the "how do you control hallucination" interview question.

### What to do when no answer is found
- Retrieval-layer fallback (score threshold, §6) catches "nothing relevant was retrieved."
- Generation-layer fallback (rule 4 in the prompt + the self-check call) catches "something was retrieved, but it doesn't actually answer the question."
- Both fallback paths return the same user-facing message but are logged with different `fallback_reason` values (`no_relevant_chunks` vs `llm_declined`) so your monitoring can distinguish a retrieval problem from a generation problem.

### Reducing hallucinations, concretely
1. Retrieval score threshold before calling the LLM at all (cheapest, catches the most common case — no relevant docs).
2. Strict, numbered-citation prompt format (cheap, catches most in-context hallucination).
3. Citation-ID verification post-processing (cheap, catches citation-to-nonexistent-source errors).
4. Self-check LLM call (more expensive, catches subtler unsupported claims that still cite a real source).
5. `temperature=0` (free, improves determinism and reduces creative drift).

---

## 8. Evaluation Plan

### Eval set design (build 30–50 questions; this is a day of work, don't skip it)
- **(a) Directly answerable (~40%)**: questions where the answer is stated clearly in one chunk. Example: "What is the maximum number of paid sick days per year?"
- **(b) Partially answerable (~30%)**: questions where the documents answer part but not all. Example: "What is the remote work policy for contractors?" when the handbook only covers full-time employees.
- **(c) Unanswerable (~30%)**: questions with no basis in the corpus at all, including some that are deliberately adjacent/tempting (plausible-sounding but not covered) to stress-test the fallback logic. Example: "What is the company's policy on cryptocurrency compensation?" when it's never mentioned.

For each question, hand-label: the expected answer (or "should decline"), and the `chunk_id`(s) that should be retrieved (your ground truth for retrieval scoring).

### Retrieval quality metrics
- **Recall@5**: is at least one ground-truth chunk in the top 5 retrieved? (Your primary retrieval metric.)
- **MRR (Mean Reciprocal Rank)**: how high does the first ground-truth chunk rank?
- Track both **pre-rerank** and **post-rerank**, so you can quantify the reranker's actual contribution — a genuinely good thing to show in a resume bullet ("reranking improved Recall@5 by X points").

### Answer quality metrics (use Ragas)
- **Faithfulness**: does the answer's claims logically follow from the retrieved context? (Ragas computes this by extracting claims and checking entailment against context.)
- **Answer relevancy**: does the answer actually address the question asked?
- **Context relevancy**: how much of the retrieved context was actually relevant/used?
- **Citation correctness** (custom metric, not in Ragas): percentage of `[Source N]` citations in the answer that map to a chunk containing the cited claim — check this by string/semantic overlap between the claim sentence and the cited chunk text.

### The evaluation loop
1. `eval_set.json` lives in the repo, version-controlled.
2. `run_eval.py` sends every question through the **actual `/query` endpoint** (not a shortcut reimplementation — testing the real code path is the point).
3. Script computes all metrics above, writes `eval_results/{git_commit_hash}_{timestamp}.json`.
4. A small comparison script diffs two eval runs side by side (e.g., before/after a chunking change) and prints a table of metric deltas.
5. Run this after every meaningful pipeline change (chunk size, reranker swap, prompt edit) — this is what turns "I tuned the prompt" into "I tuned the prompt and measured a 12-point faithfulness improvement," which is the sentence that actually lands in an interview.

---

## 9. Testing and Quality

### Unit tests
- Chunker: given known input text, assert chunk boundaries, overlap, and metadata attachment are correct.
- Duplicate-hash detection: assert exact and near-duplicate chunks are flagged correctly.
- Citation-verification post-processor: given a mock LLM answer with a citation to a non-existent source index, assert it's caught.
- RRF merge function: given known BM25/vector rank lists, assert fused ranking matches hand-computed expected order.
- Fallback threshold logic: given mocked reranker scores above/below threshold, assert correct branch is taken.

### Integration tests
- Full `/upload` → ingest → `/query` round trip on a small fixture PDF with known content, asserting the expected chunk is retrieved and cited.
- Re-upload of an unchanged file (same checksum) doesn't create duplicate chunks.
- Re-upload of a changed file correctly removes old chunks before inserting new ones.
- End-to-end run of the eval script against a frozen fixture corpus, asserting metrics don't regress below a checked-in baseline (this can literally be a CI gate — see §10).

### Failure modes and edge cases to explicitly test
- Empty/corrupted PDF upload.
- Scanned PDF with no extractable text (should either OCR — stretch goal — or fail gracefully with a clear error, not silently index nothing).
- Question with no relevant documents in the corpus at all.
- Question that's answerable but spans two documents with contradictory information (this is a great one to demo in an interview — show the answer surfaces the contradiction rather than picking one silently).
- Extremely short or malformed user queries.
- Concurrent uploads of the same file.

### Before deployment checklist
- Eval metrics meet your own baseline thresholds (define these numbers before you start tuning, so you're not moving the goalposts).
- All unit + integration tests pass in CI.
- Logging captures enough to debug a bad answer after the fact without re-running anything (see §10 monitoring).
- Secrets (API keys) are not in the repo — checked via a pre-commit hook or CI secret-scan.

---

## 10. Deployment Plan

### Local
- `docker-compose.yml` with services: `api` (FastAPI), `frontend` (Streamlit), `postgres`, `qdrant`. One `docker-compose up` should bring up the whole stack.
- `.env` file (gitignored) for `OPENAI_API_KEY`, `COHERE_API_KEY`, `DATABASE_URL`, `QDRANT_URL`, `RERANK_THRESHOLD`, `CHUNK_SIZE`, `CHUNK_OVERLAP` — every tunable parameter from this document should be an env var, not a hardcoded constant, so your eval loop can sweep them.

### Cloud (realistic for a solo builder)
- Single VM (Fly.io, Railway, or a $6/mo DigitalOcean droplet) running the same `docker-compose.yml`.
- Postgres: use the platform's managed Postgres add-on if available (Railway/Fly both have one) rather than self-hosting it — one less thing to babysit, and "I used a managed database instead of self-hosting to reduce operational burden" is a legitimate, statable trade-off.
- Qdrant: Qdrant Cloud has a free tier sufficient for a portfolio-scale corpus; use it instead of self-hosting the vector DB on your one VM to keep memory usage predictable.
- Raw file storage: S3 (or Cloudflare R2, cheaper) instead of the VM's disk, so redeploying the VM doesn't lose uploaded documents.

### CI/CD (simple, realistic)
- GitHub Actions workflow: on every push to `main`, run unit tests → run integration tests against a fixture corpus → run the eval script and fail the build if faithfulness or Recall@5 drops below your baseline threshold → if all pass, build and push the Docker image.
- Deployment itself can be manual (`git push` to Railway/Fly's git-based deploy) — full auto-deploy-on-merge is a nice-to-have, not worth the setup time relative to the eval-gate CI, which is the part that actually demonstrates engineering discipline.

---

## 11. Resume Bullets

1. Designed and built a retrieval-augmented question-answering system over unstructured PDF corpora, combining BM25 keyword search with vector similarity search via reciprocal rank fusion and a cross-encoder reranking stage, improving Recall@5 by [X]% over vector-only retrieval on a 40-question evaluation set.
2. Implemented a metadata-aware document ingestion pipeline (PyMuPDF/pdfplumber) with page-level provenance tracking, heading-based section detection, and content-hash deduplication, processing [N] documents into [N] indexed chunks with zero duplicate-content leakage in evaluation.
3. Built a hallucination-control system combining a retrieval-confidence threshold, mandatory per-claim source citations, and a citation-verification post-processor, reducing unsupported claims from [X]% to [Y]% (measured via Ragas faithfulness scoring) across a 3-category (answerable/partial/unanswerable) test set.
4. Developed a repeatable evaluation harness using the Ragas framework to score faithfulness, answer relevancy, and context relevancy, integrated into CI to gate deployments on retrieval and generation quality thresholds.
5. Deployed a Dockerized FastAPI + Streamlit RAG application with a Postgres/Qdrant storage layer to a cloud VM, with structured request logging enabling post-hoc debugging of individual answer quality issues.

*(Fill in the bracketed numbers from your actual eval results — don't publish placeholder numbers; that undermines exactly the honesty positioning you asked for.)*

---

## 12. Fifteen Hardest Interview Questions + What a Strong Answer Covers

**1. Why hybrid retrieval instead of just vector search?**
Cover: embeddings blur exact tokens (IDs, acronyms, numbers); BM25 catches those; real queries need both; RRF avoids score-normalization problems between the two.

**2. Why did you choose your chunk size, and how did you validate it?**
Cover: the size trade-off (small = precise but loses context; large = more context but dilutes retrieval precision and increases hallucination surface); that you should have actually tested 2-3 chunk sizes against your eval set and picked based on Recall@5/faithfulness, not gut feel — if you didn't do this, say you'd do it next.

**3. How do you know your system isn't hallucinating?**
Cover: the layered defense (retrieval threshold → citation-format prompt → citation-ID verification → self-check call → faithfulness metric in eval) — the point is no single layer is trusted alone.

**4. What happens when retrieval returns wrong-but-plausible chunks?**
Cover: this is the hardest case — reranking helps but isn't perfect; the self-check LLM call is your last line of defense; be honest that this failure mode isn't fully solved and describe what you'd add (e.g., a second reranker vote, or surfacing lower-confidence answers with a visible warning instead of a hard fallback).

**5. How would this scale to 100,000 documents?**
Cover: Postgres FTS starts to strain — you'd migrate to Elasticsearch/OpenSearch for BM25; you'd add metadata filtering to narrow vector search before ranking; you might need approximate nearest neighbor tuning (HNSW parameters) in Qdrant; ingestion would need to move from synchronous to a queue (Celery/SQS) — the answer should show you know where your current design's ceiling is.

**6. Why RRF instead of just averaging normalized scores?**
Cover: BM25 and cosine similarity live on different, non-comparable scales; RRF is rank-based so it sidesteps normalization entirely, at the cost of losing magnitude information (a very strong BM25 match and a barely-passing one both just contribute by rank).

**7. How do you handle a question that spans multiple documents?**
Cover: current design retrieves top-5 chunks regardless of source document, so multi-doc synthesis happens naturally if the chunks are all retrieved — but the system explicitly does NOT infer connections the documents don't state (rule 5 in the system prompt), which is a deliberate hallucination-control trade-off you should be able to defend.

**8. How do you evaluate something as subjective as "answer quality"?**
Cover: Ragas's approach (claim extraction + entailment checking against source, LLM-judged answer relevancy) turns subjective judgment into a repeatable, automatable score; acknowledge LLM-as-judge has its own bias/noise, and that's why you also hand-labeled ground truth for retrieval metrics, which are objective.

**9. What's your reranker actually buying you, quantitatively?**
Cover: you should have the number — Recall@5 pre-rerank vs post-rerank from your eval logs. If asked and you don't have it, that's a real gap; say you'd instrument it.

**10. Why Postgres FTS instead of Elasticsearch?**
Cover: operational simplicity trade-off at this corpus scale — one fewer service, one storage system doubling as metadata store and search index; name the point where you'd switch (corpus size, query volume, or need for more sophisticated text analysis like stemming/synonyms Postgres FTS handles more crudely).

**11. How do you prevent stale data (e.g., a policy is updated) from causing wrong answers?**
Cover: checksum-based re-ingestion (§5) deletes and replaces chunks by doc_id on file change; acknowledge this is manual-upload-triggered, not push-based, and describe what a real "watch a SharePoint folder" integration would add.

**12. What's your latency budget, and where does time go?**
Cover: parallel BM25+vector calls, then reranker call, then LLM call, then optional self-check LLM call — the two sequential LLM calls are your biggest latency cost; you'd cut the self-check call for latency-sensitive use cases and rely on the other three hallucination defenses instead, a real trade-off between speed and hallucination-safety.

**13. How would you add user feedback (thumbs up/down) into improving the system?**
Cover: feedback rows link back to the specific query log entry (retrieved chunks, scores, generated answer); thumbs-down on questions with high retrieval confidence but bad answers signal a generation problem; thumbs-down with low retrieval confidence signals a retrieval/corpus-coverage gap; over time, negative-feedback questions become new eval-set entries — this directly grows your evaluation rigor from real usage.

**14. What did you get wrong the first time you built this, and how did you find out?**
Cover: have a real answer ready — e.g., an initial chunk size that split answers mid-sentence, discovered via a low Recall@5 on category (a) questions, fixed by switching separator hierarchy. Interviewers weight "tell me about a mistake you caught" answers heavily; don't skip preparing this.

**15. Why not just use a long-context LLM and skip retrieval entirely (stuff the whole corpus in context)?**
Cover: cost (paying full-context tokens on every query vs. paying for embeddings once and small context per query), citation traceability (a model given the whole corpus can't point to the two paragraphs it actually used the way a 5-chunk context can), and the "needle in a haystack" problem — long-context models' recall on facts buried deep in huge contexts is empirically worse than a well-retrieved short context, which you should be able to cite as a known finding rather than an opinion.

---

## What I'd build first, in order

1. Ingestion pipeline + chunker + Postgres schema (get real chunks with metadata into storage).
2. Vector index + BM25 index, basic (non-reranked) hybrid retrieval.
3. Generation with the strict citation prompt, no reranker yet.
4. Eval set (30 questions) + Ragas scoring — measure your v1 baseline before optimizing anything.
5. Add reranker, re-run eval, record the delta.
6. Add citation-verification + self-check call, re-run eval, record the delta.
7. Streamlit UI, Docker Compose, deploy.
8. CI eval-gate, feedback capture, table parsing — polish in whatever order matches your remaining time.

This ordering matters for your resume story too: it means your eval numbers are real deltas you measured as you built, not numbers you back-filled after the fact.
