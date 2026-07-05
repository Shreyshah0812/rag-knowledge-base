"""
FastAPI app implementing the data flow described in BLUEPRINT.md §3:
- /upload: save file -> ingest (parse/chunk/embed/index)
- /query: retrieve (hybrid + rerank) -> threshold check -> generate -> log
- /feedback: thumbs up/down capture, linked back to the query log row (§2, §12 Q13)
"""
import os
import shutil
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.config import settings
from app.models import UploadResponse, QueryRequest, QueryResponse, Citation, FeedbackRequest
from app.ingestion.indexer import ingest_pdf
from app.retrieval.pipeline import retrieve
from app.generation.generator import generate_answer
from app.generation.prompts import FALLBACK_MESSAGE
from app import db
from app.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(title="RAG Knowledge Base")

os.makedirs(settings.raw_file_storage_dir, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported in this scaffold.")

    dest_path = os.path.join(settings.raw_file_storage_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_pdf(dest_path, file.filename)
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(500, f"Ingestion failed: {e}")

    return UploadResponse(**result)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.time()
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be empty.")

    retrieval_result = retrieve(question)
    chunks = retrieval_result["chunks"]

    if retrieval_result["below_threshold"] or not chunks:
        # Retrieval-layer fallback -- BLUEPRINT.md §6 -- never call the LLM here.
        latency_ms = int((time.time() - start) * 1000)
        log_id = db.insert_query_log({
            "question": question,
            "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
            "bm25_rank_list": retrieval_result["bm25_chunk_ids"],
            "vector_rank_list": retrieval_result["vector_chunk_ids"],
            "rerank_scores": {c["chunk_id"]: c.get("rerank_score") for c in chunks},
            "top_rerank_score": retrieval_result["top_rerank_score"],
            "fallback_reason": "no_relevant_chunks",
            "answer": FALLBACK_MESSAGE,
            "citations": [],
            "citation_verification_passed": True,
            "self_check_passed": True,
            "latency_ms": latency_ms,
        })
        return QueryResponse(
            answer=FALLBACK_MESSAGE,
            citations=[],
            confidence=retrieval_result["top_rerank_score"],
            fallback_reason="no_relevant_chunks",
            log_id=log_id,
        )

    gen_result = generate_answer(question, chunks)

    citations = []
    if gen_result["fallback_reason"] is None:
        for i, c in enumerate(chunks, start=1):
            if f"[Source {i}]" in gen_result["answer"]:
                citations.append(Citation(
                    source_index=i,
                    chunk_id=c["chunk_id"],
                    doc_title=c["doc_title"],
                    page_number=c["page_number"],
                ))

    latency_ms = int((time.time() - start) * 1000)
    log_id = db.insert_query_log({
        "question": question,
        "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
        "bm25_rank_list": retrieval_result["bm25_chunk_ids"],
        "vector_rank_list": retrieval_result["vector_chunk_ids"],
        "rerank_scores": {c["chunk_id"]: c.get("rerank_score") for c in chunks},
        "top_rerank_score": retrieval_result["top_rerank_score"],
        "fallback_reason": gen_result["fallback_reason"],
        "answer": gen_result["answer"],
        "citations": [c.model_dump() for c in citations],
        "citation_verification_passed": gen_result["citation_verification_passed"],
        "self_check_passed": gen_result["self_check_passed"],
        "latency_ms": latency_ms,
    })

    return QueryResponse(
        answer=gen_result["answer"],
        citations=citations,
        confidence=retrieval_result["top_rerank_score"],
        fallback_reason=gen_result["fallback_reason"],
        log_id=log_id,
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    if request.rating not in ("up", "down"):
        raise HTTPException(400, "rating must be 'up' or 'down'")
    db.insert_feedback(request.log_id, request.rating, request.comment)
    return {"status": "recorded"}
