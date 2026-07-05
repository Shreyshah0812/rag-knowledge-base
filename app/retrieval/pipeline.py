"""
Orchestrates the full retrieval flow per BLUEPRINT.md §6:
BM25 top-20 + vector top-20 -> RRF merge -> rerank -> top-5 final context,
with the score-threshold fallback check applied here so callers (the /query
endpoint and the eval harness) share one code path -- this matters for eval
validity, see BLUEPRINT.md §8.
"""
from app.retrieval import bm25, vector, fusion, reranker
from app.config import settings
from app import db


def retrieve(question: str) -> dict:
    """
    Returns:
    {
        "chunks": [...],              # final top-k chunks with full text + metadata
        "top_rerank_score": float,
        "below_threshold": bool,
        "bm25_chunk_ids": [...],      # for eval / logging
        "vector_chunk_ids": [...],
    }
    """
    bm25_results = bm25.search(question)
    vector_results = vector.search(question)

    fused = fusion.rrf_merge(bm25_results, vector_results)

    # Fused candidates from vector search don't carry 'text' yet (Qdrant payload is
    # metadata-only) -- backfill text from Postgres before reranking.
    missing_text_ids = [c["chunk_id"] for c in fused if "text" not in c]
    if missing_text_ids:
        rows = {str(r["chunk_id"]): r["text"] for r in db.get_chunks_by_ids(missing_text_ids)}
        for c in fused:
            if "text" not in c:
                c["text"] = rows.get(c["chunk_id"], "")

    reranked = reranker.rerank(question, fused[: max(settings.bm25_top_k, settings.vector_top_k)])

    top_score = reranked[0]["rerank_score"] if reranked else 0.0
    below_threshold = top_score < settings.rerank_threshold

    return {
        "chunks": reranked,
        "top_rerank_score": top_score,
        "below_threshold": below_threshold,
        "bm25_chunk_ids": [r["chunk_id"] for r in bm25_results],
        "vector_chunk_ids": [r["chunk_id"] for r in vector_results],
    }
