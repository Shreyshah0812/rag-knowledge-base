"""
RRF per BLUEPRINT.md §6: score(chunk) = sum(1 / (k + rank_i)) across both rank lists.
Rank-based (not raw-score-based) so BM25 and cosine-similarity scores -- which live
on incomparable scales -- never need to be normalized against each other.
"""
from app.config import settings


def rrf_merge(bm25_results: list[dict], vector_results: list[dict]) -> list[dict]:
    """
    bm25_results / vector_results: lists of dicts with a 'chunk_id' key, already
    ordered best-first (rank 0 = best).
    Returns a merged, deduplicated list ordered by fused RRF score, descending.
    Each returned dict carries 'chunk_id', 'rrf_score', 'in_bm25', 'in_vector'.
    """
    k = settings.rrf_k
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        meta.setdefault(cid, {}).update(r)
        meta[cid]["in_bm25"] = True

    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        meta.setdefault(cid, {}).update(r)
        meta[cid]["in_vector"] = True

    merged = []
    for cid, score in scores.items():
        entry = meta[cid]
        entry["chunk_id"] = cid
        entry["rrf_score"] = score
        entry.setdefault("in_bm25", False)
        entry.setdefault("in_vector", False)
        merged.append(entry)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged
