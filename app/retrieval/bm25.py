"""Thin wrapper around app.db.bm25_search, kept separate so retrieval/pipeline.py
reads cleanly as: bm25.search(...) + vector.search(...) -> fusion.rrf_merge(...)."""
from app import db
from app.config import settings


def search(query: str) -> list[dict]:
    """Returns [{chunk_id, doc_title, page_number, section_heading, text, score}, ...]"""
    results = db.bm25_search(query, settings.bm25_top_k)
    return [
        {
            "chunk_id": str(r["chunk_id"]),
            "doc_title": r["doc_title"],
            "page_number": r["page_number"],
            "section_heading": r["section_heading"],
            "text": r["text"],
            "score": r["score"],
        }
        for r in results
    ]
