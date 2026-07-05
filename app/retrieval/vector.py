from app.config import settings
from app.ingestion.indexer import get_qdrant_client, embed_texts


def search(query: str) -> list[dict]:
    """Returns [{chunk_id, doc_title, page_number, section_heading, score}, ...]
    (text is fetched later from Postgres by chunk_id -- Qdrant payload is metadata
    only, per the 'Postgres is the source of truth for text' rule in BLUEPRINT.md §4)."""
    query_embedding = embed_texts([query])[0]
    client = get_qdrant_client()

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_embedding,
        limit=settings.vector_top_k,
    )

    return [
        {
            "chunk_id": str(hit.payload["chunk_id"]),
            "doc_title": hit.payload["doc_title"],
            "page_number": hit.payload["page_number"],
            "section_heading": hit.payload.get("section_heading"),
            "score": hit.score,
        }
        for hit in results
    ]
