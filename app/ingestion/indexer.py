"""
Indexing per BLUEPRINT.md §5:
- Embed each chunk and write to Qdrant (payload carries all metadata for filtering).
- Write the same chunk text + metadata to Postgres (source of truth, and the BM25
  tsvector index is a generated column there -- see scripts/init_db.sql).
- Skip same-document hash collisions before indexing; record cross-document
  collisions as duplicate_of pointers.
"""
import hashlib
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_document
from app.ingestion.dedupe import content_hash
from app import db
from app.logging_utils import get_logger

logger = get_logger(__name__)

_openai_client = None
_qdrant_client = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
        _ensure_collection(_qdrant_client)
    return _qdrant_client


def _ensure_collection(client: QdrantClient) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_openai_client()
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in response.data]


def file_checksum(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(file_path: str, filename: str) -> dict:
    """
    Full ingestion flow for one uploaded PDF:
    parse -> chunk -> dedupe -> embed -> write to Qdrant + Postgres.
    Returns a summary dict used by the /upload endpoint.
    """
    checksum = file_checksum(file_path)
    doc_title = filename.rsplit(".", 1)[0]

    pages = parse_pdf(file_path)
    page_count = len(pages)

    doc_id, inserted = db.insert_document(filename, doc_title, checksum, page_count)

    if not inserted:
        # Same checksum already ingested -- re-ingest cleanly (delete + re-chunk)
        # in case the caller wants to force a refresh, otherwise this is a no-op path.
        logger.info(f"Document with checksum {checksum} already exists; re-indexing chunks.")
        db.delete_chunks_for_doc(doc_id)

    raw_chunks = chunk_document(pages)

    chunks_to_embed = []
    chunk_records = []

    for rc in raw_chunks:
        h = content_hash(rc["text"])

        if db.find_hash_within_doc(doc_id, h):
            continue  # same-document duplicate (boilerplate/footer/TOC) -- skip

        duplicate_of = db.find_hash_across_docs(h)  # None if no cross-doc match

        record = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "page_number": rc["page_number"],
            "section_heading": rc["section_heading"],
            "content_type": rc["content_type"],
            "chunk_index": rc["chunk_index"],
            "text": rc["text"],
            "char_count": rc["char_count"],
            "content_hash": h,
            "duplicate_of": duplicate_of,
        }
        chunk_records.append(record)
        chunks_to_embed.append(rc["text"])

    if not chunk_records:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "chunks_indexed": 0,
            "skipped_duplicate": not inserted,
        }

    embeddings = embed_texts(chunks_to_embed)

    qdrant = get_qdrant_client()
    points = []

    for record, embedding in zip(chunk_records, embeddings):
        chunk_id = db.insert_chunk(record)
        points.append(
            PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "doc_id": record["doc_id"],
                    "doc_title": record["doc_title"],
                    "page_number": record["page_number"],
                    "section_heading": record["section_heading"],
                    "content_type": record["content_type"],
                },
            )
        )

    qdrant.upsert(collection_name=settings.qdrant_collection, points=points)

    logger.info(f"Indexed {len(points)} chunks for document {filename}")

    return {
        "doc_id": doc_id,
        "filename": filename,
        "page_count": page_count,
        "chunks_indexed": len(points),
        "skipped_duplicate": False,
    }
