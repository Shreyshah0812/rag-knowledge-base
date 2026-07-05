"""
Postgres is the single source of truth for chunk text + metadata (BLUEPRINT.md §4).
Qdrant and the BM25 tsvector index are derived stores that can always be rebuilt
from here. Keep that invariant when you extend this file: never write chunk text
anywhere that isn't also written here first.
"""
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from contextlib import contextmanager

from app.config import settings


@contextmanager
def get_conn():
    conn = psycopg.connect(settings.database_url, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_document(filename: str, doc_title: str, checksum: str, page_count: int) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, doc_title, checksum, page_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (checksum) DO UPDATE SET filename = EXCLUDED.filename
                RETURNING doc_id, (xmax = 0) AS inserted
                """,
                (filename, doc_title, checksum, page_count),
            )
            row = cur.fetchone()
            return str(row["doc_id"]), row["inserted"]


def delete_chunks_for_doc(doc_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))


def insert_chunk(chunk: dict) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks
                    (doc_id, doc_title, page_number, section_heading, content_type,
                     chunk_index, text, char_count, content_hash, duplicate_of)
                VALUES (%(doc_id)s, %(doc_title)s, %(page_number)s, %(section_heading)s,
                        %(content_type)s, %(chunk_index)s, %(text)s, %(char_count)s,
                        %(content_hash)s, %(duplicate_of)s)
                RETURNING chunk_id
                """,
                chunk,
            )
            return str(cur.fetchone()["chunk_id"])


def find_hash_within_doc(doc_id: str, content_hash: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM chunks WHERE doc_id = %s AND content_hash = %s LIMIT 1",
                (doc_id, content_hash),
            )
            return cur.fetchone() is not None


def find_hash_across_docs(content_hash: str) -> str | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM chunks WHERE content_hash = %s LIMIT 1",
                (content_hash,),
            )
            row = cur.fetchone()
            return str(row["chunk_id"]) if row else None


def bm25_search(query: str, top_k: int) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_title, page_number, section_heading, text,
                       ts_rank(text_search, plainto_tsquery('english', %s)) AS score
                FROM chunks
                WHERE text_search @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            return cur.fetchall()


def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_title, page_number, section_heading, text, content_type
                FROM chunks WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            return cur.fetchall()


def insert_query_log(log: dict) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            params = {
                **log,
                "rerank_scores": Jsonb(log["rerank_scores"]),
                "citations": Jsonb(log["citations"]),
            }
            cur.execute(
                """
                INSERT INTO query_logs
                    (question, retrieved_chunk_ids, bm25_rank_list, vector_rank_list,
                     rerank_scores, top_rerank_score, fallback_reason, answer, citations,
                     citation_verification_passed, self_check_passed, latency_ms)
                VALUES (%(question)s, %(retrieved_chunk_ids)s, %(bm25_rank_list)s,
                        %(vector_rank_list)s, %(rerank_scores)s, %(top_rerank_score)s,
                        %(fallback_reason)s, %(answer)s, %(citations)s,
                        %(citation_verification_passed)s, %(self_check_passed)s,
                        %(latency_ms)s)
                RETURNING log_id
                """,
                params,
            )
            return str(cur.fetchone()["log_id"])


def insert_feedback(log_id: str, rating: str, comment: str | None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (log_id, rating, comment) VALUES (%s, %s, %s)",
                (log_id, rating, comment),
            )
