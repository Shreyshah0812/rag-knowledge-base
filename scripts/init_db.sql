-- Postgres schema for RAG Knowledge Base
-- Applied automatically on first container boot (see docker-compose.yml volume mount).

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    page_count INT NOT NULL,
    upload_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    doc_title TEXT NOT NULL,
    page_number INT NOT NULL,
    section_heading TEXT,
    content_type TEXT NOT NULL DEFAULT 'prose',   -- 'prose' | 'table'
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    char_count INT NOT NULL,
    content_hash TEXT NOT NULL,
    duplicate_of UUID REFERENCES chunks(chunk_id),
    upload_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_text_search ON chunks USING GIN (text_search);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks (content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

CREATE TABLE IF NOT EXISTS query_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    retrieved_chunk_ids UUID[] NOT NULL,
    bm25_rank_list UUID[],
    vector_rank_list UUID[],
    rerank_scores JSONB,
    top_rerank_score FLOAT,
    fallback_reason TEXT,          -- NULL | 'no_relevant_chunks' | 'llm_declined'
    answer TEXT,
    citations JSONB,
    citation_verification_passed BOOLEAN,
    self_check_passed BOOLEAN,
    latency_ms INT,
    created_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    log_id UUID NOT NULL REFERENCES query_logs(log_id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    created_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()
