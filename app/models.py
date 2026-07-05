from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunks_indexed: int
    skipped_duplicate: bool


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source_index: int
    chunk_id: str
    doc_title: str
    page_number: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    fallback_reason: str | None = None
    log_id: str


class FeedbackRequest(BaseModel):
    log_id: str
    rating: str  # 'up' | 'down'
    comment: str | None = None
