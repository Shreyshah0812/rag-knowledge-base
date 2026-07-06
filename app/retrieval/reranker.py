"""
Reranking per BLUEPRINT.md §6/§4: Cohere Rerank for MVP simplicity/cost.
To swap in the self-hosted cross-encoder alternative (cross-encoder/ms-marco-MiniLM-L-6-v2),
implement the same rerank(query, candidates) -> list[dict] signature using
sentence-transformers.CrossEncoder and drop it in here -- nothing else needs to change,
which is the point of keeping this behind one function.
"""
import cohere
from cohere.errors import TooManyRequestsError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from app.config import settings

_client = None


def get_client() -> cohere.Client:
    global _client
    if _client is None:
        _client = cohere.Client(settings.cohere_api_key)
    return _client


@retry(
    retry=retry_if_exception_type(TooManyRequestsError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    candidates: list of dicts with a 'text' key (chunk text) plus other metadata.
    Returns candidates re-ordered best-first, each with an added 'rerank_score' key
    (Cohere's relevance score, 0-1). Truncated to settings.rerank_top_k.

    Retries with exponential backoff on 429 rate-limit errors -- relevant on a
    Cohere trial key (10 calls/minute), but a real-world concern on paid tiers too.
    """
    if not candidates:
        return []

    client = get_client()
    docs = [c["text"] for c in candidates]

    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=min(settings.rerank_top_k, len(docs)),
    )

    reranked = []
    for result in response.results:
        candidate = dict(candidates[result.index])
        candidate["rerank_score"] = result.relevance_score
        reranked.append(candidate)

    return reranked
