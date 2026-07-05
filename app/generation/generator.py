"""
Generation flow per BLUEPRINT.md §7. This is the layer that turns retrieved chunks
into a final answer, applying every hallucination-control layer described in the
blueprint in order:
1. Retrieval-threshold fallback is checked by the caller (app/main.py) before this
   module is even invoked -- if below threshold, we never spend an LLM call.
2. Strict citation-format system prompt (this module).
3. Citation-ID verification (citation_check.verify_citations).
4. Self-check LLM call (citation_check.self_check).
5. temperature=0 throughout.
"""
from openai import OpenAI

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt, FALLBACK_MESSAGE
from app.generation import citation_check
from app.logging_utils import get_logger

logger = get_logger(__name__)

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Returns:
    {
        "answer": str,
        "fallback_reason": None | "llm_declined",
        "citation_verification_passed": bool,
        "self_check_passed": bool,
        "invalid_citations": [...],
    }
    """
    client = get_client()
    user_prompt = build_user_prompt(question, chunks)

    response = client.chat.completions.create(
        model=settings.generation_model,
        temperature=settings.generation_temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content.strip()

    citation_ok, invalid = citation_check.verify_citations(answer, len(chunks))
    if not citation_ok:
        logger.warning(f"Invalid citation indices found: {invalid}")

    self_check_ok = citation_check.self_check(client, answer, chunks)

    if answer.strip() == FALLBACK_MESSAGE:
        return {
            "answer": FALLBACK_MESSAGE,
            "fallback_reason": "llm_declined",
            "citation_verification_passed": True,
            "self_check_passed": True,
            "invalid_citations": [],
        }

    if not citation_ok or not self_check_ok:
        logger.warning(
            "Discarding generated answer due to failed hallucination check "
            f"(citation_ok={citation_ok}, self_check_ok={self_check_ok})"
        )
        return {
            "answer": FALLBACK_MESSAGE,
            "fallback_reason": "llm_declined",
            "citation_verification_passed": citation_ok,
            "self_check_passed": self_check_ok,
            "invalid_citations": invalid,
        }

    return {
        "answer": answer,
        "fallback_reason": None,
        "citation_verification_passed": True,
        "self_check_passed": True,
        "invalid_citations": [],
    }
