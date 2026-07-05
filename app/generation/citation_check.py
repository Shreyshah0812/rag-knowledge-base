"""
Two independent hallucination checks per BLUEPRINT.md §7:
1. verify_citations: regex-extracts [Source N] tags and confirms N is a valid
   index into the chunks actually sent to the model (catches citation-to-
   nonexistent-source errors -- cheap, deterministic).
2. self_check: a second LLM call asking "does every claim appear in the sources?"
   (catches subtler unsupported claims that still cite a real, existing source --
   more expensive, non-deterministic, but a meaningfully stronger check).
"""
import re

from app.generation.prompts import SELF_CHECK_PROMPT, format_sources
from app.config import settings


CITATION_PATTERN = re.compile(r"\[Source (\d+)\]")


def extract_citations(answer: str) -> list[int]:
    return [int(n) for n in CITATION_PATTERN.findall(answer)]


def verify_citations(answer: str, num_sources: int) -> tuple[bool, list[int]]:
    """Returns (all_valid, invalid_indices)."""
    cited = extract_citations(answer)
    invalid = [n for n in cited if n < 1 or n > num_sources]
    return (len(invalid) == 0, invalid)


def self_check(client, answer: str, chunks: list[dict]) -> bool:
    """
    Runs a second, cheap LLM call to verify every claim in `answer` appears in the
    sources. Returns True if the check passes (or is disabled), False otherwise.
    `client` is an OpenAI client instance, reused from generator.py to avoid a
    second client construction.
    """
    if not settings.enable_self_check:
        return True

    prompt = SELF_CHECK_PROMPT.format(sources=format_sources(chunks), answer=answer)

    response = client.chat.completions.create(
        model=settings.generation_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.choices[0].message.content.strip().upper()
    return verdict.startswith("YES")
