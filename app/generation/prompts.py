"""Prompt templates per BLUEPRINT.md §7. Kept as plain strings/functions (not a
templating framework) so the exact wording is easy to diff across eval runs."""

SYSTEM_PROMPT = """You are a document question-answering assistant. You answer ONLY using the
numbered source excerpts provided below. You do not use outside knowledge,
even if you know the answer from general training.

Rules:
1. Every factual claim in your answer must be followed by a citation tag
in the form [Source N], where N matches the source excerpt number.
2. If the answer is fully supported by the sources, answer directly and
completely, with citations.
3. If the sources only partially answer the question, answer what is
supported, cite it, and explicitly state what part of the question is
NOT answered by the provided documents.
4. If none of the sources are relevant to the question, respond exactly
with: "I don't have enough information in the uploaded documents to
answer this." Do not guess.
5. Never combine information across sources to infer something that is
not explicitly stated in at least one of them.
6. Be concise. Do not restate the question. Do not add disclaimers beyond
what these rules require."""

SELF_CHECK_PROMPT = """Does every factual claim in the following answer appear in the provided
sources? Answer with only the single word YES or NO.

Sources:
{sources}

Answer to check:
{answer}"""

FALLBACK_MESSAGE = "I don't have enough information in the uploaded documents to answer this."


def format_sources(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i} — {c['doc_title']}, p.{c['page_number']}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    return f"Source excerpts:\n{format_sources(chunks)}\n\nQuestion: {question}"
