"""
Metrics per BLUEPRINT.md §8:
- Retrieval: Recall@5, MRR (computed both pre-rerank and post-rerank so the
  reranker's actual contribution is quantifiable -- see resume bullet #1).
- Generation: Ragas faithfulness / answer_relevancy / context_relevancy.
- Citation correctness: custom metric (not in Ragas) -- % of [Source N] citations
  whose cited chunk actually contains the claim.
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision


def recall_at_k(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    """
    retrieved_texts: text of the chunks actually retrieved for this question.
    expected_keywords: phrases that should appear somewhere in a relevant chunk.
    Matching is by content, not chunk_id, because chunk_ids are randomly generated
    UUIDs that differ every time the corpus is re-ingested (e.g. a fresh database
    in CI) -- content-based matching is portable across any environment.
    """
    if not expected_keywords:
        # Nothing to recall for unanswerable questions -- treat as trivially satisfied.
        return 1.0
    combined = " ".join(retrieved_texts).lower()
    hit = any(kw.lower() in combined for kw in expected_keywords)
    return 1.0 if hit else 0.0


def mrr(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    for rank, text in enumerate(retrieved_texts, start=1):
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in expected_keywords):
            return 1.0 / rank
    return 0.0


def run_ragas_eval(records: list[dict]) -> dict:
    """
    records: [{"question": str, "answer": str, "contexts": [str, ...], "ground_truth": str}, ...]
    Returns the aggregate Ragas scores dict.
    """
    dataset = Dataset.from_list(records)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return result


def citation_correctness(answer: str, citations: list[dict], chunks_by_index: dict[int, str]) -> float:
    """
    Rough proxy: for each [Source N] cited, check that some sentence from the
    answer near that citation shares significant token overlap with the cited
    chunk's text. This is intentionally simple (token-overlap, not semantic) --
    documented as a known limitation; a stronger version would use an LLM judge.
    """
    import re

    if not citations:
        return 1.0  # no citations to check (e.g. correctly declined an unanswerable question)

    correct = 0
    for source_index in {c["source_index"] for c in citations}:
        chunk_text = chunks_by_index.get(source_index, "").lower()
        # crude sentence match: find the sentence preceding this citation tag
        pattern = re.compile(r"([^.]*\.)\s*\[Source " + str(source_index) + r"\]")
        matches = pattern.findall(answer)
        if not matches:
            continue
        claim = matches[0].lower()
        claim_tokens = set(re.findall(r"\w+", claim))
        chunk_tokens = set(re.findall(r"\w+", chunk_text))
        overlap = len(claim_tokens & chunk_tokens) / max(len(claim_tokens), 1)
        if overlap > 0.3:
            correct += 1

    total = len({c["source_index"] for c in citations})
    return correct / total if total else 1.0