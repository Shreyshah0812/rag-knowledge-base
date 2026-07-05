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
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy


def recall_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    if not expected_chunk_ids:
        # Nothing to recall for unanswerable questions -- treat as trivially satisfied.
        return 1.0
    hit = any(cid in retrieved_chunk_ids for cid in expected_chunk_ids)
    return 1.0 if hit else 0.0


def mrr(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    if not expected_chunk_ids:
        return 1.0
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in expected_chunk_ids:
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
        metrics=[faithfulness, answer_relevancy, context_relevancy],
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
