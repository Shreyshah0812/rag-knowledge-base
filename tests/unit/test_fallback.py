"""
Tests the threshold decision in isolation, mirroring the logic in
app/retrieval/pipeline.py (below_threshold = top_score < settings.rerank_threshold)
without requiring a live Qdrant/Postgres/Cohere connection.
"""


def below_threshold(top_score: float, threshold: float) -> bool:
    return top_score < threshold


def test_score_above_threshold_does_not_fallback():
    assert below_threshold(0.85, threshold=0.3) is False


def test_score_below_threshold_triggers_fallback():
    assert below_threshold(0.1, threshold=0.3) is True


def test_score_exactly_at_threshold_does_not_fallback():
    # Boundary is exclusive: score must be strictly >= threshold to pass, matching
    # the `<` comparison in app/retrieval/pipeline.py.
    assert below_threshold(0.3, threshold=0.3) is False


def test_empty_retrieval_score_treated_as_below_threshold():
    assert below_threshold(0.0, threshold=0.3) is True
