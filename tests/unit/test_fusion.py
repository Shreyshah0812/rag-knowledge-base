from app.retrieval.fusion import rrf_merge


def test_rrf_merge_favors_items_ranked_high_in_both_lists():
    bm25_results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    vector_results = [{"chunk_id": "b"}, {"chunk_id": "a"}, {"chunk_id": "d"}]

    merged = rrf_merge(bm25_results, vector_results)
    merged_ids = [m["chunk_id"] for m in merged]

    # 'a' and 'b' each appear in both lists near the top -> should outrank 'c'/'d',
    # which appear in only one list.
    assert merged_ids.index("a") < merged_ids.index("c")
    assert merged_ids.index("b") < merged_ids.index("d")


def test_rrf_merge_deduplicates_chunk_ids():
    bm25_results = [{"chunk_id": "a"}]
    vector_results = [{"chunk_id": "a"}]
    merged = rrf_merge(bm25_results, vector_results)
    assert len(merged) == 1
    assert merged[0]["in_bm25"] is True
    assert merged[0]["in_vector"] is True


def test_rrf_merge_marks_source_membership_correctly():
    bm25_results = [{"chunk_id": "only_bm25"}]
    vector_results = [{"chunk_id": "only_vector"}]
    merged = rrf_merge(bm25_results, vector_results)
    by_id = {m["chunk_id"]: m for m in merged}
    assert by_id["only_bm25"]["in_bm25"] is True
    assert by_id["only_bm25"]["in_vector"] is False
    assert by_id["only_vector"]["in_vector"] is True
    assert by_id["only_vector"]["in_bm25"] is False


def test_rrf_merge_empty_lists():
    assert rrf_merge([], []) == []
