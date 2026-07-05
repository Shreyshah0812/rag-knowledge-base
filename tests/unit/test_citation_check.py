from app.generation.citation_check import verify_citations, extract_citations


def test_extract_citations_finds_all_tags():
    answer = "Employees get 10 sick days [Source 1] and 15 vacation days [Source 2]."
    assert extract_citations(answer) == [1, 2]


def test_verify_citations_all_valid():
    answer = "Sick days are 10 per year [Source 1]."
    ok, invalid = verify_citations(answer, num_sources=3)
    assert ok is True
    assert invalid == []


def test_verify_citations_catches_out_of_range_index():
    answer = "Sick days are 10 per year [Source 5]."
    ok, invalid = verify_citations(answer, num_sources=3)
    assert ok is False
    assert invalid == [5]


def test_verify_citations_catches_zero_index():
    answer = "Some claim [Source 0]."
    ok, invalid = verify_citations(answer, num_sources=3)
    assert ok is False
    assert invalid == [0]


def test_verify_citations_no_citations_present():
    answer = "I don't have enough information in the uploaded documents to answer this."
    ok, invalid = verify_citations(answer, num_sources=3)
    assert ok is True
    assert invalid == []
