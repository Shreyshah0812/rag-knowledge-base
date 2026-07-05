from app.ingestion.chunker import chunk_document


def make_page(page_number, text, tables=None, headings=None):
    return {
        "page_number": page_number,
        "text": text,
        "tables": tables or [],
        "heading_candidates": headings or {},
    }


def test_chunk_document_basic_prose():
    pages = [make_page(1, "Paragraph one.\n\nParagraph two is here.")]
    chunks = chunk_document(pages)
    assert len(chunks) >= 1
    assert all(c["page_number"] == 1 for c in chunks)
    assert all(c["content_type"] == "prose" for c in chunks)


def test_chunk_document_respects_chunk_index_ordering():
    pages = [make_page(1, "A" * 2000)]  # forces multiple chunks given 800-char default
    chunks = chunk_document(pages)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)  # no duplicate indices


def test_table_chunk_is_not_split_by_char_count():
    long_table_md = "| a | b |\n| --- | --- |\n" + "\n".join([f"| {i} | {i} |" for i in range(200)])
    pages = [make_page(1, "", tables=[long_table_md])]
    chunks = chunk_document(pages)
    table_chunks = [c for c in chunks if c["content_type"] == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0]["text"] == long_table_md


def test_heading_carries_forward_across_pages():
    pages = [
        make_page(1, "Intro text.", headings={10.0: "1. Introduction"}),
        make_page(2, "More text under the same section, no new heading here."),
    ]
    chunks = chunk_document(pages)
    assert chunks[0]["section_heading"] == "1. Introduction"
    assert chunks[-1]["section_heading"] == "1. Introduction"


def test_new_heading_overrides_previous():
    pages = [
        make_page(1, "Intro text.", headings={10.0: "1. Introduction"}),
        make_page(2, "Body text.", headings={5.0: "2. Details"}),
    ]
    chunks = chunk_document(pages)
    page2_chunks = [c for c in chunks if c["page_number"] == 2]
    assert all(c["section_heading"] == "2. Details" for c in page2_chunks)
