"""
Chunking strategy per BLUEPRINT.md §5:
- 800 chars, 150 overlap, split on paragraph boundaries first, then recursively on
  a separator hierarchy that avoids mid-sentence splits where possible.
- Table chunks are never split by character count -- one chunk per table.
- Every chunk carries the metadata needed downstream: doc_id, page_number,
  section_heading, chunk_index, content_type.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_page(page: dict, current_heading: str | None) -> tuple[list[dict], str | None]:
    """
    Chunks a single parsed page (see parser.parse_pdf) into prose chunks + table
    chunks. Returns (chunks, updated_current_heading) -- heading state carries
    forward across pages until a new heading is detected.
    """
    splitter = build_splitter()
    chunks = []

    # Determine the heading in effect for this page (the last detected heading, or
    # carry forward from a previous page if none detected here).
    heading_candidates = page["heading_candidates"]
    if heading_candidates:
        # Take the heading nearest the top of the page as "in effect" for this page.
        current_heading = heading_candidates[min(heading_candidates.keys())]

    # Prose chunks
    paragraphs = [p for p in page["text"].split("\n\n") if p.strip()]
    prose_text = "\n\n".join(paragraphs)
    if prose_text.strip():
        for piece in splitter.split_text(prose_text):
            chunks.append({
                "page_number": page["page_number"],
                "section_heading": current_heading,
                "content_type": "prose",
                "text": piece,
            })

    # Table chunks -- one chunk per table, not split by char count.
    for table_md in page["tables"]:
        if table_md.strip():
            chunks.append({
                "page_number": page["page_number"],
                "section_heading": current_heading,
                "content_type": "table",
                "text": table_md,
            })

    return chunks, current_heading


def chunk_document(pages: list[dict]) -> list[dict]:
    """Chunks an entire parsed document, carrying heading state across pages."""
    all_chunks = []
    current_heading = None
    chunk_index = 0

    for page in pages:
        page_chunks, current_heading = chunk_page(page, current_heading)
        for c in page_chunks:
            c["chunk_index"] = chunk_index
            c["char_count"] = len(c["text"])
            all_chunks.append(c)
            chunk_index += 1

    return all_chunks
