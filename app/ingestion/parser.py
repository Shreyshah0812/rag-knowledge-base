"""
Parsing strategy per BLUEPRINT.md §5:
- PyMuPDF (fitz) for page text + font-size-based heading detection.
- pdfplumber for table extraction, kept as separate chunks (never interleaved with
  prose text — that corrupts both, per the blueprint).
"""
import fitz  # PyMuPDF
import pdfplumber
import statistics


def detect_headings(page: "fitz.Page") -> dict[int, str]:
    """
    Returns a mapping of block y-position -> heading text for blocks whose font size
    is materially larger than the page's median, or that are bold and short.
    Heuristic, not perfect -- documented trade-off, see BLUEPRINT.md §5.
    """
    blocks = page.get_text("dict")["blocks"]
    sizes = []
    for b in blocks:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span["size"])
    if not sizes:
        return {}
    median_size = statistics.median(sizes)

    headings = {}
    for b in blocks:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                is_bold = "Bold" in span.get("font", "")
                is_large = span["size"] > median_size * 1.3
                is_short = len(text) < 80
                if text and is_short and (is_large or is_bold):
                    headings[b["bbox"][1]] = text
    return headings


def parse_pdf(file_path: str) -> list[dict]:
    """
    Returns a list of page records:
    [{
        "page_number": int (1-indexed),
        "text": str,
        "heading_candidates": {y_pos: text},
        "tables": [markdown_table_str, ...]
    }, ...]
    """
    pages = []

    doc = fitz.open(file_path)
    with pdfplumber.open(file_path) as pdf:
        for i, fitz_page in enumerate(doc):
            page_number = i + 1
            text = fitz_page.get_text("text")
            headings = detect_headings(fitz_page)

            tables_md = []
            if i < len(pdf.pages):
                plumber_page = pdf.pages[i]
                for table in plumber_page.extract_tables():
                    tables_md.append(_table_to_markdown(table))

            pages.append({
                "page_number": page_number,
                "text": text,
                "heading_candidates": headings,
                "tables": tables_md,
            })

    doc.close()
    return pages


def _table_to_markdown(table: list[list[str | None]]) -> str:
    if not table:
        return ""
    rows = [[cell or "" for cell in row] for row in table]
    header = rows[0]
    body = rows[1:]

    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)
