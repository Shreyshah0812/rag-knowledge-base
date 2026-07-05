"""
Duplicate handling per BLUEPRINT.md §5:
- Normalize text (lowercase, strip whitespace/punctuation) and hash it.
- Same-document collision -> skip indexing (boilerplate headers/footers/TOC).
- Cross-document collision -> keep both chunks, but record a duplicate_of pointer
  so query-time results can be deduplicated in the citations shown to the user.
"""
import hashlib
import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
