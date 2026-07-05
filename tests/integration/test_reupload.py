"""
Per BLUEPRINT.md §9: re-upload of an unchanged file (same checksum) should not
create duplicate chunks; re-upload of a changed file should replace old chunks.
"""
import os
import requests
import pytest

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FIXTURE_PDF = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_policy.pdf")


def api_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


requires_stack = pytest.mark.skipif(not api_is_up(), reason="API stack is not running")
requires_fixture = pytest.mark.skipif(
    not os.path.exists(FIXTURE_PDF),
    reason="Fixture PDF not present -- see tests/fixtures/README.md",
)


@requires_stack
@requires_fixture
def test_reupload_unchanged_file_skips_duplicate_indexing():
    with open(FIXTURE_PDF, "rb") as f:
        first = requests.post(
            f"{API_URL}/upload",
            files={"file": ("sample_policy.pdf", f, "application/pdf")},
            timeout=120,
        ).json()

    with open(FIXTURE_PDF, "rb") as f:
        second = requests.post(
            f"{API_URL}/upload",
            files={"file": ("sample_policy.pdf", f, "application/pdf")},
            timeout=120,
        ).json()

    assert second["doc_id"] == first["doc_id"]
    # Second call should either skip entirely or cleanly re-index the same count,
    # never silently double the chunk count.
    assert second["chunks_indexed"] in (0, first["chunks_indexed"])
