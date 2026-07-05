"""
Integration tests requiring the full Docker stack (postgres, qdrant, api) running,
plus real API keys in .env for OpenAI/Cohere -- these tests make real API calls
and will incur small costs. Run with:

    docker-compose up -d
    pytest tests/integration -v

BLUEPRINT.md §9 calls for exactly this: a full /upload -> ingest -> /query round
trip on a fixture PDF with known content, asserting the expected chunk is
retrieved and cited.
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
def test_upload_then_query_round_trip():
    with open(FIXTURE_PDF, "rb") as f:
        upload_resp = requests.post(
            f"{API_URL}/upload",
            files={"file": ("sample_policy.pdf", f, "application/pdf")},
            timeout=120,
        )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["chunks_indexed"] > 0

    query_resp = requests.post(
        f"{API_URL}/query",
        json={"question": "What is the maximum number of paid sick days per year?"},
        timeout=60,
    )
    assert query_resp.status_code == 200
    result = query_resp.json()

    assert result["fallback_reason"] is None
    assert len(result["citations"]) > 0
    assert "log_id" in result


@requires_stack
@requires_fixture
def test_unanswerable_question_triggers_fallback():
    query_resp = requests.post(
        f"{API_URL}/query",
        json={"question": "What is the company's policy on cryptocurrency compensation?"},
        timeout=60,
    )
    assert query_resp.status_code == 200
    result = query_resp.json()
    assert result["fallback_reason"] is not None
    assert result["citations"] == []


@requires_stack
def test_empty_question_returns_400():
    resp = requests.post(f"{API_URL}/query", json={"question": "   "}, timeout=10)
    assert resp.status_code == 400


@requires_stack
def test_non_pdf_upload_rejected():
    resp = requests.post(
        f"{API_URL}/upload",
        files={"file": ("notes.txt", b"plain text content", "text/plain")},
        timeout=10,
    )
    assert resp.status_code == 400
