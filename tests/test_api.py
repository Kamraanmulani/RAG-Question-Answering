import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verifies that the /health endpoint reports healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "qdrant_mode" in data
    assert "collection" in data

def test_documents_stats_endpoint():
    """Verifies that /documents returns vector collection stats."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "collection_info" in data

def test_unsupported_file_extension():
    """Verifies that non-PDF and non-TXT file uploads are rejected with 422."""
    response = client.post(
        "/documents",
        files={"file": ("invalid_contract.docx", b"dummy content", "application/octet-stream")}
    )
    assert response.status_code == 422
    assert "Unsupported file format" in response.json()["detail"]

def test_empty_file_rejection():
    """Verifies that zero-byte files are rejected with 422."""
    response = client.post(
        "/documents",
        files={"file": ("empty_notes.txt", b"", "text/plain")}
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()

def test_empty_query_validation():
    """Verifies that empty questions are rejected with 422 by Pydantic."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422

def test_upload_and_query_flow():
    """
    Tests end-to-end flow:
    Upload document -> Query answerable question -> Verify grounded answer & sources.
    """
    sample_text = (
        "Name: Kamraan Faiyaz Mulani\n"
        "Assignment 5 - Distributed Mutual Exclusion\n\n"
        "The Ricart-Agrawala algorithm is a permission-based approach to distributed mutual exclusion.\n\n"
        "The algorithm requires exactly 2(N-1) messages for every single critical section entry, "
        "where N is the number of processes in the distributed system."
    )
    
    # 1. Upload
    upload_res = client.post(
        "/documents",
        files={"file": ("test_dc5.txt", sample_text.encode("utf-8"), "text/plain")}
    )
    assert upload_res.status_code == 201
    assert upload_res.json()["chunks_created"] >= 1

    # 2. Query
    query_res = client.post(
        "/query",
        json={"question": "How many messages does the Ricart-Agrawala algorithm require per critical section entry?"}
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["grounded"] is True
    assert "2(N-1)" in q_data["answer"] or "2(n-1)" in q_data["answer"].lower()
    assert len(q_data["sources"]) >= 1
    assert q_data["sources"][0]["score"] >= 0.40

def test_out_of_corpus_refusal():
    """
    Verifies the mandatory requirement:
    A question not answered in the documents returns exact refusal without hallucinating.
    """
    response = client.post(
        "/query",
        json={"question": "What is the recipe for baking chocolate chip cookies?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["answer"] == "I cannot find this in the provided documents."
    assert len(data["sources"]) == 0
    # Tier 1 short-circuit verification
    assert data["latency_ms"]["llm_ms"] == 0.0

def test_scanned_pdf_rejection():
    """Verifies that PDFs with no extractable text (scanned/empty) are rejected with 422."""
    import io
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()
    c.save()
    empty_pdf_bytes = buf.getvalue()

    response = client.post(
        "/documents",
        files={"file": ("scanned_assignment.pdf", empty_pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 422
    assert "scanned" in response.json()["detail"].lower()

def test_oversized_query_validation():
    """Verifies that questions exceeding 1000 characters are rejected with 422."""
    long_question = "Explain Ricart-Agrawala algorithm in depth. " * 30
    assert len(long_question) > 1000
    response = client.post("/query", json={"question": long_question})
    assert response.status_code == 422

def test_citation_format_verification():
    """Verifies that grounded answers contain inline [Chunk X] citations."""
    response = client.post(
        "/query",
        json={"question": "How many messages does the Ricart-Agrawala algorithm require per critical section entry?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert "[Chunk" in data["answer"]

