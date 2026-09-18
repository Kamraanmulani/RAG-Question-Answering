from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api():
    print("--- 1. Testing GET /health ---")
    res = client.get("/health")
    assert res.status_code == 200
    print(f"Health Response: {res.json()}")

    print("\n--- 2. Testing GET /documents ---")
    res = client.get("/documents")
    assert res.status_code == 200
    print(f"Documents Status: {res.json()}")

    print("\n--- 3. Testing POST /query (Answerable Question) ---")
    res = client.post("/query", json={
        "question": "How many messages does the Ricart-Agrawala algorithm require per critical section entry?"
    })
    assert res.status_code == 200
    data = res.json()
    print(f"Answer: {data['answer']}")
    print(f"Grounded: {data['grounded']}")
    print(f"Sources count: {len(data['sources'])}")
    if data['sources']:
        print(f"Top Source: {data['sources'][0]['source']} (Page {data['sources'][0]['page_number']}) - Score: {data['sources'][0]['score']}")
    print(f"Latency: {data['latency_ms']}")
    assert data['grounded'] is True
    assert "2(N-1)" in data['answer'] or "2(n-1)" in data['answer'].lower()

    print("\n--- 4. Testing POST /query (Out-of-Corpus Refusal Edge Case) ---")
    res = client.post("/query", json={
        "question": "What is the annual percentage yield (APY) for Klarna's high-yield savings account?"
    })
    assert res.status_code == 200
    data_refusal = res.json()
    print(f"Answer: {data_refusal['answer']}")
    print(f"Grounded: {data_refusal['grounded']}")
    print(f"Sources: {data_refusal['sources']}")
    print(f"Latency: {data_refusal['latency_ms']}")
    assert data_refusal['grounded'] is False
    assert data_refusal['answer'] == "I cannot find this in the provided documents."
    assert data_refusal['latency_ms']['llm_ms'] == 0.0  # Proves Tier 1 short-circuit!

    print("\n--- 5. Testing POST /documents (Validation Error Handling) ---")
    res = client.post(
        "/documents",
        files={"file": ("invalid.docx", b"some bytes", "application/octet-stream")}
    )
    assert res.status_code == 422
    print(f"Caught expected 422: {res.json()['detail']}")

    print("\n>>> ALL API TESTS PASSED 100%! <<<")

if __name__ == "__main__":
    test_api()
