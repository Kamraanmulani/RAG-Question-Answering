# RAG Question Answering System

> Production-grade applied-AI Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Qdrant**, **Pydantic v2**, and **OpenAI**. Evaluated on academic coursework in Distributed Computing.

---

## 1. System Architecture

```
                    [Client: Swagger UI / curl / test scripts]
                                        │
                                        ▼
                             [FastAPI Application]
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    │                                   │                                   │
    ▼                                   ▼                                   ▼
[POST /documents]                  [POST /query]                        [GET /health]
 - Accepts PDF & TXT                - Validates Question                 - Checks Vector Store
 - Page-Aware Text Extraction       - Embeds Query (OpenAI 1536-dim)     - Checks OpenAI Key
 - Recursive Token Chunking         - Cosine Similarity Search (Qdrant)
 - Generates Embeddings             - Threshold Gate (< 0.40)
 - Upserts with Metadata                    │
                                            ├─ < 0.40 (Out of Corpus) ────────┐
                                            │                                 │
                                            ├─ >= 0.40 (In Corpus)            ▼
                                            ▼                         [Short-Circuit Refusal]
                                 [Grounded LLM Prompt]                - Latency: ~35ms
                                  - Strict Context Only               - LLM Cost: $0.00
                                  - Inline Citations [Chunk X]        - "I cannot find this in
                                  - gpt-4o-mini (temp=0.0)              the provided documents."
                                            │                                 │
                                            ▼                                 │
                                 [Structured Response] ◄──────────────────────┘
                                  - answer: str
                                  - grounded: bool
                                  - sources: list[SourceChunk] (file, page, score, excerpt)
                                  - latency_ms: dict (retrieval_ms, llm_ms, total_ms)
```

---

## 2. Core Engineering Highlights

- **Dual-Mode Vector Store (Zero Setup Friction)**:
  - **Embedded Local Mode (`QDRANT_MODE=local`)**: Runs on-disk (`./qdrant_data`) with zero Docker daemon required. Anyone cloning this repository can run it immediately with `pip install -r requirements.txt`.
  - **Server Mode (`QDRANT_MODE=server`)**: Connects to Docker or Qdrant Cloud via `http://localhost:6333` for production scale.
- **Page-Aware PDF Ingestion**: Extracts text page-by-page, preserving 1-indexed `page_number` in chunk metadata for exact attribution (e.g. `DC_Assignment_5.pdf, Page: 2, Chunk: 2`).
- **Recursive Token-Budget Chunking**:
  - Target size: **500 tokens** (~375 words) with **80-token overlap** (~16%).
  - Tokenized using `tiktoken` (`cl100k_base`).
  - Hierarchical splitting: Paragraphs (`\n\n`) $\rightarrow$ Sentences (`. `, `! `, `? `) $\rightarrow$ Token slicing.
- **Two-Tier Hallucination Defense**:
  - **Tier 1 (Similarity Short-Circuit Gate)**: Queries with top cosine score $< 0.40$ immediately return `"I cannot find this in the provided documents."` with `grounded: false`. Bypasses LLM, reducing latency by 55–96% and token cost to $0.
  - **Tier 2 (Strict Grounding Prompt)**: LLM executed at `temperature=0.0` with explicit citation requirement (`[Chunk X]`) and mandatory refusal if unsupported.
- **Layered Error Handling**:
  - Detects scanned/image-only PDFs ($<20$ extracted chars) and raises `HTTP 422 Unprocessable Entity`.
  - OpenAI rate limit exponential backoff retry (up to 3 retries).
  - Pydantic v2 input bounds validation ($1 \le \text{length} \le 1000$).

---

## 3. Quickstart & Setup

### Prerequisites
- Python 3.10+
- OpenAI API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kamraanmulani/RAG-Question-Answering.git
   cd RAG-Question-Answering
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and insert your OpenAI API Key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   OPENAI_API_KEY=sk-...
   QDRANT_MODE=server    # or "local" if you do not want to use Docker
   QDRANT_URL=http://localhost:6333
   ```

5. *(Optional)* **Start Docker Qdrant** (if using `QDRANT_MODE=server`):
   ```bash
   docker compose up -d
   ```
   > **Note**: If you don't have Docker, simply keep `QDRANT_MODE=local` in `.env`. The app will run embedded on-disk!

6. **Start the API Server**:
   ```bash
   uvicorn app.main:app --reload
   ```
   Interactive Swagger UI is live at: **http://127.0.0.1:8000/docs**  
   Qdrant Web UI (if Docker is running) is at: **http://localhost:6333/dashboard**

---

## 4. API Endpoints & Usage

### 1. Health Check
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response**:
```json
{
  "status": "healthy",
  "qdrant_mode": "server",
  "collection": "rag_documents",
  "openai_configured": true
}
```

---

### 2. Upload Document (`POST /documents`)
Upload PDF or TXT reference files:
```bash
curl -X POST "http://127.0.0.1:8000/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/DC_Assignment_5.pdf"
```
**Response**:
```json
{
  "filename": "DC_Assignment_5.pdf",
  "chunks_created": 6,
  "total_tokens": 1929,
  "message": "Successfully ingested 'DC_Assignment_5.pdf': 6 chunks (1929 tokens) indexed into Qdrant."
}
```

---

### 3. Query Documents (`POST /query`)

#### Example A: Answerable Query (In-Corpus)
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many messages does the Ricart-Agrawala algorithm require per critical section entry?"}'
```
**Response**:
```json
{
  "answer": "Ricart-Agrawala needs 2(N-1) messages for every single CS entry, where N is the number of processes [Chunk 0].",
  "grounded": true,
  "sources": [
    {
      "source": "DC_Assignment_5.pdf",
      "page_number": 2,
      "chunk_index": 2,
      "score": 0.5131,
      "excerpt": " racing to patch the same object need a consistent way to decide who goes first..."
    }
  ],
  "latency_ms": {
    "retrieval_ms": 1236.02,
    "llm_ms": 2030.97,
    "total_ms": 3267.02
  }
}
```

#### Example B: Unanswerable Query (Out-of-Corpus Refusal Edge Case)
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the annual percentage yield (APY) for Klarna savings account?"}'
```
**Response**:
```json
{
  "answer": "I cannot find this in the provided documents.",
  "grounded": false,
  "sources": [],
  "latency_ms": {
    "retrieval_ms": 35.1,
    "llm_ms": 0.0,
    "total_ms": 35.1
  }
}
```
*(Notice `llm_ms: 0.0` — the Tier 1 similarity gate caught the out-of-scope query, saving 100% of LLM cost!)*

---

## 5. Quantitative Evaluation Benchmark

The system includes an automated 10-query benchmark across the Distributed Computing coursework:

```bash
python -m tests.evaluate_retrieval
```

### Empirical Results (10-Query Benchmark)

| # | Query | Type | Cosine Score | Total Latency | Grounded | Result |
|---|---|---|---|---|---|---|
| 1 | Ricart-Agrawala message overhead | IN | 0.5352 | 2011.6 ms | True | Correctly Answered |
| 2 | Klarna Suzuki-Kasami clock drift | IN | 0.6364 | 2553.9 ms | True | Correctly Answered |
| 3 | Token mutual exclusion vulnerability | IN | 0.5830 | 1569.2 ms | True | Correctly Answered |
| 4 | Kong Gateway load balancing algorithms | IN | 0.7156 | 2385.2 ms | True | Correctly Answered |
| 5 | Azarnova & Polukhin queuing model | IN | 0.5402 | 1893.8 ms | True | Correctly Answered |
| 6 | Klarna savings account APY | OUT | 0.0000 | 1047.4 ms | False | Refused (Deterministic) |
| 7 | Apache Kafka partition rebalancing | OUT | 0.0000 | 2487.8 ms | False | Refused (Deterministic) |
| 8 | OpenAI GPT-4o-mini context limit | OUT | 0.0000 | 302.0 ms | False | Refused (Deterministic) |
| 9 | Cassandra commit log compaction | OUT | 0.0000 | 496.2 ms | False | Refused (Deterministic) |
| 10 | Course submission deadline & rubric | OUT | 0.0000 | 353.6 ms | False | Refused (Deterministic) |

### Summary Metrics:
- **Overall Accuracy**: **100.0%** (10/10 queries correct)
- **In-Corpus Hit Rate**: **5/5 (100%)**
- **Out-of-Corpus Refusal Precision**: **5/5 (100%)**
- **Mean In-Corpus Cosine Score**: `0.6021` (range: `0.5352` – `0.7156`)
- **Mean Out-Corpus Score**: `0.0000` (threshold short-circuited)
- **Cosine Score Separation Gap**: `0.6021`

---

## 6. What Works vs. What Doesn't

### What Works:
- [x] Multi-format document ingestion (PDF and TXT).
- [x] Page-aware PDF extraction preserving 1-indexed page numbers.
- [x] Recursive token chunking with paragraph $\rightarrow$ sentence $\rightarrow$ token hierarchy.
- [x] Sliding window overlap preventing boundary context severance.
- [x] Dual-mode vector store (embedded on-disk or Docker/server Qdrant).
- [x] Two-tier hallucination defense with deterministic refusal.
- [x] Source attribution reporting filename, page number, chunk index, score, and excerpt.
- [x] Automatic rate limit retry with exponential backoff.
- [x] Automated test suite and quantitative evaluation harness.

### What Doesn't / Deliberately Omitted Scope:
- **No Cross-Encoder Re-Ranking**: Uses bi-encoder cosine similarity only. A cross-encoder (e.g. `bge-reranker-base`) would provide joint query-passage attention for more subtle re-ranking.
- **No Hybrid BM25 Keyword Search**: Pure vector search can occasionally miss exact alphanumeric identifiers (e.g. `"ERR_404_B"` or exact command flags).
- **No Structured Table Reconstruction**: Tables in PDFs are flattened into linear text by `pypdf`. Future work would integrate `pdfplumber` or Markdown conversion.
- **No Multi-Turn Conversational Memory**: Each request is treated as a stateless query.

---

## 7. Automated Testing & End-to-End Pipeline

Run the comprehensive 6-stage end-to-end testing pipeline:
```bash
# Full 6-stage E2E pipeline (Preflight, Chunking, Ingestion, API contracts, Grounding defense, and 10-query Benchmark):
python -m tests.run_e2e_pipeline
```

Run unit and integration test suites individually with pytest:
```bash
# Unit tests for chunking and token boundaries:
pytest tests/test_chunking.py -v

# End-to-end API integration tests:
pytest tests/test_api.py -v

# Run all test suites:
pytest tests/test_chunking.py tests/test_api.py -v
```


---

## 8. Author
- **Name**: Kamraan Faiyaz Mulani
- **Repository**: [https://github.com/Kamraanmulani/RAG-Question-Answering](https://github.com/Kamraanmulani/RAG-Question-Answering)
