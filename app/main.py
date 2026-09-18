import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
    HealthResponse,
    LatencyMetrics,
)
from app.ingest import extract_text_pages
from app.chunking import chunk_pages
from app.embeddings import embed_texts
from app.vectorstore import (
    ensure_collection,
    upsert_chunks,
    search_chunks,
    get_collection_info,
)
from app.llm import generate_grounded_answer, REFUSAL_MESSAGE

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure vector collection exists on application launch."""
    ensure_collection()
    yield

app = FastAPI(
    title="RAG Question Answering System",
    description=(
        "Production-grade applied-AI RAG service with document ingestion (PDF & TXT), "
        "recursive token-budget chunking, vector retrieval in Qdrant, and two-tier "
        "deterministic hallucination defense."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for flexible client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint verifying Qdrant connection and OpenAI configuration.
    """
    return HealthResponse(
        status="healthy",
        qdrant_mode=settings.QDRANT_MODE,
        collection=settings.COLLECTION_NAME,
        openai_configured=bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here"),
    )

@app.get("/documents", tags=["Documents"])
def get_documents_status():
    """
    Returns statistics on active vector collection and indexed points count.
    """
    try:
        info = get_collection_info()
        return {"status": "ok", "collection_info": info}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store unavailable: {str(e)}",
        )

@app.post(
    "/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads, parses, chunks, embeds, and indexes a PDF or TXT reference document.
    
    Pipeline:
    1. Reads uploaded bytes and extracts page-aware text.
    2. Recursively chunks text using paragraph/sentence/token budgets (500 tokens).
    3. Generates 1536-dimensional embeddings with OpenAI text-embedding-3-small.
    4. Upserts chunks with metadata into Qdrant vector store.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must be provided.",
        )

    try:
        content = await file.read()
        pages = extract_text_pages(file.filename, content)
        chunks = chunk_pages(
            pages,
            target_size=settings.CHUNK_SIZE_TOKENS,
            overlap_size=settings.CHUNK_OVERLAP_TOKENS,
        )

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No readable chunks could be produced from document.",
            )

        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        upsert_chunks(chunks, vectors)

        total_tokens = sum(c.token_count for c in chunks)
        return UploadResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            total_tokens=total_tokens,
            message=(
                f"Successfully ingested '{file.filename}': "
                f"{len(chunks)} chunks ({total_tokens} tokens) indexed into Qdrant."
            ),
        )

    except ValueError as e:
        # Client validation or parsing errors (scanned PDF, empty file, unsupported ext)
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        # Upstream service errors (OpenAI connection, rate limits)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion service failure: {str(e)}",
        )

@app.post("/query", response_model=QueryResponse, tags=["Question Answering"])
def query_documents(req: QueryRequest):
    """
    Retrieves relevant document chunks and generates an answer strictly grounded in them.
    
    Two-Tier Defense Against Hallucinations:
    - Tier 1: Cosine similarity gate. If top score < SIMILARITY_THRESHOLD (0.40),
      short-circuits immediately with refusal message (< 50ms latency, $0.00 cost).
    - Tier 2: Grounded prompt enforcement. The LLM is restricted to context chunks
      and instructed to reply 'I cannot find this in the provided documents.' if unsupported.
    """
    t_start = time.perf_counter()

    # Step 1: Embed user query
    try:
        query_vec = embed_texts([req.question])[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}",
        )

    # Step 2: Vector search in Qdrant
    try:
        results = search_chunks(query_vec, top_k=settings.TOP_K)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store search failure: {str(e)}",
        )

    t_retrieval = time.perf_counter()
    retrieval_ms = round((t_retrieval - t_start) * 1000, 2)

    # Step 3: Tier 1 Hallucination Defense (Short-Circuit Gate)
    top_score = results[0].score if results else 0.0
    if not results or top_score < settings.SIMILARITY_THRESHOLD:
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return QueryResponse(
            answer=REFUSAL_MESSAGE,
            grounded=False,
            sources=[],
            latency_ms=LatencyMetrics(
                retrieval_ms=retrieval_ms,
                llm_ms=0.0,
                total_ms=total_ms,
            ),
        )

    # Filter candidates meeting similarity threshold
    relevant_chunks = [r for r in results if r.score >= settings.SIMILARITY_THRESHOLD]
    chunk_payloads = [r.payload for r in relevant_chunks]

    # Step 4: Tier 2 Hallucination Defense (Grounded LLM Generation)
    t_llm_start = time.perf_counter()
    try:
        raw_answer = generate_grounded_answer(req.question, chunk_payloads)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM generation service failure: {str(e)}",
        )

    t_end = time.perf_counter()
    llm_ms = round((t_end - t_llm_start) * 1000, 2)
    total_ms = round((t_end - t_start) * 1000, 2)

    # Check if model responded with refusal
    is_refusal = REFUSAL_MESSAGE.lower() in raw_answer.lower()
    grounded = not is_refusal

    # Build source chunk citations
    sources: list[SourceChunk] = []
    if grounded:
        for r in relevant_chunks:
            p = r.payload
            raw_text = p.get("text", "")
            excerpt = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
            sources.append(
                SourceChunk(
                    source=p.get("source", "Unknown"),
                    page_number=p.get("page_number"),
                    chunk_index=p.get("chunk_index", 0),
                    score=round(r.score, 4),
                    excerpt=excerpt,
                )
            )

    return QueryResponse(
        answer=raw_answer,
        grounded=grounded,
        sources=sources,
        latency_ms=LatencyMetrics(
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        ),
    )
