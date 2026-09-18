from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

class QueryRequest(BaseModel):
    """
    Validates user query input.
    Guarantees non-empty input and bounds question length.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question about the ingested reference documents."
    )

class SourceChunk(BaseModel):
    """
    Attribution metadata for a retrieved document chunk.
    Fulfills Functional Requirement #6.
    """
    source: str = Field(..., description="Document filename (e.g., DC_Assignment_5.pdf)")
    page_number: Optional[int] = Field(None, description="1-indexed page number where chunk originated")
    chunk_index: int = Field(..., description="Sequence index of the chunk within the document")
    score: float = Field(..., description="Cosine similarity score against the query vector")
    excerpt: str = Field(..., description="Brief preview of the chunk text")

class LatencyMetrics(BaseModel):
    """
    End-to-end latency profiling breakdown in milliseconds.
    """
    retrieval_ms: float = Field(..., description="Time taken to embed query and search Qdrant (ms)")
    llm_ms: float = Field(..., description="Time taken by LLM to generate grounded response (ms)")
    total_ms: float = Field(..., description="Total end-to-end request processing time (ms)")

class QueryResponse(BaseModel):
    """
    Structured response contract for /query endpoint.
    """
    answer: str = Field(..., description="Grounded answer or deterministic refusal message")
    grounded: bool = Field(..., description="True if answer is supported by retrieved context; False if refused")
    sources: List[SourceChunk] = Field(default_factory=list, description="List of source chunks used for attribution")
    latency_ms: LatencyMetrics = Field(..., description="Execution latency breakdown")

class UploadResponse(BaseModel):
    """
    Response returned upon successful document ingestion.
    """
    filename: str = Field(..., description="Name of the uploaded document")
    chunks_created: int = Field(..., description="Total number of chunks produced and indexed")
    total_tokens: int = Field(..., description="Total token volume processed across all chunks")
    message: str = Field(..., description="User-friendly status confirmation")

class HealthResponse(BaseModel):
    """
    Health check response verifying service connectivity.
    """
    status: str = Field(..., description="Overall system health status")
    qdrant_mode: str = Field(..., description="Active Qdrant storage mode (local or server)")
    collection: str = Field(..., description="Active vector collection name")
    openai_configured: bool = Field(..., description="Whether a valid OpenAI API key is detected")
