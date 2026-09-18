import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings
from app.chunking import Chunk

_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    """
    Returns a singleton QdrantClient instance based on QDRANT_MODE configuration.
    
    - "local": Uses embedded on-disk storage at QDRANT_PATH (zero external dependencies).
    - "server": Connects to Docker or Qdrant Cloud at QDRANT_URL.
    """
    global _client
    if _client is None:
        if settings.QDRANT_MODE == "server":
            _client = QdrantClient(url=settings.QDRANT_URL)
        else:
            _client = QdrantClient(path=settings.QDRANT_PATH)
    return _client

def ensure_collection(client: Optional[QdrantClient] = None):
    """
    Ensures that the target collection exists with Cosine distance and correct dimensions.
    """
    qc = client or get_qdrant_client()
    collections_response = qc.get_collections()
    existing_names = [c.name for c in collections_response.collections]
    
    if settings.COLLECTION_NAME not in existing_names:
        qc.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE
            ),
        )

def upsert_chunks(
    chunks: list[Chunk],
    vectors: list[list[float]],
    client: Optional[QdrantClient] = None
):
    """
    Upserts chunk text and metadata alongside embedding vectors into Qdrant.
    
    Payload attributes stored:
    - text: chunk text content
    - source: original document filename
    - page_number: 1-indexed PDF page number (or None for TXT)
    - chunk_index: index of chunk within the document
    - token_count: token length of chunk
    """
    if not chunks or not vectors:
        return

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatched counts: {len(chunks)} chunks provided with {len(vectors)} vectors."
        )

    qc = client or get_qdrant_client()
    ensure_collection(qc)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    qc.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=points
    )

def search_chunks(
    query_vector: list[float],
    top_k: int = 4,
    client: Optional[QdrantClient] = None
):
    """
    Searches for the top-k most similar chunks using Cosine similarity.
    Uses modern Qdrant query_points API.
    Returns list of ScoredPoint objects with .score and .payload attributes.
    """
    qc = client or get_qdrant_client()
    ensure_collection(qc)

    response = qc.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    return response.points

def get_collection_info(client: Optional[QdrantClient] = None) -> dict:
    """
    Returns collection metadata including points count and vector size.
    """
    qc = client or get_qdrant_client()
    ensure_collection(qc)
    info = qc.get_collection(collection_name=settings.COLLECTION_NAME)
    return {
        "name": settings.COLLECTION_NAME,
        "points_count": info.points_count,
        "vectors_size": info.config.params.vectors.size,
        "distance": str(info.config.params.vectors.distance),
        "status": str(info.status),
    }
