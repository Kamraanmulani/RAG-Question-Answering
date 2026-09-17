import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """
    Centralized application configuration with Pydantic v2 validation.
    Reads values from environment variables or .env file with safe defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Qdrant Vector Store Configuration
    # "local" runs embedded storage at QDRANT_PATH (zero external dependencies)
    # "server" connects to Docker/remote Qdrant instance at QDRANT_URL
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", "local")
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_data")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")

    # Embedding Model Parameters
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    # Generation Model Parameters
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Chunking & Retrieval Parameters (Empirically evaluated)
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))

settings = Settings()
