import time
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError
from app.config import settings

def get_openai_client() -> OpenAI:
    """
    Returns an initialized OpenAI client.
    Raises ValueError with an actionable message if API key is not configured.
    """
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not configured or is set to placeholder. "
            "Please add your valid OpenAI API key to the .env file."
        )
    return OpenAI(api_key=settings.OPENAI_API_KEY)

def embed_texts(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """
    Generates dense vector embeddings for a list of text strings using OpenAI.
    
    Features:
    - Batch processing for maximum efficiency.
    - Exponential backoff retry on RateLimitError (429).
    - Linear backoff retry on temporary APIConnectionError.
    - Clear exception translation for upstream HTTP 503 handling.
    """
    if not texts:
        return []

    client = get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
            
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"OpenAI rate limit exceeded after {max_retries} retry attempts: {e}"
                )
            backoff = 2 ** attempt
            time.sleep(backoff)
            
        except APIConnectionError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to connect to OpenAI embedding API after {max_retries} attempts: {e}"
                )
            time.sleep(1.5 * (attempt + 1))
            
        except AuthenticationError as e:
            raise ValueError(f"OpenAI authentication failed. Invalid API key provided: {e}")
            
        except APIError as e:
            raise RuntimeError(f"OpenAI API error during embedding generation: {e}")

def embed_query(query: str) -> list[float]:
    """
    Generates an embedding vector for a single query string.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")
    vectors = embed_texts([query])
    return vectors[0]
