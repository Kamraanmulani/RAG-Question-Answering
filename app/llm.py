from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from app.config import settings
from app.embeddings import get_openai_client

SYSTEM_PROMPT = """You are a strictly grounded AI assistant. Answer the user's question using ONLY the provided numbered Context Chunks.

Rules:
1. Rely exclusively on facts directly stated in the context. Do NOT use outside knowledge, prior training data, or extrapolate.
2. For every factual claim in your answer, cite the relevant chunk using [Chunk X].
3. If the provided context does not contain the specific information needed to answer the question, reply EXACTLY with:
"I cannot find this in the provided documents."
Do not attempt to answer or explain further if the information is missing.
"""

REFUSAL_MESSAGE = "I cannot find this in the provided documents."

def generate_grounded_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generates an answer strictly grounded in the provided context chunks.
    
    Args:
        question: User query string.
        context_chunks: List of payload dicts containing 'text', 'source', 'page_number', 'chunk_index'.
        
    Returns:
        Grounded answer string with citations or exact refusal string.
    """
    if not context_chunks:
        return REFUSAL_MESSAGE

    client = get_openai_client()

    # Format chunks with source and page annotations
    formatted_chunks = []
    for idx, c in enumerate(context_chunks):
        page_info = f", Page: {c.get('page_number')}" if c.get("page_number") else ""
        source_name = c.get("source", "Unknown")
        text = c.get("text", "")
        formatted_chunks.append(
            f"--- [Chunk {idx}] (Source: {source_name}{page_info}) ---\n{text}"
        )

    context_str = "\n\n".join(formatted_chunks)
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        return answer
        
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI LLM rate limit exceeded: {e}")
    except APIConnectionError as e:
        raise RuntimeError(f"Network error connecting to OpenAI LLM API: {e}")
    except APIError as e:
        raise RuntimeError(f"OpenAI LLM generation failed: {e}")
