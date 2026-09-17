import re
import tiktoken
from dataclasses import dataclass
from app.ingest import DocumentPage

@dataclass
class Chunk:
    """
    Represents a discrete semantic chunk of text with rich attribution metadata.
    """
    text: str
    chunk_index: int
    source: str
    page_number: int | None
    token_count: int
    char_start: int
    char_end: int

_encoder = tiktoken.get_encoding("cl100k_base")

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using regex boundary matching on punctuation (. ! ?).
    Preserves trailing spaces and sentence structure.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_pages(
    pages: list[DocumentPage],
    target_size: int = 500,
    overlap_size: int = 80
) -> list[Chunk]:
    """
    Recursively chunks document pages using paragraph, sentence, and token boundaries.
    
    1. Splits pages into paragraphs (\n\n).
    2. If a paragraph exceeds target_size tokens, subdivides into sentences.
    3. If a sentence exceeds target_size tokens, falls back to token slicing.
    4. Accumulates blocks into chunks under target_size tokens with overlap_size token overlap.
    5. Preserves source filename, page number, and character offsets.
    """
    all_chunks: list[Chunk] = []
    global_index = 0
    char_cursor = 0
    
    for page in pages:
        raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        
        atomic_blocks: list[str] = []
        for p in raw_paragraphs:
            p_tokens = _encoder.encode(p)
            if len(p_tokens) <= target_size:
                atomic_blocks.append(p)
            else:
                # Subdivide oversized paragraph by sentences
                sentences = split_into_sentences(p)
                for s in sentences:
                    s_tokens = _encoder.encode(s)
                    if len(s_tokens) <= target_size:
                        atomic_blocks.append(s)
                    else:
                        # Fallback for unbroken token blocks: slice with stride
                        stride = max(1, target_size - overlap_size)
                        for i in range(0, len(s_tokens), stride):
                            slice_toks = s_tokens[i:i + target_size]
                            atomic_blocks.append(_encoder.decode(slice_toks))

        # Accumulate atomic blocks into token-bounded chunks with sliding overlap
        current_tokens: list[int] = []
        
        for block in atomic_blocks:
            block_tokens = _encoder.encode(block)
            
            # If adding this block exceeds the budget, seal current chunk
            if len(current_tokens) + len(block_tokens) > target_size and current_tokens:
                chunk_text = _encoder.decode(current_tokens)
                char_end = char_cursor + len(chunk_text)
                
                all_chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_index=global_index,
                        source=page.source,
                        page_number=page.page_number,
                        token_count=len(current_tokens),
                        char_start=char_cursor,
                        char_end=char_end,
                    )
                )
                global_index += 1
                char_cursor = char_end
                
                # Sliding window overlap: carry over the tail tokens
                if overlap_size > 0 and len(current_tokens) > overlap_size:
                    overlap_tokens = current_tokens[-overlap_size:]
                else:
                    overlap_tokens = []
                    
                current_tokens = overlap_tokens + block_tokens
            else:
                current_tokens.extend(block_tokens)
                
        # Flush any remaining tokens for this page
        if current_tokens:
            chunk_text = _encoder.decode(current_tokens)
            char_end = char_cursor + len(chunk_text)
            
            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=global_index,
                    source=page.source,
                    page_number=page.page_number,
                    token_count=len(current_tokens),
                    char_start=char_cursor,
                    char_end=char_end,
                )
            )
            global_index += 1
            char_cursor = char_end
            
    return all_chunks
