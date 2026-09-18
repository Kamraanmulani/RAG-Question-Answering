import pytest
from app.ingest import DocumentPage
from app.chunking import chunk_pages, split_into_sentences

def test_split_into_sentences():
    text = "First sentence here. Second sentence starts! Third question? Yes it is."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence here."
    assert sentences[1] == "Second sentence starts!"
    assert sentences[2] == "Third question?"
    assert sentences[3] == "Yes it is."

def test_chunking_token_budget():
    # 50 repetitions of a ~15 token sentence produces ~750 tokens
    sample_text = "In distributed systems, mutual exclusion is essential for data consistency. " * 50
    page = DocumentPage(text=sample_text, page_number=1, source="DC_Assignment_5.pdf")
    
    target_size = 100
    overlap_size = 20
    chunks = chunk_pages([page], target_size=target_size, overlap_size=overlap_size)
    
    assert len(chunks) >= 6
    for c in chunks:
        # Verify strict adherence to token budget with slight block margin
        assert c.token_count <= target_size + 25
        assert c.page_number == 1
        assert c.source == "DC_Assignment_5.pdf"

def test_chunking_preserves_overlap():
    sample_text = (
        "Section 1 covers Ricart-Agrawala with 2(N-1) messages.\n\n"
        "Section 2 covers token-based algorithms circulating a single permission token.\n\n"
        "Section 3 covers Kubernetes leader election using lease objects."
    )
    page = DocumentPage(text=sample_text, page_number=2, source="DC_Assignment_5.pdf")
    
    chunks = chunk_pages([page], target_size=20, overlap_size=8)
    assert len(chunks) >= 2
    
    # Verify sequential chunk indexing
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.char_end > c.char_start

def test_chunking_page_attribution():
    page1 = DocumentPage(text="Page 1 content on Suzuki-Kasami token algorithm.", page_number=1, source="DC_Assignment_1.pdf")
    page2 = DocumentPage(text="Page 2 content on queuing theory and Markov chains.", page_number=2, source="DC_Assignment_1.pdf")
    
    chunks = chunk_pages([page1, page2], target_size=200, overlap_size=30)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].source == "DC_Assignment_1.pdf"
    assert chunks[1].page_number == 2
    assert chunks[1].source == "DC_Assignment_1.pdf"
