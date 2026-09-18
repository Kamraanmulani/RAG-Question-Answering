from app.ingest import extract_text_pages
from app.chunking import chunk_pages
from app.embeddings import embed_texts, embed_query
from app.vectorstore import upsert_chunks, search_chunks, get_collection_info

def main():
    print("--- 1. Ingesting & Chunking DC_Assignment_5.pdf ---")
    with open("data/DC_Assignment_5.pdf", "rb") as f:
        pages = extract_text_pages("DC_Assignment_5.pdf", f.read())

    chunks = chunk_pages(pages, target_size=500, overlap_size=80)
    print(f"Extracted {len(pages)} pages -> {len(chunks)} chunks.")

    print("\n--- 2. Generating OpenAI Embeddings ---")
    vectors = embed_texts([c.text for c in chunks])
    print(f"Generated {len(vectors)} vector embeddings (1536-dim).")

    print("\n--- 3. Upserting to Docker Qdrant (http://localhost:6333) ---")
    upsert_chunks(chunks, vectors)
    info = get_collection_info()
    print(f"Docker Qdrant Collection Status: {info}")

    print("\n--- 4. Running Semantic Search in Docker Qdrant ---")
    query = "How many messages does the Ricart-Agrawala algorithm require per critical section entry?"
    q_vec = embed_query(query)
    results = search_chunks(q_vec, top_k=2)

    print(f"Search Query: '{query}'")
    for r in results:
        print(f"  -> Score: {r.score:.4f} | Chunk {r.payload['chunk_index']} (Page {r.payload['page_number']})")
        print(f"     Excerpt: {r.payload['text'][:120]}...\n")

    print(">>> DOCKER QDRANT PHASE 6 TEST COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()
