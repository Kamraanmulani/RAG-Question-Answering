import os
import time
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

DATA_DIR = "data"
BENCHMARK_FILES = ["DC_Assignment_1.pdf", "DC_Assignment_5.pdf", "DC_Assignment_7.pdf"]

# 10-Query Curated Benchmark across Kamraan's Distributed Computing Coursework
BENCHMARK_QUERIES = [
    # In-Corpus (Answerable from DC Coursework)
    ("How many messages does the Ricart-Agrawala algorithm require per critical section entry?", True),
    ("Why did Klarna adopt the Suzuki-Kasami token-based synchronization instead of timestamp-based algorithms?", True),
    ("What is the main fault tolerance vulnerability of token-based mutual exclusion?", True),
    ("What load balancing algorithms are supported by Kong Gateway according to Assignment 7?", True),
    ("How did the Azarnova and Polukhin paper model synchronization request arrivals and queue states?", True),
    # Out-of-Corpus (Unanswerable / Edge Cases)
    ("What is the annual percentage yield (APY) for Klarna's high-yield savings account?", False),
    ("How does Apache Kafka configure consumer group partition rebalancing?", False),
    ("What is the maximum context window of OpenAI's GPT-4o-mini model?", False),
    ("What storage engine does Cassandra use for commit log compaction?", False),
    ("What was the submission deadline and grading rubric for the Distributed Computing course?", False),
]

def run_benchmark():
    print("=" * 88)
    print("      DISTRIBUTED COMPUTING RAG EVALUATION BENCHMARK")
    print("=" * 88)

    # 1. Ingest documents if not already ingested
    print("\n--- Ingesting Benchmark Documents into Vector Store ---")
    for fname in BENCHMARK_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                content = f.read()
            res = client.post(
                "/documents",
                files={"file": (fname, content, "application/pdf")}
            )
            if res.status_code == 201:
                data = res.json()
                print(f"  [OK] Ingested {fname}: {data['chunks_created']} chunks ({data['total_tokens']} tokens)")
            else:
                print(f"  [Notice] {fname}: {res.json()}")
        else:
            print(f"  [Error] {fpath} not found.")

    # 2. Run benchmark queries
    print("\n--- Running 10-Query Retrieval & Grounding Benchmark ---")
    results = []

    for idx, (query, is_in_corpus) in enumerate(BENCHMARK_QUERIES, 1):
        res = client.post("/query", json={"question": query})
        assert res.status_code == 200
        data = res.json()

        top_score = data["sources"][0]["score"] if data["sources"] else 0.0
        results.append({
            "id": idx,
            "query": query,
            "expected_in_corpus": is_in_corpus,
            "grounded": data["grounded"],
            "top_score": top_score,
            "retrieval_ms": data["latency_ms"]["retrieval_ms"],
            "llm_ms": data["latency_ms"]["llm_ms"],
            "total_ms": data["latency_ms"]["total_ms"],
            "answer": data["answer"],
        })

    # 3. Print Tabular Results
    print("\n" + "-" * 88)
    print(f"{'#':<3} | {'Query':<44} | {'Exp':<5} | {'Score':<6} | {'Total(ms)':<9} | {'Grounded'}")
    print("-" * 88)
    for r in results:
        exp_str = "IN" if r["expected_in_corpus"] else "OUT"
        trunc_query = r["query"][:42] + ".." if len(r["query"]) > 44 else r["query"]
        print(f"{r['id']:<3} | {trunc_query:<44} | {exp_str:<5} | {r['top_score']:<6.4f} | {r['total_ms']:<9.1f} | {r['grounded']}")

    print("-" * 88)

    # 4. Compute Metrics Summary
    in_corpus = [r for r in results if r["expected_in_corpus"]]
    out_corpus = [r for r in results if not r["expected_in_corpus"]]

    avg_in_score = sum(r["top_score"] for r in in_corpus) / len(in_corpus)
    avg_out_score = sum(r["top_score"] for r in out_corpus) / len(out_corpus)
    avg_in_latency = sum(r["total_ms"] for r in in_corpus) / len(in_corpus)
    avg_out_latency = sum(r["total_ms"] for r in out_corpus) / len(out_corpus)

    # In-corpus recall & Out-of-corpus refusal accuracy
    in_corpus_correct = sum(1 for r in in_corpus if r["grounded"] is True)
    out_corpus_correct = sum(1 for r in out_corpus if r["grounded"] is False)
    total_accuracy = (in_corpus_correct + out_corpus_correct) / len(results) * 100

    print("\n" + "=" * 88)
    print("                     BENCHMARK SUMMARY METRICS")
    print("=" * 88)
    print(f"Overall Accuracy:                 {total_accuracy:.1f}% ({in_corpus_correct + out_corpus_correct}/10 queries correct)")
    print(f"In-Corpus Retrieval Hit Rate:     {in_corpus_correct}/{len(in_corpus)} ({in_corpus_correct / len(in_corpus) * 100:.0f}%)")
    print(f"Out-of-Corpus Refusal Precision:  {out_corpus_correct}/{len(out_corpus)} ({out_corpus_correct / len(out_corpus) * 100:.0f}%)")
    print(f"Mean In-Corpus Cosine Score:      {avg_in_score:.4f} (range: {min(r['top_score'] for r in in_corpus):.4f} - {max(r['top_score'] for r in in_corpus):.4f})")
    print(f"Mean Out-Corpus Cosine Score:     {avg_out_score:.4f} (range: {min(r['top_score'] for r in out_corpus):.4f} - {max(r['top_score'] for r in out_corpus):.4f})")
    print(f"Cosine Score Separation Gap:      {avg_in_score - avg_out_score:.4f}")
    print(f"Average In-Corpus Latency:        {avg_in_latency:.1f} ms")
    print(f"Average Out-Corpus Latency:       {avg_out_latency:.1f} ms")
    
    latency_reduction = (1 - (avg_out_latency / avg_in_latency)) * 100 if avg_in_latency > 0 else 0
    print(f"Short-Circuit Latency Reduction:  {latency_reduction:.1f}% (Zero LLM API cost on invalid queries)")
    print("=" * 88)

if __name__ == "__main__":
    run_benchmark()
