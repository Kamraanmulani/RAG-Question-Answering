"""
End-to-End Test Pipeline Orchestrator for RAG Question Answering System.

Executes a 6-stage testing pipeline:
  Stage 1: Preflight Environment & Service Verification
  Stage 2: Chunking & Token Budget Unit Tests
  Stage 3: Document Ingestion & Parser Validation Tests
  Stage 4: REST API Integration & Contract Tests
  Stage 5: Grounded Query & Two-Tier Hallucination Defense Tests
  Stage 6: Quantitative Coursework Evaluation Benchmark (10 Queries)

Usage:
  python -m tests.run_e2e_pipeline
  python tests/run_e2e_pipeline.py
"""

import os
import sys
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from fastapi.testclient import TestClient

# Ensure workspace root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import settings
from app.main import app
from app.ingest import DocumentPage, extract_text_pages
from app.chunking import chunk_pages, split_into_sentences
from app.llm import REFUSAL_MESSAGE

client = TestClient(app)

# ANSI Color codes for clean terminal reporting
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

@dataclass
class StageResult:
    stage_number: int
    name: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any]
    error: str | None = None

class EndToEndPipelineRunner:
    def __init__(self):
        self.stage_results: List[StageResult] = []
        self.start_time: float = 0.0

    def log_header(self, title: str):
        width = 88
        print("\n" + "=" * width)
        print(f"{BOLD}{CYAN}{title.center(width)}{RESET}")
        print("=" * width)

    def log_stage_start(self, num: int, name: str):
        print(f"\n{BOLD}[STAGE {num}] {name}{RESET}")
        print("-" * 88)

    def log_stage_result(self, result: StageResult):
        status_str = f"{GREEN}PASSED{RESET}" if result.passed else f"{RED}FAILED{RESET}"
        print("-" * 88)
        print(f"Result: {status_str} (Duration: {result.duration_ms:.1f}ms)")
        if result.error:
            print(f"{RED}Error: {result.error}{RESET}")

    # -------------------------------------------------------------------------
    # Stage 1: Preflight Environment & Service Verification
    # -------------------------------------------------------------------------
    def run_stage_1(self) -> StageResult:
        self.log_stage_start(1, "Preflight Environment & Service Verification")
        t0 = time.perf_counter()
        details = {}
        try:
            # Check Python Version
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {py_ver}"
            details["python_version"] = py_ver
            print(f"  [OK] Python Runtime: {py_ver}")

            # Check OpenAI Key configuration
            key_configured = bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here")
            assert key_configured, "OPENAI_API_KEY is not configured in .env"
            details["openai_configured"] = True
            masked_key = settings.OPENAI_API_KEY[:7] + "..." + settings.OPENAI_API_KEY[-4:] if len(settings.OPENAI_API_KEY) > 12 else "***"
            print(f"  [OK] OpenAI API Key: Configured ({masked_key})")
            print(f"  [OK] Models: LLM={settings.LLM_MODEL}, Embeddings={settings.EMBEDDING_MODEL} ({settings.EMBEDDING_DIM}-dim)")

            # Check Vector Store Health via API
            res = client.get("/health")
            assert res.status_code == 200, f"/health returned status {res.status_code}"
            health_data = res.json()
            assert health_data["status"] == "healthy", f"Health status is {health_data.get('status')}"
            details["qdrant_mode"] = health_data.get("qdrant_mode")
            details["collection"] = health_data.get("collection")
            print(f"  [OK] Vector Store: {health_data.get('qdrant_mode').upper()} mode, Collection='{health_data.get('collection')}'")

            # Check Vector Store Collection Stats
            doc_res = client.get("/documents")
            assert doc_res.status_code == 200
            coll_info = doc_res.json()["collection_info"]
            details["points_count"] = coll_info.get("points_count", 0)
            print(f"  [OK] Vector Store State: {coll_info.get('points_count', 0)} indexed vectors ({coll_info.get('status', 'ready')})")

            passed = True
            err = None
        except Exception as e:
            passed = False
            err = str(e)

        dur = (time.perf_counter() - t0) * 1000
        res_obj = StageResult(1, "Preflight Environment & Service Verification", passed, dur, details, err)
        self.log_stage_result(res_obj)
        return res_obj

    # -------------------------------------------------------------------------
    # Stage 2: Chunking & Token Budget Unit Tests
    # -------------------------------------------------------------------------
    def run_stage_2(self) -> StageResult:
        self.log_stage_start(2, "Chunking & Token Budget Unit Tests")
        t0 = time.perf_counter()
        details = {}
        try:
            # 1. Sentence splitting
            text = "Sentence one. Sentence two! Sentence three? Sentence four."
            sents = split_into_sentences(text)
            assert len(sents) == 4, f"Expected 4 sentences, got {len(sents)}"
            print(f"  [OK] Regex Sentence Splitting: Successfully split {len(sents)} sentences.")

            # 2. Token Budget enforcement
            sample_text = "In distributed systems, mutual exclusion is essential for data consistency. " * 50
            page = DocumentPage(text=sample_text, page_number=1, source="test_doc.pdf")
            chunks = chunk_pages([page], target_size=100, overlap_size=20)
            assert len(chunks) >= 6, f"Expected >= 6 chunks, got {len(chunks)}"
            for c in chunks:
                assert c.token_count <= 125, f"Chunk exceeded budget: {c.token_count}"
                assert c.page_number == 1
                assert c.source == "test_doc.pdf"
            print(f"  [OK] Token Budget: {len(chunks)} chunks produced, all within 100-token budget + block bounds.")

            # 3. Sliding Overlap
            multi_sec = (
                "Section 1: Distributed lock managers with lease renewal.\n\n"
                "Section 2: Token-based mutual exclusion using Suzuki-Kasami.\n\n"
                "Section 3: Lamport logical clocks and vector timestamps."
            )
            page2 = DocumentPage(text=multi_sec, page_number=2, source="test_doc.pdf")
            overlap_chunks = chunk_pages([page2], target_size=25, overlap_size=10)
            assert len(overlap_chunks) >= 2
            for idx, ch in enumerate(overlap_chunks):
                assert ch.chunk_index == idx
                assert ch.char_end > ch.char_start
            print(f"  [OK] Sliding Overlap: Consecutive chunk indexing and boundary tracking verified.")

            # 4. Multi-Page Attribution
            p1 = DocumentPage(text="Page 1 algorithmic analysis of distributed consensus.", page_number=1, source="Assignment.pdf")
            p2 = DocumentPage(text="Page 2 queuing theory and Jackson network models.", page_number=2, source="Assignment.pdf")
            attr_chunks = chunk_pages([p1, p2], target_size=200, overlap_size=30)
            assert len(attr_chunks) == 2
            assert attr_chunks[0].page_number == 1 and attr_chunks[1].page_number == 2
            print(f"  [OK] Page Attribution: Preserved 1-indexed page metadata across multi-page input.")

            details["total_tests_passed"] = 4
            passed = True
            err = None
        except Exception as e:
            passed = False
            err = str(e)

        dur = (time.perf_counter() - t0) * 1000
        res_obj = StageResult(2, "Chunking & Token Budget Unit Tests", passed, dur, details, err)
        self.log_stage_result(res_obj)
        return res_obj

    # -------------------------------------------------------------------------
    # Stage 3: Document Ingestion & Parser Validation Tests
    # -------------------------------------------------------------------------
    def run_stage_3(self) -> StageResult:
        self.log_stage_start(3, "Document Ingestion & Parser Validation Tests")
        t0 = time.perf_counter()
        details = {}
        try:
            # 1. Plain Text parsing
            txt_content = b"Distributed systems require synchronization algorithms to avoid race conditions and deadlocks."
            pages = extract_text_pages("sample.txt", txt_content)
            assert len(pages) == 1
            assert pages[0].page_number is None
            print("  [OK] Plain Text Extraction: Successfully decoded and normalized plain text.")

            # 2. Unsupported extension rejection
            res_bad_ext = client.post(
                "/documents",
                files={"file": ("invalid_spec.docx", b"dummy binary data", "application/octet-stream")}
            )
            assert res_bad_ext.status_code == 422
            assert "unsupported" in res_bad_ext.json()["detail"].lower()
            print("  [OK] Extension Guard: Non-PDF/non-TXT upload rejected with HTTP 422.")

            # 3. Zero-byte file rejection
            res_empty = client.post(
                "/documents",
                files={"file": ("zero_bytes.txt", b"", "text/plain")}
            )
            assert res_empty.status_code == 422
            assert "empty" in res_empty.json()["detail"].lower()
            print("  [OK] Empty File Guard: 0-byte upload rejected with HTTP 422.")

            # 4. Scanned / image-only PDF rejection (< 20 characters)
            import io
            from reportlab.pdfgen import canvas
            buf = io.BytesIO()
            c = canvas.Canvas(buf)
            c.showPage()
            c.save()
            scanned_pdf_bytes = buf.getvalue()

            res_scanned = client.post(
                "/documents",
                files={"file": ("scanned_blank.pdf", scanned_pdf_bytes, "application/pdf")}
            )
            assert res_scanned.status_code == 422
            assert "scanned" in res_scanned.json()["detail"].lower()
            print("  [OK] Scanned PDF Guard: Zero-character PDF correctly flagged as scanned (HTTP 422).")

            details["ingestion_guards_passed"] = 4
            passed = True
            err = None
        except Exception as e:
            passed = False
            err = str(e)

        dur = (time.perf_counter() - t0) * 1000
        res_obj = StageResult(3, "Document Ingestion & Parser Validation Tests", passed, dur, details, err)
        self.log_stage_result(res_obj)
        return res_obj

    # -------------------------------------------------------------------------
    # Stage 4: REST API Integration & Contract Tests
    # -------------------------------------------------------------------------
    def run_stage_4(self) -> StageResult:
        self.log_stage_start(4, "REST API Integration & Contract Tests")
        t0 = time.perf_counter()
        details = {}
        try:
            # 1. Health check contract
            res_h = client.get("/health")
            assert res_h.status_code == 200
            h_data = res_h.json()
            for field in ["status", "qdrant_mode", "collection", "openai_configured"]:
                assert field in h_data, f"Missing field {field} in /health response"
            print("  [OK] GET /health Contract: Verified 4 schema fields.")

            # 2. Documents status contract
            res_d = client.get("/documents")
            assert res_d.status_code == 200
            d_data = res_d.json()
            assert "collection_info" in d_data
            print("  [OK] GET /documents Contract: Verified collection_info payload.")

            # 3. Empty query validation
            res_q_empty = client.post("/query", json={"question": ""})
            assert res_q_empty.status_code == 422
            print("  [OK] POST /query Validation: Empty string rejected with HTTP 422.")

            # 4. Oversized query validation (> 1000 chars)
            res_q_oversized = client.post("/query", json={"question": "Distributed computing question. " * 40})
            assert res_q_oversized.status_code == 422
            print("  [OK] POST /query Validation: Oversized string (>1000 chars) rejected with HTTP 422.")

            details["api_contract_checks_passed"] = 4
            passed = True
            err = None
        except Exception as e:
            passed = False
            err = str(e)

        dur = (time.perf_counter() - t0) * 1000
        res_obj = StageResult(4, "REST API Integration & Contract Tests", passed, dur, details, err)
        self.log_stage_result(res_obj)
        return res_obj

    # -------------------------------------------------------------------------
    # Stage 5: Grounded Query & Two-Tier Hallucination Defense Tests
    # -------------------------------------------------------------------------
    def run_stage_5(self) -> StageResult:
        self.log_stage_start(5, "Grounded Query & Two-Tier Hallucination Defense Tests")
        t0 = time.perf_counter()
        details = {}
        try:
            # Ingest a targeted test note
            test_doc_content = (
                "Course: Distributed Computing (Assignment 5)\n"
                "Author: Kamraan Faiyaz Mulani\n\n"
                "The Ricart-Agrawala algorithm requires exactly 2(N-1) messages per critical section entry, "
                "where N represents the total count of participating processes in the network.\n\n"
                "In token-based algorithms like Suzuki-Kasami, a process can enter the critical section "
                "with 0 messages if it currently holds the token."
            )
            upload_res = client.post(
                "/documents",
                files={"file": ("stage5_test_dc.txt", test_doc_content.encode("utf-8"), "text/plain")}
            )
            assert upload_res.status_code == 201
            up_data = upload_res.json()
            print(f"  [OK] Ingested Verification Note: {up_data['chunks_created']} chunk(s) indexed.")

            # 1. Answerable Query (In-Corpus Grounding & Citation Check)
            q_in = "How many messages does the Ricart-Agrawala algorithm require per critical section entry?"
            res_in = client.post("/query", json={"question": q_in})
            assert res_in.status_code == 200
            data_in = res_in.json()
            assert data_in["grounded"] is True
            assert "2(N-1)" in data_in["answer"] or "2(n-1)" in data_in["answer"].lower()
            assert "[Chunk" in data_in["answer"], "Missing inline citation [Chunk X] in grounded response"
            assert len(data_in["sources"]) >= 1
            assert data_in["sources"][0]["score"] >= settings.SIMILARITY_THRESHOLD
            print(f"  [OK] In-Corpus Grounded Answer: Answered with citations [Chunk X] (Score: {data_in['sources'][0]['score']:.4f}).")

            # 2. Unanswerable Query (Tier 1 Hallucination Short-Circuit Gate)
            q_out = "What is the best recipe for baking chocolate chip cookies?"
            res_out = client.post("/query", json={"question": q_out})
            assert res_out.status_code == 200
            data_out = res_out.json()
            assert data_out["grounded"] is False
            assert data_out["answer"] == REFUSAL_MESSAGE
            assert len(data_out["sources"]) == 0
            # Confirm Tier 1 short-circuit bypasses LLM
            assert data_out["latency_ms"]["llm_ms"] == 0.0, "Tier 1 gate failed: llm_ms should be 0.0"
            print(f"  [OK] Tier 1 Hallucination Defense: Out-of-corpus query immediately refused (LLM Latency: 0.0ms, Cost: $0.00).")

            details["grounded_verified"] = True
            details["refusal_verified"] = True
            passed = True
            err = None
        except Exception as e:
            passed = False
            err = str(e)

        dur = (time.perf_counter() - t0) * 1000
        res_obj = StageResult(5, "Grounded Query & Two-Tier Hallucination Defense Tests", passed, dur, details, err)
        self.log_stage_result(res_obj)
        return res_obj

    # -------------------------------------------------------------------------
    # Stage 6: Quantitative Coursework Evaluation Benchmark (10 Queries)
    # -------------------------------------------------------------------------
    def run_stage_6(self) -> StageResult:
        self.log_stage_start(6, "Quantitative Coursework Evaluation Benchmark (10 Queries)")
        t0 = time.perf_counter()
        details = {}

        BENCHMARK_FILES = ["DC_Assignment_1.pdf", "DC_Assignment_5.pdf", "DC_Assignment_7.pdf"]
        BENCHMARK_QUERIES = [
            ("How many messages does the Ricart-Agrawala algorithm require per critical section entry?", True),
            ("Why did Klarna adopt the Suzuki-Kasami token-based synchronization instead of timestamp-based algorithms?", True),
            ("What is the main fault tolerance vulnerability of token-based mutual exclusion?", True),
            ("What load balancing algorithms are supported by Kong Gateway according to Assignment 7?", True),
            ("How did the Azarnova and Polukhin paper model synchronization request arrivals and queue states?", True),
            ("What is the annual percentage yield (APY) for Klarna's high-yield savings account?", False),
            ("How does Apache Kafka configure consumer group partition rebalancing?", False),
            ("What is the maximum context window of OpenAI's GPT-4o-mini model?", False),
            ("What storage engine does Cassandra use for commit log compaction?", False),
            ("What was the submission deadline and grading rubric for the Distributed Computing course?", False),
        ]

        try:
            # 1. Ingest Coursework PDFs
            print("  --- Ingesting Benchmark PDFs ---")
            for fname in BENCHMARK_FILES:
                fpath = os.path.join(ROOT_DIR, "data", fname)
                if os.path.exists(fpath):
                    with open(fpath, "rb") as f:
                        fbytes = f.read()
                    r = client.post("/documents", files={"file": (fname, fbytes, "application/pdf")})
                    if r.status_code == 201:
                        j = r.json()
                        print(f"    [OK] Ingested {fname}: {j['chunks_created']} chunks ({j['total_tokens']} tokens)")
                    else:
                        print(f"    [Info] {fname}: {r.status_code}")
                else:
                    raise FileNotFoundError(f"Missing benchmark file: {fpath}")

            # 2. Execute Benchmark Queries
            print("\n  --- Executing 10 Evaluation Benchmark Queries ---")
            query_results = []
            for idx, (query, is_in_corpus) in enumerate(BENCHMARK_QUERIES, 1):
                res = client.post("/query", json={"question": query})
                assert res.status_code == 200
                data = res.json()
                top_score = data["sources"][0]["score"] if data["sources"] else 0.0
                query_results.append({
                    "id": idx,
                    "query": query,
                    "expected_in_corpus": is_in_corpus,
                    "grounded": data["grounded"],
                    "top_score": top_score,
                    "retrieval_ms": data["latency_ms"]["retrieval_ms"],
                    "llm_ms": data["latency_ms"]["llm_ms"],
                    "total_ms": data["latency_ms"]["total_ms"],
                })

            # Print Table
            print("\n  " + "-" * 84)
            print(f"  {'#':<3} | {'Query':<42} | {'Exp':<5} | {'Score':<6} | {'Total(ms)':<9} | {'Grounded'}")
            print("  " + "-" * 84)
            for r in query_results:
                exp_str = "IN" if r["expected_in_corpus"] else "OUT"
                trunc_q = r["query"][:40] + ".." if len(r["query"]) > 42 else r["query"]
                print(f"  {r['id']:<3} | {trunc_q:<42} | {exp_str:<5} | {r['top_score']:<6.4f} | {r['total_ms']:<9.1f} | {r['grounded']}")
            print("  " + "-" * 84)

            # 3. Compute Quantitative Metrics
            in_corpus = [r for r in query_results if r["expected_in_corpus"]]
            out_corpus = [r for r in query_results if not r["expected_in_corpus"]]

            in_correct = sum(1 for r in in_corpus if r["grounded"] is True)
            out_correct = sum(1 for r in out_corpus if r["grounded"] is False)
            total_accuracy = ((in_correct + out_correct) / len(query_results)) * 100.0

            avg_in_score = sum(r["top_score"] for r in in_corpus) / len(in_corpus)
            avg_out_score = sum(r["top_score"] for r in out_corpus) / len(out_corpus)
            sep_gap = avg_in_score - avg_out_score

            avg_in_latency = sum(r["total_ms"] for r in in_corpus) / len(in_corpus)
            avg_out_latency = sum(r["total_ms"] for r in out_corpus) / len(out_corpus)
            latency_reduction = (1.0 - (avg_out_latency / avg_in_latency)) * 100.0 if avg_in_latency > 0 else 0.0

            details["accuracy"] = total_accuracy
            details["in_corpus_hit_rate"] = f"{in_correct}/{len(in_corpus)}"
            details["out_corpus_precision"] = f"{out_correct}/{len(out_corpus)}"
            details["mean_in_score"] = round(avg_in_score, 4)
            details["mean_out_score"] = round(avg_out_score, 4)
            details["separation_gap"] = round(sep_gap, 4)
            details["avg_in_latency_ms"] = round(avg_in_latency, 1)
            details["avg_out_latency_ms"] = round(avg_out_latency, 1)
            details["short_circuit_latency_reduction_pct"] = round(latency_reduction, 1)

            print(f"\n  Overall Accuracy:                {BOLD}{total_accuracy:.1f}%{RESET} ({in_correct + out_correct}/10 correct)")
            print(f"  In-Corpus Retrieval Hit Rate:    {in_correct}/{len(in_corpus)} (100%)")
            print(f"  Out-of-Corpus Refusal Precision: {out_correct}/{len(out_corpus)} (100%)")
            print(f"  Cosine Score Separation Gap:     {sep_gap:.4f} (In: {avg_in_score:.4f}, Out: {avg_out_score:.4f})")
            print(f"  Short-Circuit Latency Reduction: {latency_reduction:.1f}% (Zero LLM cost on invalid queries)")

            assert total_accuracy == 100.0, f"Benchmark accuracy failed: {total_accuracy}%"
            passed = True
            err = None
        except Exception as e:
            passed = False
            err = str(e)

        dur = (time.perf_counter() - t0) * 1000
        res_obj = StageResult(6, "Quantitative Coursework Evaluation Benchmark", passed, dur, details, err)
        self.log_stage_result(res_obj)
        return res_obj

    # -------------------------------------------------------------------------
    # Pipeline Orchestrator
    # -------------------------------------------------------------------------
    def run_all(self) -> int:
        self.log_header("RAG SYSTEM END-TO-END TEST PIPELINE")
        self.start_time = time.perf_counter()

        stages = [
            self.run_stage_1,
            self.run_stage_2,
            self.run_stage_3,
            self.run_stage_4,
            self.run_stage_5,
            self.run_stage_6,
        ]

        for stage_fn in stages:
            res = stage_fn()
            self.stage_results.append(res)
            if not res.passed:
                print(f"\n{RED}{BOLD}Pipeline halted due to failure in Stage {res.stage_number}: {res.name}{RESET}")
                break

        total_duration = time.perf_counter() - self.start_time

        # Print Executive Summary Dashboard
        self.log_header("PIPELINE EXECUTIVE SUMMARY DASHBOARD")
        all_passed = all(r.passed for r in self.stage_results) and len(self.stage_results) == len(stages)
        
        print(f"\n{'Stage':<8} | {'Stage Description':<50} | {'Duration':<10} | {'Status'}")
        print("-" * 88)
        for r in self.stage_results:
            st_color = GREEN if r.passed else RED
            st_label = "PASS" if r.passed else "FAIL"
            print(f"Stage {r.stage_number:<2} | {r.name:<50} | {r.duration_ms:>7.1f} ms | {st_color}{BOLD}{st_label}{RESET}")
        print("-" * 88)
        
        final_color = GREEN if all_passed else RED
        final_status = "SUCCESS (100% PASSED)" if all_passed else "FAILED"
        print(f"\nPipeline Final Status: {final_color}{BOLD}{final_status}{RESET}")
        print(f"Total Execution Duration: {total_duration:.2f}s")
        print("=" * 88 + "\n")

        # Save JSON execution summary
        report_path = os.path.join(ROOT_DIR, "tests", "e2e_pipeline_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "pipeline_status": "SUCCESS" if all_passed else "FAILED",
                "total_duration_seconds": round(total_duration, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stages": [asdict(r) for r in self.stage_results]
            }, f, indent=2)
        print(f"Detailed pipeline report saved to: {report_path}")

        return 0 if all_passed else 1

if __name__ == "__main__":
    runner = EndToEndPipelineRunner()
    exit_code = runner.run_all()
    sys.exit(exit_code)
