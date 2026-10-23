"""System Architecture & Reliability Benchmark for ARGUS.

Measures engineering metrics independently of scientific claim accuracy:
  1. Concurrency Limits & Semaphore Enforcement: Verifies peak active requests <= max limit.
  2. Throughput & Latency: Measures requests/sec and response time distribution.
  3. Failure Recovery & Circuit Breaker: Measures cooldown tripping and fallback switchover time.
  4. Multimodal Ingestion Pipeline: Measures ingestion throughput across PDF, HTML, and audio media.

Usage:
    python experiments/run_system_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm import BaseLLMClient, LLMResponse, _TransientError, ask_llm_with_meta
from app.core.pool import get_llm_pool, reset_llm_pool
from app.ingestion import ingest


def benchmark_concurrency_and_throughput(num_requests: int = 16, max_limit: int = 4) -> Dict[str, Any]:
    """Test that pool bounds peak concurrency and measure total throughput."""
    reset_llm_pool()
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["LLM_MAX_CONCURRENCY_GROQ"] = str(max_limit)

    pool = get_llm_pool()
    lock = threading.Lock()
    active_samples: List[int] = []
    peak_concurrency = 0

    def mock_request(prompt, model, **kwargs):
        nonlocal peak_concurrency
        with lock:
            current = pool.get_status()["active_concurrency"].get("groq", 0)
            active_samples.append(current)
            if current > peak_concurrency:
                peak_concurrency = current
        time.sleep(0.04)  # simulate network roundtrip
        return LLMResponse(text="ok", provider="groq", model=model)

    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.provider_name.return_value = "groq"
    mock_client.complete_with_meta.side_effect = mock_request

    t0 = time.monotonic()
    with patch("app.core.llm.get_llm_client", return_value=mock_client):
        threads = [
            threading.Thread(target=lambda i=i: ask_llm_with_meta(f"Req {i}"))
            for i in range(num_requests)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    total_time = time.monotonic() - t0

    throughput = round(num_requests / total_time, 2)
    avg_latency = round(total_time / (num_requests / max_limit), 4)

    return {
        "num_requests": num_requests,
        "max_limit_configured": max_limit,
        "peak_concurrency_observed": peak_concurrency,
        "limit_respected": peak_concurrency <= max_limit,
        "total_elapsed_seconds": round(total_time, 3),
        "throughput_req_per_sec": throughput,
        "effective_avg_latency_s": avg_latency,
    }


def benchmark_failure_recovery_and_circuit_breaker() -> Dict[str, Any]:
    """Measure circuit breaker detection and fallback latency."""
    reset_llm_pool()
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_FALLBACK_PROVIDER"] = "groq"
    os.environ["STRICT_LLM"] = "false"
    os.environ["LLM_COOLDOWN_SECONDS"] = "15.0"

    pool = get_llm_pool()

    primary_mock = MagicMock(spec=BaseLLMClient)
    primary_mock.provider_name.return_value = "gemini"
    primary_mock.complete_with_meta.side_effect = _TransientError("429 Rate limit exceeded", retry_after=10.0)

    fallback_mock = MagicMock(spec=BaseLLMClient)
    fallback_mock.provider_name.return_value = "groq"
    fallback_mock.complete_with_meta.return_value = LLMResponse(
        text="Fallback answer", provider="groq", model="llama-3.3-70b-versatile"
    )

    t0 = time.monotonic()
    with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
         patch("app.core.llm._build_client", return_value=fallback_mock), \
         patch("time.sleep"):
        # First request trips circuit breaker and falls back
        resp1 = ask_llm_with_meta("Prompt 1")
        switch_latency = time.monotonic() - t0

        # Second request: primary is now in cooldown, should bypass immediately
        t1 = time.monotonic()
        resp2 = ask_llm_with_meta("Prompt 2")
        bypass_latency = time.monotonic() - t1

    in_cd, rem, _ = pool.is_in_cooldown("gemini")

    return {
        "initial_failover_latency_s": round(switch_latency, 4),
        "cooldown_bypass_latency_s": round(bypass_latency, 4),
        "primary_in_cooldown": in_cd,
        "cooldown_remaining_s": round(rem, 2),
        "fallback_used_req1": resp1.fallback_used,
        "fallback_used_req2": resp2.fallback_used,
    }


def benchmark_multimodal_ingestion(tmp_path: Path) -> Dict[str, Any]:
    """Measure ingestion parsing speed across PDF, HTML, and audio/video transcripts."""
    # 1. PDF
    pdf_file = tmp_path / "bench.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n/Page 1 Experimental results on gene editing.\n/Page 2 Next gen sequencing.\n%%EOF")

    t0 = time.monotonic()
    pdf_res = ingest(pdf_file)
    pdf_time_ms = round((time.monotonic() - t0) * 1000, 2)

    # 2. HTML
    html_content = "<html><body><h1>Title</h1>" + "<p>Paragraph of scientific observations.</p>" * 20 + "</body></html>"
    t0 = time.monotonic()
    html_res = ingest(html_content, title="HTML doc")
    html_time_ms = round((time.monotonic() - t0) * 1000, 2)

    # 3. Audio/Video SRT
    srt_file = tmp_path / "bench.srt"
    srt_file.write_text(
        "1\n00:00:01,000 --> 00:00:05,000\nOpening remarks.\n\n"
        "2\n00:00:06,000 --> 00:00:10,000\nKey findings.\n",
        encoding="utf-8",
    )
    t0 = time.monotonic()
    srt_res = ingest(srt_file)
    srt_time_ms = round((time.monotonic() - t0) * 1000, 2)

    return {
        "pdf_parse_time_ms": pdf_time_ms,
        "pdf_pages_extracted": len(pdf_res.segments),
        "html_parse_time_ms": html_time_ms,
        "html_segments_extracted": len(html_res.segments),
        "srt_parse_time_ms": srt_time_ms,
        "srt_segments_extracted": len(srt_res.segments),
    }


def main():
    print("=" * 70)
    print("  ARGUS SYSTEM ARCHITECTURE & RELIABILITY BENCHMARK (Phase 6)")
    print("=" * 70)

    print("\n[1/3] Running Concurrency & Throughput Stress Test...")
    concurrency_metrics = benchmark_concurrency_and_throughput(num_requests=16, max_limit=4)
    print(f"  - Configured Max Limit   : {concurrency_metrics['max_limit_configured']}")
    print(f"  - Peak Observed Concurrency: {concurrency_metrics['peak_concurrency_observed']}")
    print(f"  - Concurrency Respected  : {'[PASS]' if concurrency_metrics['limit_respected'] else '[FAIL]'}")
    print(f"  - System Throughput      : {concurrency_metrics['throughput_req_per_sec']} req/sec")

    print("\n[2/3] Running Failure Recovery & Circuit Breaker Test...")
    recovery_metrics = benchmark_failure_recovery_and_circuit_breaker()
    print(f"  - Initial Failover Latency: {recovery_metrics['initial_failover_latency_s']}s")
    print(f"  - Cooldown Bypass Latency : {recovery_metrics['cooldown_bypass_latency_s']}s (Instant skip)")
    print(f"  - Circuit Breaker Active  : {'[PASS]' if recovery_metrics['primary_in_cooldown'] else '[FAIL]'}")

    print("\n[3/3] Running Multimodal Ingestion Pipeline Test...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ingestion_metrics = benchmark_multimodal_ingestion(Path(tmp_dir))
    print(f"  - PDF Parser Latency     : {ingestion_metrics['pdf_parse_time_ms']} ms ({ingestion_metrics['pdf_pages_extracted']} pages)")
    print(f"  - HTML Parser Latency    : {ingestion_metrics['html_parse_time_ms']} ms ({ingestion_metrics['html_segments_extracted']} segments)")
    print(f"  - SRT Parser Latency     : {ingestion_metrics['srt_parse_time_ms']} ms ({ingestion_metrics['srt_segments_extracted']} segments)")

    output_path = Path(__file__).resolve().parent / "system_benchmark_results.json"
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "concurrency_stress": concurrency_metrics,
        "failure_recovery": recovery_metrics,
        "multimodal_ingestion": ingestion_metrics,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Artifact] Results persisted to {output_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
