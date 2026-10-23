"""Reliability, Concurrency Stress, and Multimodal Provenance Audit (Phase 6 Items 3, 4, 5)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.core.llm import (
    BaseLLMClient,
    LLMError,
    LLMResponse,
    _AuthError,
    _TransientError,
    ask_llm_with_meta,
)
from app.core.pool import get_llm_pool
from app.core.provenance import compute_content_hash, verify_quote_in_text
from app.ingestion import ingest
from app.models.research import Evidence, Source, SourceType


# =============================================================================
# Item 3: Concurrency Stress Test
# =============================================================================

class TestConcurrencyStress:
    def test_high_concurrency_stress_strictly_respects_provider_limits(self, monkeypatch):
        """20 simultaneous agent worker threads must never exceed the semaphore bound."""
        max_allowed_concurrency = 3
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_MAX_CONCURRENCY_GEMINI", str(max_allowed_concurrency))
        monkeypatch.setenv("LLM_RPM_GEMINI", "1000")  # High RPM so test focuses on concurrency, not rate pacing

        pool = get_llm_pool()
        lock = threading.Lock()
        active_counts = []
        peak_concurrency = 0

        def simulated_llm_worker(prompt, model, **kwargs):
            nonlocal peak_concurrency
            with lock:
                current = pool.get_status()["active_concurrency"].get("gemini", 0)
                active_counts.append(current)
                if current > peak_concurrency:
                    peak_concurrency = current
            # Simulate real network work
            time.sleep(0.02)
            return LLMResponse(text=f"Processed: {prompt}", provider="gemini", model=model)

        mock_client = MagicMock(spec=BaseLLMClient)
        mock_client.provider_name.return_value = "gemini"
        mock_client.complete_with_meta.side_effect = simulated_llm_worker

        with patch("app.core.llm.get_llm_client", return_value=mock_client):
            threads = [
                threading.Thread(target=lambda i=i: ask_llm_with_meta(f"Concurrent Agent {i}"))
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Concurrency must NEVER exceed max_allowed_concurrency
        assert peak_concurrency <= max_allowed_concurrency
        assert len(active_counts) == 20
        # Active concurrency must drain back to 0
        assert pool.get_status()["active_concurrency"].get("gemini", 0) == 0


# =============================================================================
# Item 4: Failure Recovery & Circuit Breaker Simulation
# =============================================================================

class TestFailureRecovery:
    @pytest.mark.parametrize(
        "error_type,error_msg,expect_cooldown",
        [
            ("429", "429 Too Many Requests: Rate limit exceeded", True),
            ("503", "503 Service Unavailable: High load", True),
            ("timeout", "Connection timeout after 120s", True),
            ("quota", "Resource has been exhausted: check your quota or billing", True),
        ],
    )
    def test_transient_failures_trigger_cooldown_and_fallback(
        self, monkeypatch, error_type, error_msg, expect_cooldown
    ):
        """Simulate 429, 503, timeouts, and quota errors: in normal mode, fallback is taken."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "false")
        monkeypatch.setenv("LLM_COOLDOWN_SECONDS", "30.0")

        pool = get_llm_pool()

        primary_mock = MagicMock(spec=BaseLLMClient)
        primary_mock.provider_name.return_value = "gemini"
        primary_mock.complete_with_meta.side_effect = _TransientError(error_msg, retry_after=15.0)

        fallback_mock = MagicMock(spec=BaseLLMClient)
        fallback_mock.provider_name.return_value = "groq"
        fallback_mock.complete_with_meta.return_value = LLMResponse(
            text="Fallback succeeded", provider="groq", model="llama-3.3-70b-versatile"
        )

        with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
             patch("app.core.llm._build_client", return_value=fallback_mock), \
             patch("time.sleep"):
            resp = ask_llm_with_meta("Test prompt")

        assert resp.fallback_used is True
        assert resp.fallback_from == "gemini"
        assert resp.text == "Fallback succeeded"

        # Circuit breaker should be open
        in_cd, remaining, _ = pool.is_in_cooldown("gemini")
        assert in_cd is expect_cooldown
        assert remaining > 0.0

    def test_strict_mode_fails_fast_on_quota_and_cooldown(self, monkeypatch):
        """In strict benchmark mode, 429/quota error trips cooldown and raises LLMError immediately."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "true")

        primary_mock = MagicMock(spec=BaseLLMClient)
        primary_mock.provider_name.return_value = "gemini"
        primary_mock.complete_with_meta.side_effect = _TransientError("429 Quota Exceeded")

        fallback_mock = MagicMock(spec=BaseLLMClient)

        with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
             patch("app.core.llm._build_client", return_value=fallback_mock), \
             patch("time.sleep"):
            with pytest.raises(LLMError, match="STRICT_LLM=true disables fallback"):
                ask_llm_with_meta("Strict prompt")

        # Fallback was strictly blocked
        fallback_mock.assert_not_called()


# =============================================================================
# Item 5: Multimodal Provenance & Verbatim Verification Validation
# =============================================================================

class TestMultimodalProvenance:
    def test_pdf_page_number_and_content_hash_integrity(self, tmp_path):
        """PDF ingestion generates valid content hash and page locators."""
        pdf_path = tmp_path / "test_doc.pdf"
        sample_bytes = b"%PDF-1.4\n/Page 1: Discovery of Cas12a endonuclease mechanisms.\n%%EOF"
        pdf_path.write_bytes(sample_bytes)

        result = ingest(pdf_path, title="Cas12a Mechanism")

        # 1. Source content hash matches deterministic SHA-256 of page_text
        expected_hash = compute_content_hash(result.source.page_text)
        assert result.source.content_hash == expected_hash

        # 2. Page locator preserved
        assert result.segments[0].page == 1
        assert "Discovery of Cas12a" in result.segments[0].text

        # 3. Text quote verification against source page_text succeeds
        quote = "Discovery of Cas12a endonuclease mechanisms."
        is_valid, score = verify_quote_in_text(quote, result.source.page_text)
        assert is_valid is True
        assert score >= 0.95

    def test_audio_video_timestamp_integrity(self, tmp_path):
        """SRT transcript preserves second-level start/end times and location strings."""
        video_path = tmp_path / "interview.mp4"
        video_path.write_bytes(b"mockvideodata")
        srt_path = tmp_path / "interview.srt"
        srt_path.write_text(
            "1\n00:02:15,000 --> 00:02:20,500\nWe observed zero off-targets across 200 sites.\n",
            encoding="utf-8",
        )

        result = ingest(video_path)

        assert result.source.source_type == SourceType.VIDEO
        seg = result.segments[0]
        assert seg.start_time == 135.0  # 2m 15s = 135s
        assert seg.end_time == 140.5    # 2m 20.5s = 140.5s
        assert seg.location == "02:15 - 02:20"

        # Evidence model instantiates correctly with timestamps
        ev = Evidence(
            source_id=result.source.id,
            quote=seg.text,
            start_time=seg.start_time,
            end_time=seg.end_time,
            location=seg.location,
        )
        assert ev.start_time == 135.0
        assert ev.end_time == 140.5

    def test_verbatim_quote_vs_generated_description_distinction(self):
        """Verbatim check correctly identifies that generated descriptions are not quotes."""
        raw_source_text = "The western blot assay showed high expression of protein X in kidney cells."

        # Verbatim quote: passes
        quote = "western blot assay showed high expression of protein X"
        is_verb, score = verify_quote_in_text(quote, raw_source_text)
        assert is_verb is True

        # AI-generated summary/description: correctly fails verbatim check
        generated_summary = "An image depicting strong protein X band intensity in renal tissue."
        is_verb, score = verify_quote_in_text(generated_summary, raw_source_text)
        assert is_verb is False
        assert score < 0.5
