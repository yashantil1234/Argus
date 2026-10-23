"""Comprehensive tests for the shared, quota-aware LLMPool (Phase 3).

Coverage:
  1. Concurrency enforcement: Provider semaphore bounds active concurrent requests.
  2. Cooldown tripping: 429/transient error triggers circuit breaker cooldown.
  3. Cooldown bypass: Subsequent calls in normal mode immediately bypass primary to fallback.
  4. Strict benchmark isolation: In STRICT_LLM=true, circuit breaker fails fast (no fallback).
  5. Cooldown expiration & circuit recovery: Primary can be retried once cooldown expires.
  6. Observability: Pool status reflects active concurrency, counts, and cooldowns accurately.
  7. Thread safety: Multi-threaded concurrent worker execution without deadlocks.
"""

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
from app.core.pool import LLMPool, get_llm_pool, reset_llm_pool


class TestLLMPoolConcurrency:
    def test_provider_concurrency_bounded_by_semaphore(self, monkeypatch):
        """Ollama (max_concurrency=1) must serialize concurrent worker calls."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MAX_CONCURRENCY_OLLAMA", "1")

        active_counts = []
        peak_concurrency = 0
        lock = threading.Lock()

        def slow_complete(*args, **kwargs):
            nonlocal peak_concurrency
            with lock:
                current = pool.get_status()["active_concurrency"].get("ollama", 0)
                active_counts.append(current)
                if current > peak_concurrency:
                    peak_concurrency = current
            time.sleep(0.05)
            return LLMResponse(text="response", provider="ollama", model="llama3.2:3b")

        pool = get_llm_pool()
        mock_client = MagicMock(spec=BaseLLMClient)
        mock_client.provider_name.return_value = "ollama"
        mock_client.complete_with_meta.side_effect = slow_complete

        with patch("app.core.llm.get_llm_client", return_value=mock_client):
            threads = [
                threading.Thread(target=lambda: ask_llm_with_meta(f"Worker {i}"))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Concurrency must never exceed 1
        assert peak_concurrency <= 1
        assert len(active_counts) == 4


class TestLLMPoolCircuitBreaker:
    def test_transient_failure_trips_cooldown(self, monkeypatch):
        """A transient 429 error on primary trips cooldown for that provider."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "false")
        monkeypatch.setenv("LLM_COOLDOWN_SECONDS", "10.0")

        pool = get_llm_pool()

        primary_mock = MagicMock(spec=BaseLLMClient)
        primary_mock.provider_name.return_value = "gemini"
        primary_mock.complete_with_meta.side_effect = _TransientError("429 Quota Exceeded", retry_after=5.0)

        fallback_mock = MagicMock(spec=BaseLLMClient)
        fallback_mock.provider_name.return_value = "groq"
        fallback_mock.complete_with_meta.return_value = LLMResponse(
            text="groq answer", provider="groq", model="llama-3.3-70b-versatile"
        )

        with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
             patch("app.core.llm._build_client", return_value=fallback_mock), \
             patch("time.sleep"):
            resp = ask_llm_with_meta("Query 1")

        assert resp.fallback_used is True
        assert resp.provider == "groq"

        # Check circuit breaker state
        in_cd, remaining, reason = pool.is_in_cooldown("gemini")
        assert in_cd is True
        assert remaining > 0.0
        assert "429" in reason

    def test_subsequent_calls_bypass_primary_during_cooldown(self, monkeypatch):
        """While primary is in cooldown, subsequent calls route to fallback immediately."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "false")

        pool = get_llm_pool()
        # Manually trip cooldown on gemini
        pool.trip_cooldown("gemini", duration=20.0, reason="Previous 429")

        primary_mock = MagicMock(spec=BaseLLMClient)
        primary_mock.provider_name.return_value = "gemini"

        fallback_mock = MagicMock(spec=BaseLLMClient)
        fallback_mock.provider_name.return_value = "groq"
        fallback_mock.complete_with_meta.return_value = LLMResponse(
            text="instant fallback", provider="groq", model="llama-3.3-70b-versatile"
        )

        with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
             patch("app.core.llm._build_client", return_value=fallback_mock):
            resp = ask_llm_with_meta("Query during cooldown")

        # Primary was completely bypassed without any call
        primary_mock.complete_with_meta.assert_not_called()
        assert resp.fallback_used is True
        assert resp.text == "instant fallback"

    def test_strict_mode_blocks_fallback_even_during_cooldown(self, monkeypatch):
        """In benchmark mode (STRICT_LLM=true), cooldown raises LLMError instead of falling back."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "true")

        pool = get_llm_pool()
        pool.trip_cooldown("gemini", duration=15.0, reason="Benchmark throttle")

        fallback_mock = MagicMock(spec=BaseLLMClient)

        with patch("app.core.llm._build_client", return_value=fallback_mock):
            with pytest.raises(LLMError, match="STRICT_LLM=true disables fallback"):
                ask_llm_with_meta("Benchmark query")

        fallback_mock.assert_not_called()

    def test_cooldown_recovery_clears_on_success(self, monkeypatch):
        """When provider succeeds after cooldown expires, cooldown is cleared."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "true")

        pool = get_llm_pool()
        # Set short expired cooldown
        pool.trip_cooldown("groq", duration=0.001, reason="Short test cooldown")
        time.sleep(0.01)  # Allow cooldown to expire

        mock_client = MagicMock(spec=BaseLLMClient)
        mock_client.provider_name.return_value = "groq"
        mock_client.complete_with_meta.return_value = LLMResponse(
            text="Recovered answer", provider="groq", model="llama-3.3-70b-versatile"
        )

        with patch("app.core.llm.get_llm_client", return_value=mock_client):
            resp = ask_llm_with_meta("Recovery query")

        assert resp.text == "Recovered answer"
        in_cd, _, _ = pool.is_in_cooldown("groq")
        assert in_cd is False


class TestLLMPoolDiagnostics:
    def test_pool_status_tracking(self, monkeypatch):
        """Pool status accurately reports total requests, success, trips, and fallbacks."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")

        pool = get_llm_pool()
        mock_client = MagicMock(spec=BaseLLMClient)
        mock_client.provider_name.return_value = "groq"
        mock_client.complete_with_meta.return_value = LLMResponse(
            text="ok", provider="groq", model="llama-3.3-70b-versatile"
        )

        with patch("app.core.llm.get_llm_client", return_value=mock_client):
            ask_llm_with_meta("Status test 1")
            ask_llm_with_meta("Status test 2")

        status = pool.get_status()
        assert status["total_requests"] == 2
        assert status["successful_requests"] == 2
        assert status["fallback_requests"] == 0
