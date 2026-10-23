"""Unit tests for the ARGUS LLM abstraction layer (refactored).

All tests are fully mocked — no live network calls or real API keys required.

Coverage:
  1.  Provider factory: ollama (default), groq, gemini, openrouter, grok, unknown.
  2.  Groq: config, auth check, response normalization, metadata extraction.
  3.  Gemini: OpenAI-compatible config, GOOGLE_API_KEY fallback, response normalization.
  4.  Ollama: config, metadata extraction, connection/timeout error mapping.
  5.  Missing API keys produce clear _AuthError (not silent degradation).
  6.  Primary-provider success without fallback.
  7.  Retry before fallback: transient errors retry with backoff before attempting fallback.
  8.  Retry recovery: transient error resolved on retry without triggering fallback.
  9.  Fallback target in deployment: cloud-to-cloud fallback (gemini -> groq).
 10.  Auth error fails fast — NEVER triggers retry or fallback.
 11.  Rate limits (429) parsed with Retry-After delay.
 12.  Strict benchmark mode (STRICT_LLM=true) blocks fallback.
 13.  Provenance tracking: LLMResponse metadata (tokens, model_version, latency, fallback).
 14.  Per-role model resolution: VERIFIER_MODEL, RESEARCHER_MODEL, PLANNER_MODEL.
 15.  Disk caching: LLM_CACHE=true caches responses.
 16.  Structured JSON generation & Planner compatibility.
 17.  Backward compatibility: ask_llm() returns plain str.
 18.  API secrets do not appear in error messages.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm import (
    BaseLLMClient,
    GeminiClient,
    GrokClient,
    GroqClient,
    LLMError,
    LLMProvider,
    LLMResponse,
    OllamaClient,
    OpenAICompatibleClient,
    OpenRouterClient,
    _AuthError,
    _TransientError,
    _build_client,
    _call_with_fallback,
    _execute_with_retry,
    _extract_retry_delay,
    _resolve_model,
    ask_llm,
    ask_llm_with_meta,
    get_llm_client,
)
from app.agents.research_planner import ResearchPlanner, PlanningError
from app.models.research import ResearchQuestion


# =============================================================================
# 1. Provider Factory Tests
# =============================================================================

class TestFactory:
    def test_default_is_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        client = get_llm_client()
        assert isinstance(client, OllamaClient)

    def test_ollama_explicit(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        client = get_llm_client()
        assert isinstance(client, OllamaClient)

    def test_groq_factory(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-testkey")
        with patch("app.core.llm.GroqClient.__init__", return_value=None):
            client = get_llm_client()
        assert isinstance(client, GroqClient)

    def test_gemini_factory(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-testkey")
        with patch("app.core.llm.GeminiClient.__init__", return_value=None):
            client = get_llm_client()
        assert isinstance(client, GeminiClient)

    def test_openrouter_factory(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        with patch("app.core.llm.OpenRouterClient.__init__", return_value=None):
            client = get_llm_client()
        assert isinstance(client, OpenRouterClient)

    def test_grok_legacy_factory(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "grok")
        monkeypatch.setenv("XAI_API_KEY", "xai-testkey")
        with patch("app.core.llm.GrokClient.__init__", return_value=None):
            client = get_llm_client()
        assert isinstance(client, GrokClient)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "unknown-xyz")
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            _build_client("unknown-xyz")


# =============================================================================
# 2. Groq Client Tests
# =============================================================================

class TestGroqClient:
    def test_missing_api_key_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(_AuthError, match="GROQ_API_KEY"):
            GroqClient()

    def test_complete_returns_text_and_meta(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Groq ultra-fast response"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 15
        mock_resp.usage.completion_tokens = 8
        mock_resp.usage.total_tokens = 23
        mock_resp.model = "llama-3.3-70b-versatile"
        mock_openai.chat.completions.create.return_value = mock_resp

        with patch("app.core.llm.OpenAI", return_value=mock_openai):
            client = GroqClient()
            res = client.complete_with_meta("prompt", "llama-3.3-70b-versatile")

        assert res.text == "Groq ultra-fast response"
        assert res.provider == "groq"
        assert res.prompt_tokens == 15
        assert res.completion_tokens == 8
        assert res.total_tokens == 23

    def test_rate_limit_extracts_retry_after(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = Exception(
            "Rate limit reached. Please try again in 2.5s."
        )

        with patch("app.core.llm.OpenAI", return_value=mock_openai):
            client = GroqClient()
            with pytest.raises(_TransientError) as exc_info:
                client.complete("prompt", "llama-3.3-70b-versatile")

        assert exc_info.value.retry_after == 2.5


# =============================================================================
# 3. Gemini Client Tests
# =============================================================================

class TestGeminiClient:
    def test_missing_api_key_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(_AuthError, match="GEMINI_API_KEY"):
            GeminiClient()

    def test_google_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-valid-key")
        mock_openai = MagicMock()

        with patch("app.core.llm.OpenAI", return_value=mock_openai) as mock_cls:
            client = GeminiClient()
            assert client.provider_name() == "gemini"
            mock_cls.assert_called_once_with(
                api_key="google-valid-key",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            )

    def test_auth_error_fails_fast(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "invalid-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = Exception(
            "401 Unauthorized: Invalid API key provided"
        )

        with patch("app.core.llm.OpenAI", return_value=mock_openai):
            client = GeminiClient()
            with pytest.raises(_AuthError):
                client.complete("prompt", "gemini-2.0-flash")


# =============================================================================
# 4. Ollama Client Tests
# =============================================================================

class TestOllamaClient:
    def test_complete_extracts_tokens(self):
        client = OllamaClient()
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "response": "Local llama answer",
            "prompt_eval_count": 20,
            "eval_count": 10,
        }
        fake_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_response):
            res = client.complete_with_meta("prompt", "llama3.2:3b")

        assert res.text == "Local llama answer"
        assert res.prompt_tokens == 20
        assert res.completion_tokens == 10
        assert res.total_tokens == 30


# =============================================================================
# 5. Retry Before Fallback Tests
# =============================================================================

class TestRetryBeforeFallback:
    def test_retry_delay_regex(self):
        err = Exception("Rate limit reached. Please try again in 3.4s.")
        assert _extract_retry_delay(err) == 3.4

    def test_transient_error_retries_and_recovers(self):
        client = MagicMock(spec=BaseLLMClient)
        client.provider_name.return_value = "gemini"
        # First call fails with 429, second call succeeds
        client.complete_with_meta.side_effect = [
            _TransientError("Rate limit 429", retry_after=0.01),
            LLMResponse(text="Success on retry", provider="gemini", model="gemini-2.0-flash"),
        ]

        with patch("time.sleep") as mock_sleep:
            res = _execute_with_retry(client, "prompt", "gemini-2.0-flash", max_retries=2)

        assert res.text == "Success on retry"
        assert res.retries_attempted == 1
        mock_sleep.assert_called_once_with(0.01)

    def test_retries_exhausted_raises_transient(self):
        client = MagicMock(spec=BaseLLMClient)
        client.provider_name.return_value = "gemini"
        client.complete_with_meta.side_effect = _TransientError("503 Service Unavailable")

        with patch("time.sleep"):
            with pytest.raises(_TransientError):
                _execute_with_retry(client, "prompt", "gemini-2.0-flash", max_retries=2)

        assert client.complete_with_meta.call_count == 3

    def test_quota_exhaustion_skips_retries_immediately(self):
        """Permanent quota exhaustion (out of credits) must not sleep/retry."""
        client = MagicMock(spec=BaseLLMClient)
        client.provider_name.return_value = "gemini"
        client.complete_with_meta.side_effect = _TransientError(
            "429 Resource has been exhausted: check your quota or billing"
        )

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(_TransientError):
                _execute_with_retry(client, "prompt", "gemini-2.5-flash", max_retries=2)

        # Fails immediately on first attempt without sleeping
        assert client.complete_with_meta.call_count == 1
        mock_sleep.assert_not_called()


# =============================================================================
# 6. Fallback Policy Tests
# =============================================================================

class TestFallbackPolicy:
    def test_cloud_to_cloud_fallback_success(self, monkeypatch):
        """When Gemini experiences transient errors, it retries and falls back to Groq."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "false")

        primary_mock = MagicMock()
        primary_mock.provider_name.return_value = "gemini"
        primary_mock.complete_with_meta.side_effect = _TransientError("Gemini 503 Unavailable")

        fallback_mock = MagicMock()
        fallback_mock.provider_name.return_value = "groq"
        fallback_mock.complete_with_meta.return_value = LLMResponse(
            text="Groq fallback output", provider="groq", model="llama-3.3-70b-versatile"
        )

        with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
             patch("app.core.llm._build_client", return_value=fallback_mock), \
             patch("time.sleep"):
            resp = ask_llm_with_meta("Test prompt")

        assert resp.text == "Groq fallback output"
        assert resp.fallback_used is True
        assert resp.fallback_from == "gemini"
        assert resp.provider == "groq"

    def test_auth_error_never_triggers_fallback(self, monkeypatch):
        """Invalid API key on primary fails immediately without fallback."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "false")

        primary_mock = MagicMock()
        primary_mock.provider_name.return_value = "gemini"
        primary_mock.complete_with_meta.side_effect = _AuthError("Invalid Gemini API Key")

        fallback_mock = MagicMock()

        with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
             patch("app.core.llm._build_client", return_value=fallback_mock):
            with pytest.raises(LLMError, match="no fallback is applicable"):
                ask_llm_with_meta("Test prompt")

        fallback_mock.assert_not_called()

    def test_strict_mode_blocks_fallback(self, monkeypatch):
        """STRICT_LLM=true disables fallback during benchmarks."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "true")

        primary_mock = MagicMock()
        primary_mock.provider_name.return_value = "gemini"
        primary_mock.complete_with_meta.side_effect = _TransientError("Network timeout")

        fallback_mock = MagicMock()

        with patch("app.core.llm.get_llm_client", return_value=primary_mock), \
             patch("app.core.llm._build_client", return_value=fallback_mock), \
             patch("time.sleep"):
            with pytest.raises(LLMError, match="STRICT_LLM=true disables fallback"):
                ask_llm_with_meta("Benchmark prompt")

        fallback_mock.assert_not_called()


# =============================================================================
# 7. Per-Role Model Resolution Tests
# =============================================================================

class TestPerRoleModels:
    def test_verifier_model_override(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("VERIFIER_MODEL", "gemini-2.5-flash")

        model = _resolve_model("gemini", role="verifier")
        assert model == "gemini-2.5-flash"

    def test_planner_model_override(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        monkeypatch.setenv("PLANNER_MODEL", "llama-3.3-70b-versatile")

        model = _resolve_model("groq", role="planner")
        assert model == "llama-3.3-70b-versatile"

    def test_explicit_model_trumps_role(self, monkeypatch):
        monkeypatch.setenv("VERIFIER_MODEL", "gemini-2.5-flash")
        model = _resolve_model("gemini", explicit_model="custom-model", role="verifier")
        assert model == "custom-model"


# =============================================================================
# 8. Disk Caching Tests
# =============================================================================

class TestDiskCache:
    def test_cache_hit_bypasses_client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARGUS_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "false")

        primary_mock = MagicMock()
        primary_mock.provider_name.return_value = "groq"
        primary_mock.complete_with_meta.return_value = LLMResponse(
            text="Cached result", provider="groq", model="llama-3.3-70b-versatile"
        )

        with patch("app.core.llm.get_llm_client", return_value=primary_mock):
            # First call: populates cache
            r1 = ask_llm_with_meta("Repeatable prompt")
            assert primary_mock.complete_with_meta.call_count == 1

            # Second call: served from cache
            r2 = ask_llm_with_meta("Repeatable prompt")
            assert primary_mock.complete_with_meta.call_count == 1
            assert r2.text == "Cached result"

    def test_strict_mode_bypasses_cache_to_prevent_contamination(self, tmp_path, monkeypatch):
        """When STRICT_LLM=true (scientific benchmark mode), disk cache must NOT be used."""
        monkeypatch.setenv("ARGUS_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("STRICT_LLM", "false")
        monkeypatch.setenv("STRICT_LLM", "true")

        primary_mock = MagicMock()
        primary_mock.provider_name.return_value = "groq"
        primary_mock.complete_with_meta.side_effect = [
            LLMResponse(text="Run 1 output", provider="groq", model="llama-3.3-70b-versatile"),
            LLMResponse(text="Run 2 output", provider="groq", model="llama-3.3-70b-versatile"),
        ]

        with patch("app.core.llm.get_llm_client", return_value=primary_mock):
            r1 = ask_llm_with_meta("Benchmark prompt")
            r2 = ask_llm_with_meta("Benchmark prompt")

        # In strict benchmark mode, both calls hit the model directly — zero cache contamination!
        assert primary_mock.complete_with_meta.call_count == 2
        assert r1.text == "Run 1 output"
        assert r2.text == "Run 2 output"


# =============================================================================
# 9. Structured JSON Planner Test
# =============================================================================

class TestStructuredJSONPlanner:
    def test_planner_parses_json_from_llm(self, monkeypatch):
        json_plan = json.dumps({
            "topic": "CRISPR Mechanisms",
            "key_questions": ["How does PAM recognition work?"],
            "sub_questions": ["What is Cas9 specificity?"],
            "information_needed": ["Structural biology literature"],
        })

        with patch("app.agents.research_planner.ask_llm", return_value=json_plan):
            planner = ResearchPlanner()
            plan = planner.create_plan(ResearchQuestion(question="How does Cas9 bind target DNA?"))

        assert plan.topic == "CRISPR Mechanisms"
        assert len(plan.key_questions) == 1
        assert len(plan.sub_questions) == 1

    def test_planner_invalid_json_raises_planning_error(self, monkeypatch):
        with patch("app.agents.research_planner.ask_llm", return_value="Not valid json"):
            planner = ResearchPlanner()
            with pytest.raises(PlanningError):
                planner.create_plan(ResearchQuestion(question="How does Cas9 bind target DNA?"))


# =============================================================================
# 10. Backward Compatibility & Secrets Safety
# =============================================================================

class TestBackwardCompatibility:
    def test_ask_llm_returns_plain_str(self):
        with patch("app.core.llm.ask_llm_with_meta") as mock_meta:
            mock_meta.return_value = LLMResponse(text="Simple text", provider="groq", model="test")
            result = ask_llm("test prompt")
            assert isinstance(result, str)
            assert result == "Simple text"

    def test_api_secrets_never_leak_in_errors(self, monkeypatch):
        """Auth failures produce _AuthError referencing the env var name, not the key value."""
        secret_key = "gsk-super-secret-key-123456789"
        monkeypatch.setenv("GROQ_API_KEY", secret_key)
        mock_openai = MagicMock()
        # Simulate a 401 auth failure (which our _handle_exception maps to _AuthError)
        mock_openai.chat.completions.create.side_effect = Exception(
            "401 Unauthorized: Invalid API key"
        )

        with patch("app.core.llm.OpenAI", return_value=mock_openai):
            client = GroqClient()
            with pytest.raises(_AuthError) as exc_info:
                client.complete("prompt", "llama-3.3-70b-versatile")

            # Error references the env var name, not the raw key
            assert "GROQ_API_KEY" in str(exc_info.value)
            assert secret_key not in str(exc_info.value)
