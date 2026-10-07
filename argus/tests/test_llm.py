"""Unit tests for the ARGUS LLM abstraction layer.

All tests are fully mocked — no real network calls, no running Ollama or OpenRouter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from unittest.mock import MagicMock, patch

import pytest

from app.core.llm import (
    BaseLLMClient,
    LLMProvider,
    OllamaClient,
    OpenRouterClient,
    ask_llm,
    get_llm_client,
)


# ---------------------------------------------------------------------------
# 1. Structure tests — verify the contract is correct
# ---------------------------------------------------------------------------

def test_ollama_client_has_complete_method():
    """OllamaClient must implement BaseLLMClient.complete()."""
    assert issubclass(OllamaClient, BaseLLMClient)
    assert callable(getattr(OllamaClient, "complete", None))


def test_openrouter_client_has_complete_method():
    """OpenRouterClient must implement BaseLLMClient.complete()."""
    assert issubclass(OpenRouterClient, BaseLLMClient)
    assert callable(getattr(OpenRouterClient, "complete", None))


# ---------------------------------------------------------------------------
# 2. Factory tests — verify LLM_PROVIDER env var routes correctly
# ---------------------------------------------------------------------------

def test_factory_returns_ollama_by_default(monkeypatch):
    """get_llm_client() defaults to OllamaClient when LLM_PROVIDER is unset."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = get_llm_client()
    assert isinstance(client, OllamaClient)


def test_factory_returns_ollama_when_set(monkeypatch):
    """get_llm_client() returns OllamaClient when LLM_PROVIDER=ollama."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = get_llm_client()
    assert isinstance(client, OllamaClient)


def test_factory_returns_openrouter_when_set(monkeypatch):
    """get_llm_client() returns OpenRouterClient when LLM_PROVIDER=openrouter."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    # Patch OpenAI so OpenRouterClient.__init__ doesn't make network calls
    with patch("app.core.llm.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        client = get_llm_client()

    assert isinstance(client, OpenRouterClient)


# ---------------------------------------------------------------------------
# 3. ask_llm delegates to the active client
# ---------------------------------------------------------------------------

def test_ask_llm_delegates_to_active_client(monkeypatch):
    """ask_llm() should call .complete() on whatever get_llm_client() returns."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.2")

    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.complete.return_value = "Mocked LLM response"

    with patch("app.core.llm.get_llm_client", return_value=mock_client):
        result = ask_llm("What is RAG?")

    mock_client.complete.assert_called_once_with("What is RAG?", "llama3.2")
    assert result == "Mocked LLM response"


# ---------------------------------------------------------------------------
# 4. Ollama connection error gives a clear message
# ---------------------------------------------------------------------------

def test_ollama_connection_error_has_clear_message(monkeypatch):
    """OllamaClient.complete() raises ConnectionError with a helpful message."""
    import httpx

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = OllamaClient(base_url="http://localhost:11434")

    with patch("app.core.llm.httpx.post") as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(ConnectionError) as exc_info:
            client.complete("test prompt", "llama3.2")

    assert "Ollama" in str(exc_info.value)
    assert "ollama serve" in str(exc_info.value)
