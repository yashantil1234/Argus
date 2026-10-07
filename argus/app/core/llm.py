"""LLM abstraction layer for ARGUS.

Provides a single `ask_llm()` interface that routes to the configured backend.
Switch providers by setting LLM_PROVIDER in .env — agents never change.

Architecture:
    ask_llm(prompt)
        └── get_llm_client()          ← reads LLM_PROVIDER from env
                ├── OllamaClient      ← local (default)
                └── OpenRouterClient  ← cloud
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


# ---------------------------------------------------------------------------
# Abstract base — the contract every backend must honour
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def complete(self, prompt: str, model: str) -> str:
        """Send *prompt* to the LLM and return the text response."""
        ...


# ---------------------------------------------------------------------------
# Ollama backend (local)
# ---------------------------------------------------------------------------

class OllamaClient(BaseLLMClient):
    """Calls a locally-running Ollama server via its REST API."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, model: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 350, "temperature": 0.1},
                },
                timeout=httpx.Timeout(240.0, connect=15.0),
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`) and the model is pulled."
            ) from None
        except httpx.TimeoutException:
            raise TimeoutError(
                f"Ollama request timed out after 240s for model {model} at {self.base_url}. "
                "Local CPU load may be high."
            ) from None
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc


# ---------------------------------------------------------------------------
# OpenRouter backend (cloud)
# ---------------------------------------------------------------------------

class OpenRouterClient(BaseLLMClient):
    """Calls OpenRouter via the OpenAI-compatible API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file to use the OpenRouter backend."
            )
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def complete(self, prompt: str, model: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory — reads LLM_PROVIDER from environment
# ---------------------------------------------------------------------------

def get_llm_client() -> BaseLLMClient:
    """Return the configured LLM backend.

    Controlled by the ``LLM_PROVIDER`` environment variable:
    - ``ollama``      → OllamaClient (default)
    - ``openrouter``  → OpenRouterClient
    """
    provider = os.getenv("LLM_PROVIDER", LLMProvider.OLLAMA).lower().strip()
    if provider == LLMProvider.OPENROUTER:
        return OpenRouterClient()
    return OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )


# ---------------------------------------------------------------------------
# Public interface — the only function agents should call
# ---------------------------------------------------------------------------

def ask_llm(prompt: str, model: str | None = None) -> str:
    """Send *prompt* to the active LLM backend and return the response text.

    The backend is selected by ``LLM_PROVIDER`` in .env.
    The model defaults to ``DEFAULT_LLM_MODEL`` if not explicitly provided.
    """
    client = get_llm_client()
    resolved_model = model or os.getenv("DEFAULT_LLM_MODEL", "llama3.2")
    return client.complete(prompt, resolved_model)


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("DEFAULT_LLM_MODEL", "llama3.2")
    print(f"[ARGUS] LLM provider: {provider} | model: {model}")
    try:
        answer = ask_llm("In one sentence: what is retrieval-augmented generation?")
        print(f"[ARGUS] Response: {answer}")
    except (ConnectionError, EnvironmentError, RuntimeError) as err:
        print(f"[ARGUS] LLM unavailable: {err}")