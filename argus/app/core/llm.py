"""LLM abstraction layer for ARGUS.

Provides a unified ``ask_llm()`` / ``ask_llm_with_meta()`` interface that routes
to the configured backend. Switch providers by setting ``LLM_PROVIDER`` in .env —
agents never change.

Supported Providers:
    - Groq (groq.com)         ← Fast inference, generous free tier (default cloud alternative)
    - Google Gemini            ← Google Gemini API via official OpenAI-compatible endpoint
    - Ollama                   ← Local offline inference (default offline fallback)
    - OpenRouter               ← Optional legacy cloud aggregator
    - xAI Grok                 ← Optional legacy cloud endpoint

Architecture:
    ask_llm(prompt, model=None, role=None)
        └── ask_llm_with_meta()
                ├── Optional Disk Cache (LLM_CACHE=true / ARGUS_CACHE_DIR)
                ├── _execute_with_retry()  ← Retries transient errors (429/503/timeout) with backoff
                └── _call_with_fallback()  ← Falls back to LLM_FALLBACK_PROVIDER only after retries exhaust
                        ├── OpenAICompatibleClient
                        │       ├── GroqClient
                        │       ├── GeminiClient
                        │       ├── OpenRouterClient
                        │       └── GrokClient (legacy xAI)
                        └── OllamaClient (native REST / options control)

Fallback Policy:
    1. Primary provider is attempted with exponential backoff (honoring Retry-After).
    2. If primary encounters transient errors (429 rate limit, 503, network timeout)
       and retries are exhausted, fallback provider is attempted.
    3. Authentication errors (401, invalid API keys) FAIL FAST — never trigger fallback.
    4. STRICT_LLM=true disables fallback — mandatory for scientific benchmark runs.

Provenance Tracking:
    LLMResponse records: text, provider, model, model_version, prompt_tokens,
    completion_tokens, total_tokens, temperature, latency_s, fallback_used,
    fallback_from, and retries_attempted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider Enum
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GROK = "grok"  # legacy xAI alias


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Raised when all configured LLM providers fail."""


class _AuthError(RuntimeError):
    """Internal: invalid API key or permission denied — must NOT trigger fallback."""


class _TransientError(RuntimeError):
    """Internal: network/timeout/rate-limit error — may trigger retry/fallback."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMConnectionError(ConnectionError, _TransientError):
    """Connection error to an LLM provider."""


class LLMTimeoutError(TimeoutError, _TransientError):
    """Timeout error from an LLM provider."""


# ---------------------------------------------------------------------------
# Provenance & Response Dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Rich response containing text and full provenance metadata."""

    text: str
    provider: str
    model: str
    fallback_used: bool = False
    fallback_from: Optional[str] = None
    latency_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_version: Optional[str] = None
    temperature: Optional[float] = None
    retries_attempted: int = 0


# ---------------------------------------------------------------------------
# Abstract Base Client
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def complete(self, prompt: str, model: str) -> str:
        """Send *prompt* to the LLM and return the text response."""
        ...

    def complete_with_meta(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Return full LLMResponse with text and available metadata."""
        t0 = time.monotonic()
        text = self.complete(prompt, model)
        latency = time.monotonic() - t0
        return LLMResponse(
            text=text,
            provider=self.provider_name(),
            model=model,
            latency_s=latency,
            temperature=temperature,
        )

    def provider_name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# Generic OpenAI-Compatible Client
# Powers Groq, Gemini, OpenRouter, and xAI
# ---------------------------------------------------------------------------

class OpenAICompatibleClient(BaseLLMClient):
    """Generic client for any service exposing an OpenAI-compatible REST API.

    Unifies Groq, Gemini, OpenRouter, and xAI Grok under a single codebase.
    """

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str,
        auth_env_hint: str = "",
        timeout: float = 120.0,
    ) -> None:
        self._provider = provider_name
        self.base_url = base_url.rstrip("/")
        self.auth_env_hint = auth_env_hint
        self.timeout = timeout

        if not api_key:
            raise _AuthError(
                f"{auth_env_hint} is not set. Add it to your .env file to use the {provider_name} backend."
            )
        if OpenAI is None:
            raise ImportError(
                "The `openai` package is not installed. Run: pip install openai"
            )
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def provider_name(self) -> str:
        return self._provider

    def list_available_models(self) -> list[str]:
        """Query provider's live /models endpoint to verify supported model IDs."""
        try:
            page = self._client.models.list()
            return sorted(m.id for m in getattr(page, "data", []))
        except Exception as exc:
            logger.debug("[%s] Failed to fetch live models: %s", self._provider, exc)
            return []

    def complete(self, prompt: str, model: str) -> str:
        return self.complete_with_meta(prompt, model).text

    def complete_with_meta(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            latency = time.monotonic() - t0
            choice = response.choices[0]
            text = choice.message.content or ""

            p_tokens: Optional[int] = None
            c_tokens: Optional[int] = None
            t_tokens: Optional[int] = None
            if getattr(response, "usage", None):
                p_tokens = getattr(response.usage, "prompt_tokens", None)
                c_tokens = getattr(response.usage, "completion_tokens", None)
                t_tokens = getattr(response.usage, "total_tokens", None)

            m_version = getattr(response, "model", model)

            return LLMResponse(
                text=text,
                provider=self._provider,
                model=model,
                model_version=m_version,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                temperature=temperature,
                latency_s=latency,
            )
        except Exception as exc:
            self._handle_exception(exc)

    def _handle_exception(self, exc: Exception) -> None:
        """Categorize OpenAI-compatible errors into auth, transient, or fatal."""
        msg = str(exc)
        lowered = msg.lower()

        # Check for authentication or permission issues (fail fast)
        if any(kw in lowered for kw in (
            "unauthorized", "invalid api key", "authentication", "permission denied",
            "401", "403", "forbidden"
        )):
            raise _AuthError(
                f"{self._provider.upper()} authentication failed. Check {self.auth_env_hint}. Detail: {msg}"
            ) from exc

        # Check for rate limits (429) and extract Retry-After if available
        if "429" in lowered or "rate limit" in lowered or "quota" in lowered:
            retry_delay = _extract_retry_delay(exc)
            raise _TransientError(
                f"{self._provider.upper()} rate limit exceeded: {msg}",
                retry_after=retry_delay,
            ) from exc

        # Check for transient operational errors (500, 502, 503, 504, timeout, connection)
        if any(kw in lowered for kw in (
            "timeout", "timed out", "connection", "connect", "500", "502", "503", "504", "unavailable"
        )):
            raise _TransientError(
                f"{self._provider.upper()} transient service error: {msg}"
            ) from exc

        # Unrecognized error
        raise RuntimeError(f"{self._provider.upper()} request failed: {msg}") from exc


# ---------------------------------------------------------------------------
# Specialized Provider Clients
# ---------------------------------------------------------------------------

class GroqClient(OpenAICompatibleClient):
    """Calls Groq Cloud API via its OpenAI-compatible endpoint.

    Free tier available with no credit card required at https://console.groq.com/
    Fastest inference speed with high rate limits for open-weight models.
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self) -> None:
        key = os.getenv("GROQ_API_KEY", "").strip()
        base_url = os.getenv("GROQ_BASE_URL", self.DEFAULT_BASE_URL).strip()
        super().__init__(
            provider_name=LLMProvider.GROQ,
            api_key=key,
            base_url=base_url,
            auth_env_hint="GROQ_API_KEY",
        )


class GeminiClient(OpenAICompatibleClient):
    """Calls Google Gemini API via its official OpenAI-compatible endpoint.

    Official endpoint: https://generativelanguage.googleapis.com/v1beta/openai/
    Accepts GEMINI_API_KEY or GOOGLE_API_KEY from https://aistudio.google.com/apikey
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self) -> None:
        key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        base_url = os.getenv("GEMINI_BASE_URL", self.DEFAULT_BASE_URL).strip()
        super().__init__(
            provider_name=LLMProvider.GEMINI,
            api_key=key,
            base_url=base_url,
            auth_env_hint="GEMINI_API_KEY (or GOOGLE_API_KEY)",
        )


class OpenRouterClient(OpenAICompatibleClient):
    """Calls OpenRouter API via its OpenAI-compatible endpoint.

    Retained for backward compatibility during migration.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self) -> None:
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", self.DEFAULT_BASE_URL).strip()
        super().__init__(
            provider_name=LLMProvider.OPENROUTER,
            api_key=key,
            base_url=base_url,
            auth_env_hint="OPENROUTER_API_KEY",
        )


class GrokClient(OpenAICompatibleClient):
    """Calls xAI Grok API via its OpenAI-compatible endpoint (console.x.ai)."""

    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(self) -> None:
        key = os.getenv("XAI_API_KEY", "").strip()
        base_url = os.getenv("XAI_BASE_URL", self.DEFAULT_BASE_URL).strip()
        super().__init__(
            provider_name=LLMProvider.GROK,
            api_key=key,
            base_url=base_url,
            auth_env_hint="XAI_API_KEY",
        )


class OllamaClient(BaseLLMClient):
    """Calls a locally-running Ollama server via its native REST API."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def provider_name(self) -> str:
        return LLMProvider.OLLAMA

    def complete(self, prompt: str, model: str) -> str:
        return self.complete_with_meta(prompt, model).text

    def complete_with_meta(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        t0 = time.monotonic()
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": temperature},
                },
                timeout=httpx.Timeout(240.0, connect=15.0),
            )
            response.raise_for_status()
            latency = time.monotonic() - t0
            data = response.json()
            text = data.get("response", "")

            p_tokens = data.get("prompt_eval_count")
            c_tokens = data.get("eval_count")
            t_tokens = (p_tokens + c_tokens) if (p_tokens is not None and c_tokens is not None) else None

            return LLMResponse(
                text=text,
                provider=LLMProvider.OLLAMA,
                model=model,
                model_version=model,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                temperature=temperature,
                latency_s=latency,
            )
        except httpx.ConnectError:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`) and the model is pulled."
            ) from None
        except httpx.TimeoutException:
            raise LLMTimeoutError(
                f"Ollama request timed out after 240s for model '{model}' at {self.base_url}. "
                "Local CPU/GPU load may be high."
            ) from None
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc


# ---------------------------------------------------------------------------
# Model Resolution & Defaults
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OLLAMA: "llama3.2:3b",
    LLMProvider.OPENROUTER: "mistralai/mistral-7b-instruct",
    LLMProvider.GROK: "grok-3-mini",
}

_PROVIDER_MODEL_ENVVARS: dict[str, str] = {
    LLMProvider.GROQ: "GROQ_MODEL",
    LLMProvider.GEMINI: "GEMINI_MODEL",
    LLMProvider.OLLAMA: "OLLAMA_MODEL",
    LLMProvider.OPENROUTER: "OPENROUTER_MODEL",
    LLMProvider.GROK: "XAI_MODEL",
}

_ROLE_ENVVARS: dict[str, str] = {
    "verifier": "VERIFIER_MODEL",
    "researcher": "RESEARCHER_MODEL",
    "planner": "PLANNER_MODEL",
    "writer": "WRITER_MODEL",
}


def _resolve_model(
    provider: str,
    explicit_model: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    """Resolve the effective model ID for *provider*.

    Resolution Order:
        1. ``explicit_model`` argument passed at call time.
        2. Per-role env var (e.g. VERIFIER_MODEL, RESEARCHER_MODEL, PLANNER_MODEL).
        3. Provider-specific env var (e.g. GEMINI_MODEL, GROQ_MODEL, OLLAMA_MODEL).
        4. Legacy ``DEFAULT_LLM_MODEL`` (backward compatibility).
        5. Hard-coded provider default.
    """
    if explicit_model:
        return explicit_model

    if role:
        role_env = _ROLE_ENVVARS.get(role.lower().strip())
        if role_env:
            val = os.getenv(role_env, "").strip()
            if val:
                return val

    provider_env = _PROVIDER_MODEL_ENVVARS.get(provider.lower().strip(), "")
    if provider_env:
        val = os.getenv(provider_env, "").strip()
        if val:
            return val

    legacy = os.getenv("DEFAULT_LLM_MODEL", "").strip()
    if legacy:
        return legacy

    return _PROVIDER_DEFAULT_MODELS.get(provider.lower().strip(), "llama3.2:3b")


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

def _build_client(provider: str) -> BaseLLMClient:
    """Instantiate a client for the requested provider."""
    p = provider.lower().strip()
    if p == LLMProvider.GROQ:
        return GroqClient()
    if p == LLMProvider.GEMINI:
        return GeminiClient()
    if p == LLMProvider.OLLAMA:
        return OllamaClient(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    if p == LLMProvider.OPENROUTER:
        return OpenRouterClient()
    if p == LLMProvider.GROK:
        return GrokClient()

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        f"Valid values: {', '.join(p.value for p in LLMProvider)}"
    )


def get_llm_client() -> BaseLLMClient:
    """Return the configured primary LLM backend."""
    provider = os.getenv("LLM_PROVIDER", LLMProvider.OLLAMA).lower().strip()
    return _build_client(provider)


# ---------------------------------------------------------------------------
# Disk Caching & Isolation Safeguards
# ---------------------------------------------------------------------------

def _get_cache_key(
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    role: Optional[str] = None,
) -> str:
    """Generate SHA-256 cache key including provider, model, role, temperature, and prompt."""
    key_str = f"{provider}:{model}:{role or 'none'}:{temperature:.2f}:{prompt}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def _read_cache(cache_key: str, strict: bool = False) -> Optional[LLMResponse]:
    """Read cached response if caching is active and benchmark isolation is preserved."""
    # Disallow cache reads in strict benchmark mode to prevent cross-experiment contamination
    if strict and os.getenv("FORCE_BENCHMARK_CACHE", "false").lower() not in ("1", "true", "yes"):
        return None

    cache_dir = os.getenv("ARGUS_CACHE_DIR", "")
    is_enabled = bool(cache_dir) or os.getenv("LLM_CACHE", "false").lower() in ("1", "true", "yes")
    if not is_enabled:
        return None

    path = Path(cache_dir or "./data/cache/llm") / f"{cache_key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["latency_s"] = 0.0
            return LLMResponse(**data)
        except Exception:
            return None
    return None


def _write_cache(cache_key: str, resp: LLMResponse, strict: bool = False) -> None:
    """Persist response to disk cache if allowed."""
    if strict and os.getenv("FORCE_BENCHMARK_CACHE", "false").lower() not in ("1", "true", "yes"):
        return

    cache_dir = os.getenv("ARGUS_CACHE_DIR", "")
    is_enabled = bool(cache_dir) or os.getenv("LLM_CACHE", "false").lower() in ("1", "true", "yes")
    if not is_enabled:
        return

    try:
        target_dir = Path(cache_dir or "./data/cache/llm")
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{cache_key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(resp), f)
    except Exception as exc:
        logger.debug("Failed to write to LLM cache: %s", exc)


# ---------------------------------------------------------------------------
# Retry with Backoff & Fallback Policy
# ---------------------------------------------------------------------------

_TRANSIENT_TYPES = (_TransientError, ConnectionError, TimeoutError)
_AUTH_TYPES = (_AuthError,)


def _is_transient(exc: BaseException) -> bool:
    """Return True if *exc* represents a recoverable transient failure."""
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "connection", "429", "rate limit", "503", "unavailable"))


def _is_auth_failure(exc: BaseException) -> bool:
    """Return True if *exc* indicates an authentication or authorization failure."""
    if isinstance(exc, _AUTH_TYPES):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("unauthorized", "invalid api key", "authentication", "401", "403"))


def _is_quota_exhausted(exc: BaseException) -> bool:
    """Return True if the error indicates hard quota exhaustion (unrecoverable by retries)."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "insufficient_quota",
            "exceeded your current quota",
            "quota exceeded",
            "quota_exceeded",
            "out of credits",
            "billing",
            "credit balance",
        )
    )


def _extract_retry_delay(exc: BaseException) -> Optional[float]:
    """Extract retry delay seconds from exception headers or message."""
    delay = getattr(exc, "retry_after", None)
    if isinstance(delay, (int, float)) and delay > 0:
        return float(delay)

    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {})
        ra = headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass

    msg = str(exc)
    match = re.search(
        r"(?:try again in|retry after|retry-after|wait)\s*([0-9]+(?:\.[0-9]+)?)\s*s?",
        msg,
        re.IGNORECASE,
    )
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def _execute_with_retry(
    client: BaseLLMClient,
    prompt: str,
    model: str,
    temperature: float = 0.1,
    max_retries: int = 2,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
) -> LLMResponse:
    """Execute LLM call, retrying transient errors (429/503/timeout) with backoff.

    Honors Retry-After headers if provided by the provider.
    Fails fast immediately on authentication errors or exhausted quotas.
    """
    last_exc: Optional[BaseException] = None
    retries = 0

    for attempt in range(max_retries + 1):
        try:
            if hasattr(client, "complete_with_meta"):
                res = client.complete_with_meta(prompt, model, temperature=temperature)
                if isinstance(res, LLMResponse):
                    resp = res
                else:
                    text = client.complete(prompt, model)
                    resp = LLMResponse(
                        text=str(text) if not isinstance(text, str) else text,
                        provider=client.provider_name() if hasattr(client, "provider_name") and callable(client.provider_name) else "client",
                        model=model,
                        temperature=temperature,
                    )
            else:
                text = client.complete(prompt, model)
                resp = LLMResponse(
                    text=str(text),
                    provider=client.provider_name() if hasattr(client, "provider_name") and callable(client.provider_name) else "client",
                    model=model,
                    temperature=temperature,
                )
            resp.retries_attempted = retries
            return resp
        except _AuthError:
            # Auth errors must fail immediately — no retry, no fallback
            raise
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc):
                # Non-transient errors (syntax error, model not found) fail fast
                raise

            # Differentiate transient throttling from permanent quota exhaustion
            if _is_quota_exhausted(exc):
                logger.warning(
                    "[%s] Quota exhausted for model '%s'. Skipping sleep retries.",
                    client.provider_name(), model,
                )
                raise exc

            if attempt < max_retries:
                retries += 1
                delay = _extract_retry_delay(exc)
                if delay is None or delay <= 0:
                    delay = initial_backoff * (backoff_factor ** (attempt))
                delay = min(delay, max_delay)
                logger.warning(
                    "[%s] Transient rate limit/error on attempt %d/%d (%s). Retrying in %.2fs...",
                    client.provider_name(), attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] All %d retries exhausted for model '%s'. Last error: %s",
                    client.provider_name(), max_retries + 1, model, exc,
                )

    assert last_exc is not None
    raise last_exc


def _call_with_fallback(
    prompt: str,
    primary_provider: str,
    fallback_provider: Optional[str],
    model: Optional[str],
    role: Optional[str],
    strict: bool,
    temperature: float,
) -> LLMResponse:
    """Execute the LLM request through the shared, quota-aware LLMPool."""
    from app.core.pool import get_llm_pool
    return get_llm_pool().execute(
        prompt=prompt,
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        model=model,
        role=role,
        strict=strict,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Public Agent Interface
# ---------------------------------------------------------------------------

def ask_llm(
    prompt: str,
    model: Optional[str] = None,
    role: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    """Send *prompt* to the active LLM backend and return the response text.

    Backward-compatible, provider-agnostic interface used by all agents.
    """
    return ask_llm_with_meta(prompt, model=model, role=role, temperature=temperature).text


def ask_llm_with_meta(
    prompt: str,
    model: Optional[str] = None,
    role: Optional[str] = None,
    temperature: float = 0.1,
) -> LLMResponse:
    """Send *prompt* to the active LLM backend and return rich LLMResponse metadata.

    Records provider, model, tokens, latency, and fallback provenance.
    """
    primary_provider = os.getenv("LLM_PROVIDER", LLMProvider.OLLAMA).lower().strip()
    fallback_provider_raw = os.getenv("LLM_FALLBACK_PROVIDER", "").lower().strip()
    fallback_provider: Optional[str] = fallback_provider_raw or None
    strict = os.getenv("STRICT_LLM", "false").lower().strip() in ("1", "true", "yes")

    return _call_with_fallback(
        prompt=prompt,
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        model=model,
        role=role,
        strict=strict,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Quick CLI Diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model_id = _resolve_model(provider)
    print(f"[ARGUS] LLM provider: {provider} | model: {model_id}")
    try:
        res = ask_llm_with_meta("In one sentence: what is retrieval-augmented generation?")
        print(f"[ARGUS] Response ({res.provider}/{res.model}): {res.text}")
        if res.fallback_used:
            print(f"[ARGUS] Fallback was used — primary was: {res.fallback_from}")
    except LLMError as err:
        print(f"[ARGUS] LLM unavailable: {err}", file=sys.stderr)
        sys.exit(1)