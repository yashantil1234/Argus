"""Shared, Quota-Aware LLM Client Pool for ARGUS.

Coordinates concurrent LLM requests across agents, enforces per-provider
concurrency limits, bounds request rates (RPM) via a rolling window, and provides
a circuit breaker with cooldown to prevent hammering exhausted providers.

Features:
  1. Concurrency Control: Bounded semaphores per provider (e.g. Ollama=1, Gemini=5, Groq=8).
  2. Rate Limiting: Sliding-window RPM tracker pacing calls before triggering API 429s.
  3. Circuit Breaker & Cooldown:
     - Tripped on 429 rate limit, hard quota exhaustion, or 503 service unavailable.
     - Respects `Retry-After` headers or configurable default cooldown (e.g. 30s).
     - When tripped, subsequent calls in normal mode immediately bypass to fallback
       without waiting or sleeping on dead endpoints.
     - In strict benchmark mode (STRICT_LLM=true), circuit breaker fails fast with LLMError
       to maintain experimental purity (zero fallback leakage).
  4. Observability: Thread-safe metrics on active concurrency, cooldowns, and fallbacks.
  5. Zero Agent Disruption: Directly powers `ask_llm()` and `ask_llm_with_meta()`.
"""

from __future__ import annotations

import collections
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider Quota Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProviderQuotaConfig:
    """Configured rate and concurrency limits for a specific provider."""

    provider: str
    max_concurrency: int
    rpm: int
    default_cooldown: float = 30.0


# Default limits based on provider tiers
_DEFAULT_LIMITS: Dict[str, Tuple[int, int, float]] = {
    # provider: (max_concurrency, rpm, default_cooldown)
    "ollama": (1, 120, 10.0),       # Local inference: 1 at a time to prevent CPU/GPU saturation
    "gemini": (5, 15, 60.0),        # Free tier: 15 RPM
    "groq": (8, 30, 30.0),          # Groq free tier: 30 RPM
    "openrouter": (4, 60, 30.0),
    "grok": (4, 30, 30.0),
}


def _get_provider_config(provider: str) -> ProviderQuotaConfig:
    """Load provider quota settings from environment variables or safe defaults."""
    p = provider.lower().strip()
    def_concurrency, def_rpm, def_cooldown = _DEFAULT_LIMITS.get(p, (4, 30, 30.0))

    concurrency_env = os.getenv(f"LLM_MAX_CONCURRENCY_{p.upper()}", "")
    concurrency = int(concurrency_env) if concurrency_env.isdigit() and int(concurrency_env) > 0 else def_concurrency

    rpm_env = os.getenv(f"LLM_RPM_{p.upper()}", "")
    rpm = int(rpm_env) if rpm_env.isdigit() and int(rpm_env) > 0 else def_rpm

    cooldown_env = os.getenv("LLM_COOLDOWN_SECONDS", "")
    cooldown = float(cooldown_env) if cooldown_env.replace(".", "", 1).isdigit() else def_cooldown

    return ProviderQuotaConfig(
        provider=p,
        max_concurrency=concurrency,
        rpm=rpm,
        default_cooldown=cooldown,
    )


# ---------------------------------------------------------------------------
# Shared LLM Pool
# ---------------------------------------------------------------------------

class LLMPool:
    """Thread-safe, quota-aware LLM client pool managing concurrency and circuit breaking."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._semaphores: Dict[str, threading.BoundedSemaphore] = {}
        self._cooldown_until: Dict[str, float] = {}
        self._cooldown_reason: Dict[str, str] = {}
        self._request_timestamps: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

        # Metrics
        self._total_requests: int = 0
        self._successful_requests: int = 0
        self._circuit_trips: int = 0
        self._fallback_requests: int = 0
        self._active_concurrency: Dict[str, int] = collections.defaultdict(int)

    def _get_semaphore(self, provider: str) -> threading.BoundedSemaphore:
        """Retrieve or initialize bounded semaphore for a provider."""
        p = provider.lower().strip()
        with self._lock:
            if p not in self._semaphores:
                cfg = _get_provider_config(p)
                self._semaphores[p] = threading.BoundedSemaphore(cfg.max_concurrency)
            return self._semaphores[p]

    # -----------------------------------------------------------------------
    # Circuit Breaker & Cooldown Management
    # -----------------------------------------------------------------------

    def is_in_cooldown(self, provider: str) -> Tuple[bool, float, str]:
        """Check if provider is currently in circuit-breaker cooldown.

        Returns:
            (in_cooldown, remaining_seconds, reason)
        """
        p = provider.lower().strip()
        with self._lock:
            until = self._cooldown_until.get(p, 0.0)
            now = time.monotonic()
            if now < until:
                remaining = until - now
                reason = self._cooldown_reason.get(p, "Rate limit / transient error cooldown")
                return True, remaining, reason
            elif until > 0.0:
                # Cooldown expired, half-open circuit
                del self._cooldown_until[p]
                self._cooldown_reason.pop(p, None)
                logger.info("[LLMPool] Cooldown expired for provider '%s'. Circuit half-opened.", p)
            return False, 0.0, ""

    def trip_cooldown(self, provider: str, duration: Optional[float] = None, reason: str = "") -> None:
        """Put provider into cooldown (open circuit) for specified or default duration."""
        p = provider.lower().strip()
        cfg = _get_provider_config(p)
        cooldown_secs = duration if (duration is not None and duration > 0) else cfg.default_cooldown

        with self._lock:
            self._cooldown_until[p] = time.monotonic() + cooldown_secs
            self._cooldown_reason[p] = reason or f"Rate limit / transient error on {p}"
            self._circuit_trips += 1
            logger.warning(
                "[LLMPool] Circuit breaker TRIPPED for provider '%s' (cooldown: %.1fs). Reason: %s",
                p, cooldown_secs, self._cooldown_reason[p],
            )

    def clear_cooldown(self, provider: str) -> None:
        """Clear cooldown for a provider upon successful request."""
        p = provider.lower().strip()
        with self._lock:
            if p in self._cooldown_until:
                del self._cooldown_until[p]
                self._cooldown_reason.pop(p, None)
                logger.debug("[LLMPool] Provider '%s' cooldown cleared.", p)

    # -----------------------------------------------------------------------
    # Rate Limiting & Concurrency Slot Acquisition
    # -----------------------------------------------------------------------

    def _acquire_slot(self, provider: str, timeout: float = 60.0) -> bool:
        """Acquire both concurrency semaphore and rate-limit window clearance."""
        p = provider.lower().strip()
        cfg = _get_provider_config(p)
        sem = self._get_semaphore(p)

        # 1. Acquire concurrency semaphore
        acquired = sem.acquire(timeout=timeout)
        if not acquired:
            logger.error("[LLMPool] Concurrency timeout (%.1fs) waiting for provider '%s' slot.", timeout, p)
            return False

        # 2. Rate pacing check (sliding 60-second window)
        now = time.monotonic()
        sleep_needed = 0.0
        with self._lock:
            dq = self._request_timestamps[p]
            # Prune timestamps older than 60 seconds
            while dq and now - dq[0] > 60.0:
                dq.popleft()

            if len(dq) >= cfg.rpm:
                oldest = dq[0]
                sleep_needed = max(0.0, (oldest + 60.0) - now + 0.05)

        if sleep_needed > 0:
            if sleep_needed > timeout:
                sem.release()
                logger.warning(
                    "[LLMPool] Provider '%s' RPM limit reached (%d/%d). Required wait %.2fs > timeout.",
                    p, len(dq), cfg.rpm, sleep_needed,
                )
                return False
            logger.info(
                "[LLMPool] Provider '%s' approaching RPM limit (%d/%d). Pacing request by sleeping %.2fs...",
                p, len(dq), cfg.rpm, sleep_needed,
            )
            time.sleep(sleep_needed)

        with self._lock:
            self._request_timestamps[p].append(time.monotonic())
            self._active_concurrency[p] += 1

        return True

    def _release_slot(self, provider: str) -> None:
        """Release concurrency semaphore and decrement active counter."""
        p = provider.lower().strip()
        with self._lock:
            if self._active_concurrency[p] > 0:
                self._active_concurrency[p] -= 1
        sem = self._get_semaphore(p)
        try:
            sem.release()
        except ValueError:
            pass

    # -----------------------------------------------------------------------
    # Execution Engine
    # -----------------------------------------------------------------------

    def execute(
        self,
        prompt: str,
        primary_provider: str,
        fallback_provider: Optional[str],
        model: Optional[str],
        role: Optional[str],
        strict: bool,
        temperature: float,
    ) -> Any:
        """Execute request through pool with concurrency, circuit breaking, and fallback."""
        # Dynamic import to avoid circular dependency and respect mocks on app.core.llm
        core_llm = sys.modules.get("app.core.llm")
        if core_llm is None:
            import app.core.llm as core_llm

        resolved_primary_model = core_llm._resolve_model(primary_provider, model, role)

        with self._lock:
            self._total_requests += 1

        # 1. Check disk cache before any slot acquisition (conserves quota)
        cache_key = core_llm._get_cache_key(
            primary_provider, resolved_primary_model, temperature, prompt, role=role
        )
        cached_resp = core_llm._read_cache(cache_key, strict=strict)
        if cached_resp is not None:
            return cached_resp

        primary_exc: Optional[BaseException] = None
        primary_in_cooldown = False

        # 2. Check if primary provider circuit breaker is currently open
        in_cooldown, remaining, reason = self.is_in_cooldown(primary_provider)
        if in_cooldown:
            primary_in_cooldown = True
            primary_exc = core_llm._TransientError(
                f"Provider '{primary_provider}' circuit breaker is OPEN (cooldown remaining: {remaining:.1f}s. Reason: {reason})"
            )
            logger.warning("[LLMPool] Primary provider '%s' is in cooldown (%.1fs remaining).", primary_provider, remaining)

        # 3. If primary is NOT in cooldown, attempt primary
        if not primary_in_cooldown:
            primary_client: Optional[Any] = None
            try:
                primary_client = core_llm.get_llm_client()
                p_name_fn = getattr(primary_client, "provider_name", None)
                if callable(p_name_fn):
                    val = p_name_fn()
                    if isinstance(val, str):
                        primary_provider = val
            except core_llm._AuthError as exc:
                primary_exc = exc
                logger.error("[LLMPool] Primary provider '%s' auth failure: %s", primary_provider, exc)
            except Exception as exc:
                primary_exc = exc
                logger.error("[LLMPool] Primary provider '%s' init failed: %s", primary_provider, exc)

            if primary_client is not None:
                slot_acquired = self._acquire_slot(primary_provider)
                if not slot_acquired:
                    primary_exc = core_llm._TransientError(f"Concurrency/RPM timeout for provider '{primary_provider}'")
                else:
                    try:
                        resp = core_llm._execute_with_retry(
                            client=primary_client,
                            prompt=prompt,
                            model=resolved_primary_model,
                            temperature=temperature,
                        )
                        self.clear_cooldown(primary_provider)
                        core_llm._write_cache(cache_key, resp, strict=strict)
                        with self._lock:
                            self._successful_requests += 1
                        return resp
                    except core_llm._AuthError as exc:
                        # Auth errors fail fast: never trip cooldown, never fallback
                        primary_exc = exc
                    except Exception as exc:
                        primary_exc = exc
                        # Trip circuit breaker on rate limit, quota exhaustion, or service outage
                        if core_llm._is_transient(exc):
                            delay = core_llm._extract_retry_delay(exc)
                            self.trip_cooldown(primary_provider, duration=delay, reason=str(exc))
                    finally:
                        self._release_slot(primary_provider)

        # 4. Fallback decision
        can_fallback = (
            fallback_provider is not None
            and fallback_provider.lower().strip() != primary_provider.lower().strip()
            and not strict
            and primary_exc is not None
            and not core_llm._is_auth_failure(primary_exc)
            and core_llm._is_transient(primary_exc)
        )

        if can_fallback:
            assert fallback_provider is not None
            fb_provider = fallback_provider.lower().strip()
            fb_model = core_llm._resolve_model(fb_provider, model, role)

            # Check if fallback provider is also in cooldown
            fb_in_cd, fb_rem, fb_reason = self.is_in_cooldown(fb_provider)
            if fb_in_cd:
                raise core_llm.LLMError(
                    f"Primary provider '{primary_provider}' failed: {primary_exc}\n"
                    f"Fallback provider '{fb_provider}' is also in cooldown ({fb_rem:.1f}s remaining: {fb_reason})."
                )

            logger.info(
                "[LLMPool] Diverting to fallback provider '%s' (model: %s). Primary '%s' error: %s",
                fb_provider, fb_model, primary_provider, primary_exc,
            )

            fb_slot = self._acquire_slot(fb_provider)
            if not fb_slot:
                raise core_llm.LLMError(
                    f"Primary provider '{primary_provider}' failed: {primary_exc}\n"
                    f"Fallback provider '{fb_provider}' slot acquisition timed out."
                )

            try:
                fallback_client = core_llm._build_client(fb_provider)
                resp = core_llm._execute_with_retry(
                    client=fallback_client,
                    prompt=prompt,
                    model=fb_model,
                    temperature=temperature,
                )
                resp.fallback_used = True
                resp.fallback_from = primary_provider
                self.clear_cooldown(fb_provider)
                core_llm._write_cache(cache_key, resp, strict=strict)
                with self._lock:
                    self._successful_requests += 1
                    self._fallback_requests += 1
                return resp
            except core_llm._AuthError as fb_exc:
                raise core_llm.LLMError(
                    f"Primary provider '{primary_provider}' failed: {primary_exc}\n"
                    f"Fallback provider '{fb_provider}' authentication failed: {fb_exc}\n"
                    "Please verify API keys in your .env file."
                ) from fb_exc
            except Exception as fb_exc:
                if core_llm._is_transient(fb_exc):
                    delay = core_llm._extract_retry_delay(fb_exc)
                    self.trip_cooldown(fb_provider, duration=delay, reason=str(fb_exc))
                raise core_llm.LLMError(
                    f"Primary provider '{primary_provider}' failed: {primary_exc}\n"
                    f"Fallback provider '{fb_provider}' also failed: {fb_exc}"
                ) from fb_exc
            finally:
                self._release_slot(fb_provider)

        # 5. Fallback forbidden or unavailable
        if strict and fallback_provider:
            raise core_llm.LLMError(
                f"Provider '{primary_provider}' failed and STRICT_LLM=true disables fallback.\n"
                f"Primary error: {primary_exc}"
            )

        raise core_llm.LLMError(
            f"Provider '{primary_provider}' failed and no fallback is applicable.\n"
            f"Error: {primary_exc}"
        )

    # -----------------------------------------------------------------------
    # Status & Diagnostic Reporting
    # -----------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return snapshot of pool health, active concurrency, and cooldowns."""
        with self._lock:
            now = time.monotonic()
            active_cooldowns = {
                prov: {
                    "remaining_seconds": max(0.0, round(until - now, 2)),
                    "reason": self._cooldown_reason.get(prov, ""),
                }
                for prov, until in self._cooldown_until.items()
                if until > now
            }
            return {
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "fallback_requests": self._fallback_requests,
                "circuit_trips": self._circuit_trips,
                "active_concurrency": dict(self._active_concurrency),
                "cooldowns": active_cooldowns,
            }

    def reset(self) -> None:
        """Reset all state, semaphores, and cooldowns (used in test isolation)."""
        with self._lock:
            self._semaphores.clear()
            self._cooldown_until.clear()
            self._cooldown_reason.clear()
            self._request_timestamps.clear()
            self._active_concurrency.clear()
            self._total_requests = 0
            self._successful_requests = 0
            self._circuit_trips = 0
            self._fallback_requests = 0


# ---------------------------------------------------------------------------
# Global Pool Accessors
# ---------------------------------------------------------------------------

_GLOBAL_POOL: Optional[LLMPool] = None
_GLOBAL_POOL_LOCK = threading.Lock()


def get_llm_pool() -> LLMPool:
    """Retrieve the shared singleton LLMPool instance."""
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        with _GLOBAL_POOL_LOCK:
            if _GLOBAL_POOL is None:
                _GLOBAL_POOL = LLMPool()
    return _GLOBAL_POOL


def reset_llm_pool() -> None:
    """Reset the global LLMPool state."""
    global _GLOBAL_POOL
    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.reset()
