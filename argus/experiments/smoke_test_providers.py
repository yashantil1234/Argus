"""Live smoke tests for ARGUS LLM providers.

Tests providers independently with minimal real API calls, validates credentials,
and verifies structured JSON generation via ResearchPlanner.

Usage:
    python experiments/smoke_test_providers.py [target]

Supported targets:
    groq      - Test Groq Cloud API (groq.com)
    gemini    - Test Google Gemini API (aistudio.google.com)
    ollama    - Test local Ollama daemon
    planner   - Test structured JSON planning across active provider
    fallback  - Test configured fallback & error recovery
    all       - Test all configured providers (default)

Security:
    API keys are verified for presence and length, but NEVER printed or logged.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.core.llm import (
    BaseLLMClient,
    GeminiClient,
    GroqClient,
    LLMError,
    LLMProvider,
    OllamaClient,
    _resolve_model,
    ask_llm,
    ask_llm_with_meta,
)
from app.agents.research_planner import ResearchPlanner, PlanningError
from app.models.research import ResearchQuestion

PROMPT = "Respond in one word: PONG"
PLANNER_QUESTION = "What are the molecular mechanisms of Cas9 off-target cleavage?"


def _mask_presence(val: str | None) -> str:
    """Safely report presence without leaking key bytes."""
    if not val or not val.strip():
        return "[NOT SET]"
    return f"[SET, length={len(val.strip())}]"


# ---------------------------------------------------------------------------
# Provider Smoke Tests
# ---------------------------------------------------------------------------

def smoke_test_groq() -> bool:
    print("\n" + "=" * 65)
    print("  SMOKE TEST: Groq (Cloud — Fast Inference & Free Tier)")
    print("=" * 65)
    key = os.getenv("GROQ_API_KEY", "")
    model = _resolve_model(LLMProvider.GROQ)
    print(f"  API Key  : {_mask_presence(key)}")
    print(f"  Model    : {model}")

    if not key.strip():
        print("  Status   : [SKIPPED] GROQ_API_KEY is not set.")
        print("  Hint     : Get a free API key (no credit card required) at:")
        print("             https://console.groq.com/keys")
        print("             Then add `GROQ_API_KEY=your-key` to your .env file.")
        return False

    try:
        client = GroqClient()
        available = client.list_available_models()
        if available:
            print(f"  Live Models : {len(available)} supported (e.g. {', '.join(available[:4])})")
            if model not in available:
                print(f"  Notice      : Configured model '{model}' not found in active list.")
        t0 = time.monotonic()
        res = client.complete_with_meta(PROMPT, model=model)
        latency = time.monotonic() - t0
        print(f"  Latency  : {latency:.2f}s")
        print(f"  Response : {res.text.strip()}")
        print(f"  Tokens   : prompt={res.prompt_tokens}, completion={res.completion_tokens}, total={res.total_tokens}")
        print("  Status   : [PASSED]")
        return True
    except Exception as exc:
        print(f"  Status   : [FAILED]")
        print(f"  Error    : {exc}")
        print("  Hint     : Check GROQ_API_KEY at https://console.groq.com/keys")
        return False


def smoke_test_gemini() -> bool:
    print("\n" + "=" * 65)
    print("  SMOKE TEST: Google Gemini (Cloud — OpenAI-Compatible API)")
    print("=" * 65)
    key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    model = _resolve_model(LLMProvider.GEMINI)
    print(f"  API Key  : {_mask_presence(key)}")
    print(f"  Model    : {model}")

    if not key.strip():
        print("  Status   : [SKIPPED] GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        print("  Hint     : Obtain an API key at https://aistudio.google.com/apikey")
        print("             Then add `GEMINI_API_KEY=your-key` to your .env file.")
        return False

    try:
        client = GeminiClient()
        available = client.list_available_models()
        if available:
            print(f"  Live Models : {len(available)} supported (e.g. {', '.join(available[:4])})")
            matching = [m for m in available if model in m]
            if not matching:
                print(f"  Notice      : Configured model '{model}' not found in active list.")
        t0 = time.monotonic()
        res = client.complete_with_meta(PROMPT, model=model)
        latency = time.monotonic() - t0
        print(f"  Latency  : {latency:.2f}s")
        print(f"  Response : {res.text.strip()}")
        print(f"  Tokens   : prompt={res.prompt_tokens}, completion={res.completion_tokens}, total={res.total_tokens}")
        print("  Status   : [PASSED]")
        return True
    except Exception as exc:
        print(f"  Status   : [FAILED]")
        print(f"  Error    : {exc}")
        print("  Hint     : Check your key at https://aistudio.google.com/apikey")
        return False


def smoke_test_ollama() -> bool:
    print("\n" + "=" * 65)
    print("  SMOKE TEST: Ollama (Local Daemon)")
    print("=" * 65)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = _resolve_model(LLMProvider.OLLAMA)
    print(f"  Base URL : {base_url}")
    print(f"  Model    : {model}")

    try:
        client = OllamaClient(base_url=base_url)
        t0 = time.monotonic()
        res = client.complete_with_meta(PROMPT, model=model)
        latency = time.monotonic() - t0
        print(f"  Latency  : {latency:.2f}s")
        print(f"  Response : {res.text.strip()}")
        print(f"  Tokens   : prompt={res.prompt_tokens}, completion={res.completion_tokens}, total={res.total_tokens}")
        print("  Status   : [PASSED]")
        return True
    except Exception as exc:
        print(f"  Status   : [FAILED/OFFLINE]")
        print(f"  Error    : {exc}")
        print("  Hint     : Ensure Ollama is running (`ollama serve`) and model is pulled (`ollama pull llama3.2:3b`).")
        return False


def smoke_test_planner_json(provider_name: str | None = None) -> bool:
    print("\n" + "=" * 65)
    active_provider = provider_name or os.getenv("LLM_PROVIDER", "ollama")
    print(f"  SMOKE TEST: Structured JSON Generation (Planner check on '{active_provider}')")
    print("=" * 65)

    try:
        planner = ResearchPlanner()
        t0 = time.monotonic()
        plan = planner.create_plan(
            question=ResearchQuestion(question=PLANNER_QUESTION, domain="biology")
        )
        latency = time.monotonic() - t0
        print(f"  Latency      : {latency:.2f}s")
        print(f"  Topic        : {plan.topic}")
        print(f"  Key Questions: {len(plan.key_questions)} items")
        print(f"  Sub Questions: {len(plan.sub_questions)} items")
        print(f"  Info Needed  : {len(plan.information_needed)} items")
        print("  Status       : [PASSED — Valid Schema & JSON]")
        return True
    except PlanningError as pe:
        print(f"  Status       : [FAILED — Schema/JSON Parsing Error]")
        print(f"  Error        : {pe}")
        return False
    except Exception as exc:
        print(f"  Status       : [FAILED]")
        print(f"  Error        : {exc}")
        return False


def smoke_test_fallback() -> bool:
    print("\n" + "=" * 65)
    print("  SMOKE TEST: Configured End-to-End Fallback Policy")
    print("=" * 65)
    primary = os.getenv("LLM_PROVIDER", "ollama")
    fallback = os.getenv("LLM_FALLBACK_PROVIDER", "[none]")
    strict = os.getenv("STRICT_LLM", "false")
    print(f"  Primary  : {primary}")
    print(f"  Fallback : {fallback}")
    print(f"  Strict   : {strict}")

    try:
        t0 = time.monotonic()
        res = ask_llm_with_meta(PROMPT)
        latency = time.monotonic() - t0
        print(f"  Latency  : {latency:.2f}s")
        print(f"  Response : {res.text.strip()}")
        print(f"  Provider : {res.provider}")
        print(f"  Model    : {res.model}")
        print(f"  Fallback : used={res.fallback_used} (from={res.fallback_from})")
        print(f"  Retries  : {res.retries_attempted}")
        print("  Status   : [PASSED]")
        return True
    except LLMError as exc:
        print(f"  Status   : [FAILED]")
        print(f"  Error    : {exc}")
        return False


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

def main() -> None:
    target = sys.argv[1].lower().strip() if len(sys.argv) > 1 else "all"

    print("=" * 65)
    print("  ARGUS LLM Provider & Schema Verification Suite")
    print(f"  Target: {target}")
    print("=" * 65)

    results: dict[str, bool] = {}

    if target in ("groq", "all"):
        results["groq"] = smoke_test_groq()
    if target in ("gemini", "all"):
        results["gemini"] = smoke_test_gemini()
    if target in ("ollama", "all"):
        results["ollama"] = smoke_test_ollama()
    if target in ("planner", "all"):
        results["planner_json"] = smoke_test_planner_json()
    if target in ("fallback", "all"):
        results["fallback"] = smoke_test_fallback()

    print("\n" + "=" * 65)
    print("  Summary:")
    for name, ok in results.items():
        status = "PASSED" if ok else "FAILED/SKIPPED"
        print(f"    - {name:<14}: {status}")
    print("=" * 65)


if __name__ == "__main__":
    main()
