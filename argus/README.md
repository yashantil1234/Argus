# Argus (Autonomous Unified Research Agent)

Argus is an evidence-first, research-pipeline autonomous agent system designed with strict traceability and bounded execution.

## Architecture

Rather than an over-engineered multi-agent orchestrator, Argus is designed around an **evidence-first pipeline**:

```text
                     USER
                       │
                       ▼
              RESEARCH QUESTION
                       │
                       ▼
                 ┌───────────┐
                 │  PLANNER  │
                 └─────┬─────┘
                       │
                  Research Plan
                       │
                       ▼
                ┌─────────────┐
                │  RESEARCHER │
                └──────┬──────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
         Web Search             RAG
           (later)            (later)
             │                   │
             └─────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │ CLAIM + EVIDENCE│
              └────────┬────────┘
                       │
                       ▼
                ┌────────────┐
                │  VERIFIER  │
                └─────┬──────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
           VALID            INVALID
             │                 │
             │           Research Again (Bounded)
             │                 │
             └────────┬────────┘
                      ▼
                 ┌─────────┐
                 │ WRITER  │
                 └────┬────┘
                      ▼
                FINAL REPORT
                      │
                      ▼
             ┌─────────────────┐
             │   EVALUATION    │
             └─────────────────┘
```

## Evidence Traceability Backbone

Every factual statement in the final report traces back through a transparent, immutable provenance chain:

$$\text{Report} \longrightarrow \text{Claim} \longrightarrow \text{Evidence} \longrightarrow \text{Source}$$

- **`ResearchQuestion`**: The user-provided inquiry and optional domain tag.
- **`ResearchPlan`**: Structured sub-questions and information requirements.
- **`Source`**: Provenance metadata (URL, title, type, timestamp).
- **`Evidence`**: Granular extracted text snippets, page numbers, and locations.
- **`Claim`**: Atomic factual claims with calculated confidence scores.
- **`VerificationResult`**: Validation status (`VALID`, `INVALID`, `PARTIAL`, `UNVERIFIABLE`), reasoning, and supporting evidence.
- **`Report`**: Synthesized output directly referencing verified claims and sources.

## Bounded Execution

To prevent unbounded loops, research retries are strictly bounded:
- `MAX_RESEARCH_ROUNDS = 2`

## Getting Started

### 1. Installation

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Tests & Evaluation

```bash
pytest tests/ -q
```

---

## LLM Provider Architecture

ARGUS features a multi-provider LLM abstraction layer with controlled fallback, backoff retries, and rich provenance tracking. All agents (Planner, Researcher, Verifier, Writer) call a single, provider-agnostic interface:

```python
from app.core.llm import ask_llm, ask_llm_with_meta

# Plain text response (backward-compatible)
text = ask_llm("Explain CRISPR PAM specificity", role="researcher")

# Rich response with tokens and provenance metadata for experiment logging
response = ask_llm_with_meta("Explain CRISPR PAM specificity", role="verifier")
print(response.text, response.provider, response.model, response.prompt_tokens, response.fallback_used)
```

### Supported Providers

| Provider | Type | API Endpoint | Typical Models | Key Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **Google Gemini** | Cloud (Primary) | Official OpenAI-compatible (`v1beta/openai`) | `gemini-2.0-flash`, `gemini-2.5-flash` | `GEMINI_API_KEY` ([AI Studio](https://aistudio.google.com/apikey)) |
| **Groq** | Cloud (Alternative / Hosted Fallback) | Official OpenAI-compatible (`api.groq.com/openai/v1`) | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | `GROQ_API_KEY` ([Groq Console](https://console.groq.com/keys) — free tier, no credit card) |
| **Ollama** | Local (Offline Fallback) | Native REST (`/api/generate`) | `llama3.2:3b`, `mistral:7b` | Local daemon running (`ollama serve`) |
| **OpenRouter** | Cloud (Legacy) | OpenAI-compatible (`openrouter.ai/api/v1`) | `mistralai/mistral-7b-instruct` | `OPENROUTER_API_KEY` ([OpenRouter](https://openrouter.ai/)) |

---

### Configuration Guide

All configuration is controlled via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

#### 1. Google Gemini (Cloud Primary) + Groq (Cloud Fallback)
```env
LLM_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=groq
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile
```

#### 2. Groq (Cloud Primary) + Gemini (Cloud Fallback)
```env
LLM_PROVIDER=groq
LLM_FALLBACK_PROVIDER=gemini
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash
```

#### 3. Ollama (Offline / Local Edge)
```env
LLM_PROVIDER=ollama
LLM_FALLBACK_PROVIDER=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

#### 4. Per-Role Model Overrides (Optional)
```env
VERIFIER_MODEL=gemini-2.5-flash
RESEARCHER_MODEL=llama-3.3-70b-versatile
PLANNER_MODEL=gemini-2.0-flash
```

---

### Retry-Then-Fallback & Research Integrity

ARGUS implements a strict, resilient execution order:
1. **Exponential Backoff Retries on Transient Failures:**
   If the primary provider encounters a rate limit (`429`) or temporary service unavailability (`503`, timeouts), ARGUS **retries the primary provider up to 2 times** with exponential backoff, honoring `Retry-After` headers.
2. **Controlled Fallback:**
   Only when primary retries are fully exhausted does ARGUS fall back to `LLM_FALLBACK_PROVIDER`. In production deployments where Ollama does not exist, the fallback target can be another cloud provider (e.g. `groq`).
3. **Fast-Failure on Authentication Errors:**
   `401 Unauthorized` or invalid API keys **never trigger fallback**. They fail immediately with clear instructions to prevent silent model switching due to configuration bugs.
4. **Strict Benchmark Mode (`STRICT_LLM=true`):**
   In scientific evaluations, mixing models invalidates experimental rigor. Benchmark scripts enforce `STRICT_LLM=true` by default, disabling all fallback so runs fail visibly if a model is unavailable.

---

### Smoke Tests & Schema Validation

ARGUS includes an isolated smoke test script to verify live connectivity, credentials, and JSON planning schema generation without logging secrets:

```bash
# Test Groq (groq.com)
python experiments/smoke_test_providers.py groq

# Test Gemini (Google AI Studio)
python experiments/smoke_test_providers.py gemini

# Test Local Ollama
python experiments/smoke_test_providers.py ollama

# Test Structured JSON Planning
python experiments/smoke_test_providers.py planner

# Test End-to-End Fallback Policy
python experiments/smoke_test_providers.py fallback

# Run Full Diagnostics
python experiments/smoke_test_providers.py all
```

---

### Security Best Practices

- **Never commit `.env`:** The repository `.gitignore` blocks `.env`.
- **Key Rotation:** If an API key is ever exposed, rotate it immediately in the respective cloud console ([Google AI Studio](https://aistudio.google.com/apikey) or [Groq Console](https://console.groq.com/keys)).
- **Token Provenance:** Inspect `LLMResponse.prompt_tokens`, `completion_tokens`, and `model_version` in experiment logs to monitor quota consumption.
