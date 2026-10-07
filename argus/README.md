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
pytest argus/tests/
```
