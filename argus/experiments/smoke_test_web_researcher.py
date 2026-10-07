"""Live smoke test — Researcher Stage 4B with real web search + extraction.

Run from argus/ directory:
    python experiments/smoke_test_web_researcher.py

Pipeline:
    ResearchPlan -> search_web -> fetch_page_content -> Ollama extraction -> ResearchResult
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from app.agents.researcher import conduct_research
from app.models.research import ResearchPlan

plan = ResearchPlan(
    topic="CRISPR-Cas9 PAM specificity",
    key_questions=[
        "What is the role of PAM sequences in CRISPR Cas9 target recognition?",
    ],
)

provider = os.getenv("LLM_PROVIDER", "ollama")
model = os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")

print("=" * 64)
print("  ARGUS Researcher Stage 4B -- Web Research Live Test")
print(f"  Provider : {provider}")
print(f"  Model    : {model}")
print("=" * 64)
print(f"\nPlan topic: {plan.topic}\n")

try:
    result = conduct_research(plan=plan, max_sources=1)

    print(f"\n[OK] ResearchResult id : {result.id}")
    print(f"     plan_id           : {result.research_plan_id}")
    print(f"     sources count     : {len(result.sources)}")
    for s in result.sources:
        print(f"       - {s.title} ({s.url})")

    print(f"\n--- Extracted Evidence ({len(result.evidence)}) ---")
    for i, ev in enumerate(result.evidence):
        print(f"  [{i}] quote     : {ev.quote[:90]!r}...")
        print(f"       relevance : {ev.relevance}")

    print(f"\n--- Extracted Claims ({len(result.claims)}) ---")
    for i, cl in enumerate(result.claims):
        print(f"  [{i}] claim     : {cl.claim_text}")
        print(f"       confidence: {cl.confidence}")
        print(f"       ev_ids    : {cl.evidence_ids}")

    # Traceability check
    evidence_ids = {ev.id for ev in result.evidence}
    traceable = all(
        all(eid in evidence_ids for eid in cl.evidence_ids)
        for cl in result.claims
    )
    print(f"\n[OK] Traceability check passed: {traceable}")

except Exception as err:
    print(f"\n[FAIL] Live web research failed: {err}")

print("\n" + "=" * 64)
