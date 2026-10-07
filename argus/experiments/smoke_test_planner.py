"""Live end-to-end smoke test: Planner → Ollama → ResearchPlan.

Run from argus/ directory:
    python experiments/smoke_test_planner.py

Reads LLM_PROVIDER and DEFAULT_LLM_MODEL from .env.
Falls back with a clear PlanningError if the LLM response is malformed.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from app.agents.research_planner import plan_research, PlanningError

QUESTION = "How does CRISPR-Cas9 achieve genome-editing specificity?"

provider = os.getenv("LLM_PROVIDER", "ollama")
model    = os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")

print("=" * 62)
print("  ARGUS Planner — Live Smoke Test")
print(f"  Provider : {provider}")
print(f"  Model    : {model}")
print("=" * 62)
print(f"\nQuestion:\n  {QUESTION}\n")

try:
    plan = plan_research(QUESTION)

    print(f"[OK] topic              : {plan.topic}")
    print(f"     plan id            : {plan.id}")
    print(f"\n     key_questions ({len(plan.key_questions)}):")
    for q in plan.key_questions:
        print(f"       - {q}")
    print(f"\n     sub_questions ({len(plan.sub_questions)}):")
    for q in plan.sub_questions:
        print(f"       - {q}")
    print(f"\n     information_needed ({len(plan.information_needed)}):")
    for i in plan.information_needed:
        print(f"       - {i}")
    print(f"\n     initial_hypotheses ({len(plan.initial_hypotheses)}):")
    for h in plan.initial_hypotheses:
        print(f"       - {h}")

except ConnectionError as err:
    print(f"\n[FAIL] Ollama unreachable:\n       {err}")
except PlanningError as err:
    print(f"\n[FAIL] Planning failed (LLM returned bad JSON):\n       {err}")

print("\n" + "=" * 62)
