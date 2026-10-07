"""Live smoke test — Verifier Agent Stage 5.5 against Ollama llama3.2:3b.

Run from argus/ directory:
    python experiments/smoke_test_verifier.py

Verifies 4 distinct verification scenarios:
  1. Corroborated claim      -> Expected: VERIFIED
  2. Contradicted claim       -> Expected: REJECTED
  3. Insufficient evidence   -> Expected: UNCERTAIN
  4. Missing/orphan evidence -> Expected: UNCERTAIN (short-circuits)
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from app.agents.verifier import verify_research
from app.models.research import Claim, Evidence, ResearchResult, Source, SourceType

provider = os.getenv("LLM_PROVIDER", "ollama")
model = os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")

print("=" * 66)
print("  ARGUS Verifier Stage 5.5 -- Adversarial Verification Live Test")
print(f"  Provider : {provider}")
print(f"  Model    : {model}")
print("=" * 66)

# Create mock research artifact representing different claim truth values
source = Source(
    title="Mechanistic Studies on CRISPR-Cas9 Target Recognition",
    url="https://doi.org/10.1038/example-study",
    source_type=SourceType.ACADEMIC,
)

ev1 = Evidence(
    source_id=source.id,
    quote="Cas9 requires recognition of a PAM sequence adjacent to the target site for DNA binding.",
    relevance=0.98,
)
ev2 = Evidence(
    source_id=source.id,
    quote="Cas9 cleavage strictly cannot occur in the absence of a canonical PAM motif.",
    relevance=0.95,
)
ev3 = Evidence(
    source_id=source.id,
    quote="Researchers monitored Cas9 conformational transitions at room temperature over standard assays.",
    relevance=0.60,
)

# 1. Corroborated
cl_valid = Claim(
    claim_text="Cas9 requires an adjacent PAM sequence for DNA target binding.",
    evidence_ids=[ev1.id],
)

# 2. Contradicted
cl_contradicted = Claim(
    claim_text="Cas9 cleavage can occur efficiently even when no PAM motif is present.",
    evidence_ids=[ev2.id],
)

# 3. Insufficient / tangential
cl_weak = Claim(
    claim_text="High temperatures above 60C permanently denature Cas9 within seconds.",
    evidence_ids=[ev3.id],
)

# 4. No evidence supplied
cl_orphan = Claim(
    claim_text="Engineered Cas9 variants operate in eukaryotic chromatin.",
    evidence_ids=[],
)

research_result = ResearchResult(
    research_plan_id="plan-crispr-spec",
    sources=[source],
    evidence=[ev1, ev2, ev3],
    claims=[cl_valid, cl_contradicted, cl_weak, cl_orphan],
)

try:
    print(f"\nEvaluating ResearchResult ({len(research_result.claims)} claims)...")
    report = verify_research(research_result)

    print("\n" + "-" * 66)
    print("  VERIFICATION REPORT")
    print("-" * 66)
    print(f"Report ID        : {report.id}")
    print(f"Total Evaluated  : {len(report.results)}")
    print(f"VERIFIED (Pass)  : {report.verified_count}")
    print(f"REJECTED (Fail)  : {report.rejected_count}")
    print(f"UNCERTAIN        : {report.uncertain_count}")
    print("-" * 66)

    for i, res in enumerate(report.results):
        matching_claim = next(c for c in research_result.claims if c.id == res.claim_id)
        print(f"\n[{i + 1}] Claim: {matching_claim.claim_text}")
        print(f"    Status    : {res.status.upper()} (supported={res.supported})")
        print(f"    Confidence: {res.confidence}")
        print(f"    Reasoning : {res.reasoning}")

    print("\n[OK] Verifier live evaluation complete.")

except Exception as err:
    print(f"\n[FAIL] Verifier test failed: {err}")

print("\n" + "=" * 66)
