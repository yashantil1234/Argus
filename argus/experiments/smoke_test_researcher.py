"""Live smoke test — Researcher against Ollama llama3.2:3b.

Run from argus/ directory:
    python experiments/smoke_test_researcher.py

Uses a fixed source text (controlled experiment).
Prints the full ResearchResult with traceability chain.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from app.agents.researcher import extract_from_source, ExtractionError
from app.models.research import ResearchPlan, Source, SourceType

# ---------------------------------------------------------------------------
# Fixed controlled source — same CRISPR text used in unit tests
# ---------------------------------------------------------------------------

SOURCE_TEXT = (
    "Cas9 requires recognition of a PAM sequence adjacent to the target site. "
    "The guide RNA directs Cas9 to complementary DNA sequences. "
    "Mismatches in the seed region significantly reduce cleavage efficiency. "
    "Off-target sites with PAM sequences and partial complementarity can be cleaved. "
    "High-fidelity Cas9 variants have been engineered to reduce off-target activity "
    "while maintaining on-target efficiency. SpCas9 recognizes the NGG PAM sequence, "
    "while other Cas9 orthologs recognize different PAM motifs, expanding the range "
    "of targetable genomic sites."
)

plan = ResearchPlan(
    topic="CRISPR-Cas9 Genome Editing Specificity",
    key_questions=[
        "How does PAM recognition influence Cas9 specificity?",
        "What mechanisms cause off-target cleavage?",
    ],
)

source = Source(
    url="https://doi.org/10.1038/example-crispr-review",
    title="CRISPR-Cas9 Specificity Review (controlled test)",
    source_type=SourceType.ACADEMIC,
)

provider = os.getenv("LLM_PROVIDER", "ollama")
model    = os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")

print("=" * 64)
print("  ARGUS Researcher -- Live Smoke Test")
print(f"  Provider : {provider}")
print(f"  Model    : {model}")
print("=" * 64)
print(f"\nPlan topic : {plan.topic}")
print(f"Source     : {source.title}\n")

try:
    result = extract_from_source(plan=plan, source_text=SOURCE_TEXT, source=source)

    print(f"[OK] ResearchResult id : {result.id}")
    print(f"     plan_id           : {result.research_plan_id}")
    print(f"     created_at        : {result.created_at}")

    print(f"\n--- Evidence ({len(result.evidence)}) ---")
    for i, ev in enumerate(result.evidence):
        print(f"  [{i}] id         : {ev.id}")
        print(f"       source_id  : {ev.source_id}  (== source.id: {ev.source_id == source.id})")
        print(f"       relevance  : {ev.relevance}")
        print(f"       quote      : {ev.quote[:90]!r}...")
        print()

    print(f"--- Claims ({len(result.claims)}) ---")
    for i, cl in enumerate(result.claims):
        print(f"  [{i}] id            : {cl.id}")
        print(f"       confidence    : {cl.confidence}")
        print(f"       evidence_ids  : {cl.evidence_ids}")
        print(f"       claim_text    : {cl.claim_text}")
        # Verify traceability
        evidence_id_set = {ev.id for ev in result.evidence}
        all_resolved = all(eid in evidence_id_set for eid in cl.evidence_ids)
        print(f"       traceable?    : {all_resolved}")
        print()

    print(f"--- Traceability check ---")
    evidence_id_set = {ev.id for ev in result.evidence}
    broken = [
        (cl.claim_text[:50], eid)
        for cl in result.claims
        for eid in cl.evidence_ids
        if eid not in evidence_id_set
    ]
    if broken:
        print(f"[FAIL] Broken links: {broken}")
    else:
        print(f"[OK]  All claim.evidence_ids resolve to valid Evidence objects.")

except ConnectionError as err:
    print(f"\n[FAIL] Ollama unreachable:\n       {err}")
except ExtractionError as err:
    print(f"\n[FAIL] Extraction failed:\n       {err}")

print("\n" + "=" * 64)
