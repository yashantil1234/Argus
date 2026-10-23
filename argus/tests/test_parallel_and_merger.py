"""Unit tests for EvidenceMerger, ParallelResearchCoordinator, and Parallel Verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agents.evidence_merger import merge_research_results
from app.agents.parallel_coordinator import ParallelResearchCoordinator
from app.agents.verifier import Verifier
from app.models.research import (
    Claim,
    Evidence,
    ResearchPlan,
    ResearchResult,
    Source,
    SourceType,
    VerificationStatus,
)


def test_evidence_merger_deduplicates_sources_and_quotes():
    """EvidenceMerger merges duplicate sources and duplicate quotes across workers."""
    plan_id = "test-plan-123"

    # Worker 1 result
    src1 = Source(url="https://nature.com/article1", title="Nature Paper", page_text="Sample text A")
    ev1 = Evidence(source_id=src1.id, quote="Cas9 produces double-strand breaks.")
    cl1 = Claim(claim_text="Cas9 cleaves target DNA.", evidence_ids=[ev1.id])
    res1 = ResearchResult(research_plan_id=plan_id, sources=[src1], evidence=[ev1], claims=[cl1])

    # Worker 2 result: duplicate source URL, identical quote, but new claim
    src2 = Source(url="https://nature.com/article1", title="Nature Paper Copy", page_text="Sample text A")
    ev2 = Evidence(source_id=src2.id, quote="Cas9 produces double-strand breaks.")
    cl2 = Claim(claim_text="Cas9 induces double stranded cuts.", evidence_ids=[ev2.id])
    res2 = ResearchResult(research_plan_id=plan_id, sources=[src2], evidence=[ev2], claims=[cl2])

    merged = merge_research_results([res1, res2], plan_id=plan_id)

    # Only 1 unique source and 1 unique evidence should survive
    assert len(merged.sources) == 1
    assert len(merged.evidence) == 1
    assert len(merged.claims) == 2

    # Both claims must link to the single surviving evidence ID
    surviving_ev_id = merged.evidence[0].id
    assert merged.claims[0].evidence_ids == [surviving_ev_id]
    assert merged.claims[1].evidence_ids == [surviving_ev_id]


def test_evidence_merger_clusters_identical_claims():
    """Identical claims from different workers are merged with unioned evidence references."""
    plan_id = "test-plan-456"

    src = Source(url="https://science.org/paper", title="Science Paper", page_text="Context")
    ev1 = Evidence(source_id=src.id, quote="Quote 1: Off-target cleavage observed.")
    ev2 = Evidence(source_id=src.id, quote="Quote 2: Fidelity increased with engineering.")

    cl1 = Claim(claim_text="Engineering reduces off-target cleavage.", evidence_ids=[ev1.id], confidence=0.8)
    cl2 = Claim(claim_text="Engineering reduces off-target cleavage.", evidence_ids=[ev2.id], confidence=0.95)

    res1 = ResearchResult(research_plan_id=plan_id, sources=[src], evidence=[ev1], claims=[cl1])
    res2 = ResearchResult(research_plan_id=plan_id, sources=[src], evidence=[ev2], claims=[cl2])

    merged = merge_research_results([res1, res2], plan_id=plan_id)

    assert len(merged.claims) == 1
    # Evidence IDs from both claims are combined
    assert len(merged.claims[0].evidence_ids) == 2
    assert merged.claims[0].confidence == 0.95


def test_parallel_research_coordinator_runs_workers_concurrently():
    """ParallelResearchCoordinator executes extraction across sources and consolidates."""
    plan = ResearchPlan(topic="CRISPR specificity", key_questions=["What is the off-target rate?"])
    src1 = Source(url="https://source1.com", title="Source 1")
    src2 = Source(url="https://source2.com", title="Source 2")

    coordinator = ParallelResearchCoordinator(max_workers=2)

    # Mock researcher.extract
    def mock_extract(plan, source_text, source):
        ev = Evidence(source_id=source.id, quote=f"Quote from {source.title}")
        cl = Claim(claim_text=f"Claim from {source.title}", evidence_ids=[ev.id])
        return ResearchResult(research_plan_id=plan.id, sources=[source], evidence=[ev], claims=[cl])

    with patch.object(coordinator.researcher, "extract", side_effect=mock_extract):
        result = coordinator.execute_subtasks(
            plan=plan,
            sources_with_text=[(src1, "Text for 1"), (src2, "Text for 2")],
        )

    assert len(result.sources) == 2
    assert len(result.evidence) == 2
    assert len(result.claims) == 2


def test_parallel_verification_preserves_order():
    """Verifier.verify_research_result with max_workers > 1 verifies claims concurrently and preserves order."""
    verifier = Verifier()
    src = Source(title="Source")
    ev1 = Evidence(source_id=src.id, quote="Quote A")
    ev2 = Evidence(source_id=src.id, quote="Quote B")

    cl1 = Claim(claim_text="Claim A", evidence_ids=[ev1.id])
    cl2 = Claim(claim_text="Claim B", evidence_ids=[ev2.id])
    cl3 = Claim(claim_text="Claim C", evidence_ids=[])

    res = ResearchResult(
        research_plan_id="plan-789",
        sources=[src],
        evidence=[ev1, ev2],
        claims=[cl1, cl2, cl3],
    )

    from app.models.research import VerificationResult

    with patch.object(verifier, "verify_claim") as mock_vc:
        mock_vc.side_effect = lambda claim, evidence_items: VerificationResult(
            claim_id=claim.id,
            status=VerificationStatus.VERIFIED if evidence_items else VerificationStatus.UNCERTAIN,
            reasoning="mock verified",
        )

        report = verifier.verify_research_result(res, max_workers=3)

    assert len(report.results) == 3
    # Preserves exact claim order
    assert report.results[0].claim_id == cl1.id
    assert report.results[1].claim_id == cl2.id
    assert report.results[2].claim_id == cl3.id
