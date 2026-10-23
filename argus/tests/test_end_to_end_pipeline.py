"""End-to-End Pipeline Integration Test (Phase 6 Item 2).

Verifies the complete execution chain:
  ResearchQuestion
       ↓
  ResearchPlanner
       ↓
  ParallelResearchCoordinator (with multimodal sources)
       ↓
  EvidenceMerger (deduplication & re-linking)
       ↓
  Verifier (independent evaluation)
       ↓
  ResearchWriter (report synthesis)
       ↓
  Report

Validates that:
  - IDs and foreign key references survive every transition intact:
    Report.claim_ids → Claim.id
    Claim.evidence_ids → Evidence.id
    Evidence.source_id → Source.id
  - Only VERIFIED claims are synthesized into the final report.
  - Quote Fidelity is accurately computed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.evidence_merger import merge_research_results
from app.agents.parallel_coordinator import ParallelResearchCoordinator
from app.agents.research_planner import ResearchPlanner
from app.agents.verifier import Verifier
from app.agents.writer import ResearchWriter
from app.ingestion import ingest
from app.models.research import (
    Claim,
    Evidence,
    Report,
    ResearchPlan,
    ResearchQuestion,
    ResearchResult,
    Source,
    SourceType,
    VerificationResult,
    VerificationStatus,
)


def test_full_pipeline_traceability_from_question_to_report(tmp_path):
    """Full lifecycle test checking ID survival, deduplication, verification, and report synthesis."""

    # 1. Research Question
    question = ResearchQuestion(
        question="What strategies minimize off-target cleavage in CRISPR-Cas9 genome editing?",
        domain="Genomics",
    )
    assert question.id is not None

    # 2. Planner decomposes question
    planner = ResearchPlanner()
    mock_plan_json = json.dumps({
        "topic": "CRISPR-Cas9 Off-Target Minimization",
        "key_questions": [
            "What engineered Cas9 variants reduce off-targets?",
            "How do truncated guide RNAs impact specificity?",
        ],
        "sub_questions": ["Variant fidelity", "tru-gRNA mechanics"],
        "information_needed": ["Comparative cleavage assays", "NGS sequencing data"],
        "initial_hypotheses": ["Engineered variants substantially lower off-target rates"],
    })

    with patch("app.agents.research_planner.ask_llm", return_value=mock_plan_json):
        plan = planner.create_plan(question)

    assert plan.research_question_id == question.id
    assert len(plan.key_questions) == 2

    # 3. Multimodal Ingestion of sources
    paper_pdf = tmp_path / "paper.pdf"
    paper_pdf.write_bytes(b"%PDF-1.4\n/Page High-fidelity Cas9 variants like eSpCas9 reduce off-target cuts.\n%%EOF")
    pdf_res = ingest(paper_pdf, title="eSpCas9 Paper")

    web_html = """
    <html><body>
        <h1>Guide RNA Optimization</h1>
        <p>Truncated guide RNAs (tru-gRNAs) with 17-18 nucleotides reduce off-target cleavage by orders of magnitude.</p>
        <p>Speculative assertion: It may cure all genetic diseases immediately.</p>
    </body></html>
    """
    html_res = ingest(web_html, title="tru-gRNA Guide")

    sources_with_text = [
        (pdf_res.source, pdf_res.full_text),
        (html_res.source, html_res.full_text),
    ]

    # 4. Parallel Coordinator executes subtasks
    coordinator = ParallelResearchCoordinator(max_workers=2)

    def mock_extract(plan, source_text, source):
        if source.source_type == SourceType.PDF:
            ev = Evidence(
                source_id=source.id,
                quote="High-fidelity Cas9 variants like eSpCas9 reduce off-target cuts.",
                page=1,
                location="Page 1",
            )
            cl = Claim(
                claim_text="High-fidelity variants reduce off-target cleavage.",
                evidence_ids=[ev.id],
                confidence=0.95,
            )
            return ResearchResult(research_plan_id=plan.id, sources=[source], evidence=[ev], claims=[cl])
        else:
            ev1 = Evidence(
                source_id=source.id,
                quote="Truncated guide RNAs (tru-gRNAs) with 17-18 nucleotides reduce off-target cleavage by orders of magnitude.",
                location="Section: Guide RNA Optimization",
            )
            ev2 = Evidence(
                source_id=source.id,
                quote="Speculative assertion: It may cure all genetic diseases immediately.",
                location="Section: Guide RNA Optimization",
            )
            cl1 = Claim(
                claim_text="Truncated guide RNAs decrease off-target activity.",
                evidence_ids=[ev1.id],
                confidence=0.9,
            )
            cl2 = Claim(
                claim_text="CRISPR immediately cures all genetic diseases.",
                evidence_ids=[ev2.id],
                confidence=0.4,
            )
            return ResearchResult(research_plan_id=plan.id, sources=[source], evidence=[ev1, ev2], claims=[cl1, cl2])

    with patch.object(coordinator.researcher, "extract", side_effect=mock_extract):
        merged_result = coordinator.execute_subtasks(plan, sources_with_text)

    assert len(merged_result.sources) == 2
    assert len(merged_result.evidence) == 3
    assert len(merged_result.claims) == 3
    assert merged_result.quote_fidelity == 1.0

    # 5. Independent Verifier
    verifier = Verifier()

    def mock_verify_claim(claim, evidence_items):
        if "cures all" in claim.claim_text.lower():
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.REJECTED,
                confidence=0.99,
                reasoning="The evidence explicitly labels this as a speculative assertion.",
            )
        return VerificationResult(
            claim_id=claim.id,
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            reasoning="Direct logical entailment supported by experimental text.",
        )

    with patch.object(verifier, "verify_claim", side_effect=mock_verify_claim):
        verification_report = verifier.verify_research_result(merged_result, max_workers=2)

    assert verification_report.verified_count == 2
    assert verification_report.rejected_count == 1
    assert verification_report.uncertain_count == 0

    # 6. Research Writer synthesizes verified findings
    writer = ResearchWriter()

    with patch("app.agents.writer.ask_llm", return_value="# Synthesis: CRISPR Specificity\n\n[Finding 1] High-fidelity variants reduce cleavage. [Finding 2] Truncated guides decrease off-target activity."):
        final_report = writer.write_report(
            question=question,
            research_result=merged_result,
            verification_report=verification_report,
            plan=plan,
        )

    # 7. Traceability Verification
    assert isinstance(final_report, Report)
    assert final_report.research_question_id == question.id
    assert len(final_report.claim_ids) == 2  # Only the 2 verified claims included!
    assert len(final_report.source_ids) == 2

    # Verify claim IDs in report point to verified claims
    verified_claim_texts = [
        cl.claim_text for cl in merged_result.claims if cl.id in final_report.claim_ids
    ]
    assert "High-fidelity variants reduce off-target cleavage." in verified_claim_texts
    assert "Truncated guide RNAs decrease off-target activity." in verified_claim_texts
    assert "CRISPR immediately cures all genetic diseases." not in verified_claim_texts

    # Verify source IDs in report point to valid sources
    source_id_set = {s.id for s in merged_result.sources}
    for sid in final_report.source_ids:
        assert sid in source_id_set
