"""Writer Agent — Stage 6: Report Synthesis with Verified Provenance.

Synthesizes verified claims and cited evidence into an academic research report.
Enforces strict citation integrity:
  1. Only VERIFIED claims are included as factual assertions.
  2. Claims marked REJECTED or UNCERTAIN are either excluded or placed in a
     critical limitations section.
  3. All claim_ids and source_ids are preserved in the final Report artifact,
     completing the end-to-end traceability chain:
     Report -> Claim -> Evidence -> Source
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.core.llm import ask_llm
from app.models.research import (
    Claim,
    Evidence,
    Report,
    ResearchPlan,
    ResearchQuestion,
    ResearchResult,
    VerificationReport,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

_WRITER_PROMPT = """\
You are an objective scientific research writer. Your task is to synthesize \
verified factual findings into a structured research report for the topic:
"{topic}"

The following findings have been independently VERIFIED against source evidence:
{verified_findings}

{limitations_block}

Instructions:
1. Write a clear, scholarly synthesis organized into sections (Summary, Key Findings, Discussion).
2. Reference the numbered findings directly in your text (e.g. "[Finding 1]").
3. DO NOT introduce external claims that are not in the verified findings list above.
4. If limitations are noted, include a brief "Caveats & Uncertainties" section.
5. Provide a cohesive, academic narrative.
"""


class ResearchWriter:
    """Agent that writes a grounded Research Report from verified claims."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    def write_report(
        self,
        question: ResearchQuestion | str,
        research_result: ResearchResult,
        verification_report: VerificationReport,
        plan: Optional[ResearchPlan] = None,
    ) -> Report:
        """Synthesize verified claims into a final Report.

        Args:
            question: Original ResearchQuestion or question string.
            research_result: The consolidated research artifact.
            verification_report: The independent verification results.
            plan: Optional ResearchPlan for topic context.

        Returns:
            Report with explicit claim_ids and source_ids traceability.
        """
        question_id = question.id if isinstance(question, ResearchQuestion) else research_result.research_plan_id
        topic = plan.topic if plan else (question.question if isinstance(question, ResearchQuestion) else str(question))

        # Map verification status by claim ID
        status_by_claim_id = {r.claim_id: r.status for r in verification_report.results}

        verified_claims: List[Claim] = []
        unverified_claims: List[Claim] = []

        for claim in research_result.claims:
            st = status_by_claim_id.get(claim.id, VerificationStatus.UNCERTAIN)
            if st == VerificationStatus.VERIFIED:
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)

        # Build prompt inputs
        findings_lines = [
            f"[{idx}] {cl.claim_text} (Confidence: {cl.confidence:.2f})"
            for idx, cl in enumerate(verified_claims, start=1)
        ]
        verified_text = "\n".join(findings_lines) if findings_lines else "No verified findings available."

        limitations_block = ""
        if unverified_claims:
            limitations_block = "The following findings were REJECTED or UNCERTAIN upon verification:\n" + "\n".join(
                f"- {cl.claim_text}" for cl in unverified_claims
            )

        prompt = _WRITER_PROMPT.format(
            topic=topic,
            verified_findings=verified_text,
            limitations_block=limitations_block,
        )

        logger.info(
            "ResearchWriter: synthesizing report for topic %r with %d verified claims (model=%s)",
            topic, len(verified_claims), self.model,
        )

        content = ask_llm(prompt, model=self.model, role="writer")

        # Traceability IDs
        surviving_claim_ids = [cl.id for cl in verified_claims]
        # Resolve which sources backed the verified claims
        evidence_map = {ev.id: ev for ev in research_result.evidence}
        used_source_ids = set()
        for cl in verified_claims:
            for eid in cl.evidence_ids:
                if eid in evidence_map:
                    used_source_ids.add(evidence_map[eid].source_id)

        report = Report(
            research_question_id=question_id,
            content=content,
            claim_ids=surviving_claim_ids,
            source_ids=sorted(list(used_source_ids)),
        )

        logger.info(
            "ResearchWriter: report generated. claims=%d sources=%d",
            len(report.claim_ids), len(report.source_ids),
        )
        return report
