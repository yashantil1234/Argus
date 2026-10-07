import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.research import (
    Claim,
    Evidence,
    Report,
    ResearchPlan,
    ResearchQuestion,
    ResearchResult,
    Source,
    SourceType,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Module-level fixtures (used by the original test)
# ---------------------------------------------------------------------------

question = ResearchQuestion(
    question="How is generative AI changing software development?"
)

plan = ResearchPlan(
    topic="Impact of generative AI on software development",
    key_questions=[
        "How does generative AI affect developer productivity?",
        "Which tasks are most affected?",
    ],
    information_needed=[
        "Productivity studies",
        "Adoption statistics",
    ],
    initial_hypotheses=[
        "Generative AI can reduce time spent on routine coding tasks."
    ],
)


# ---------------------------------------------------------------------------
# 1. Original test — must still pass unchanged
# ---------------------------------------------------------------------------

def test_models_instantiation():
    """Pytest validation for ResearchQuestion and ResearchPlan models."""
    assert question.question.startswith("How is generative AI")
    assert plan.topic == "Impact of generative AI on software development"
    assert len(plan.key_questions) == 2
    assert len(plan.sub_questions) == 2        # synced by model_validator
    assert len(plan.information_needed) == 2
    assert len(plan.initial_hypotheses) == 1


# ---------------------------------------------------------------------------
# 2. Evidence — quote field and relevance
# ---------------------------------------------------------------------------

def test_evidence_uses_quote_field():
    source = Source(title="Nature 2023", source_type=SourceType.ACADEMIC)
    evidence = Evidence(
        source_id=source.id,
        quote="Cas9 requires recognition of a PAM sequence adjacent to the target.",
        relevance=0.95,
        page=3,
        location="Section 2.1",
    )
    assert evidence.quote == "Cas9 requires recognition of a PAM sequence adjacent to the target."
    assert evidence.relevance == 0.95
    assert evidence.source_id == source.id
    assert evidence.id is not None  # system-generated


def test_evidence_has_no_text_field():
    """Verify the old `text` field no longer exists."""
    assert not hasattr(Evidence.model_fields, "text"), \
        "Evidence.text was renamed to Evidence.quote — old field must not exist"
    assert "text" not in Evidence.model_fields


def test_evidence_relevance_bounds():
    source = Source(title="Test Source")
    with pytest.raises(ValidationError):
        Evidence(source_id=source.id, quote="some quote", relevance=1.5)
    with pytest.raises(ValidationError):
        Evidence(source_id=source.id, quote="some quote", relevance=-0.1)


# ---------------------------------------------------------------------------
# 3. Claim — evidence_ids list
# ---------------------------------------------------------------------------

def test_claim_uses_evidence_ids_list():
    ev1 = Evidence(source_id="src-1", quote="Quote A")
    ev2 = Evidence(source_id="src-1", quote="Quote B")
    claim = Claim(
        claim_text="Cas9 specificity arises from PAM recognition and guide RNA binding.",
        evidence_ids=[ev1.id, ev2.id],
        confidence=0.92,
    )
    assert len(claim.evidence_ids) == 2
    assert ev1.id in claim.evidence_ids
    assert ev2.id in claim.evidence_ids


def test_claim_has_no_evidence_id_singular():
    """Verify the old singular `evidence_id` field no longer exists."""
    assert "evidence_id" not in Claim.model_fields


def test_claim_defaults_to_empty_evidence_ids():
    claim = Claim(claim_text="A claim with no evidence yet.")
    assert claim.evidence_ids == []


def test_claim_confidence_bounds():
    with pytest.raises(ValidationError):
        Claim(claim_text="Bad confidence", confidence=1.5)
    with pytest.raises(ValidationError):
        Claim(claim_text="Bad confidence", confidence=-0.1)


# ---------------------------------------------------------------------------
# 4. ResearchResult — wrapper and traceability chain
# ---------------------------------------------------------------------------

def test_research_result_chains_claim_to_evidence_to_source():
    """Verify the full traceability chain is representable in the data model."""
    source = Source(
        url="https://doi.org/10.1038/s41586-021-03819-2",
        title="CRISPR-Cas9 specificity review",
        source_type=SourceType.ACADEMIC,
    )
    evidence = Evidence(
        source_id=source.id,
        quote="Cas9 requires recognition of a PAM sequence adjacent to the target.",
        relevance=0.97,
    )
    claim = Claim(
        claim_text="Cas9 specificity depends partly on PAM recognition.",
        evidence_ids=[evidence.id],
        confidence=0.95,
    )
    result = ResearchResult(
        research_plan_id="plan-abc",
        sources=[source],
        evidence=[evidence],
        claims=[claim],
    )

    # Traceability: claim → evidence → source
    assert claim.evidence_ids[0] == evidence.id
    assert evidence.source_id == source.id
    assert result.research_plan_id == "plan-abc"
    assert result.id is not None          # system-generated UUID
    assert result.created_at is not None  # system-generated timestamp


def test_research_result_ids_are_system_generated():
    """LLM never generates IDs — verify they come from uuid defaults."""
    result = ResearchResult(research_plan_id="plan-xyz")
    assert len(result.id) == 36   # uuid4 string length
    assert result.id != "plan-xyz"


def test_research_result_empty_lists_are_valid():
    """A ResearchResult with no findings is structurally valid."""
    result = ResearchResult(research_plan_id="plan-empty")
    assert result.sources == []
    assert result.evidence == []
    assert result.claims == []


# ---------------------------------------------------------------------------
# 5. Stage 5.1: Verification contracts (Status, Result, Report)
# ---------------------------------------------------------------------------

def test_verification_status_values():
    """Verify exact VerificationStatus enum values."""
    assert VerificationStatus.VERIFIED == "verified"
    assert VerificationStatus.REJECTED == "rejected"
    assert VerificationStatus.UNCERTAIN == "uncertain"


def test_verification_result_supported_flag_sync():
    """`supported` should be True only when status is VERIFIED."""
    v_verified = VerificationResult(
        claim_id="cl-1",
        status=VerificationStatus.VERIFIED,
        reasoning="Quote directly corroborates the claim statement.",
        evidence_ids=["ev-1"],
    )
    assert v_verified.supported is True
    assert v_verified.status == VerificationStatus.VERIFIED
    assert len(v_verified.id) == 36

    v_rejected = VerificationResult(
        claim_id="cl-2",
        status=VerificationStatus.REJECTED,
        reasoning="Evidence explicitly contradicts the claim.",
        evidence_ids=["ev-2"],
    )
    assert v_rejected.supported is False
    assert v_rejected.status == VerificationStatus.REJECTED

    v_uncertain = VerificationResult(
        claim_id="cl-3",
        status=VerificationStatus.UNCERTAIN,
        reasoning="Evidence discusses related topic but does not establish claim.",
    )
    assert v_uncertain.supported is False
    assert v_uncertain.status == VerificationStatus.UNCERTAIN


def test_verification_result_status_normalization():
    """Legacy strings 'valid', 'invalid', 'partial' normalize gracefully."""
    v1 = VerificationResult(claim_id="cl-1", status="valid", reason="All good")
    assert v1.status == VerificationStatus.VERIFIED
    assert v1.supported is True
    assert v1.reasoning == "All good"

    v2 = VerificationResult(claim_id="cl-2", status="invalid")
    assert v2.status == VerificationStatus.REJECTED
    assert v2.supported is False

    v3 = VerificationResult(claim_id="cl-3", status="unverifiable")
    assert v3.status == VerificationStatus.UNCERTAIN
    assert v3.supported is False


def test_verification_result_confidence_bounds():
    with pytest.raises(ValidationError):
        VerificationResult(claim_id="cl-1", status=VerificationStatus.VERIFIED, confidence=1.5)
    with pytest.raises(ValidationError):
        VerificationResult(claim_id="cl-1", status=VerificationStatus.VERIFIED, confidence=-0.1)


def test_verification_report_aggregates_counts():
    """VerificationReport computes aggregate counts automatically."""
    r1 = VerificationResult(claim_id="cl-1", status=VerificationStatus.VERIFIED)
    r2 = VerificationResult(claim_id="cl-2", status=VerificationStatus.VERIFIED)
    r3 = VerificationResult(claim_id="cl-3", status=VerificationStatus.REJECTED)
    r4 = VerificationResult(claim_id="cl-4", status=VerificationStatus.UNCERTAIN)

    report = VerificationReport(
        research_result_id="res-123",
        results=[r1, r2, r3, r4],
    )
    assert report.verified_count == 2
    assert report.rejected_count == 1
    assert report.uncertain_count == 1
    assert len(report.results) == 4
    assert len(report.id) == 36