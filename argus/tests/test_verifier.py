"""Unit tests for the Verifier agent (Stage 5).

All tests are fully mocked — no live external LLM calls.
Covers:
  1. Strong evidence → VERIFIED (supported == True)
  2. Contradictory evidence → REJECTED (supported == False)
  3. Weak/insufficient evidence → UNCERTAIN (supported == False)
  4. Missing evidence → UNCERTAIN (no LLM call triggered)
  5. Hallucinated claim / outside knowledge refusal → UNCERTAIN
  6. Malformed JSON response → VerificationError
  7. Unrecognized status string → VerificationError
  8. Markdown-fenced JSON is handled properly
  9. Full ResearchResult verification generates VerificationReport with correct counts
 10. verify_claim and verify_research convenience functions
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.agents.verifier import (
    VerificationError,
    Verifier,
    verify_claim,
    verify_research,
)
from app.models.research import (
    Claim,
    Evidence,
    ResearchResult,
    Source,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_evidence():
    return Evidence(
        source_id="src-1",
        quote="Cas9 requires recognition of a PAM sequence adjacent to the target site.",
        relevance=0.98,
    )


@pytest.fixture
def sample_claim(sample_evidence):
    return Claim(
        claim_text="Cas9 target recognition depends on adjacent PAM sequences.",
        evidence_ids=[sample_evidence.id],
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# 1. Strong evidence → VERIFIED
# ---------------------------------------------------------------------------

def test_strong_evidence_yields_verified(sample_claim, sample_evidence):
    mock_payload = {
        "status": "VERIFIED",
        "confidence": 0.98,
        "reasoning": "The quote directly states that Cas9 requires recognition of a PAM sequence.",
    }
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm", return_value=json.dumps(mock_payload)):
        result = verifier.verify_claim(sample_claim, [sample_evidence])

    assert isinstance(result, VerificationResult)
    assert result.status == VerificationStatus.VERIFIED
    assert result.supported is True
    assert result.claim_id == sample_claim.id
    assert sample_evidence.id in result.evidence_ids
    assert result.confidence == 0.98


# ---------------------------------------------------------------------------
# 2. Contradictory evidence → REJECTED
# ---------------------------------------------------------------------------

def test_contradictory_evidence_yields_rejected():
    evidence = Evidence(
        source_id="src-1",
        quote="Cas9 activity is completely blocked in the presence of calcium ions.",
    )
    claim = Claim(
        claim_text="Calcium ions significantly enhance Cas9 cleavage activity.",
        evidence_ids=[evidence.id],
    )
    mock_payload = {
        "status": "REJECTED",
        "confidence": 0.95,
        "reasoning": "The quote states calcium ions block activity, directly contradicting the enhancement claim.",
    }
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm", return_value=json.dumps(mock_payload)):
        result = verifier.verify_claim(claim, [evidence])

    assert result.status == VerificationStatus.REJECTED
    assert result.supported is False
    assert "contradicting" in result.reasoning


# ---------------------------------------------------------------------------
# 3. Weak / insufficient evidence → UNCERTAIN
# ---------------------------------------------------------------------------

def test_insufficient_evidence_yields_uncertain():
    evidence = Evidence(
        source_id="src-1",
        quote="Some researchers have studied enzymatic dynamics in modified buffer conditions.",
    )
    claim = Claim(
        claim_text="Buffer pH 7.4 maximizes SpCas9 kinetic rate by threefold.",
        evidence_ids=[evidence.id],
    )
    mock_payload = {
        "status": "UNCERTAIN",
        "confidence": 0.90,
        "reasoning": "The quote mentions buffer studies generally but gives no specifics on pH or kinetic rate.",
    }
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm", return_value=json.dumps(mock_payload)):
        result = verifier.verify_claim(claim, [evidence])

    assert result.status == VerificationStatus.UNCERTAIN
    assert result.supported is False


# ---------------------------------------------------------------------------
# 4. Missing evidence → UNCERTAIN (short-circuits, no LLM call)
# ---------------------------------------------------------------------------

def test_missing_evidence_short_circuits_to_uncertain():
    claim = Claim(claim_text="An orphan claim with no evidence.")
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm") as mock_llm:
        result = verifier.verify_claim(claim, [])
        mock_llm.assert_not_called()

    assert result.status == VerificationStatus.UNCERTAIN
    assert result.supported is False
    assert result.evidence_ids == []
    assert "No supporting evidence" in result.reasoning


# ---------------------------------------------------------------------------
# 5. Hallucinated claim / outside knowledge refusal → UNCERTAIN
# ---------------------------------------------------------------------------

def test_hallucinated_claim_yields_uncertain_or_rejected():
    evidence = Evidence(
        source_id="src-1",
        quote="Cas9 recognizes the NGG PAM sequence in S. pyogenes.",
    )
    claim = Claim(
        claim_text="The average human genome size is approximately 3.2 billion base pairs.",
        evidence_ids=[evidence.id],
    )
    # The claim is true in real biology, but completely absent from the quote
    mock_payload = {
        "status": "UNCERTAIN",
        "confidence": 0.99,
        "reasoning": "The quote only discusses Cas9 PAM recognition; it contains no information about human genome size.",
    }
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm", return_value=json.dumps(mock_payload)):
        result = verifier.verify_claim(claim, [evidence])

    assert result.status == VerificationStatus.UNCERTAIN
    assert result.supported is False


# ---------------------------------------------------------------------------
# 6. Malformed JSON response → VerificationError
# ---------------------------------------------------------------------------

def test_malformed_json_raises_verification_error(sample_claim, sample_evidence):
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm", return_value="Not valid json {status: invalid"):
        with pytest.raises(VerificationError, match="valid JSON"):
            verifier.verify_claim(sample_claim, [sample_evidence])


# ---------------------------------------------------------------------------
# 7. Unrecognized status string → VerificationError
# ---------------------------------------------------------------------------

def test_unrecognized_status_raises_verification_error(sample_claim, sample_evidence):
    bad_payload = {"status": "MAYBE", "confidence": 0.5, "reasoning": "Not sure"}
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm", return_value=json.dumps(bad_payload)):
        with pytest.raises(VerificationError, match="Unrecognized verification status"):
            verifier.verify_claim(sample_claim, [sample_evidence])


# ---------------------------------------------------------------------------
# 8. Markdown-fenced JSON is handled properly
# ---------------------------------------------------------------------------

def test_markdown_fenced_json_is_parsed(sample_claim, sample_evidence):
    fenced = f"```json\n{{\"status\": \"VERIFIED\", \"confidence\": 0.9, \"reasoning\": \"Supported.\"}}\n```"
    verifier = Verifier()

    with patch("app.agents.verifier.ask_llm", return_value=fenced):
        result = verifier.verify_claim(sample_claim, [sample_evidence])

    assert result.status == VerificationStatus.VERIFIED
    assert result.supported is True


# ---------------------------------------------------------------------------
# 9. Batch verification on ResearchResult artifact
# ---------------------------------------------------------------------------

def test_verify_research_result_generates_report():
    source = Source(title="Test Source")
    ev1 = Evidence(source_id=source.id, quote="Cas9 binds PAM.")
    ev2 = Evidence(source_id=source.id, quote="Seed mutations reduce binding.")

    cl1 = Claim(claim_text="Cas9 binds PAM motif.", evidence_ids=[ev1.id])
    cl2 = Claim(claim_text="Seed mutations reduce binding.", evidence_ids=[ev2.id])
    cl3 = Claim(claim_text="Cas9 functions in extreme cold.", evidence_ids=[])

    res = ResearchResult(
        research_plan_id="plan-1",
        sources=[source],
        evidence=[ev1, ev2],
        claims=[cl1, cl2, cl3],
    )

    verifier = Verifier()

    # cl1 and cl2 call LLM with VERIFIED; cl3 short-circuits without LLM
    mock_verified = json.dumps({"status": "VERIFIED", "confidence": 0.95, "reasoning": "Valid."})
    with patch("app.agents.verifier.ask_llm", return_value=mock_verified):
        report = verifier.verify_research_result(res)

    assert isinstance(report, VerificationReport)
    assert report.research_result_id == res.id
    assert len(report.results) == 3
    assert report.verified_count == 2
    assert report.uncertain_count == 1  # cl3 had no evidence
    assert report.rejected_count == 0


# ---------------------------------------------------------------------------
# 10. Convenience functions
# ---------------------------------------------------------------------------

def test_convenience_functions(sample_claim, sample_evidence):
    mock_payload = {"status": "VERIFIED", "confidence": 0.9, "reasoning": "Good."}
    with patch("app.agents.verifier.ask_llm", return_value=json.dumps(mock_payload)):
        res = verify_claim(sample_claim, [sample_evidence])
    assert res.status == VerificationStatus.VERIFIED

    rr = ResearchResult(
        research_plan_id="p-1",
        evidence=[sample_evidence],
        claims=[sample_claim],
    )
    with patch("app.agents.verifier.ask_llm", return_value=json.dumps(mock_payload)):
        rep = verify_research(rr)
    assert rep.verified_count == 1
