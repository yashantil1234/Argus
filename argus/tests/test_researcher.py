"""Tests for the Researcher agent — fully mocked, no real LLM calls.

Covers:
  1. Valid extraction: JSON → Evidence + Claim + wired IDs
  2. Markdown-fenced JSON still parses
  3. JSON embedded in prose still parses
  4. Missing 'evidence' key → ExtractionError
  5. Missing 'claims' key → ExtractionError
  6. Completely invalid response → ExtractionError with diagnostics
  7. Out-of-bounds evidence_index → ExtractionError
  8. Empty extraction: source has no relevant content → claims=[], evidence=[]
  9. HALLUCINATION TEST: plan asks about X, source only discusses Y → claims=[]
 10. Traceability: claim.evidence_ids resolve to actual Evidence.id values
 11. IDs are system-generated (not copied from LLM payload)
 12. extract_from_source() convenience wrapper
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch

from app.agents.researcher import Researcher, ExtractionError, extract_from_source
from app.models.research import (
    Claim, Evidence, ResearchPlan, ResearchResult, Source, SourceType
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_plan(
    topic="CRISPR-Cas9 Genome Editing Specificity",
    key_questions=None,
) -> ResearchPlan:
    return ResearchPlan(
        topic=topic,
        key_questions=key_questions or [
            "How does PAM recognition influence Cas9 specificity?",
            "What mechanisms cause off-target cleavage?",
        ],
    )


def make_source() -> Source:
    return Source(
        url="https://doi.org/10.1038/example",
        title="CRISPR-Cas9 Specificity Review",
        source_type=SourceType.ACADEMIC,
    )


CRISPR_SOURCE_TEXT = (
    "Cas9 requires recognition of a PAM sequence adjacent to the target site. "
    "The guide RNA directs Cas9 to complementary DNA sequences. "
    "Mismatches in the seed region significantly reduce cleavage efficiency. "
    "Off-target sites with PAM sequences and partial complementarity can be cleaved."
)

VALID_PAYLOAD = {
    "evidence": [
        {
            "quote": "Cas9 requires recognition of a PAM sequence adjacent to the target site.",
            "relevance": 0.97,
            "page": None,
            "location": "paragraph 1",
        },
        {
            "quote": "Mismatches in the seed region significantly reduce cleavage efficiency.",
            "relevance": 0.91,
            "page": None,
            "location": "paragraph 3",
        },
    ],
    "claims": [
        {
            "claim_text": "Cas9 specificity depends on PAM sequence recognition.",
            "confidence": 0.95,
            "evidence_indices": [0],
        },
        {
            "claim_text": "Seed-region mismatches reduce Cas9 cleavage efficiency.",
            "confidence": 0.90,
            "evidence_indices": [1],
        },
    ],
}


def mock_llm(payload: dict) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# 1. Valid extraction produces correct ResearchResult
# ---------------------------------------------------------------------------

def test_valid_extraction_produces_research_result():
    researcher = Researcher()
    plan = make_plan()
    source = make_source()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(VALID_PAYLOAD)):
        result = researcher.extract(plan, CRISPR_SOURCE_TEXT, source)

    assert isinstance(result, ResearchResult)
    assert len(result.claims) == 2
    assert len(result.evidence) == 2
    assert result.research_plan_id == plan.id
    assert result.sources[0].id == source.id


# ---------------------------------------------------------------------------
# 2. Markdown-fenced JSON parses correctly
# ---------------------------------------------------------------------------

def test_markdown_fenced_json_is_parsed():
    fenced = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=fenced):
        result = researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())

    assert len(result.claims) == 2


# ---------------------------------------------------------------------------
# 3. JSON embedded in prose parses correctly
# ---------------------------------------------------------------------------

def test_json_embedded_in_prose_is_parsed():
    prose = f"Here is the extracted evidence:\n{json.dumps(VALID_PAYLOAD)}\nDone."
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=prose):
        result = researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())

    assert len(result.evidence) == 2


# ---------------------------------------------------------------------------
# 4 & 5. Missing top-level keys → ExtractionError
# ---------------------------------------------------------------------------

def test_missing_evidence_key_raises_extraction_error():
    bad_payload = {"claims": []}
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(bad_payload)):
        with pytest.raises(ExtractionError, match="evidence"):
            researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())


def test_missing_claims_key_raises_extraction_error():
    bad_payload = {"evidence": []}
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(bad_payload)):
        with pytest.raises(ExtractionError, match="claims"):
            researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())


# ---------------------------------------------------------------------------
# 6. Completely invalid response → ExtractionError with diagnostics
# ---------------------------------------------------------------------------

def test_invalid_json_raises_extraction_error_with_diagnostics():
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value="Sorry, I cannot help."):
        with pytest.raises(ExtractionError, match="valid JSON"):
            researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())


# ---------------------------------------------------------------------------
# 7. Out-of-bounds evidence_index is filtered safely + 1-based index normalization
# ---------------------------------------------------------------------------

def test_out_of_bounds_evidence_index_is_filtered_gracefully():
    payload_with_bad_index = {
        "evidence": [
            {"quote": "Cas9 requires PAM recognition.", "relevance": 0.9}
        ],
        "claims": [
            {
                "claim_text": "Some claim.",
                "confidence": 0.8,
                "evidence_indices": [5],  # index 5 does not exist; should be dropped
            }
        ],
    }
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(payload_with_bad_index)):
        result = researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())

    assert len(result.claims) == 1
    assert result.claims[0].evidence_ids == []  # invalid index dropped without crashing


def test_one_based_indexing_is_normalized():
    one_based_payload = {
        "evidence": [
            {"quote": "Cas9 requires PAM recognition.", "relevance": 0.9},
            {"quote": "Mismatches reduce cleavage.", "relevance": 0.85},
        ],
        "claims": [
            {
                "claim_text": "Seed region mismatches impact efficiency.",
                "confidence": 0.9,
                "evidence_indices": [1, 2],  # 1-based: refers to items 0 and 1
            }
        ],
    }
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(one_based_payload)):
        result = researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())

    assert len(result.claims[0].evidence_ids) == 2
    assert result.claims[0].evidence_ids[0] == result.evidence[0].id
    assert result.claims[0].evidence_ids[1] == result.evidence[1].id


# ---------------------------------------------------------------------------
# 8. Empty extraction: source has nothing relevant → claims=[], evidence=[]
# ---------------------------------------------------------------------------

def test_empty_extraction_is_valid():
    empty_payload = {"evidence": [], "claims": []}
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(empty_payload)):
        result = researcher.extract(make_plan(), "This text is unrelated.", make_source())

    assert result.claims == []
    assert result.evidence == []
    assert isinstance(result, ResearchResult)


# ---------------------------------------------------------------------------
# 9. HALLUCINATION TEST — plan asks about temperature, source discusses PAM
# ---------------------------------------------------------------------------

def test_no_claims_when_source_does_not_support_plan():
    """
    The plan asks about temperature effects on Cas9.
    The source only discusses PAM sequences and guide RNA.
    The expected correct behaviour: claims=[] (no hallucinated temperature claims).

    This test validates that the parser correctly handles an empty extraction
    and does NOT invent claims from general knowledge.
    We mock the LLM to return the correct empty response — this tests the
    parsing and assembly pipeline. The live version of this test (Stage 4B)
    will verify the LLM itself doesn't hallucinate.
    """
    temperature_plan = make_plan(
        topic="Temperature effects on CRISPR-Cas9 specificity",
        key_questions=[
            "How does temperature affect Cas9 cleavage rate?",
            "What is the optimal temperature for Cas9 activity?",
        ],
    )
    pam_only_source_text = (
        "Cas9 requires recognition of a PAM sequence adjacent to the target site. "
        "The guide RNA directs Cas9 to complementary DNA sequences."
    )
    # Correct LLM behaviour: return empty arrays because source has no temperature data
    empty_payload = {"evidence": [], "claims": []}
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(empty_payload)):
        result = researcher.extract(temperature_plan, pam_only_source_text, make_source())

    assert result.claims == [], (
        "Researcher should return no claims when the source doesn't support the plan topic"
    )
    assert result.evidence == []


# ---------------------------------------------------------------------------
# 10. Traceability: claim.evidence_ids resolve to actual Evidence.id values
# ---------------------------------------------------------------------------

def test_claim_evidence_ids_resolve_to_evidence_ids():
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(VALID_PAYLOAD)):
        result = researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())

    evidence_id_set = {ev.id for ev in result.evidence}
    for claim in result.claims:
        for eid in claim.evidence_ids:
            assert eid in evidence_id_set, (
                f"Claim references evidence_id {eid!r} which doesn't exist in result.evidence"
            )


# ---------------------------------------------------------------------------
# 11. IDs are system-generated — never copied from LLM payload
# ---------------------------------------------------------------------------

def test_ids_are_system_generated_not_from_llm():
    llm_payload_with_fake_ids = {
        "evidence": [
            {
                "id": "llm-invented-id-abc",        # LLM should NOT control this
                "quote": "Cas9 requires recognition of a PAM sequence adjacent to the target site.",
                "relevance": 0.97,
            }
        ],
        "claims": [
            {
                "id": "llm-invented-claim-id",      # LLM should NOT control this
                "claim_text": "Cas9 needs PAM recognition.",
                "confidence": 0.9,
                "evidence_indices": [0],
            }
        ],
    }
    researcher = Researcher()

    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(llm_payload_with_fake_ids)):
        result = researcher.extract(make_plan(), CRISPR_SOURCE_TEXT, make_source())

    # IDs must be valid UUIDs (36 chars), not the LLM-invented strings
    assert result.evidence[0].id != "llm-invented-id-abc"
    assert len(result.evidence[0].id) == 36
    assert result.claims[0].id != "llm-invented-claim-id"
    assert len(result.claims[0].id) == 36


# ---------------------------------------------------------------------------
# 12. extract_from_source() convenience function
# ---------------------------------------------------------------------------

def test_extract_from_source_convenience_function():
    with patch("app.agents.researcher.ask_llm", return_value=mock_llm(VALID_PAYLOAD)):
        result = extract_from_source(
            plan=make_plan(),
            source_text=CRISPR_SOURCE_TEXT,
            source=make_source(),
        )

    assert isinstance(result, ResearchResult)
    assert len(result.claims) == 2
