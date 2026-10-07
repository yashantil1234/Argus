"""Unit tests for the Provenance Layer and Deterministic Quote Verification."""

import sys
from pathlib import Path
from unittest.mock import patch
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.core.provenance import (
    compute_content_hash,
    normalize_text,
    verify_quote_in_text,
)
from app.models.research import Evidence, ResearchPlan, ResearchResult, Source
from app.agents.researcher import Researcher


# ---------------------------------------------------------------------------
# 1. Content Hashing
# ---------------------------------------------------------------------------

def test_compute_content_hash_deterministic():
    text = "CRISPR-Cas9 genome editing specificity."
    hash1 = compute_content_hash(text)
    hash2 = compute_content_hash(text)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string


def test_source_auto_hashes_page_text():
    src = Source(title="Nature Study", page_text="Sample text content")
    assert src.content_hash is not None
    assert src.content_hash == compute_content_hash("Sample text content")


# ---------------------------------------------------------------------------
# 2. Text Normalization
# ---------------------------------------------------------------------------

def test_normalize_text():
    raw = "  “Cas9—Derived   from S. pyogenes…” \n\t  "
    norm = normalize_text(raw)
    assert norm == '"cas9-derived from s. pyogenes..."'


# ---------------------------------------------------------------------------
# 3. Deterministic Quote Matching
# ---------------------------------------------------------------------------

def test_verify_quote_exact_match():
    source_text = "Cas9 requires recognition of a PAM sequence adjacent to the target."
    quote = "Cas9 requires recognition of a PAM sequence"
    is_valid, score = verify_quote_in_text(quote, source_text)
    assert is_valid is True
    assert score == 1.0


def test_verify_quote_with_formatting_differences():
    source_text = "The Cas9 nuclease (derived from S. pyogenes) cleaves targeted DNA."
    quote = "the cas9 nuclease (derived from s. pyogenes) cleaves"
    is_valid, score = verify_quote_in_text(quote, source_text)
    assert is_valid is True
    assert score >= 0.98


def test_verify_quote_hallucinated_fails():
    source_text = "Photosynthetic enzymes in plants capture photon energy."
    quote = "Cas9 precisely edits genomic double-stranded breaks"
    is_valid, score = verify_quote_in_text(quote, source_text)
    assert is_valid is False
    assert score < 0.5


def test_verify_quote_empty():
    assert verify_quote_in_text("", "Some text")[0] is False
    assert verify_quote_in_text("Some quote", "")[0] is False


# ---------------------------------------------------------------------------
# 4. Quote Fidelity Metric on ResearchResult
# ---------------------------------------------------------------------------

def test_research_result_quote_fidelity():
    src = Source(title="Test Source")
    ev1 = Evidence(source_id=src.id, quote="Valid quote 1", is_verbatim=True)
    ev2 = Evidence(source_id=src.id, quote="Hallucinated quote 2", is_verbatim=False)

    res = ResearchResult(
        research_plan_id="p-1",
        sources=[src],
        evidence=[ev1, ev2],
    )
    assert res.quote_fidelity == 0.5  # 1 out of 2 is verbatim


def test_research_result_perfect_fidelity():
    src = Source(title="Test Source")
    ev1 = Evidence(source_id=src.id, quote="Valid quote 1", is_verbatim=True)
    res = ResearchResult(
        research_plan_id="p-1",
        sources=[src],
        evidence=[ev1],
    )
    assert res.quote_fidelity == 1.0


# ---------------------------------------------------------------------------
# 5. Researcher extraction sets is_verbatim accurately
# ---------------------------------------------------------------------------

def test_researcher_flags_non_verbatim_quote():
    source_text = "Actual source text about CRISPR PAM sequences."
    mock_payload = {
        "evidence": [
            {"quote": "Actual source text about CRISPR PAM sequences.", "relevance": 1.0},
            {"quote": "Fabricated quote never found in page.", "relevance": 0.8},
        ],
        "claims": [
            {"claim_text": "Claim 1", "confidence": 0.9, "evidence_indices": [0]},
            {"claim_text": "Claim 2", "confidence": 0.8, "evidence_indices": [1]},
        ],
    }

    researcher = Researcher()
    plan = ResearchPlan(topic="CRISPR")
    source = Source(title="CRISPR Paper")

    with patch("app.agents.researcher.ask_llm", return_value=json.dumps(mock_payload)):
        res = researcher.extract(plan=plan, source_text=source_text, source=source)

    assert res.evidence[0].is_verbatim is True
    assert res.evidence[1].is_verbatim is False
    assert res.quote_fidelity == 0.5
    assert source.page_text == source_text
    assert source.content_hash == compute_content_hash(source_text)
