"""Tests for ResearchPlanner — fully mocked, no real LLM calls.

Covers:
  - Valid JSON response → validated ResearchPlan
  - Markdown-fenced JSON response → still parses correctly
  - Inline brace-only JSON extraction
  - Completely invalid response → PlanningError with diagnostic message
  - ResearchQuestion object as input → question_id propagated
  - domain → appears in prompt
  - plan_research() convenience function
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
from unittest.mock import patch

from app.agents.research_planner import ResearchPlanner, plan_research, PlanningError
from app.models.research import ResearchPlan, ResearchQuestion

# ---------------------------------------------------------------------------
# Shared fixture: a valid LLM JSON payload
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "topic": "CRISPR-Cas9 Genome Editing Specificity",
    "key_questions": [
        "How does PAM recognition influence Cas9 target binding?",
        "What mechanisms cause off-target double-strand breaks?",
    ],
    "sub_questions": [
        "What structural features of the guide RNA determine specificity?",
        "How do mismatches in the seed region affect cleavage efficiency?",
    ],
    "information_needed": [
        "Structural biology studies on Cas9-RNA-DNA complex",
        "Genome-wide off-target profiling datasets (GUIDE-seq, CIRCLE-seq)",
    ],
    "initial_hypotheses": [
        "Seed-region mismatches disproportionately reduce Cas9 cleavage fidelity."
    ],
}


def _mock_llm(payload: dict) -> str:
    """Return a clean JSON string as the LLM would."""
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# 1. Valid plain JSON response
# ---------------------------------------------------------------------------

def test_valid_json_produces_research_plan():
    planner = ResearchPlanner()
    with patch("app.agents.research_planner.ask_llm", return_value=_mock_llm(VALID_PAYLOAD)):
        plan = planner.create_plan("How does CRISPR-Cas9 achieve specificity?")

    assert isinstance(plan, ResearchPlan)
    assert plan.topic == "CRISPR-Cas9 Genome Editing Specificity"
    assert len(plan.key_questions) == 2
    assert len(plan.sub_questions) == 2
    assert len(plan.information_needed) == 2
    assert len(plan.initial_hypotheses) == 1


# ---------------------------------------------------------------------------
# 2. Markdown-fenced response is correctly stripped
# ---------------------------------------------------------------------------

def test_markdown_fenced_json_is_parsed():
    fenced = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    planner = ResearchPlanner()
    with patch("app.agents.research_planner.ask_llm", return_value=fenced):
        plan = planner.create_plan("How does CRISPR-Cas9 achieve specificity?")

    assert plan.topic == "CRISPR-Cas9 Genome Editing Specificity"


# ---------------------------------------------------------------------------
# 3. JSON embedded in prose (brace extraction)
# ---------------------------------------------------------------------------

def test_json_embedded_in_prose_is_extracted():
    prose = f"Sure! Here is your plan:\n{json.dumps(VALID_PAYLOAD)}\nLet me know if you need more."
    planner = ResearchPlanner()
    with patch("app.agents.research_planner.ask_llm", return_value=prose):
        plan = planner.create_plan("How does CRISPR-Cas9 achieve specificity?")

    assert plan.topic == "CRISPR-Cas9 Genome Editing Specificity"


# ---------------------------------------------------------------------------
# 4. Completely invalid response raises PlanningError with diagnostics
# ---------------------------------------------------------------------------

def test_invalid_response_raises_planning_error():
    planner = ResearchPlanner()
    with patch("app.agents.research_planner.ask_llm", return_value="Sorry, I cannot help with that."):
        with pytest.raises(PlanningError) as exc_info:
            planner.create_plan("Irrelevant question")

    assert "LLM did not return valid JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. ResearchQuestion object propagates question_id
# ---------------------------------------------------------------------------

def test_research_question_id_is_propagated():
    rq = ResearchQuestion(question="What causes antibiotic resistance?")
    planner = ResearchPlanner()
    with patch("app.agents.research_planner.ask_llm", return_value=_mock_llm(VALID_PAYLOAD)):
        plan = planner.create_plan(rq)

    assert plan.research_question_id == rq.id


# ---------------------------------------------------------------------------
# 6. Domain is included in the prompt sent to the LLM
# ---------------------------------------------------------------------------

def test_domain_is_included_in_prompt():
    captured_prompt = {}

    def fake_llm(prompt: str, model=None) -> str:
        captured_prompt["text"] = prompt
        return _mock_llm(VALID_PAYLOAD)

    planner = ResearchPlanner()
    with patch("app.agents.research_planner.ask_llm", side_effect=fake_llm):
        planner.create_plan("What causes antibiotic resistance?", domain="microbiology")

    assert "microbiology" in captured_prompt["text"]


# ---------------------------------------------------------------------------
# 7. plan_research() convenience wrapper works correctly
# ---------------------------------------------------------------------------

def test_plan_research_convenience_function():
    with patch("app.agents.research_planner.ask_llm", return_value=_mock_llm(VALID_PAYLOAD)):
        plan = plan_research("How does CRISPR-Cas9 achieve specificity?")

    assert isinstance(plan, ResearchPlan)
    assert len(plan.key_questions) >= 2
