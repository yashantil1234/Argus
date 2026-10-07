"""Research Planner Agent.

Turns a ResearchQuestion into a validated ResearchPlan by prompting the LLM
for structured JSON and parsing it with Pydantic.

Pipeline:
    ResearchQuestion
          ↓
      _build_prompt()
          ↓
       ask_llm()          ← provider-agnostic (Ollama / OpenRouter)
          ↓
      raw LLM text
          ↓
      _extract_json()     ← strips markdown fences, finds JSON block
          ↓
      json.loads()
          ↓
    ResearchPlan.model_validate()
          ↓
      ResearchPlan  ✅   (or PlanningError on failure)
"""

from __future__ import annotations

import json
import re
import logging
from typing import Optional

from app.core.llm import ask_llm
from app.models.research import ResearchPlan, ResearchQuestion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception — raised when planning fails so callers can decide
# ---------------------------------------------------------------------------

class PlanningError(Exception):
    """Raised when the Planner cannot produce a valid ResearchPlan."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """\
You are a rigorous academic research planner. Your only job is to decompose \
a research question into a structured plan.

Research Question: "{question}"{domain_line}

You MUST respond with ONLY a valid JSON object — no prose, no explanation, \
no markdown — matching this exact schema:

{{
  "topic": "A concise title for the research topic (string)",
  "key_questions": [
    "Specific sub-question 1 that must be answered?",
    "Specific sub-question 2 that must be answered?"
  ],
  "sub_questions": [
    "A more granular breakdown of sub-question 1",
    "A more granular breakdown of sub-question 2"
  ],
  "information_needed": [
    "Type of source, data, or literature needed",
    "Another type of source or data"
  ],
  "initial_hypotheses": [
    "A testable working hypothesis based on the question"
  ]
}}

Rules:
- Return ONLY the JSON object. Nothing before it, nothing after it.
- All arrays must have at least 2 items.
- Do not include any explanation or reasoning outside the JSON.
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ResearchPlanner:
    """Agent that decomposes a research question into a structured ResearchPlan."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model  # None → resolved from DEFAULT_LLM_MODEL in .env

    def create_plan(
        self,
        question: str | ResearchQuestion,
        domain: Optional[str] = None,
    ) -> ResearchPlan:
        """Generate a validated ResearchPlan.

        Raises:
            PlanningError: If the LLM response cannot be parsed into a valid plan.
        """
        q_text, q_id, domain = self._resolve_question(question, domain)
        prompt = self._build_prompt(q_text, domain)

        logger.info("Planner: calling LLM for question=%r model=%s", q_text, self.model)
        raw = ask_llm(prompt, model=self.model)
        logger.debug("Planner: raw LLM response:\n%s", raw)

        return self._parse_response(raw, q_text, q_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_question(
        self,
        question: str | ResearchQuestion,
        domain: Optional[str],
    ) -> tuple[str, Optional[str], Optional[str]]:
        if isinstance(question, ResearchQuestion):
            return question.question, question.id, domain or question.domain
        return question, None, domain

    def _build_prompt(self, question: str, domain: Optional[str]) -> str:
        domain_line = f'\nDomain: {domain}' if domain else ""
        return _PLAN_PROMPT.format(question=question, domain_line=domain_line)

    def _extract_json(self, text: str) -> str:
        """Extract raw JSON from LLM output, stripping markdown fences if present."""
        text = text.strip()

        # Case 1: wrapped in ```json ... ``` or ``` ... ```
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fenced:
            return fenced.group(1).strip()

        # Case 2: find first { ... } block
        brace_match = re.search(r"\{[\s\S]+\}", text)
        if brace_match:
            return brace_match.group(0).strip()

        return text  # let json.loads produce a useful error

    def _parse_response(
        self, raw: str, question_text: str, question_id: Optional[str]
    ) -> ResearchPlan:
        """Parse raw LLM text into a validated ResearchPlan."""
        # Step 1: extract JSON
        try:
            json_str = self._extract_json(raw)
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise PlanningError(
                f"LLM did not return valid JSON.\n"
                f"Error: {exc}\n"
                f"Raw response:\n{raw}"
            ) from exc

        # Step 2: Pydantic validation
        try:
            plan = ResearchPlan.model_validate(
                {
                    "research_question_id": question_id,
                    "topic": data.get("topic", question_text),
                    "key_questions": data.get("key_questions", []),
                    "sub_questions": data.get("sub_questions", []),
                    "information_needed": data.get("information_needed", []),
                    "initial_hypotheses": data.get("initial_hypotheses", []),
                }
            )
        except Exception as exc:
            raise PlanningError(
                f"LLM JSON failed Pydantic validation.\n"
                f"Error: {exc}\n"
                f"Parsed data: {data}"
            ) from exc

        logger.info(
            "Planner: plan created topic=%r questions=%d",
            plan.topic,
            len(plan.key_questions),
        )
        return plan


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def plan_research(
    question: str | ResearchQuestion,
    domain: Optional[str] = None,
    model: str | None = None,
) -> ResearchPlan:
    """Create a ResearchPlan for the given question.

    Raises:
        PlanningError: if the LLM response cannot be parsed.
        ConnectionError: if the LLM backend is unreachable.
    """
    planner = ResearchPlanner(model=model)
    return planner.create_plan(question=question, domain=domain)
