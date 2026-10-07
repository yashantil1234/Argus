"""Researcher Agent — Stage 4A (extraction) + Stage 4B (web research).

Pipeline:
    ResearchPlan
         ↓
    Web Search (Stage 4B)
         ↓
    Fetch Page Content
         ↓
    _build_prompt()
         ↓
      ask_llm()           ← provider-agnostic (Ollama / OpenRouter)
         ↓
      raw JSON
         ↓
    _extract_json()       ← strip fences, find { } block
         ↓
    json.loads()
         ↓
   _validate_payload()    ← check structure + index bounds
         ↓
   _assemble_result()     ← Python generates all IDs + wires relationships
         ↓
    ResearchResult        ← clean, traceable, no invented provenance
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, List, Optional

from app.core.llm import ask_llm
from app.models.research import (
    Claim,
    Evidence,
    ResearchPlan,
    ResearchResult,
    Source,
    SourceType,
)
from app.core.provenance import verify_quote_in_text
from app.tools.web_search import fetch_page_content, search_web

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Raised when the Researcher cannot parse a valid ResearchResult."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a rigorous research evidence extractor. Your task is to read the \
supplied source text and extract factual claims that are directly supported \
by the text, in the context of the research plan provided.

Research Topic: {topic}

Research Questions:
{questions}

Source Title: {source_title}
Source URL: {source_url}

--- BEGIN SOURCE TEXT ---
{source_text}
--- END SOURCE TEXT ---

Instructions:
1. Extract ONLY claims that are directly and explicitly supported by the text above.
2. Every "quote" must be copied VERBATIM from the source text. Do not paraphrase.
3. Do not use your general knowledge. If the source does not support a claim, do not make it.
4. If the source contains no relevant evidence, return empty arrays.
5. "evidence_indices" must refer to array indices of the "evidence" list you return (0-based).
6. Return ONLY a valid JSON object. No markdown, no prose, no explanation.

Return this exact JSON schema:
{{
  "evidence": [
    {{
      "quote": "verbatim text copied from the source",
      "relevance": 0.95,
      "page": null,
      "location": "section or paragraph identifier, or null"
    }}
  ],
  "claims": [
    {{
      "claim_text": "A factual claim directly supported by the evidence",
      "confidence": 0.9,
      "evidence_indices": [0]
    }}
  ]
}}

Rules:
- "relevance" and "confidence" are floats between 0.0 and 1.0.
- "evidence_indices" contains only valid indices into the "evidence" array you return.
- If no relevant evidence exists, return: {{"evidence": [], "claims": []}}
- Return ONLY the JSON. Nothing before it. Nothing after it.
"""


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

class Researcher:
    """Extracts Claims and Evidence from sources using an LLM."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model  # None → resolved from DEFAULT_LLM_MODEL in .env

    def extract(
        self,
        plan: ResearchPlan,
        source_text: str,
        source: Source,
    ) -> ResearchResult:
        """Stage 4A: Extract Claims and Evidence from a single source text.

        Args:
            plan: The ResearchPlan defining what to look for.
            source_text: Raw text content of the source (article, paper, page).
            source: Source metadata (title, URL, type). Must already have a system ID.

        Returns:
            ResearchResult with all IDs system-generated and relationships wired.

        Raises:
            ExtractionError: If the LLM response cannot be parsed or validated.
        """
        prompt = self._build_prompt(plan, source_text, source)

        logger.info(
            "Researcher: extracting from source=%r plan_topic=%r model=%s",
            source.title, plan.topic, self.model,
        )
        raw = ask_llm(prompt, model=self.model)
        logger.debug("Researcher: raw LLM response:\n%s", raw)

        payload = self._parse_and_validate(raw)
        if not source.page_text and source_text:
            source.page_text = source_text
        result = self._assemble_result(payload, plan, source, source_text=source_text)

        logger.info(
            "Researcher: extracted claims=%d evidence=%d",
            len(result.claims), len(result.evidence),
        )
        return result

    def research(
        self,
        plan: ResearchPlan,
        max_sources: int = 2,
    ) -> ResearchResult:
        """Stage 4B: Autonomous web search + extraction pipeline.

        1. Searches the web for plan queries.
        2. Retrieves and cleans page text.
        3. Extracts evidence and claims per source.
        4. Consolidates into a unified ResearchResult.
        """
        queries = plan.key_questions[:max_sources] or [plan.topic]
        logger.info("Researcher: starting web research with queries=%r", queries)

        discovered_candidates: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        # Step 1: Retrieve search results
        for q in queries:
            results = search_web(q, max_results=2)
            for item in results:
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    discovered_candidates.append(item)
                    if len(discovered_candidates) >= max_sources:
                        break
            if len(discovered_candidates) >= max_sources:
                break

        # Step 2: Fetch and extract per candidate source
        all_sources: list[Source] = []
        all_evidence: list[Evidence] = []
        all_claims: list[Claim] = []

        for candidate in discovered_candidates[:max_sources]:
            source = Source(
                url=candidate["url"],
                title=candidate.get("title", "Web Source"),
                source_type=SourceType.WEB,
            )

            # Try to fetch live webpage text, fallback to snippet
            content = fetch_page_content(candidate["url"])
            if not content or len(content.strip()) < 50:
                content = candidate.get("snippet", "")

            if not content:
                continue

            try:
                sub_res = self.extract(plan=plan, source_text=content, source=source)
                all_sources.append(source)
                all_evidence.extend(sub_res.evidence)
                all_claims.extend(sub_res.claims)
            except Exception as exc:
                logger.warning(
                    "Extraction failed for source %s: %s", source.url, exc
                )

        return ResearchResult(
            research_plan_id=plan.id,
            sources=all_sources,
            evidence=all_evidence,
            claims=all_claims,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self, plan: ResearchPlan, source_text: str, source: Source
    ) -> str:
        questions = "\n".join(f"- {q}" for q in plan.key_questions)
        return _EXTRACTION_PROMPT.format(
            topic=plan.topic,
            questions=questions,
            source_title=source.title,
            source_url=source.url or "not available",
            source_text=source_text.strip(),
        )

    def _extract_json(self, text: str) -> str:
        """Extract raw JSON from LLM output, stripping markdown fences if present."""
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fenced:
            return fenced.group(1).strip()
        brace_match = re.search(r"\{[\s\S]+\}", text)
        if brace_match:
            return brace_match.group(0).strip()
        return text

    def _parse_and_validate(self, raw: str) -> dict[str, Any]:
        """Parse raw LLM text and validate the payload structure."""
        try:
            json_str = self._extract_json(raw)
            payload = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"LLM did not return valid JSON.\nError: {exc}\nRaw:\n{raw}"
            ) from exc

        if not isinstance(payload.get("evidence"), list):
            raise ExtractionError(
                f"Payload missing 'evidence' list.\nParsed: {payload}"
            )
        if not isinstance(payload.get("claims"), list):
            raise ExtractionError(
                f"Payload missing 'claims' list.\nParsed: {payload}"
            )

        evidence_count = len(payload["evidence"])
        
        # Detect if model used 1-based indexing (e.g. indices 1..N instead of 0..N-1)
        all_raw_indices = [
            idx
            for cl in payload["claims"]
            for idx in (cl.get("evidence_indices") if isinstance(cl.get("evidence_indices"), list) else [])
            if isinstance(idx, int)
        ]
        is_one_based = (
            evidence_count > 0
            and bool(all_raw_indices)
            and min(all_raw_indices) >= 1
            and max(all_raw_indices) == evidence_count
        )

        for i, claim in enumerate(payload["claims"]):
            raw_indices = claim.get("evidence_indices", [])
            if not isinstance(raw_indices, list):
                claim["evidence_indices"] = []
                continue

            valid_indices = []
            for idx in raw_indices:
                if not isinstance(idx, int):
                    continue
                norm_idx = idx - 1 if is_one_based else idx
                if 0 <= norm_idx < evidence_count:
                    valid_indices.append(norm_idx)
                else:
                    logger.warning(
                        "Claim[%d] has out-of-bounds evidence_index %d (evidence count: %d). Dropping index.",
                        i, idx, evidence_count
                    )
            claim["evidence_indices"] = valid_indices

        return payload

    def _assemble_result(
        self,
        payload: dict[str, Any],
        plan: ResearchPlan,
        source: Source,
        source_text: str = "",
    ) -> ResearchResult:
        """Build a ResearchResult — Python generates all IDs, deterministically verifies quotes, and wires relationships."""
        evidence_objects: list[Evidence] = []
        for raw_ev in payload["evidence"]:
            raw_quote = raw_ev.get("quote", "")
            is_verb, score = (
                verify_quote_in_text(raw_quote, source_text)
                if source_text
                else (True, 1.0)
            )
            if not is_verb and source_text:
                logger.warning(
                    "Deterministic quote verification failed for quote %r (score=%.2f) from source %s",
                    raw_quote[:60],
                    score,
                    source.url,
                )
            ev = Evidence(
                source_id=source.id,
                quote=raw_quote,
                relevance=float(raw_ev.get("relevance", 1.0)),
                page=raw_ev.get("page"),
                location=raw_ev.get("location"),
                is_verbatim=is_verb,
                match_score=score,
            )
            evidence_objects.append(ev)

        claim_objects: list[Claim] = []
        for raw_cl in payload["claims"]:
            indices = raw_cl.get("evidence_indices", [])
            evidence_ids = [evidence_objects[idx].id for idx in indices]
            cl = Claim(
                claim_text=raw_cl.get("claim_text", ""),
                evidence_ids=evidence_ids,
                confidence=float(raw_cl.get("confidence", 1.0)),
            )
            claim_objects.append(cl)

        return ResearchResult(
            research_plan_id=plan.id,
            sources=[source],
            evidence=evidence_objects,
            claims=claim_objects,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def extract_from_source(
    plan: ResearchPlan,
    source_text: str,
    source: Source,
    model: str | None = None,
) -> ResearchResult:
    """Extract claims and evidence from a single source."""
    researcher = Researcher(model=model)
    return researcher.extract(plan=plan, source_text=source_text, source=source)


def conduct_research(
    plan: ResearchPlan,
    max_sources: int = 2,
    model: str | None = None,
) -> ResearchResult:
    """Perform autonomous web research for a plan."""
    researcher = Researcher(model=model)
    return researcher.research(plan=plan, max_sources=max_sources)
