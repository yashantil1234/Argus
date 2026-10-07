"""Verifier Agent — Stage 5: Independent Adversarial Verification.

Pipeline:
    Claim + Evidence Quotes
             ↓
      _build_prompt()     ← V1 (strict literal) or V2 (calibrated semantic entailment)
             ↓
        ask_llm()         ← provider-agnostic (Ollama / OpenRouter)
             ↓
        raw JSON
             ↓
      _extract_json()     ← strip fences, extract { }
             ↓
      json.loads()
             ↓
    VerificationResult    ← Python sets system IDs, supported flag, timestamps
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from app.core.llm import ask_llm
from app.models.research import (
    Claim,
    Evidence,
    ResearchResult,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class VerificationError(Exception):
    """Raised when the Verifier cannot parse or produce a valid VerificationResult."""


# ---------------------------------------------------------------------------
# Prompt Templates: V1 (Strict Literal) & V2 (Calibrated Entailment)
# ---------------------------------------------------------------------------

_VERIFICATION_PROMPT_V1 = """\
You are an independent, adversarial fact-checking verifier in a scientific research system.
Your SOLE duty is to determine whether the provided evidence quotes explicitly support the claim.

Claim to evaluate:
"{claim_text}"

Provided Evidence Quotes:
{evidence_text}

Evaluation Rules:
1. DO NOT assume the claim is true.
2. DO NOT use your outside knowledge or general training data. Evaluate ONLY what is written in the quotes above. Even if the claim is scientifically true in the real world, if the quotes do not state it, you must NOT mark it VERIFIED.
3. Choose exactly one status:
   - "VERIFIED": The quotes explicitly, directly, and adequately prove the claim statement.
   - "REJECTED": The quotes directly contradict or refute the claim statement.
   - "UNCERTAIN": The quotes are tangential, weak, speculative, or insufficient to prove or disprove the claim.
4. "confidence" must be a float between 0.0 and 1.0 representing your certainty in this verdict.
5. "reasoning" must be a concise explanation (1-2 sentences) referencing the quote contents.
6. Return ONLY a valid JSON object matching this schema. No markdown fences, no explanatory prose before or after.

JSON Schema:
{{
  "status": "VERIFIED" | "REJECTED" | "UNCERTAIN",
  "confidence": 0.95,
  "reasoning": "Concise justification strictly based on the text."
}}
"""


_VERIFICATION_PROMPT_V2 = """\
You are an independent, rigorous fact-checking verifier in a scientific research system.
Your SOLE duty is to determine whether the provided evidence quotes factually substantiate the claim.

Claim to evaluate:
"{claim_text}"

Provided Evidence Quotes:
{evidence_text}

Core Evaluation Standard:
Test for logical entailment, NOT literal word-for-word string matching.
- Accept clear semantic entailment where synonyms or paraphrases convey the identical factual meaning (e.g. "substantially lower" entails "reduce", "canonical target recognition requires NGG" entails "specifically targets NGG").
- DO NOT infer unstated facts, relationships, populations, conditions, magnitude, causality, or scope.

Calibration Rules:
1. "VERIFIED": The quotes logically guarantee and substantiate the claim. Paraphrases with equivalent meaning are VERIFIED.
   Example: Quote: "Variants substantially lower off-target cleavage." -> Claim: "Variants reduce off-target cleavage." -> VERIFIED.
2. "REJECTED": The quotes directly contradict, negate, or disprove the claim.
   Example: Quote: "Activity is completely blocked by calcium." -> Claim: "Calcium enhances activity." -> REJECTED.
3. "UNCERTAIN": The quotes are insufficient, speculative, unproven, or overextended:
   - Scope overreach (Quote says "in mice" -> Claim says "in humans") -> UNCERTAIN.
   - Causal overreach (Quote says "correlated with" -> Claim says "causes") -> UNCERTAIN.
   - Magnitude exaggeration (Quote says "substantially lower" -> Claim says "eliminate") -> UNCERTAIN.
   - Speculation (Quote says "we speculate / may" -> Claim states as established fact) -> UNCERTAIN.
   - Missing or irrelevant evidence -> UNCERTAIN.
4. DO NOT use outside world knowledge. Judge solely on the provided quotes.
5. Return ONLY a valid JSON object matching the schema below. No markdown fences, no explanatory text outside JSON.

JSON Schema:
{{
  "status": "VERIFIED" | "REJECTED" | "UNCERTAIN",
  "confidence": 0.95,
  "reasoning": "Concise justification referencing text facts and logical relation."
}}
"""

# Default prompt alias
_VERIFICATION_PROMPT = _VERIFICATION_PROMPT_V2


# ---------------------------------------------------------------------------
# Verifier Agent
# ---------------------------------------------------------------------------

class Verifier:
    """Independent agent that evaluates Claims against their cited Evidence."""

    def __init__(self, model: str | None = None, prompt_version: str = "v2") -> None:
        self.model = model  # None → resolved from DEFAULT_LLM_MODEL in .env
        self.prompt_version = prompt_version.lower().strip()

    def verify_claim(
        self,
        claim: Claim,
        evidence_items: List[Evidence],
    ) -> VerificationResult:
        """Evaluate whether the given evidence items adequately support a single claim.

        If no evidence items are provided, short-circuits to UNCERTAIN without an LLM call.
        """
        # Rule: Missing evidence is immediately UNCERTAIN
        if not evidence_items:
            logger.info("Verifier: claim %s has no evidence. Marking UNCERTAIN.", claim.id)
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.UNCERTAIN,
                confidence=1.0,
                reasoning="No supporting evidence was provided for this claim.",
                evidence_ids=[],
            )

        prompt = self._build_prompt(claim, evidence_items)

        logger.info(
            "Verifier (prompt=%s): evaluating claim %r against %d evidence quotes (model=%s)",
            self.prompt_version,
            claim.claim_text[:60],
            len(evidence_items),
            self.model,
        )
        raw = ask_llm(prompt, model=self.model)
        logger.debug("Verifier: raw response:\n%s", raw)

        payload = self._parse_and_validate(raw)

        return VerificationResult(
            claim_id=claim.id,
            status=payload["status"],
            confidence=payload["confidence"],
            reasoning=payload["reasoning"],
            evidence_ids=[ev.id for ev in evidence_items],
        )

    def verify_research_result(
        self,
        research_result: ResearchResult,
    ) -> VerificationReport:
        """Evaluate all claims in a ResearchResult artifact and produce a VerificationReport."""
        evidence_map = {ev.id: ev for ev in research_result.evidence}
        results: List[VerificationResult] = []

        for claim in research_result.claims:
            matched_evidence = [
                evidence_map[eid]
                for eid in claim.evidence_ids
                if eid in evidence_map
            ]
            v_res = self.verify_claim(claim=claim, evidence_items=matched_evidence)
            results.append(v_res)

        report = VerificationReport(
            research_result_id=research_result.id,
            results=results,
        )

        logger.info(
            "Verifier: verification complete. total=%d verified=%d rejected=%d uncertain=%d",
            len(results),
            report.verified_count,
            report.rejected_count,
            report.uncertain_count,
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, claim: Claim, evidence_items: List[Evidence]) -> str:
        quotes_formatted = "\n".join(
            f'[{i + 1}] "{ev.quote}"'
            for i, ev in enumerate(evidence_items)
        )
        template = (
            _VERIFICATION_PROMPT_V1
            if self.prompt_version == "v1"
            else _VERIFICATION_PROMPT_V2
        )
        return template.format(
            claim_text=claim.claim_text.strip(),
            evidence_text=quotes_formatted,
        )

    def _extract_json(self, text: str) -> str:
        """Strip markdown code blocks or extract JSON substring."""
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fenced:
            return fenced.group(1).strip()
        brace_match = re.search(r"\{[\s\S]+\}", text)
        if brace_match:
            return brace_match.group(0).strip()
        return text

    def _parse_and_validate(self, raw: str) -> dict[str, Any]:
        """Parse raw LLM response and extract status, confidence, and reasoning."""
        try:
            json_str = self._extract_json(raw)
            payload = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"Verifier LLM did not return valid JSON.\nError: {exc}\nRaw:\n{raw}"
            ) from exc

        raw_status = payload.get("status")
        if not raw_status or not isinstance(raw_status, str):
            raise VerificationError(
                f"Payload missing valid 'status' string.\nParsed: {payload}"
            )

        status_norm = raw_status.lower().strip()
        if status_norm in ("verified", "valid", "pass", "true"):
            status = VerificationStatus.VERIFIED
        elif status_norm in ("rejected", "invalid", "fail", "false"):
            status = VerificationStatus.REJECTED
        elif status_norm in ("uncertain", "partial", "unverifiable", "inconclusive"):
            status = VerificationStatus.UNCERTAIN
        else:
            raise VerificationError(
                f"Unrecognized verification status '{raw_status}'. "
                "Expected VERIFIED, REJECTED, or UNCERTAIN."
            )

        try:
            confidence = float(payload.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 1.0

        reasoning = str(payload.get("reasoning") or payload.get("reason") or "").strip()

        return {
            "status": status,
            "confidence": confidence,
            "reasoning": reasoning,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def verify_claim(
    claim: Claim,
    evidence_items: List[Evidence],
    model: str | None = None,
    prompt_version: str = "v2",
) -> VerificationResult:
    """Convenience helper to verify a single claim."""
    verifier = Verifier(model=model, prompt_version=prompt_version)
    return verifier.verify_claim(claim=claim, evidence_items=evidence_items)


def verify_research(
    research_result: ResearchResult,
    model: str | None = None,
    prompt_version: str = "v2",
) -> VerificationReport:
    """Convenience helper to verify an entire ResearchResult artifact."""
    verifier = Verifier(model=model, prompt_version=prompt_version)
    return verifier.verify_research_result(research_result=research_result)
