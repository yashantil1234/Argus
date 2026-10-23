"""Agents module: Autonomous research pipeline agents."""

from app.agents.evidence_merger import merge_research_results
from app.agents.parallel_coordinator import ParallelResearchCoordinator
from app.agents.research_planner import ResearchPlanner, plan_research
from app.agents.researcher import (
    ExtractionError,
    Researcher,
    conduct_research,
    extract_from_source,
)
from app.agents.verifier import (
    VerificationError,
    Verifier,
    verify_claim,
    verify_research,
)
from app.agents.writer import ResearchWriter

__all__ = [
    "ResearchPlanner",
    "plan_research",
    "Researcher",
    "ExtractionError",
    "extract_from_source",
    "conduct_research",
    "Verifier",
    "VerificationError",
    "verify_claim",
    "verify_research",
    "merge_research_results",
    "ParallelResearchCoordinator",
    "ResearchWriter",
]
