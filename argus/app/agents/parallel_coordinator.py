"""Parallel Research Coordinator for ARGUS (Layer 3).

Coordinates concurrent, independent research workers across subtasks with:
  1. Deterministic Subtask IDs (e.g. `subtask_{plan_id[:8]}_{idx:02d}`).
  2. Bounded Concurrency: ThreadPoolExecutor with controlled worker limits.
  3. Shared Quota Awareness: All worker requests funnel through the shared LLMPool.
  4. Deduplication & Reconciliation: Merges multi-worker results via EvidenceMerger.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Callable, List, Optional

from app.agents.evidence_merger import merge_research_results
from app.agents.researcher import Researcher
from app.models.research import ResearchPlan, ResearchResult, Source, SourceType

logger = logging.getLogger(__name__)


class ParallelResearchCoordinator:
    """Orchestrates concurrent research workers over a ResearchPlan."""

    def __init__(
        self,
        model: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.model = model
        env_workers = os.getenv("ARGUS_PARALLEL_WORKERS", "")
        self.max_workers = (
            int(env_workers)
            if env_workers.isdigit() and int(env_workers) > 0
            else (max_workers or 4)
        )
        self.researcher = Researcher(model=model)

    def execute_subtasks(
        self,
        plan: ResearchPlan,
        sources_with_text: List[tuple[Source, str]],
    ) -> ResearchResult:
        """Execute independent extraction workers concurrently across candidate sources.

        Args:
            plan: The master ResearchPlan.
            sources_with_text: List of (Source, source_text) pairs to analyze.

        Returns:
            Unified, deduplicated ResearchResult artifact.
        """
        if not sources_with_text:
            return ResearchResult(research_plan_id=plan.id)

        worker_results: List[ResearchResult] = []
        bounded_workers = min(len(sources_with_text), self.max_workers)

        logger.info(
            "[Coordinator] Launching %d parallel workers for plan '%s' (max_concurrency=%d)",
            len(sources_with_text), plan.topic, bounded_workers,
        )

        def _worker_task(idx: int, src: Source, txt: str) -> ResearchResult:
            subtask_id = f"subtask_{plan.id[:8]}_{idx:02d}"
            logger.debug("[Worker %s] Starting extraction on '%s'", subtask_id, src.title)
            try:
                result = self.researcher.extract(plan=plan, source_text=txt, source=src)
                logger.debug(
                    "[Worker %s] Completed: %d claims, %d evidence",
                    subtask_id, len(result.claims), len(result.evidence),
                )
                return result
            except Exception as exc:
                logger.warning("[Worker %s] Extraction failed for '%s': %s", subtask_id, src.title, exc)
                return ResearchResult(research_plan_id=plan.id, sources=[src])

        with concurrent.futures.ThreadPoolExecutor(max_workers=bounded_workers) as executor:
            future_to_idx = {
                executor.submit(_worker_task, idx, src, txt): idx
                for idx, (src, txt) in enumerate(sources_with_text, start=1)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                res = future.result()
                worker_results.append(res)

        # Merge and deduplicate across all worker outputs
        return merge_research_results(worker_results, plan_id=plan.id)
