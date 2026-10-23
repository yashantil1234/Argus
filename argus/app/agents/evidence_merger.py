"""Evidence and Claim Deduplication Merger for ARGUS.

Consolidates outputs from multiple parallel research workers into a single,
coherent ResearchResult. Ensures:
  1. Deduplication of identical sources by URL or content hash.
  2. Deduplication of evidence quotes.
  3. Clustering/merging of redundant claims.
  4. Repair and re-wiring of claim -> evidence -> source reference IDs.
  5. Accurate deterministic recomputation of Quote Fidelity.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Set

from app.core.provenance import normalize_text, verify_quote_in_text
from app.models.research import Claim, Evidence, ResearchResult, Source

logger = logging.getLogger(__name__)


def merge_research_results(
    results: List[ResearchResult],
    plan_id: str,
) -> ResearchResult:
    """Consolidate multiple ResearchResult artifacts into one deduplicated result.

    Args:
        results: List of ResearchResult instances from individual workers.
        plan_id: ID of the overarching ResearchPlan.

    Returns:
        Unified ResearchResult with clean FK relationships and deduplicated claims.
    """
    if not results:
        return ResearchResult(research_plan_id=plan_id)

    # -------------------------------------------------------------------------
    # 1. Deduplicate Sources
    # -------------------------------------------------------------------------
    surviving_sources: List[Source] = []
    source_id_remap: Dict[str, str] = {}  # old_source_id -> new_source_id
    seen_urls: Dict[str, str] = {}        # normalized_url -> surviving_source_id
    seen_hashes: Dict[str, str] = {}      # content_hash -> surviving_source_id

    for r in results:
        for src in r.sources:
            norm_url = (src.url or "").strip().lower()
            chash = src.content_hash or ""

            # Check if this source matches an existing one by URL or Content Hash
            matched_id: Optional[str] = None
            if norm_url and norm_url in seen_urls:
                matched_id = seen_urls[norm_url]
            elif chash and chash in seen_hashes:
                matched_id = seen_hashes[chash]

            if matched_id:
                source_id_remap[src.id] = matched_id
            else:
                surviving_sources.append(src)
                source_id_remap[src.id] = src.id
                if norm_url:
                    seen_urls[norm_url] = src.id
                if chash:
                    seen_hashes[chash] = src.id

    # -------------------------------------------------------------------------
    # 2. Deduplicate Evidence Quotes
    # -------------------------------------------------------------------------
    surviving_evidence: List[Evidence] = []
    evidence_id_remap: Dict[str, str] = {}   # old_ev_id -> surviving_ev_id
    # (remapped_source_id, normalized_quote) -> surviving_evidence_id
    seen_quotes: Dict[tuple[str, str], str] = {}

    for r in results:
        for ev in r.evidence:
            remapped_src_id = source_id_remap.get(ev.source_id, ev.source_id)
            norm_q = normalize_text(ev.quote)
            key = (remapped_src_id, norm_q)

            if key in seen_quotes:
                surviving_id = seen_quotes[key]
                evidence_id_remap[ev.id] = surviving_id
            else:
                ev.source_id = remapped_src_id
                surviving_evidence.append(ev)
                evidence_id_remap[ev.id] = ev.id
                seen_quotes[key] = ev.id

    # -------------------------------------------------------------------------
    # 3. Deduplicate and Cluster Claims
    # -------------------------------------------------------------------------
    surviving_claims: List[Claim] = []
    # normalized_claim_text -> Claim
    seen_claims: Dict[str, Claim] = {}

    for r in results:
        for cl in r.claims:
            norm_c = normalize_text(cl.claim_text)
            # Remap evidence IDs to surviving evidence items
            remapped_eids = [
                evidence_id_remap[eid]
                for eid in cl.evidence_ids
                if eid in evidence_id_remap
            ]

            if norm_c in seen_claims:
                existing = seen_claims[norm_c]
                # Merge evidence references without duplicates
                combined_eids = list(dict.fromkeys(existing.evidence_ids + remapped_eids))
                existing.evidence_ids = combined_eids
                existing.confidence = round(max(existing.confidence, cl.confidence), 3)
            else:
                cl.evidence_ids = list(dict.fromkeys(remapped_eids))
                surviving_claims.append(cl)
                seen_claims[norm_c] = cl

    consolidated = ResearchResult(
        research_plan_id=plan_id,
        sources=surviving_sources,
        evidence=surviving_evidence,
        claims=surviving_claims,
    )

    logger.info(
        "[EvidenceMerger] Merged %d worker results -> sources=%d evidence=%d claims=%d (fidelity=%.2f)",
        len(results),
        len(consolidated.sources),
        len(consolidated.evidence),
        len(consolidated.claims),
        consolidated.quote_fidelity,
    )

    return consolidated
