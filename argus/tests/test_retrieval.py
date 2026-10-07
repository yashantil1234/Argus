"""Tests for the web retrieval layer and Researcher multi-source orchestration (Stage 4B).

All tests are fully mocked — no live external HTTP requests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.agents.researcher import Researcher, conduct_research
from app.models.research import ResearchPlan, ResearchResult
from app.tools.web_search import _unwrap_ddg_url, fetch_page_content, search_web


# ---------------------------------------------------------------------------
# 1. URL Unwrapping
# ---------------------------------------------------------------------------

def test_unwrap_ddg_url():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nature.com%2Farticles%2Fnature123&rut=xyz"
    unwrapped = _unwrap_ddg_url(wrapped)
    assert unwrapped == "https://www.nature.com/articles/nature123"

    direct = "https://example.org/study"
    assert _unwrap_ddg_url(direct) == direct


# ---------------------------------------------------------------------------
# 2. search_web parsing
# ---------------------------------------------------------------------------

MOCK_DDG_HTML = """
<html>
<body>
  <div class="result">
    <h2 class="result__title">
      <a href="https://example.com/crispr-study">CRISPR Mechanics and Specificity</a>
    </h2>
    <div class="result__snippet">
      Comprehensive overview of Cas9 PAM binding motifs and cleavage precision.
    </div>
  </div>
  <div class="result">
    <h2 class="result__title">
      <a href="https://example.com/off-target-study">Off-Target Cleavage Profiling</a>
    </h2>
    <div class="result__snippet">
      Genome-wide DSB detection assays for modified guide RNAs.
    </div>
  </div>
</body>
</html>
"""


def test_search_web_parses_results():
    with patch("app.tools.web_search.httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_DDG_HTML
        mock_post.return_value = mock_resp

        results = search_web("CRISPR Cas9 specificity", max_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "CRISPR Mechanics and Specificity"
    assert results[0]["url"] == "https://example.com/crispr-study"
    assert "PAM binding" in results[0]["snippet"]


# ---------------------------------------------------------------------------
# 3. fetch_page_content stripping and truncation
# ---------------------------------------------------------------------------

MOCK_PAGE_HTML = """
<html>
<head><style>.ad { color: red; }</style></head>
<body>
  <nav><a href="/">Home</a></nav>
  <script>console.log("tracking");</script>
  <article>
    <p>CRISPR-Cas9 acts as RNA-guided endonuclease recognizing specific target genomic sites.</p>
    <p>Target recognition strictly requires an adjacent protospacer adjacent motif.</p>
  </article>
  <footer>Copyright 2026</footer>
</body>
</html>
"""


def test_fetch_page_content_cleans_noise():
    with patch("app.tools.web_search.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_PAGE_HTML
        mock_get.return_value = mock_resp

        content = fetch_page_content("https://example.com/article", max_chars=500)

    assert content is not None
    assert "RNA-guided endonuclease" in content
    assert "protospacer adjacent motif" in content
    assert "tracking" not in content  # script stripped
    assert "Copyright" not in content  # footer stripped


# ---------------------------------------------------------------------------
# 4. Multi-source Researcher orchestration (Stage 4B)
# ---------------------------------------------------------------------------

MOCK_EXTRACTION_JSON = """{
  "evidence": [
    {
      "quote": "Target recognition strictly requires an adjacent protospacer adjacent motif.",
      "relevance": 0.95,
      "page": null,
      "location": "p2"
    }
  ],
  "claims": [
    {
      "claim_text": "Cas9 target recognition requires an adjacent PAM motif.",
      "confidence": 0.96,
      "evidence_indices": [0]
    }
  ]
}"""


def test_researcher_multi_source_orchestration():
    plan = ResearchPlan(
        topic="CRISPR Specificity",
        key_questions=["How does PAM recognition work?"],
    )

    mock_search_results = [
        {"title": "Study 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
        {"title": "Study 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
    ]

    with patch("app.agents.researcher.search_web", return_value=mock_search_results), \
         patch("app.agents.researcher.fetch_page_content", return_value="Target recognition strictly requires an adjacent protospacer adjacent motif."), \
         patch("app.agents.researcher.ask_llm", return_value=MOCK_EXTRACTION_JSON):

        researcher = Researcher()
        result = researcher.research(plan=plan, max_sources=2)

    assert isinstance(result, ResearchResult)
    assert len(result.sources) == 2
    assert len(result.evidence) == 2
    assert len(result.claims) == 2

    # Verify traceability across all claims
    evidence_ids = {ev.id for ev in result.evidence}
    for claim in result.claims:
        for eid in claim.evidence_ids:
            assert eid in evidence_ids
