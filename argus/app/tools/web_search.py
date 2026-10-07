"""Web search and page content retrieval tools for ARGUS research pipeline."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any, List, Optional

import bs4
import httpx

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def search_web(query: str, max_results: int = 3, timeout: float = 10.0) -> List[dict[str, str]]:
    """Search the web for a query and return top results.

    Returns a list of dicts with keys:
        - title: Title of the web page
        - url: Direct destination URL
        - snippet: Short text summary/snippet
    """
    logger.info("Searching web for query: %r (max_results=%d)", query, max_results)
    results: List[dict[str, str]] = []

    try:
        response = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()

        soup = bs4.BeautifulSoup(response.text, "html.parser")
        for res in soup.select(".result"):
            title_tag = res.select_one(".result__title a")
            snippet_tag = res.select_one(".result__snippet")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            raw_url = title_tag.get("href", "")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            # Unwrap DuckDuckGo redirect URL
            actual_url = _unwrap_ddg_url(raw_url)

            if actual_url and actual_url.startswith("http"):
                results.append({
                    "title": title,
                    "url": actual_url,
                    "snippet": snippet,
                })
                if len(results) >= max_results:
                    break

    except Exception as exc:
        logger.warning("Web search failed for query %r: %s", query, exc)

    return results


def _unwrap_ddg_url(url: str) -> str:
    """Extract destination URL from DuckDuckGo wrapper."""
    if "uddg=" in url:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return qs["uddg"][0]
    return url


def fetch_page_content(url: str, max_chars: int = 4000, timeout: float = 10.0) -> Optional[str]:
    """Fetch and clean the visible main text of a web page.

    Removes scripts, styles, navigation, headers, and footers.
    Truncates to `max_chars` to fit within small local model context windows.
    """
    logger.info("Fetching page content from: %s", url)
    try:
        response = httpx.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()

        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # Strip unneeded noise
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()

        # Extract text from paragraph and article elements if available
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "article"])]
        full_text = "\n\n".join(p for p in paragraphs if len(p) > 20)

        # Fallback to general text if no substantial paragraphs were found
        if len(full_text) < 100:
            full_text = soup.get_text(" ", strip=True)

        # Clean multiple spaces / newlines
        cleaned = " ".join(full_text.split())

        if not cleaned:
            return None

        # Truncate safely
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "..."

        return cleaned

    except Exception as exc:
        logger.warning("Failed to fetch page %s: %s", url, exc)
        return None
