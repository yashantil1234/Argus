"""Tools module: External retrieval, search, and content extraction tools."""

from app.tools.web_search import fetch_page_content, search_web

__all__ = ["search_web", "fetch_page_content"]
