"""Text, Markdown, and HTML ingestion for ARGUS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

import bs4

from app.ingestion.base import BaseIngester, ContentSegment, IngestionResult
from app.models.research import Source, SourceType


class TextDocumentIngester(BaseIngester):
    """Ingests plain text, Markdown, and local HTML files."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm"}

    def can_ingest(self, target: Union[str, Path]) -> bool:
        path = Path(target)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS or str(target).startswith(("<html", "<!doctype"))

    def ingest(self, target: Union[str, Path], **kwargs) -> IngestionResult:
        if isinstance(target, str) and target.strip().startswith(("<html", "<!doctype")):
            content = target
            title = kwargs.get("title", "HTML Document")
            url = kwargs.get("url")
            ext = ".html"
        else:
            path = Path(target)
            if not path.exists():
                raise FileNotFoundError(f"Document not found at {path}")
            content = path.read_text(encoding="utf-8", errors="replace")
            title = kwargs.get("title", path.stem)
            url = kwargs.get("url", str(path.resolve()))
            ext = path.suffix.lower()

        segments: List[ContentSegment] = []

        if ext in (".html", ".htm"):
            soup = bs4.BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Extract headings, paragraphs, and tables as distinct locators
            current_section = "Introduction"
            for element in soup.find_all(["h1", "h2", "h3", "p", "table"]):
                text = element.get_text(separator=" ", strip=True)
                if not text:
                    continue
                if element.name in ("h1", "h2", "h3"):
                    current_section = text
                else:
                    loc = f"Section: {current_section}"
                    if element.name == "table":
                        loc += " (Table)"
                    segments.append(ContentSegment(text=text, location=loc))
        else:
            # Markdown / plain text: segment by paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs, start=1):
                segments.append(ContentSegment(text=para, location=f"Paragraph {idx}"))

        source = Source(
            url=url,
            title=title,
            source_type=SourceType.DOCUMENT,
            page_text="\n\n".join(s.text for s in segments),
        )

        return IngestionResult(source=source, segments=segments)
