"""Ingestion Router for ARGUS multimodal processing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from app.ingestion.base import BaseIngester, ContentSegment, IngestionResult
from app.ingestion.media import AudioVideoIngester, ImageIngester
from app.ingestion.pdf import PDFIngester
from app.ingestion.text_doc import TextDocumentIngester

_INGESTERS: List[BaseIngester] = [
    PDFIngester(),
    ImageIngester(),
    AudioVideoIngester(),
    TextDocumentIngester(),
]


def ingest(target: Union[str, Path], **kwargs) -> IngestionResult:
    """Ingest a document or media asset using the appropriate registered ingester.

    Args:
        target: File path, HTML string, or raw text.
        **kwargs: Optional metadata (e.g. title, url, transcript, description).

    Returns:
        IngestionResult containing the Source model and list of granular ContentSegments.
    """
    for ingester in _INGESTERS:
        if ingester.can_ingest(target):
            return ingester.ingest(target, **kwargs)

    # Fallback to TextDocumentIngester
    return TextDocumentIngester().ingest(target, **kwargs)
