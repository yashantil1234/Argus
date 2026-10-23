"""ARGUS Multimodal Ingestion Engine."""

from app.ingestion.base import BaseIngester, ContentSegment, IngestionResult
from app.ingestion.media import AudioVideoIngester, ImageIngester
from app.ingestion.pdf import PDFIngester
from app.ingestion.router import ingest
from app.ingestion.text_doc import TextDocumentIngester

__all__ = [
    "BaseIngester",
    "ContentSegment",
    "IngestionResult",
    "PDFIngester",
    "ImageIngester",
    "AudioVideoIngester",
    "TextDocumentIngester",
    "ingest",
]
