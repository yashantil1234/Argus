"""Unit tests for ARGUS Multimodal Ingestion Engine (Layer 2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.ingestion import (
    AudioVideoIngester,
    ImageIngester,
    PDFIngester,
    TextDocumentIngester,
    ingest,
)
from app.models.research import SourceType


def test_text_and_markdown_ingestion(tmp_path: Path):
    """TextDocumentIngester segments markdown/text into paragraphs with locators."""
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Title\n\nParagraph 1: Background info.\n\nParagraph 2: Results found.", encoding="utf-8")

    result = ingest(md_file)
    assert result.source.source_type == SourceType.DOCUMENT
    assert len(result.segments) >= 2
    assert "Background info" in result.segments[1].text
    assert result.segments[1].location == "Paragraph 2"


def test_html_document_ingestion_with_sections():
    """HTML ingestion preserves section headings and table locators."""
    html_content = """
    <html>
        <body>
            <h1>Section A</h1>
            <p>Data paragraph in section A.</p>
            <table><tr><td>Table cell content</td></tr></table>
        </body>
    </html>
    """
    ingester = TextDocumentIngester()
    result = ingester.ingest(html_content, title="Test Web Doc")

    assert result.source.source_type == SourceType.DOCUMENT
    assert len(result.segments) == 2
    assert result.segments[0].location == "Section: Section A"
    assert "Data paragraph" in result.segments[0].text
    assert result.segments[1].location == "Section: Section A (Table)"
    assert "Table cell content" in result.segments[1].text


def test_pdf_ingestion_preserves_page_numbers(tmp_path: Path):
    """PDFIngester assigns page numbers to segments."""
    pdf_file = tmp_path / "paper.pdf"
    # Create simulated multi-page PDF stream
    pdf_file.write_bytes(b"%PDF-1.4\n/Page 1 text on first page\n/Page 2 text on second page\n%%EOF")

    result = ingest(pdf_file)
    assert result.source.source_type == SourceType.PDF
    assert len(result.segments) >= 2
    assert result.segments[0].page == 1
    assert result.segments[0].location == "Page 1"
    assert result.segments[1].page == 2
    assert result.segments[1].location == "Page 2"


def test_image_ingestion_with_ocr_and_description(tmp_path: Path):
    """ImageIngester extracts OCR text and visual description."""
    img_file = tmp_path / "diagram.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")
    # Companion OCR sidecar
    sidecar = tmp_path / "diagram.txt"
    sidecar.write_text("Architecture Diagram showing Model Pipeline", encoding="utf-8")

    result = ingest(img_file, description="A system flowchart")
    assert result.source.source_type == SourceType.IMAGE
    assert any("Architecture Diagram" in s.text for s in result.segments)
    assert any("flowchart" in s.text for s in result.segments)


def test_audio_video_srt_timestamp_extraction(tmp_path: Path):
    """AudioVideoIngester extracts second-level timestamps and time formatted locators."""
    video_file = tmp_path / "lecture.mp4"
    video_file.write_bytes(b"fakevideobuffer")
    srt_file = tmp_path / "lecture.srt"
    srt_file.write_text(
        "1\n00:01:10,000 --> 00:01:15,500\nWelcome to the CRISPR gene editing seminar.\n\n"
        "2\n00:01:16,000 --> 00:01:22,000\nToday we discuss off-target mutations.\n",
        encoding="utf-8",
    )

    result = ingest(video_file)
    assert result.source.source_type == SourceType.VIDEO
    assert len(result.segments) == 2
    assert result.segments[0].start_time == 70.0
    assert result.segments[0].end_time == 75.5
    assert result.segments[0].location == "01:10 - 01:15"
    assert "CRISPR" in result.segments[0].text
    assert result.segments[1].start_time == 76.0
    assert result.segments[1].end_time == 82.0
    assert "off-target" in result.segments[1].text
