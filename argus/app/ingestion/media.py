"""Image, Audio, and Video multimodal ingestion for ARGUS."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Union

from app.ingestion.base import BaseIngester, ContentSegment, IngestionResult
from app.models.research import Source, SourceType

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ImageIngester(BaseIngester):
    """Ingests image files, extracting visual descriptions and OCR text."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"}

    def can_ingest(self, target: Union[str, Path]) -> bool:
        path = Path(target)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def ingest(self, target: Union[str, Path], **kwargs) -> IngestionResult:
        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")

        title = kwargs.get("title", path.stem)
        url = kwargs.get("url", str(path.resolve()))
        description = kwargs.get("description", "")
        ocr_text = kwargs.get("ocr_text", "")

        segments: List[ContentSegment] = []

        # If OCR text is provided or discovered in accompanying sidecar (.txt/.json)
        sidecar_txt = path.with_suffix(".txt")
        if not ocr_text and sidecar_txt.exists():
            ocr_text = sidecar_txt.read_text(encoding="utf-8", errors="replace")

        if ocr_text:
            segments.append(
                ContentSegment(
                    text=ocr_text.strip(),
                    location="Image OCR Text",
                )
            )

        if description:
            segments.append(
                ContentSegment(
                    text=description.strip(),
                    location="Visual Description",
                )
            )

        if not segments:
            # Basic fallback representation
            segments.append(
                ContentSegment(
                    text=f"Visual asset: {path.name}",
                    location="Metadata",
                )
            )

        source = Source(
            url=url,
            title=title,
            source_type=SourceType.IMAGE,
            page_text="\n\n".join(s.text for s in segments),
        )

        return IngestionResult(source=source, segments=segments)


class AudioVideoIngester(BaseIngester):
    """Ingests audio and video media, extracting timestamped spoken segments."""

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

    def can_ingest(self, target: Union[str, Path]) -> bool:
        path = Path(target)
        ext = path.suffix.lower()
        return ext in self.AUDIO_EXTENSIONS or ext in self.VIDEO_EXTENSIONS or ext in {".srt", ".vtt"}

    def ingest(self, target: Union[str, Path], **kwargs) -> IngestionResult:
        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found at {path}")

        ext = path.suffix.lower()
        is_video = ext in self.VIDEO_EXTENSIONS
        source_type = SourceType.VIDEO if is_video else SourceType.AUDIO
        title = kwargs.get("title", path.stem)
        url = kwargs.get("url", str(path.resolve()))

        segments: List[ContentSegment] = []

        # Check for sidecar subtitle/transcript files (.srt, .vtt, .json)
        srt_path = path.with_suffix(".srt")
        vtt_path = path.with_suffix(".vtt")
        transcript_path = path.with_suffix(".transcript.json")

        if ext == ".srt" or srt_path.exists():
            target_srt = path if ext == ".srt" else srt_path
            segments = self._parse_srt(target_srt)
        elif ext == ".vtt" or vtt_path.exists():
            target_vtt = path if ext == ".vtt" else vtt_path
            segments = self._parse_vtt(target_vtt)
        elif transcript_path.exists():
            segments = self._parse_json_transcript(transcript_path)
        elif "transcript" in kwargs:
            raw_transcript = kwargs["transcript"]
            if isinstance(raw_transcript, list):
                for item in raw_transcript:
                    st = float(item.get("start", 0.0))
                    et = float(item.get("end", st + 5.0))
                    txt = item.get("text", "")
                    loc = f"{_format_timestamp(st)} - {_format_timestamp(et)}"
                    segments.append(ContentSegment(text=txt, start_time=st, end_time=et, location=loc))
            elif isinstance(raw_transcript, str):
                segments.append(ContentSegment(text=raw_transcript, location="Transcript"))

        if not segments:
            segments.append(
                ContentSegment(
                    text=f"Media recording: {path.name}",
                    start_time=0.0,
                    end_time=0.0,
                    location="00:00",
                )
            )

        full_text = "\n\n".join(f"[{s.location}]\n{s.text}" for s in segments)
        source = Source(
            url=url,
            title=title,
            source_type=source_type,
            page_text=full_text,
        )

        return IngestionResult(source=source, segments=segments)

    def _parse_srt(self, path: Path) -> List[ContentSegment]:
        content = path.read_text(encoding="utf-8", errors="replace")
        blocks = re.split(r"\n\s*\n", content.strip())
        segments: List[ContentSegment] = []

        for b in blocks:
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            if len(lines) < 2:
                continue
            # Look for 00:00:20,000 --> 00:00:24,400
            m = re.search(
                r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
                lines[1] if len(lines) > 1 and "-->" in lines[1] else lines[0],
            )
            if m:
                st = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/1000.0
                et = int(m.group(5))*3600 + int(m.group(6))*60 + int(m.group(7)) + int(m.group(8))/1000.0
                text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
                text = " ".join(text_lines)
                loc = f"{_format_timestamp(st)} - {_format_timestamp(et)}"
                segments.append(ContentSegment(text=text, start_time=st, end_time=et, location=loc))
        return segments

    def _parse_vtt(self, path: Path) -> List[ContentSegment]:
        content = path.read_text(encoding="utf-8", errors="replace")
        blocks = re.split(r"\n\s*\n", content.strip())
        segments: List[ContentSegment] = []

        for b in blocks:
            if "WEBVTT" in b:
                continue
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            if not lines:
                continue
            m = re.search(
                r"(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})",
                lines[0],
            )
            if m:
                h1 = int(m.group(1) or 0)
                st = h1*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/1000.0
                h2 = int(m.group(5) or 0)
                et = h2*3600 + int(m.group(6))*60 + int(m.group(7)) + int(m.group(8))/1000.0
                text = " ".join(lines[1:])
                loc = f"{_format_timestamp(st)} - {_format_timestamp(et)}"
                segments.append(ContentSegment(text=text, start_time=st, end_time=et, location=loc))
        return segments

    def _parse_json_transcript(self, path: Path) -> List[ContentSegment]:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments: List[ContentSegment] = []
        for item in data.get("segments", data if isinstance(data, list) else []):
            st = float(item.get("start", 0.0))
            et = float(item.get("end", st))
            txt = item.get("text", "").strip()
            loc = f"{_format_timestamp(st)} - {_format_timestamp(et)}"
            segments.append(ContentSegment(text=txt, start_time=st, end_time=et, location=loc))
        return segments
