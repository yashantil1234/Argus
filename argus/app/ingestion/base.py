"""Base interfaces and data structures for multimodal ingestion in ARGUS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from app.models.research import Source, SourceType


@dataclass
class ContentSegment:
    """A granular segment extracted from an ingested source with exact locators."""

    text: str
    page: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    location: Optional[str] = None


@dataclass
class IngestionResult:
    """The result of ingesting any multimodal document or file."""

    source: Source
    segments: List[ContentSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenated text of all segments."""
        return "\n\n".join(s.text for s in self.segments if s.text.strip())


class BaseIngester(ABC):
    """Abstract base class for all file and media ingesters."""

    @abstractmethod
    def can_ingest(self, target: Union[str, Path]) -> bool:
        """Check if this ingester can handle the specified file or URL."""
        ...

    @abstractmethod
    def ingest(self, target: Union[str, Path], **kwargs) -> IngestionResult:
        """Extract text and locators from the target."""
        ...
