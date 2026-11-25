"""Base interfaces for extraction and QA models.

Every extraction back-end (OCR + regex baseline, LayoutLMv3, etc.)
returns the same ``ExtractionResult`` structure so the rest of the
application never depends on a specific model implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class ExtractionResult:
    """Unified output of any extraction pipeline."""

    fields: dict[str, Any] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    line_items: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    method: str = "unknown"
    model_name: str | None = None
    processing_time: float = 0.0
    metadata: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None


@dataclass
class QAResult:
    """Single answer returned by a document QA model."""

    question: str = ""
    answer: str | None = None
    confidence: float = 0.0
    confidence_level: str | None = None
    source_text: str | None = None
    page: int | None = None
    explanation: str | None = None
    model_name: str | None = None
    source_type: str | None = None
    field_name: str | None = None


class BaseExtractor(ABC):
    """Interface that every extraction back-end must implement."""

    @abstractmethod
    def extract(
        self,
        image: Image.Image,
        ocr_text: str = "",
        document_type: str = "Unknown",
    ) -> ExtractionResult:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return *True* if the required libraries are installed."""
        ...

    def is_loaded(self) -> bool:
        """Return *True* if model weights are loaded into memory."""
        return False


class BaseQA(ABC):
    """Interface for document question answering."""

    @abstractmethod
    def answer(
        self,
        image: Image.Image,
        question: str,
        ocr_text: str = "",
    ) -> QAResult:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    def is_loaded(self) -> bool:
        return False
