"""Pydantic response and request schemas used across the API.

These models enforce a consistent JSON contract between the backend
and any consumer (React frontend, tests, or external integrations).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request bodies ───────────────────────────────────────────────────

class FieldUpdate(BaseModel):
    invoice_fields: dict | None = None
    extracted_fields: dict | None = None


class VerifyRequest(BaseModel):
    verified: bool = True


class GroundTruthRequest(BaseModel):
    fields: dict = Field(default_factory=dict)


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class ReprocessRequest(BaseModel):
    """Choose which extraction method to (re-)run."""
    method: str = Field(
        default="all",
        description="'baseline', 'layoutlm', or 'all'",
    )


# ── Nested response pieces ──────────────────────────────────────────

class ConfidenceDetail(BaseModel):
    score: float
    level: str  # "high" | "medium" | "low"
    calibrated: bool = False


class QAAnswer(BaseModel):
    question: str
    answer: str | None = None
    confidence: float | None = None
    confidence_level: str | None = None
    source_text: str | None = None
    page: int | None = None
    explanation: str | None = None
    model: str | None = None
    source_type: str | None = None
    field_name: str | None = None
    page_number: int | None = None
    pages_searched: int | None = None


class ExtractionMethodResult(BaseModel):
    method: str
    model_name: str | None = None
    fields: dict = Field(default_factory=dict)
    confidence_scores: dict = Field(default_factory=dict)
    line_items: list[dict] = Field(default_factory=list)
    processing_time: float | None = None
    metadata: dict | None = None
    validation: dict | None = None


class AuditEntry(BaseModel):
    field_name: str
    old_value: str | None = None
    new_value: str | None = None
    confidence: float | None = None
    action: str = "edit"  # "edit" | "approve" | "reject"
    timestamp: str | None = None
    document_id: str | None = None


# ── Top-level API responses ─────────────────────────────────────────

class ModelStatusResponse(BaseModel):
    layoutlm_available: bool = False
    layoutlm_loaded: bool = False
    layoutlm_model: str | None = None
    qa_available: bool = False
    qa_loaded: bool = False
    qa_model: str | None = None
    device: str = "cpu"
