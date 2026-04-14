"""
app/models.py
─────────────
Pydantic request / response models for the IDP API.
All I/O is strictly typed for validation and OpenAPI docs.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    SUMMARIZE = "summarize"
    QA = "qa"
    EXTRACT = "extract"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


# ── Request Models ────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    task: TaskType = Field(
        default=TaskType.SUMMARIZE,
        description="Processing task to perform on the document."
    )
    question: str | None = Field(
        default=None,
        description="Question to answer (required when task=qa)."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MARKDOWN,
        description="Desired output format."
    )
    extraction_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema hint for structured extraction (optional)."
    )


# ── Response Models ───────────────────────────────────────────────────────────

class DocumentMeta(BaseModel):
    filename: str
    extension: str
    size_bytes: int
    page_count: int | None = None
    word_count: int | None = None
    chunk_count: int | None = None


class ProcessResponse(BaseModel):
    success: bool
    task: TaskType
    document: DocumentMeta
    result: str | dict[str, Any]
    model_used: str
    processing_time_ms: float
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    gemini_configured: bool
    supported_formats: list[str]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None
