"""
app/models.py
─────────────
Pydantic request / response models for the IDP API.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    SUMMARIZE = "summarize"
    QA = "qa"
    EXTRACT = "extract"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class ProcessRequest(BaseModel):
    task: TaskType = Field(default=TaskType.SUMMARIZE)
    question: str | None = Field(default=None)
    output_format: OutputFormat = Field(default=OutputFormat.MARKDOWN)
    extraction_schema: dict[str, Any] | None = Field(default=None)


class DocumentMeta(BaseModel):
    filename: str
    extension: str
    size_bytes: int
    page_count: int | None = None
    word_count: int | None = None
    chunk_count: int | None = None


class ProcessResponse(BaseModel):
    # Suppress Pydantic's warning about fields starting with "model_"
    model_config = ConfigDict(protected_namespaces=())

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
