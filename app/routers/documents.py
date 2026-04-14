"""
app/routers/documents.py
─────────────────────────
FastAPI router handling document upload and processing endpoints.

Endpoints:
  POST /api/v1/process   – Upload + process a document
  GET  /api/v1/health    – Health check
"""

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from loguru import logger

from app.config import get_settings
from app.models import (
    DocumentMeta,
    ErrorResponse,
    HealthResponse,
    OutputFormat,
    ProcessRequest,
    ProcessResponse,
    TaskType,
)
from app.services.chunker import create_chunks, select_representative_chunks
from app.services.file_parser import parse_document
from app.services.llm_service import process_document
from app.services.text_processor import count_words, preprocess_document

router = APIRouter(prefix="/api/v1", tags=["IDP"])
settings = get_settings()


# ── Health Check ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        gemini_configured=bool(settings.gemini_api_key),
        supported_formats=settings.supported_extensions,
    )


# ── Main Processing Endpoint ──────────────────────────────────────────────────

@router.post(
    "/process",
    response_model=ProcessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Upload and process a document",
    description=(
        "Upload a document (.txt / .pdf / .docx / .xlsx / .pptx) and apply "
        "one of three AI tasks: **summarize**, **qa** (question answering), "
        "or **extract** (structured JSON extraction)."
    ),
)
async def process_document_endpoint(
    file: Annotated[UploadFile, File(description="Document to process")],
    task: Annotated[str, Form(description="Task: summarize | qa | extract")] = "summarize",
    question: Annotated[str | None, Form(description="Question (required for qa)")] = None,
    output_format: Annotated[str, Form(description="Output format: text | json | markdown")] = "markdown",
) -> ProcessResponse:

    t_start = time.perf_counter()
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    logger.info(f"Received file: {filename} | task={task} | format={output_format}")

    # ── Input validation ──────────────────────────────────────────────────────
    if ext not in settings.supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(settings.supported_extensions)}"
            ),
        )

    # Validate task
    try:
        task_enum = TaskType(task.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid task '{task}'. Choose from: summarize, qa, extract",
        )

    if task_enum == TaskType.QA and not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A 'question' is required when task='qa'.",
        )

    # Validate output format
    try:
        fmt_enum = OutputFormat(output_format.lower())
    except ValueError:
        fmt_enum = OutputFormat.MARKDOWN  # Graceful fallback

    # ── File size check ───────────────────────────────────────────────────────
    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {settings.max_file_size_mb} MB."
            ),
        )

    # ── Parse document ────────────────────────────────────────────────────────
    try:
        raw_text, page_count = parse_document(filename, content)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Unexpected parse error for '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document parsing failed: {str(exc)}",
        )

    # ── Pre-process text ──────────────────────────────────────────────────────
    cleaned_text, word_count = preprocess_document(raw_text)

    # ── Chunking ──────────────────────────────────────────────────────────────
    all_chunks = create_chunks(cleaned_text)
    chunks = select_representative_chunks(all_chunks)

    warning: str | None = None
    if len(chunks) < len(all_chunks):
        warning = (
            f"Document was very large ({len(all_chunks)} chunks). "
            f"Processed a representative sample of {len(chunks)} chunks."
        )

    # ── LLM Processing ────────────────────────────────────────────────────────
    try:
        result, model_used = await process_document(
            chunks=chunks,
            task=task_enum,
            question=question,
            output_format=fmt_enum,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("LLM processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI processing failed: {str(exc)}",
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return ProcessResponse(
        success=True,
        task=task_enum,
        document=DocumentMeta(
            filename=filename,
            extension=ext,
            size_bytes=len(content),
            page_count=page_count,
            word_count=word_count,
            chunk_count=len(chunks),
        ),
        result=result,
        model_used=model_used,
        processing_time_ms=round(elapsed_ms, 1),
        warning=warning,
    )
