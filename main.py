"""
main.py
────────
IDP System – FastAPI application entry point.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import get_settings
from app.routers.documents import router as documents_router

settings = get_settings()

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level=settings.log_level,
    colorize=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═══════════════════════════════════════")
    logger.info("  IDP System Backend  –  Starting up  ")
    logger.info("═══════════════════════════════════════")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"Gemini key  : {'✓ Configured' if settings.gemini_api_key else '✗ MISSING!'}")
    logger.info(f"Port        : {os.environ.get('PORT', '8000')}")
    yield
    logger.info("IDP System shutting down cleanly.")


app = FastAPI(
    title="IDP System API",
    description=(
        "**Intelligent Document Processing** system powered by Google Gemini.\n\n"
        "Accepts .txt / .pdf / .docx / .xlsx / .pptx files and performs:\n"
        "- **Summarization**\n"
        "- **Question Answering**\n"
        "- **Structured Key-Information Extraction**"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)},
    )


app.include_router(documents_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "IDP System API", "version": "1.0.0", "status": "running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    # Read PORT from environment — Render sets this automatically
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
