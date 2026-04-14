"""
main.py
────────
IDP System – FastAPI application entry point.

Starts the application, configures middleware, mounts routers,
and sets up structured logging.
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import get_settings
from app.routers.documents import router as documents_router

# ── Logging Setup ─────────────────────────────────────────────────────────────
settings = get_settings()

logger.remove()  # Remove default handler
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


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("═══════════════════════════════════════")
    logger.info("  IDP System Backend  –  Starting up  ")
    logger.info("═══════════════════════════════════════")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"Gemini key  : {'✓ Configured' if settings.gemini_api_key else '✗ MISSING!'}")
    logger.info(f"Max file    : {settings.max_file_size_mb} MB")
    yield
    logger.info("IDP System shutting down cleanly.")


# ── Application ───────────────────────────────────────────────────────────────

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


# ── CORS ──────────────────────────────────────────────────────────────────────
# In production, replace "*" with your Streamlit deployment URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(documents_router)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "IDP System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


# ── Dev runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
