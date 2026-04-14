"""
app/config.py
─────────────
Centralised settings loaded from environment variables.
Uses pydantic-settings for validation and type safety.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # ── AI Keys ──────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # ── App Behaviour ─────────────────────────────────────────────────────────
    app_env: str = "production"
    max_file_size_mb: int = 20
    log_level: str = "INFO"

    # ── Gemini Model Config ───────────────────────────────────────────────────
    gemini_model: str = "gemini-1.5-flash-latest"          # Fast & cost-effective
    gemini_model_pro: str = "gemini-1.5-pro"        # Higher quality fallback

    # ── Chunking Config ───────────────────────────────────────────────────────
    chunk_size_tokens: int = 1500
    chunk_overlap_tokens: int = 150
    max_chunks_per_request: int = 10               # Avoid Gemini rate limits

    # ── Supported MIME types ──────────────────────────────────────────────────
    supported_extensions: list[str] = [
        ".txt", ".pdf", ".docx", ".xlsx", ".pptx"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton – settings are read once at startup."""
    return Settings()
