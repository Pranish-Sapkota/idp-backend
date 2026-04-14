"""
app/services/text_processor.py
────────────────────────────────
Cleans and normalises raw extracted text before AI processing.
"""

import re
from loguru import logger


def clean_text(raw: str) -> str:
    """
    Normalise extracted text:
      - Collapse 3+ blank lines → 2
      - Strip trailing whitespace per line
      - Remove null bytes / non-printable control chars (keep newlines/tabs)
      - Normalise Unicode whitespace
    """
    # Remove null bytes and non-printable chars (except \n \t \r)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", raw)

    # Normalise unicode spaces (non-breaking space etc.) → regular space
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)

    # Strip trailing spaces on each line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces (but not newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def count_words(text: str) -> int:
    """Return approximate word count."""
    return len(text.split())


def truncate_for_display(text: str, max_chars: int = 500) -> str:
    """Return a short preview of the extracted text."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [{len(text) - max_chars:,} more characters]"


def preprocess_document(raw_text: str) -> tuple[str, int]:
    """
    Full preprocessing pipeline.

    Returns:
        (cleaned_text, word_count)
    """
    logger.debug(f"Preprocessing {len(raw_text):,} raw chars")
    cleaned = clean_text(raw_text)
    words = count_words(cleaned)
    logger.debug(f"After cleaning: {len(cleaned):,} chars, ~{words:,} words")
    return cleaned, words
