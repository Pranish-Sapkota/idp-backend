"""
app/services/chunker.py
────────────────────────
Token-aware text chunker using tiktoken.
Splits on paragraph boundaries first, then falls back to sentence-level
splitting to preserve semantic coherence.

Strategy:
  1. Split text into paragraphs (double newline)
  2. Accumulate paragraphs until we hit chunk_size_tokens
  3. On overflow, commit current chunk and start a new one with
     overlap taken from the tail of the previous chunk
"""

from __future__ import annotations
import re
import tiktoken
from loguru import logger
from app.config import get_settings


# Use cl100k_base encoder (GPT-4 / Gemini approximation)
_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _token_count(text: str) -> int:
    return len(_get_encoder().encode(text))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines; fall back to sentences for very long paragraphs."""
    raw_paras = re.split(r"\n{2,}", text)
    paragraphs: list[str] = []

    for para in raw_paras:
        para = para.strip()
        if not para:
            continue
        # If a single paragraph is already huge, split into sentences
        if _token_count(para) > get_settings().chunk_size_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            paragraphs.extend(s.strip() for s in sentences if s.strip())
        else:
            paragraphs.append(para)

    return paragraphs


def create_chunks(text: str) -> list[str]:
    """
    Split `text` into token-bounded chunks with overlap.

    Returns:
        List of chunk strings (each ≤ chunk_size_tokens).
    """
    settings = get_settings()
    max_tokens = settings.chunk_size_tokens
    overlap_tokens = settings.chunk_overlap_tokens

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens: int = 0

    for para in paragraphs:
        para_tokens = _token_count(para)

        # If adding this paragraph would overflow, commit the chunk
        if current_tokens + para_tokens > max_tokens and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(chunk_text)

            # Build overlap: take tail paragraphs that fit within overlap budget
            overlap_parts: list[str] = []
            overlap_running = 0
            for part in reversed(current_parts):
                pt = _token_count(part)
                if overlap_running + pt <= overlap_tokens:
                    overlap_parts.insert(0, part)
                    overlap_running += pt
                else:
                    break

            current_parts = overlap_parts
            current_tokens = overlap_running

        current_parts.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    logger.info(
        f"Chunked document into {len(chunks)} chunk(s) "
        f"(max {max_tokens} tokens each, {overlap_tokens} token overlap)"
    )
    return chunks


def select_representative_chunks(
    chunks: list[str],
    max_chunks: int | None = None,
) -> list[str]:
    """
    When the document has more chunks than `max_chunks`, intelligently
    select a representative sample: first, last, and evenly-spaced middle.
    """
    if max_chunks is None:
        max_chunks = get_settings().max_chunks_per_request

    if len(chunks) <= max_chunks:
        return chunks

    # Always include first and last chunks for context
    if max_chunks <= 2:
        return [chunks[0], chunks[-1]][:max_chunks]

    middle_count = max_chunks - 2
    middle_indices = [
        int(round(i * (len(chunks) - 2) / (middle_count + 1)))
        for i in range(1, middle_count + 1)
    ]
    middle_indices = [min(max(1, idx), len(chunks) - 2) for idx in middle_indices]

    selected = [chunks[0]]
    seen = {0}
    for idx in middle_indices:
        if idx not in seen:
            selected.append(chunks[idx])
            seen.add(idx)
    if len(chunks) - 1 not in seen:
        selected.append(chunks[-1])

    logger.warning(
        f"Document has {len(chunks)} chunks; selected {len(selected)} "
        f"representative chunks (max_chunks={max_chunks})."
    )
    return selected
