"""
app/services/llm_service.py
────────────────────────────
Gemini API integration using the new google-genai SDK (v1 endpoint).
  - Primary:  Google Gemini 2.0 Flash
  - Fallback: Groq (llama-3.1-70b) if GROQ_API_KEY is set
  - Retry:    tenacity (3 attempts, exponential backoff)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx
from google import genai
from google.genai import types
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.models import OutputFormat, TaskType


# ── Gemini client ─────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Set it in your .env file or Render environment variables."
        )
    return genai.Client(api_key=settings.gemini_api_key)


# ── Prompt Templates ──────────────────────────────────────────────────────────

SUMMARIZE_PROMPT = """\
You are an expert document analyst. Analyse the following document content and produce a \
comprehensive, well-structured summary.

Requirements:
- Include the main topics and key points
- Preserve important facts, figures, and dates
- Use clear headings and bullet points where appropriate
- Length: concise but complete (aim for 20-30% of original content)

Document Content:
─────────────────
{text}
─────────────────

Summary:"""

QA_PROMPT = """\
You are an expert document analyst. Answer the following question using ONLY the information \
provided in the document content below. If the answer cannot be found, say so clearly.

Question: {question}

Document Content:
─────────────────
{text}
─────────────────

Answer (cite relevant sections where possible):"""

EXTRACT_PROMPT = """\
You are an expert information extraction system. Extract structured key information from \
the document below.

{schema_hint}

Extract the following categories (return ONLY valid JSON, no markdown fences):
{{
  "title": "document title if identifiable",
  "document_type": "report/contract/invoice/email/presentation/spreadsheet/other",
  "key_entities": {{
    "people": [],
    "organizations": [],
    "locations": [],
    "dates": [],
    "monetary_values": []
  }},
  "main_topics": [],
  "key_facts": [],
  "action_items": [],
  "summary_one_line": "one sentence summary"
}}

Document Content:
─────────────────
{text}
─────────────────

JSON Output:"""

SYNTHESIS_PROMPT = """\
You are synthesising partial analyses of different sections of a large document.

Task: {task}
{question_line}

Partial results from each section:
{partial_results}

Produce a single, coherent, final {output_type} that integrates all the above.
Do not repeat duplicate information. Be concise and well-structured."""


# ── Core Gemini Call ──────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _call_gemini(prompt: str, use_pro: bool = False) -> str:
    """Single async Gemini call with retry logic using the new google-genai SDK."""
    settings = get_settings()
    model_name = settings.gemini_model_pro if use_pro else settings.gemini_model

    client = _get_client()
    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


# ── Fallback: Groq ────────────────────────────────────────────────────────────

async def _call_groq(prompt: str) -> str:
    """Groq fallback using httpx async client."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured for fallback.")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 4096,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def _call_llm(prompt: str, use_pro: bool = False) -> tuple[str, str]:
    """Try Gemini first, fall back to Groq on failure."""
    settings = get_settings()
    model_label = settings.gemini_model_pro if use_pro else settings.gemini_model

    try:
        result = await _call_gemini(prompt, use_pro=use_pro)
        return result, model_label
    except Exception as gemini_err:
        logger.warning(f"Gemini failed ({gemini_err}), trying Groq fallback…")
        if settings.groq_api_key:
            try:
                result = await _call_groq(prompt)
                return result, "groq/llama-3.1-70b-versatile"
            except Exception as groq_err:
                logger.error(f"Groq fallback also failed: {groq_err}")
        raise gemini_err


# ── Per-Chunk Processing ──────────────────────────────────────────────────────

async def _process_chunk(
    chunk: str,
    task: TaskType,
    question: str | None,
    extraction_schema: dict | None,
) -> str:
    if task == TaskType.SUMMARIZE:
        prompt = SUMMARIZE_PROMPT.format(text=chunk)
    elif task == TaskType.QA:
        if not question:
            raise ValueError("A question is required for the QA task.")
        prompt = QA_PROMPT.format(question=question, text=chunk)
    elif task == TaskType.EXTRACT:
        schema_hint = ""
        if extraction_schema:
            schema_hint = f"Use this schema as a guide:\n{json.dumps(extraction_schema, indent=2)}\n"
        prompt = EXTRACT_PROMPT.format(text=chunk, schema_hint=schema_hint)
    else:
        raise ValueError(f"Unknown task: {task}")

    result, _ = await _call_llm(prompt)
    return result


# ── Synthesis ─────────────────────────────────────────────────────────────────

async def _synthesise(
    partial_results: list[str],
    task: TaskType,
    question: str | None,
) -> tuple[str, str]:
    numbered = "\n\n".join(
        f"--- Section {i + 1} ---\n{r}" for i, r in enumerate(partial_results)
    )
    output_type = {
        TaskType.SUMMARIZE: "summary",
        TaskType.QA: "answer",
        TaskType.EXTRACT: "JSON extraction",
    }[task]
    question_line = f"Question: {question}" if task == TaskType.QA and question else ""

    prompt = SYNTHESIS_PROMPT.format(
        task=task.value,
        question_line=question_line,
        partial_results=numbered,
        output_type=output_type,
    )
    return await _call_llm(prompt, use_pro=True)


# ── JSON Post-processing ──────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"```(?:json)?\s*", "", text)
    clean = re.sub(r"```", "", clean).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("Could not parse JSON from LLM response.")


# ── Public API ────────────────────────────────────────────────────────────────

async def process_document(
    chunks: list[str],
    task: TaskType,
    question: str | None = None,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    extraction_schema: dict | None = None,
) -> tuple[str | dict[str, Any], str]:
    start = time.perf_counter()
    logger.info(f"LLM processing: task={task.value}, chunks={len(chunks)}")

    if len(chunks) == 1:
        result_text, model_used = await _call_llm(
            _build_prompt(chunks[0], task, question, extraction_schema)
        )
    else:
        partial_tasks = [
            _process_chunk(chunk, task, question, extraction_schema)
            for chunk in chunks
        ]
        partial_results = await asyncio.gather(*partial_tasks, return_exceptions=True)

        valid_results: list[str] = []
        for i, res in enumerate(partial_results):
            if isinstance(res, Exception):
                logger.warning(f"Chunk {i} failed: {res}")
            else:
                valid_results.append(res)

        if not valid_results:
            raise RuntimeError("All chunks failed to process. Check your API key.")

        if len(valid_results) == 1:
            result_text, model_used = valid_results[0], get_settings().gemini_model
        else:
            result_text, model_used = await _synthesise(valid_results, task, question)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.success(f"LLM processing done in {elapsed_ms:.0f}ms using {model_used}")

    if task == TaskType.EXTRACT:
        try:
            return _extract_json(result_text), model_used
        except Exception as exc:
            logger.warning(f"JSON parse failed for extraction result: {exc}")
            return result_text, model_used

    return result_text, model_used


def _build_prompt(
    text: str,
    task: TaskType,
    question: str | None,
    extraction_schema: dict | None,
) -> str:
    if task == TaskType.SUMMARIZE:
        return SUMMARIZE_PROMPT.format(text=text)
    elif task == TaskType.QA:
        return QA_PROMPT.format(question=question or "", text=text)
    elif task == TaskType.EXTRACT:
        schema_hint = ""
        if extraction_schema:
            schema_hint = f"Use this schema as a guide:\n{json.dumps(extraction_schema, indent=2)}\n"
        return EXTRACT_PROMPT.format(text=text, schema_hint=schema_hint)
    raise ValueError(f"Unknown task: {task}")
