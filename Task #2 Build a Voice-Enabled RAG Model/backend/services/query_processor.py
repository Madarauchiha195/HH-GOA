"""
Query processor — language detection, normalization, embedding.
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Optional, Tuple

import structlog

from backend.schemas.models import LatencyBreakdown
from backend.services.retrieval.index_manager import IndexManager

logger = structlog.get_logger(__name__)

# Indic Unicode blocks for code-mix detection
_INDIC_RANGE_START = 0x0900
_INDIC_RANGE_END = 0x0DFF


def _detect_language(text: str) -> str:
    """Detect language using langdetect with fallback."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def _is_code_mixed(text: str) -> bool:
    has_latin = any("A" <= c <= "z" for c in text)
    has_indic = any(_INDIC_RANGE_START <= ord(c) <= _INDIC_RANGE_END for c in text)
    return has_latin and has_indic


def _normalize(text: str) -> str:
    """Normalize whitespace and unicode."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class QueryProcessor:
    """
    Processes raw query text into normalized form + embeddings.
    """

    async def process(
        self,
        query: str,
        language_hint: Optional[str] = None,
    ) -> Tuple[str, str, bool, object, LatencyBreakdown]:
        """
        Returns:
            normalized_query, language, is_code_mixed, embedding, partial_timings
        """
        t0 = time.monotonic()

        # 1. Normalize
        normalized = _normalize(query)

        # 2. Language detection
        if language_hint and language_hint.strip():
            language = language_hint.strip()
        else:
            language = _detect_language(normalized)

        # 3. Code-mix detection
        code_mixed = _is_code_mixed(normalized)

        lang_ms = (time.monotonic() - t0) * 1000

        # 4. Embed
        t_embed = time.monotonic()
        mgr = await IndexManager.get_instance()
        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, mgr.embed, [normalized])
        embed_ms = (time.monotonic() - t_embed) * 1000

        query_proc_ms = lang_ms
        embed_ms = round(embed_ms, 2)

        logger.debug(
            "Query processed",
            language=language,
            code_mixed=code_mixed,
            embed_ms=embed_ms,
        )

        timings = LatencyBreakdown(
            query_processing_ms=round(query_proc_ms, 2),
            embedding_ms=embed_ms,
        )

        return normalized, language, code_mixed, embedding, timings
