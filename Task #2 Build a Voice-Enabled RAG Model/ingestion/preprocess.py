"""
Preprocessor — deduplication, normalization, metadata enrichment.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Set


class Preprocessor:
    """Cleans and deduplicates raw records."""

    def process(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_hashes: Set[str] = set()
        clean: List[Dict[str, Any]] = []

        for record in records:
            passage = self._normalize_text(record.get("passage", ""))
            if not passage or len(passage) < 20:
                continue

            h = hashlib.md5(passage.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            clean.append({
                **record,
                "passage": passage,
                "query": self._normalize_text(record.get("query", "")),
                "answer": self._normalize_text(record.get("answer", "")),
                "passage_hash": h,
            })

        return clean

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
