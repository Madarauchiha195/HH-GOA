"""
Streaming dataset downloader for ai4bharat/MSMARCO-XI.
Uses HuggingFace datasets with streaming to avoid OOM on large files.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn

logger = structlog.get_logger(__name__)
console = Console()

# Language code → MSMARCO-XI split name mapping
LANGUAGE_MAP = {
    "en": "train.en",
    "hi": "train.hi",
    "kn": "train.kn",
    "mr": "train.mr",
    "ta": "train.ta",
    "te": "train.te",
    "bn": "train.bn",
    "gu": "train.gu",
    "ml": "train.ml",
    "pa": "train.pa",
    "as": "train.as",
    "or": "train.or",
    "ur": "train.ur",
    "ne": "train.ne",
    "si": "train.si",
}

DATASET_NAME = "ai4bharat/MSMARCO-XI"


class DatasetDownloader:
    """
    Streams MSMARCO-XI from HuggingFace without loading full dataset into memory.
    """

    def __init__(self, languages: List[str], sample_size: int = 10_000):
        self.languages = languages
        self.sample_size = sample_size

    def load(self) -> List[Dict[str, Any]]:
        """
        Stream and collect records for selected languages.
        Returns list of normalized record dicts.
        """
        from datasets import load_dataset

        records = []

        for lang in self.languages:
            split = LANGUAGE_MAP.get(lang)
            if not split:
                console.print(f"  [yellow]⚠ Unknown language: {lang}, skipping[/yellow]")
                continue

            console.print(f"  Loading {lang} ({split})…")

            try:
                dataset = load_dataset(
                    DATASET_NAME,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )

                count = 0
                for row in dataset:
                    if self.sample_size and count >= self.sample_size:
                        break
                    normalized = self._normalize_row(row, lang)
                    if normalized:
                        records.append(normalized)
                        count += 1

                console.print(f"  ✅ {lang}: {count:,} records loaded")

            except Exception as exc:
                console.print(f"  [red]❌ Failed to load {lang}: {exc}[/red]")
                logger.error("Dataset load failed", language=lang, error=str(exc))

        return records

    def _normalize_row(self, row: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Normalize a raw MSMARCO-XI row into a consistent schema."""
        # MSMARCO-XI columns vary slightly; try multiple key names
        query = (
            row.get("query")
            or row.get("question")
            or row.get("query_text")
            or ""
        )
        passage = (
            row.get("passage")
            or row.get("positive_passages", [{}])[0].get("text", "")
            if row.get("positive_passages")
            else row.get("context", "")
            or row.get("passage_text", "")
            or ""
        )
        answer = row.get("answer") or row.get("answers", [""])[0] if isinstance(row.get("answers"), list) else row.get("answer", "")

        # Derive passage from positive_passages if available
        if not passage and "positive_passages" in row:
            pos = row["positive_passages"]
            if pos and isinstance(pos, list):
                passage = pos[0].get("text", "")

        if not passage:
            return None

        query_id = row.get("query_id") or row.get("id") or hashlib.md5(query.encode()).hexdigest()[:12]
        doc_id = row.get("doc_id") or f"{language}_{query_id}"

        return {
            "document_id": str(doc_id),
            "query_id": str(query_id),
            "language": language,
            "query": str(query).strip(),
            "passage": str(passage).strip(),
            "answer": str(answer).strip() if answer else "",
            "source": DATASET_NAME,
        }
