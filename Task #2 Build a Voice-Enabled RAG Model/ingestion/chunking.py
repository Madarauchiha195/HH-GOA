"""
Chunking strategies for MSMARCO-XI passages.

Available strategies:
1. SentenceChunker       — split by sentence boundaries
2. SlidingWindowChunker  — fixed-size with overlap
3. SemanticChunker       — group semantically similar sentences
4. MetadataAwareChunker  — wraps any chunker, ensures metadata propagation

All chunks implement BaseChunk and include full metadata.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseChunk:
    chunk_id: str
    document_id: str
    language: str
    strategy: str
    chunk_index: int
    text: str
    token_count: int
    parent_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make_id(cls, document_id: str, strategy: str, index: int, text: str) -> str:
        h = hashlib.md5(f"{document_id}:{strategy}:{index}:{text[:50]}".encode()).hexdigest()[:12]
        return f"{strategy[:3]}_{h}"


class BaseChunker(ABC):
    """Abstract base for all chunkers."""

    @abstractmethod
    def chunk(self, record: Dict[str, Any]) -> List[BaseChunk]:
        """Split a record's passage into chunks."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sentence Chunker
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_ENDINGS = re.compile(r"(?<=[.!?।؟])\s+")


def _split_sentences(text: str) -> List[str]:
    """Language-agnostic sentence splitter."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback: regex split
        parts = _SENTENCE_ENDINGS.split(text)
        return [p.strip() for p in parts if p.strip()]


class SentenceChunker(BaseChunker):
    """
    Splits passages by sentence boundaries.
    Groups N sentences per chunk for context.
    """

    def __init__(self, sentences_per_chunk: int = 3, overlap: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

    def chunk(self, record: Dict[str, Any]) -> List[BaseChunk]:
        passage = record.get("passage", "").strip()
        if not passage:
            return []

        sentences = _split_sentences(passage)
        if not sentences:
            return []

        chunks: List[BaseChunk] = []
        step = max(1, self.sentences_per_chunk - self.overlap)

        for i in range(0, len(sentences), step):
            group = sentences[i:i + self.sentences_per_chunk]
            text = " ".join(group).strip()
            if not text:
                continue

            chunk_idx = len(chunks)
            chunks.append(BaseChunk(
                chunk_id=BaseChunk.make_id(record["document_id"], "sentence", chunk_idx, text),
                document_id=record["document_id"],
                language=record.get("language", "unknown"),
                strategy="sentence",
                chunk_index=chunk_idx,
                text=text,
                token_count=len(text.split()),
                parent_metadata={k: v for k, v in record.items() if k not in ("passage",)},
            ))

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sliding Window Chunker
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowChunker(BaseChunker):
    """
    Fixed-size sliding window over tokens with configurable overlap.
    Useful for passage context spanning sentence boundaries.
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, record: Dict[str, Any]) -> List[BaseChunk]:
        passage = record.get("passage", "").strip()
        if not passage:
            return []

        words = passage.split()
        if not words:
            return []

        chunks: List[BaseChunk] = []
        step = max(1, self.chunk_size - self.overlap)

        for i in range(0, len(words), step):
            window = words[i:i + self.chunk_size]
            text = " ".join(window).strip()
            if not text:
                continue

            chunk_idx = len(chunks)
            chunks.append(BaseChunk(
                chunk_id=BaseChunk.make_id(record["document_id"], "sliding_window", chunk_idx, text),
                document_id=record["document_id"],
                language=record.get("language", "unknown"),
                strategy="sliding_window",
                chunk_index=chunk_idx,
                text=text,
                token_count=len(window),
                parent_metadata={k: v for k, v in record.items() if k not in ("passage",)},
            ))

            if len(window) < self.chunk_size:
                break  # Last chunk

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Semantic Chunker
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Groups semantically similar sentences using cosine similarity.
    Starts a new chunk when similarity drops below threshold.
    """

    def __init__(self, similarity_threshold: float = 0.5, max_chunk_sentences: int = 6):
        self.threshold = similarity_threshold
        self.max_chunk_sentences = max_chunk_sentences
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model for chunking (not the main retrieval model)
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def chunk(self, record: Dict[str, Any]) -> List[BaseChunk]:
        passage = record.get("passage", "").strip()
        if not passage:
            return []

        sentences = _split_sentences(passage)
        if len(sentences) <= 2:
            # Too short to split meaningfully — return as single chunk
            return self._single_chunk(record, passage, 0)

        try:
            model = self._get_model()
            embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            # Fall back to sentence chunker on error
            return SentenceChunker().chunk(record)

        # Group sentences into semantic chunks
        groups: List[List[str]] = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = float(embeddings[i] @ embeddings[i - 1])  # cosine similarity
            if sim >= self.threshold and len(current_group) < self.max_chunk_sentences:
                current_group.append(sentences[i])
            else:
                groups.append(current_group)
                current_group = [sentences[i]]

        if current_group:
            groups.append(current_group)

        chunks: List[BaseChunk] = []
        for idx, group in enumerate(groups):
            text = " ".join(group).strip()
            if text:
                chunks.append(BaseChunk(
                    chunk_id=BaseChunk.make_id(record["document_id"], "semantic", idx, text),
                    document_id=record["document_id"],
                    language=record.get("language", "unknown"),
                    strategy="semantic",
                    chunk_index=idx,
                    text=text,
                    token_count=len(text.split()),
                    parent_metadata={k: v for k, v in record.items() if k not in ("passage",)},
                ))

        return chunks

    def _single_chunk(self, record: Dict[str, Any], text: str, idx: int) -> List[BaseChunk]:
        return [BaseChunk(
            chunk_id=BaseChunk.make_id(record["document_id"], "semantic", idx, text),
            document_id=record["document_id"],
            language=record.get("language", "unknown"),
            strategy="semantic",
            chunk_index=idx,
            text=text,
            token_count=len(text.split()),
            parent_metadata={k: v for k, v in record.items() if k not in ("passage",)},
        )]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metadata-Aware Chunker
# ─────────────────────────────────────────────────────────────────────────────

class MetadataAwareChunker(BaseChunker):
    """
    Wraps any chunker and ensures all metadata fields are propagated.
    Also includes the query as context in metadata.
    """

    def __init__(self, inner: BaseChunker):
        self.inner = inner

    def chunk(self, record: Dict[str, Any]) -> List[BaseChunk]:
        chunks = self.inner.chunk(record)
        for chunk in chunks:
            # Ensure query is included in metadata for context
            chunk.parent_metadata["original_query"] = record.get("query", "")
            chunk.parent_metadata["original_answer"] = record.get("answer", "")
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

class ChunkerFactory:
    @staticmethod
    def create(strategy: str, **kwargs) -> BaseChunker:
        strategies = {
            "sentence": lambda: SentenceChunker(**kwargs),
            "sliding_window": lambda: SlidingWindowChunker(**kwargs),
            "semantic": lambda: SemanticChunker(**kwargs),
            "metadata": lambda: MetadataAwareChunker(SlidingWindowChunker(**kwargs)),
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from {list(strategies)}")
        return strategies[strategy]()
