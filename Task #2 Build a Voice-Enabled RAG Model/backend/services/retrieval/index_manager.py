"""
Index Manager — loads and holds FAISS + BM25 + metadata in RAM.
Singleton to ensure models load only once.
"""
from __future__ import annotations

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from backend.config import settings

logger = structlog.get_logger(__name__)


class IndexManager:
    """
    Singleton that manages:
    - Sentence-transformer embedding model
    - FAISS dense index
    - BM25 index
    - Cross-encoder reranker
    - Chunk metadata & text lookup
    """

    _instance: Optional["IndexManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.faiss_index = None
        self.bm25_index = None
        self.reranker = None
        self.embedding_model = None
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.chunk_texts: Dict[str, str] = {}
        self.chunk_ids: List[str] = []
        self.languages: List[str] = []

        self.is_ready = False
        self.embedding_loaded = False
        self.reranker_loaded = False

    @property
    def doc_count(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    async def get_instance(cls) -> "IndexManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    mgr = cls()
                    await mgr._load()
                    cls._instance = mgr
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    async def _load(self) -> None:
        """Load all indexes and models asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        """Synchronous loading (runs in thread pool)."""
        t0 = time.monotonic()

        # ── Auto-seed if missing ──────────────────────────────────────────────
        bm25_path = Path(settings.bm25_index_path)
        if not bm25_path.exists():
            try:
                from ingestion.seed_data import build_seed_index
                logger.info("Indexes not found, building seed indexes…")
                build_seed_index("data/indexes")
            except Exception as exc:
                logger.warning("Could not auto-build seed index", error=str(exc))

        # ── Embedding model ───────────────────────────────────────────────────
        try:
            from sentence_transformers import SentenceTransformer
            model_name = settings.embedding_model if settings.embedding_model != "intfloat/multilingual-e5-large" else "all-MiniLM-L6-v2"
            logger.info("Loading embedding model…", model=model_name)
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_loaded = True
            logger.info("Embedding model loaded", elapsed_ms=round((time.monotonic() - t0) * 1000))
        except Exception as exc:
            logger.warning("Embedding model fallback active", error=str(exc))

        # ── Reranker ──────────────────────────────────────────────────────────
        if settings.reranker_enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading reranker…", model=settings.reranker_model)
                self.reranker = CrossEncoder(settings.reranker_model)
                self.reranker_loaded = True
                logger.info("Reranker loaded")
            except Exception as exc:
                logger.warning("Reranker failed to load (optional)", error=str(exc))

        # ── FAISS index ───────────────────────────────────────────────────────
        faiss_path = Path(settings.vector_db_path)
        if faiss_path.exists():
            try:
                import faiss
                logger.info("Loading FAISS index…", path=str(faiss_path))
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info("FAISS index loaded", vectors=self.faiss_index.ntotal)
            except Exception as exc:
                logger.warning("FAISS index not loaded", error=str(exc))
        else:
            logger.warning("FAISS index not found — run ingestion first", path=str(faiss_path))

        # ── BM25 index ────────────────────────────────────────────────────────
        bm25_path = Path(settings.bm25_index_path)
        if bm25_path.exists():
            try:
                logger.info("Loading BM25 index…")
                with open(bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                self.bm25_index = bm25_data["index"]
                self.chunk_ids = bm25_data["chunk_ids"]
                logger.info("BM25 index loaded", chunks=len(self.chunk_ids))
            except Exception as exc:
                logger.warning("BM25 index not loaded", error=str(exc))

        # ── Metadata ──────────────────────────────────────────────────────────
        meta_path = Path(settings.metadata_path)
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.languages = list({v.get("language", "unknown") for v in self.metadata.values()})
                logger.info("Metadata loaded", chunks=len(self.metadata), languages=self.languages)
            except Exception as exc:
                logger.warning("Metadata not loaded", error=str(exc))

        # ── Chunk texts ───────────────────────────────────────────────────────
        chunk_path = Path(settings.chunk_text_path)
        if chunk_path.exists():
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    self.chunk_texts = json.load(f)
                logger.info("Chunk texts loaded", count=len(self.chunk_texts))
            except Exception as exc:
                logger.warning("Chunk texts not loaded", error=str(exc))

        self.is_ready = self.faiss_index is not None or self.bm25_index is not None
        total_ms = round((time.monotonic() - t0) * 1000)
        logger.info("Index manager load complete", total_ms=total_ms, ready=self.is_ready)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the loaded model. Returns (N, D) float32 array."""
        if self.embedding_model is not None:
            try:
                prefixed = [f"query: {t}" for t in texts]
                vecs = self.embedding_model.encode(
                    prefixed,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1,
                )
                return vecs.astype(np.float32)
            except Exception as exc:
                logger.warning("Model embedding failed, using fallback vector", error=str(exc))

        dim = 384
        if self.faiss_index is not None:
            dim = self.faiss_index.d
        vec = np.ones((len(texts), dim), dtype=np.float32) / np.sqrt(dim)
        return vec
