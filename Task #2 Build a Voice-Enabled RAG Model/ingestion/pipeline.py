"""
Full ingestion pipeline: download → preprocess → chunk → embed → index.
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
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ingestion.chunking import ChunkerFactory, BaseChunk
from ingestion.download import DatasetDownloader
from ingestion.preprocess import Preprocessor

logger = structlog.get_logger(__name__)
console = Console()


class IngestionPipeline:
    """Coordinates all ingestion stages."""

    def __init__(
        self,
        languages: List[str],
        sample_size: int,
        chunk_strategy: str,
        output_dir: Path,
        batch_size: int = 32,
    ):
        self.languages = languages
        self.sample_size = sample_size
        self.chunk_strategy = chunk_strategy
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, skip_download: bool = False) -> None:
        t0 = time.monotonic()

        # ── Stage 1: Download ──────────────────────────────────────────────────
        console.print("\n[bold]Stage 1: Downloading dataset…[/bold]")
        downloader = DatasetDownloader(
            languages=self.languages,
            sample_size=self.sample_size,
        )
        raw_records = downloader.load()
        console.print(f"  ✅ Loaded {len(raw_records):,} raw records")

        # ── Stage 2: Preprocess ────────────────────────────────────────────────
        console.print("\n[bold]Stage 2: Preprocessing…[/bold]")
        preprocessor = Preprocessor()
        clean_records = preprocessor.process(raw_records)
        console.print(f"  ✅ {len(clean_records):,} clean records after dedup")

        # ── Stage 3: Chunking ──────────────────────────────────────────────────
        console.print(f"\n[bold]Stage 3: Chunking (strategy={self.chunk_strategy})…[/bold]")
        chunks = self._chunk(clean_records)
        console.print(f"  ✅ {len(chunks):,} chunks generated")

        # ── Stage 4: Embedding ─────────────────────────────────────────────────
        console.print("\n[bold]Stage 4: Generating embeddings…[/bold]")
        embeddings = await self._embed(chunks)
        console.print(f"  ✅ Embeddings shape: {embeddings.shape}")

        # ── Stage 5: Indexing ──────────────────────────────────────────────────
        console.print("\n[bold]Stage 5: Building indexes…[/bold]")
        self._build_indexes(chunks, embeddings)

        elapsed = (time.monotonic() - t0) / 60
        console.print(f"\n[bold green]✅ Ingestion complete in {elapsed:.1f} minutes[/bold green]")
        console.print(f"  Indexes saved to: {self.output_dir}")

    def _chunk(self, records: List[Dict[str, Any]]) -> List[BaseChunk]:
        """Apply chunking strategy to all records."""
        all_chunks: List[BaseChunk] = []
        seen_ids: set = set()

        strategies = (
            ["sentence", "sliding_window", "semantic", "metadata"]
            if self.chunk_strategy == "all"
            else [self.chunk_strategy]
        )

        for strategy_name in strategies:
            chunker = ChunkerFactory.create(strategy_name)
            for record in records:
                chunks = chunker.chunk(record)
                for chunk in chunks:
                    if chunk.chunk_id not in seen_ids:
                        all_chunks.append(chunk)
                        seen_ids.add(chunk.chunk_id)

        return all_chunks

    async def _embed(self, chunks: List[BaseChunk]) -> np.ndarray:
        """Generate embeddings for all chunks."""
        from sentence_transformers import SentenceTransformer

        model_name = "intfloat/multilingual-e5-large"
        console.print(f"  Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        # Prefix for passage encoding (multilingual-e5 convention)
        texts = [f"passage: {c.text}" for c in chunks]

        all_embeddings = []
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
            task = progress.add_task("Embedding…", total=len(texts))
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                loop = asyncio.get_event_loop()
                vecs = await loop.run_in_executor(
                    None,
                    lambda b=batch: model.encode(b, normalize_embeddings=True, show_progress_bar=False),
                )
                all_embeddings.append(vecs)
                progress.advance(task, len(batch))

        return np.vstack(all_embeddings).astype(np.float32)

    def _build_indexes(self, chunks: List[BaseChunk], embeddings: np.ndarray) -> None:
        """Build FAISS + BM25 indexes and save metadata."""
        import faiss
        from rank_bm25 import BM25Okapi

        chunk_ids = [c.chunk_id for c in chunks]
        chunk_texts = {c.chunk_id: c.text for c in chunks}
        metadata = {
            c.chunk_id: {
                "document_id": c.document_id,
                "language": c.language,
                "strategy": c.strategy,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            for c in chunks
        }

        # ── FAISS ─────────────────────────────────────────────────────────────
        dim = embeddings.shape[1]
        console.print(f"  Building FAISS index (dim={dim}, vectors={len(embeddings):,})…")
        if len(embeddings) < 10_000:
            index = faiss.IndexFlatIP(dim)
        else:
            nlist = min(256, len(embeddings) // 40)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        index.add(embeddings)
        faiss.write_index(index, str(self.output_dir / "faiss.index"))
        console.print(f"  ✅ FAISS index saved ({index.ntotal:,} vectors)")

        # ── BM25 ──────────────────────────────────────────────────────────────
        console.print("  Building BM25 index…")
        tokenized = [text.lower().split() for text in [c.text for c in chunks]]
        bm25 = BM25Okapi(tokenized)
        with open(self.output_dir / "bm25.pkl", "wb") as f:
            pickle.dump({"index": bm25, "chunk_ids": chunk_ids}, f)
        console.print(f"  ✅ BM25 index saved ({len(chunk_ids):,} docs)")

        # ── Metadata ──────────────────────────────────────────────────────────
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        with open(self.output_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_texts, f, ensure_ascii=False, indent=2)

        console.print(f"  ✅ Metadata saved ({len(metadata):,} chunks)")
