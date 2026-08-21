"""
Ingestion pipeline — downloads, chunks, embeds, and indexes MSMARCO-XI.

Usage:
    python -m ingestion.run --languages en hi kn mr --sample 10000
    python -m ingestion.run --languages en hi --sample 100000 --strategy semantic
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(help="MSMARCO-XI ingestion pipeline")
console = Console()


@app.command()
def run(
    languages: List[str] = typer.Option(
        ["en", "hi"], "--languages", "-l",
        help="Language codes to ingest (e.g. en hi kn mr)"
    ),
    sample: int = typer.Option(
        10_000, "--sample", "-s",
        help="Number of rows per language to ingest (0 = all)"
    ),
    strategy: str = typer.Option(
        "sliding_window", "--strategy",
        help="Chunking strategy: sentence | sliding_window | semantic | all"
    ),
    output_dir: str = typer.Option(
        "data/indexes", "--output", "-o",
        help="Output directory for indexes"
    ),
    batch_size: int = typer.Option(
        32, "--batch-size",
        help="Embedding batch size"
    ),
    skip_download: bool = typer.Option(
        False, "--skip-download",
        help="Skip download if data already exists"
    ),
):
    """Run the full ingestion pipeline: download → preprocess → chunk → embed → index."""
    console.print("[bold blue]HH Goa Voice RAG — Ingestion Pipeline[/bold blue]")
    console.print(f"  Languages: {languages}")
    console.print(f"  Sample size: {sample:,} per language")
    console.print(f"  Chunk strategy: {strategy}")
    console.print(f"  Output: {output_dir}")

    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        languages=languages,
        sample_size=sample,
        chunk_strategy=strategy,
        output_dir=Path(output_dir),
        batch_size=batch_size,
    )

    asyncio.run(pipeline.run(skip_download=skip_download))


if __name__ == "__main__":
    app()
