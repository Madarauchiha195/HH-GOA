"""
Benchmark suite — runs N queries, measures per-stage latency, outputs P50/P70/P95/P100.

Usage:
    python -m evaluation.benchmark --queries evaluation/test_questions.json --runs 50
    python -m evaluation.benchmark --runs 100 --text-only
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer()
console = Console()

RESULTS_DIR = Path("evaluation/results")


def percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0, "p70": 0, "p95": 0, "p100": 0, "mean": 0, "min": 0, "max": 0, "count": 0}
    s = sorted(values)
    n = len(s)
    def p(pct): return s[min(int(pct / 100 * n), n - 1)]
    return {
        "p50": round(p(50), 2),
        "p70": round(p(70), 2),
        "p95": round(p(95), 2),
        "p100": round(s[-1], 2),
        "mean": round(statistics.mean(s), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "count": n,
        "stddev": round(statistics.stdev(s) if n > 1 else 0, 2),
    }


@app.command()
def run(
    queries_file: str = typer.Option("evaluation/test_questions.json", "--queries", "-q"),
    runs: int = typer.Option(50, "--runs", "-r", help="Number of queries to run"),
    backend_url: str = typer.Option("http://localhost:8000", "--url"),
    output: str = typer.Option("evaluation/results/latest.json", "--output", "-o"),
    text_only: bool = typer.Option(True, "--text-only", help="Use text API (no audio)"),
):
    """Run the latency benchmark and output P50/P70/P95/P100."""
    asyncio.run(_run_benchmark(queries_file, runs, backend_url, output, text_only))


async def _run_benchmark(
    queries_file: str,
    runs: int,
    backend_url: str,
    output: str,
    text_only: bool,
):
    import httpx

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test questions
    qs = json.loads(Path(queries_file).read_text(encoding="utf-8"))
    # Filter only non-edge-case for fair benchmarking
    qs = [q for q in qs if not q.get("should_abstain") and q.get("query", "").strip()]
    # Repeat to reach `runs` count
    queries = (qs * ((runs // len(qs)) + 1))[:runs]

    console.print(f"\n[bold blue]HH Goa Benchmark — {runs} queries[/bold blue]")
    console.print(f"  Backend: {backend_url}")

    results = []
    all_timings = {
        "embedding_ms": [],
        "dense_search_ms": [],
        "bm25_ms": [],
        "fusion_ms": [],
        "rerank_ms": [],
        "generation_ms": [],
        "total_rag_ms": [],
    }
    errors = 0

    async with httpx.AsyncClient(timeout=60) as client:
        for i, q in enumerate(queries):
            t_start = time.monotonic()
            try:
                resp = await client.post(
                    f"{backend_url}/api/v1/text/query",
                    json={"query": q["query"]},
                )
                elapsed = (time.monotonic() - t_start) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    timings = data.get("timings", {})
                    record = {
                        "query_id": q["id"],
                        "query": q["query"],
                        "language": q.get("language", "en"),
                        "category": q.get("category"),
                        "status": data.get("status"),
                        "grounded": data.get("grounded"),
                        "should_answer": data.get("should_answer"),
                        "wall_ms": round(elapsed, 2),
                        **{k: timings.get(k, 0) for k in all_timings},
                    }
                    results.append(record)
                    for k in all_timings:
                        v = timings.get(k, 0)
                        if v:
                            all_timings[k].append(v)

                    console.print(
                        f"  [{i+1:3d}/{runs}] {q['id']:<12} "
                        f"rag={timings.get('total_rag_ms', 0):.0f}ms  "
                        f"status={data.get('status', 'unknown')}"
                    )
                else:
                    errors += 1
                    console.print(f"  [{i+1:3d}/{runs}] [red]HTTP {resp.status_code}[/red]")

            except Exception as exc:
                errors += 1
                console.print(f"  [{i+1:3d}/{runs}] [red]Error: {exc}[/red]")

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(f"\n[bold]Benchmark Results — {len(results)} successful / {errors} errors[/bold]\n")

    table = Table(title="Latency Percentiles (ms)")
    table.add_column("Stage", style="cyan")
    table.add_column("P50", justify="right")
    table.add_column("P70", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("P100", justify="right")
    table.add_column("Mean", justify="right")

    summary = {}
    for stage, values in all_timings.items():
        if values:
            stats = percentiles(values)
            summary[stage] = stats
            table.add_row(
                stage,
                str(stats["p50"]),
                str(stats["p70"]),
                str(stats["p95"]),
                str(stats["p100"]),
                str(stats["mean"]),
            )

    # Wall clock (total round-trip including HTTP)
    wall_times = [r["wall_ms"] for r in results if "wall_ms" in r]
    if wall_times:
        stats = percentiles(wall_times)
        summary["wall_ms"] = stats
        table.add_row(
            "wall_clock",
            str(stats["p50"]),
            str(stats["p70"]),
            str(stats["p95"]),
            str(stats["p100"]),
            str(stats["mean"]),
        )

    console.print(table)

    # ── Save results ──────────────────────────────────────────────────────────
    output_data = {
        "run_at": datetime.utcnow().isoformat(),
        "total_queries": runs,
        "successful": len(results),
        "errors": errors,
        "backend_url": backend_url,
        "percentiles": summary,
        "records": results,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    console.print(f"\n✅ Results saved to: {output}")


if __name__ == "__main__":
    app()
