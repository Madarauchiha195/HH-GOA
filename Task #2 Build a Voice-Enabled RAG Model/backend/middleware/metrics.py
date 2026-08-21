"""In-memory metrics collector for latency P50/P70/P95/P100 tracking."""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, List, Optional

from backend.schemas.models import MetricsResponse, PercentileStats


@dataclass
class RequestMetrics:
    request_id: str
    timestamp: float = field(default_factory=time.time)
    retrieval_ms: Optional[float] = None
    rag_ms: Optional[float] = None
    voice_ms: Optional[float] = None
    grounded: Optional[bool] = None
    abstained: bool = False


class MetricsCollector:
    """Thread-safe rolling window metrics collector."""

    _instance: Optional["MetricsCollector"] = None
    _lock = Lock()

    def __init__(self, window: int = 1000):
        self._window = window
        self._records: Deque[RequestMetrics] = deque(maxlen=window)
        self._start_time = time.monotonic()
        self._lock = Lock()

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._records.append(metrics)

    def _percentiles(self, values: List[float]) -> PercentileStats:
        if not values:
            return PercentileStats(p50=0, p70=0, p95=0, p100=0, mean=0, min=0, max=0, count=0)
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def pct(p: float) -> float:
            idx = min(int(p / 100 * n), n - 1)
            return round(sorted_vals[idx], 2)

        return PercentileStats(
            p50=pct(50),
            p70=pct(70),
            p95=pct(95),
            p100=pct(100),
            mean=round(statistics.mean(sorted_vals), 2),
            min=round(min(sorted_vals), 2),
            max=round(max(sorted_vals), 2),
            count=n,
        )

    def get_metrics(self) -> MetricsResponse:
        with self._lock:
            records = list(self._records)

        retrieval_ms = [r.retrieval_ms for r in records if r.retrieval_ms is not None]
        rag_ms = [r.rag_ms for r in records if r.rag_ms is not None]
        voice_ms = [r.voice_ms for r in records if r.voice_ms is not None]
        grounded = [r.grounded for r in records if r.grounded is not None]
        abstained = [r.abstained for r in records]

        grounding_rate = sum(grounded) / len(grounded) if grounded else 0.0
        abstention_rate = sum(abstained) / len(abstained) if abstained else 0.0

        return MetricsResponse(
            retrieval_latency=self._percentiles(retrieval_ms),
            rag_latency=self._percentiles(rag_ms),
            voice_latency=self._percentiles(voice_ms) if voice_ms else None,
            grounding_rate=round(grounding_rate, 4),
            abstention_rate=round(abstention_rate, 4),
            total_requests=len(records),
            uptime_seconds=round(time.monotonic() - self._start_time, 1),
        )
