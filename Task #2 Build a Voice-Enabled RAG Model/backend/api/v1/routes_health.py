"""API routes — health, metrics, config, debug."""
from __future__ import annotations

import time

from fastapi import APIRouter

from backend.config import settings
from backend.middleware.metrics import MetricsCollector
from backend.schemas.models import ConfigResponse, HealthResponse, MetricsResponse
from backend.services.retrieval.index_manager import IndexManager

router = APIRouter()

_start_time = time.monotonic()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """System health check."""
    mgr = await IndexManager.get_instance()
    return HealthResponse(
        status="healthy" if mgr.is_ready else "degraded",
        indexes_loaded=mgr.is_ready,
        embedding_model_loaded=mgr.embedding_loaded,
        reranker_loaded=mgr.reranker_loaded,
        llm_configured=bool(settings.gemini_api_key),
        sarvam_configured=bool(settings.sarvam_api_key),
        index_doc_count=mgr.doc_count,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Latency percentiles and system metrics."""
    return MetricsCollector.get_instance().get_metrics()


@router.get("/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """Current retrieval configuration (no secrets)."""
    mgr = await IndexManager.get_instance()
    return ConfigResponse(
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        reranker_enabled=settings.reranker_enabled,
        top_k_dense=settings.top_k_dense,
        top_k_bm25=settings.top_k_bm25,
        top_k_fused=settings.top_k_fused,
        top_k_rerank=settings.top_k_rerank,
        rrf_k=settings.rrf_k,
        retrieval_confidence_threshold=settings.retrieval_confidence_threshold,
        fast_mode=settings.fast_mode,
        llm_model=settings.llm_model,
        loaded_languages=mgr.languages,
        chunk_count=mgr.doc_count,
    )


@router.get("/retrieval/debug/{request_id}")
async def retrieval_debug(request_id: str):
    """Return per-request debug trace (stored in memory, last 500 requests)."""
    from backend.services.orchestrator import Orchestrator
    trace = Orchestrator.get_trace(request_id)
    if trace is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Trace not found for request_id={request_id}")
    return trace
