"""Text query endpoint."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from backend.schemas.models import RAGResponse, TextQueryRequest
from backend.services.orchestrator import Orchestrator

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/text/query", response_model=RAGResponse)
async def text_query(request: TextQueryRequest) -> RAGResponse:
    """
    Run the full RAG pipeline on a text query.

    - Query processing & language detection
    - Guardrail check
    - Hybrid retrieval (dense + BM25 + RRF + reranker)
    - Grounded generation
    - Response validation
    """
    try:
        orchestrator = Orchestrator()
        response = await orchestrator.run_text_pipeline(
            query=request.query,
            language_hint=request.language,
            top_k_override=request.top_k,
        )
        return response
    except Exception as exc:
        logger.error("Text query failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error — please try again.") from exc
