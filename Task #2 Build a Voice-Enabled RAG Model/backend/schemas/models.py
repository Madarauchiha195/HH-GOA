"""Pydantic schemas for HH Goa Voice RAG."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query text")
    language: Optional[str] = Field(None, description="Hint: ISO 639-1 language code (e.g. 'hi')")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override default top_k")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


# ── Citation / Source ──────────────────────────────────────────────────────────

class Citation(BaseModel):
    chunk_id: str
    document_id: str
    language: str = "unknown"
    strategy: str = "unknown"
    text_excerpt: str = ""
    score: float
    rank: int


# ── Latency Breakdown ─────────────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    stt_ms: Optional[float] = None
    query_processing_ms: float = 0
    embedding_ms: float = 0
    dense_search_ms: float = 0
    bm25_ms: float = 0
    fusion_ms: float = 0
    rerank_ms: float = 0
    generation_ms: float = 0
    grounding_ms: float = 0
    validation_ms: float = 0
    total_rag_ms: float = 0
    total_voice_ms: Optional[float] = None


# ── STT Result ────────────────────────────────────────────────────────────────

class STTResult(BaseModel):
    transcript: str
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    provider: str = "sarvam"
    provider_request_id: Optional[str] = None
    latency_ms: float = 0
    is_code_mixed: bool = False


# ── Retrieval Result ──────────────────────────────────────────────────────────

class RetrievalCandidate(BaseModel):
    chunk_id: str
    document_id: str
    language: str
    strategy: str
    text: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float
    rank: int


class RetrievalResult(BaseModel):
    candidates: List[RetrievalCandidate]
    confidence: float
    confidence_reason: str
    total_dense: int = 0
    total_bm25: int = 0
    total_fused: int = 0
    total_reranked: int = 0


# ── Generation Result ─────────────────────────────────────────────────────────

class GenerationResult(BaseModel):
    answer: str
    should_answer: bool = True
    confidence: float = 0.0
    grounded: bool = False
    citations: List[Citation] = []
    abstention_reason: Optional[str] = None


# ── Grounding Result ──────────────────────────────────────────────────────────

class GroundingResult(BaseModel):
    grounded: bool
    confidence: float
    unsupported_claims: List[str] = []
    reason: str = ""


# ── Guardrail Results ─────────────────────────────────────────────────────────

class GuardrailResult(BaseModel):
    passed: bool
    guardrail_type: str
    reason: str = ""
    action: str = "allow"  # allow | reject | abstain


# ── Full RAG Response ─────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Input
    query: str
    transcript: Optional[str] = None  # Only set for voice queries
    detected_language: Optional[str] = None

    # Output
    answer: str
    should_answer: bool = True
    confidence: float = 0.0
    grounded: bool = False
    abstention_reason: Optional[str] = None

    # Sources
    sources: List[Citation] = []

    # Diagnostics
    timings: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    guardrail_decisions: List[GuardrailResult] = []

    # Status
    status: str = "success"  # success | abstained | error
    error: Optional[str] = None


# ── Dashboard / Metrics ───────────────────────────────────────────────────────

class PercentileStats(BaseModel):
    p50: float
    p70: float
    p95: float
    p100: float
    mean: float
    min: float
    max: float
    count: int


class MetricsResponse(BaseModel):
    retrieval_latency: PercentileStats
    rag_latency: PercentileStats
    voice_latency: Optional[PercentileStats] = None
    grounding_rate: float
    abstention_rate: float
    total_requests: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    indexes_loaded: bool
    embedding_model_loaded: bool
    reranker_loaded: bool
    llm_configured: bool
    sarvam_configured: bool
    index_doc_count: int = 0
    uptime_seconds: float


class ConfigResponse(BaseModel):
    embedding_model: str
    reranker_model: str
    reranker_enabled: bool
    top_k_dense: int
    top_k_bm25: int
    top_k_fused: int
    top_k_rerank: int
    rrf_k: int
    retrieval_confidence_threshold: float
    fast_mode: bool
    llm_model: str
    loaded_languages: List[str] = []
    chunk_count: int = 0
