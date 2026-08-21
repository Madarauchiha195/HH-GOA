"""
Central orchestration harness — runs the full pipeline with per-stage timing.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Optional

import structlog

from backend.middleware.metrics import MetricsCollector, RequestMetrics
from backend.schemas.models import (
    GuardrailResult,
    LatencyBreakdown,
    RAGResponse,
    RetrievalResult,
)
from backend.services.generation.llm import LLMGenerator
from backend.services.guardrails.input_guard import InputGuard
from backend.services.guardrails.output_guard import OutputGuard
from backend.services.guardrails.retrieval_guard import RetrievalGuard
from backend.services.query_processor import QueryProcessor
from backend.services.retrieval.hybrid_retriever import HybridRetriever
from backend.services.stt.sarvam import get_sarvam_stt

logger = structlog.get_logger(__name__)

# Rolling trace store (last 500 requests)
_trace_store: OrderedDict[str, dict] = OrderedDict()
_TRACE_LIMIT = 500


class Orchestrator:
    """
    Runs the complete RAG pipeline:
    validate → STT → process → guard → retrieve → rerank → generate → ground → respond
    """

    def __init__(self):
        self._input_guard = InputGuard()
        self._retrieval_guard = RetrievalGuard()
        self._output_guard = OutputGuard()
        self._query_processor = QueryProcessor()
        self._retriever = HybridRetriever()
        self._generator = LLMGenerator()

    # ── Voice pipeline ─────────────────────────────────────────────────────────

    async def run_voice_pipeline(
        self,
        audio_bytes: bytes,
        audio_content_type: str,
        language_hint: Optional[str] = None,
    ) -> RAGResponse:
        t_pipeline_start = time.monotonic()
        response = RAGResponse(query="", answer="", status="success")

        # ── STT ───────────────────────────────────────────────────────────────
        t = time.monotonic()
        try:
            stt_svc = get_sarvam_stt()
            stt_result = await stt_svc.transcribe(audio_bytes, audio_content_type, language_hint)
            response.transcript = stt_result.transcript
            response.detected_language = stt_result.detected_language
            response.timings.stt_ms = stt_result.latency_ms
            language_hint = language_hint or stt_result.detected_language
            query = stt_result.transcript
        except Exception as exc:
            logger.error("STT failed", error=str(exc))
            response.answer = "I couldn't process the audio. Please try recording again."
            response.status = "error"
            response.error = str(exc)
            return response

        response.query = query

        # ── Run RAG pipeline ──────────────────────────────────────────────────
        response = await self._run_rag(response, query, language_hint)

        # ── Total voice latency ───────────────────────────────────────────────
        total_voice_ms = (time.monotonic() - t_pipeline_start) * 1000
        response.timings.total_voice_ms = round(total_voice_ms, 2)

        # ── Record metrics ────────────────────────────────────────────────────
        MetricsCollector.get_instance().record(
            RequestMetrics(
                request_id=response.request_id,
                retrieval_ms=response.timings.total_rag_ms,
                rag_ms=response.timings.total_rag_ms,
                voice_ms=total_voice_ms,
                grounded=response.grounded,
                abstained=not response.should_answer,
            )
        )

        self._store_trace(response)
        return response

    # ── Text pipeline ──────────────────────────────────────────────────────────

    async def run_text_pipeline(
        self,
        query: str,
        language_hint: Optional[str] = None,
        top_k_override: Optional[int] = None,
    ) -> RAGResponse:
        t_pipeline_start = time.monotonic()
        response = RAGResponse(query=query, answer="", status="success")

        response = await self._run_rag(response, query, language_hint, top_k_override)

        # ── Record metrics ─────────────────────────────────────────────────────
        MetricsCollector.get_instance().record(
            RequestMetrics(
                request_id=response.request_id,
                retrieval_ms=response.timings.total_rag_ms,
                rag_ms=response.timings.total_rag_ms,
                grounded=response.grounded,
                abstained=not response.should_answer,
            )
        )

        self._store_trace(response)
        return response

    # ── Core RAG pipeline ──────────────────────────────────────────────────────

    async def _run_rag(
        self,
        response: RAGResponse,
        query: str,
        language_hint: Optional[str],
        top_k_override: Optional[int] = None,
    ) -> RAGResponse:
        t_rag_start = time.monotonic()
        guardrail_decisions = []

        # ── Input guardrail ────────────────────────────────────────────────────
        input_guard_result = self._input_guard.check(query)
        guardrail_decisions.append(input_guard_result)

        if not input_guard_result.passed:
            return self._build_abstain_response(
                response, query, input_guard_result, guardrail_decisions, t_rag_start
            )

        # ── Query processing ───────────────────────────────────────────────────
        normalized, language, is_code_mixed, embedding, proc_timings = (
            await self._query_processor.process(query, language_hint)
        )
        response.detected_language = response.detected_language or language
        response.timings.query_processing_ms = proc_timings.query_processing_ms
        response.timings.embedding_ms = proc_timings.embedding_ms

        # ── Hybrid retrieval ───────────────────────────────────────────────────
        retrieval_result, retrieval_timings = await self._retriever.retrieve(
            normalized, embedding
        )
        response.timings.dense_search_ms = retrieval_timings.get("dense_search_ms", 0)
        response.timings.bm25_ms = retrieval_timings.get("bm25_ms", 0)
        response.timings.fusion_ms = retrieval_timings.get("fusion_ms", 0)
        response.timings.rerank_ms = retrieval_timings.get("rerank_ms", 0)

        # ── Retrieval confidence guardrail ─────────────────────────────────────
        retrieval_guard_result = self._retrieval_guard.check(retrieval_result)
        guardrail_decisions.append(retrieval_guard_result)

        if not retrieval_guard_result.passed:
            return self._build_abstain_response(
                response, query, retrieval_guard_result, guardrail_decisions, t_rag_start
            )

        # ── Generation ─────────────────────────────────────────────────────────
        t_gen = time.monotonic()
        generation = await self._generator.generate(
            query=normalized,
            candidates=retrieval_result.candidates,
            retrieval_confidence=retrieval_result.confidence,
        )
        response.timings.generation_ms = round((time.monotonic() - t_gen) * 1000, 2)

        # ── Output validation ──────────────────────────────────────────────────
        t_val = time.monotonic()
        output_guard_result = self._output_guard.check_output(generation)
        guardrail_decisions.append(output_guard_result)

        # ── Grounding check ────────────────────────────────────────────────────
        grounding = self._output_guard.check_grounding(
            generation.answer, retrieval_result.candidates, generation
        )
        response.timings.grounding_ms = round((time.monotonic() - t_val) * 1000, 2)
        response.timings.validation_ms = response.timings.grounding_ms

        # ── Assemble response ──────────────────────────────────────────────────
        total_rag_ms = (time.monotonic() - t_rag_start) * 1000
        response.timings.total_rag_ms = round(total_rag_ms, 2)

        response.answer = generation.answer
        response.should_answer = generation.should_answer and output_guard_result.passed
        response.confidence = generation.confidence
        response.grounded = grounding.grounded
        response.sources = generation.citations
        response.guardrail_decisions = guardrail_decisions
        response.abstention_reason = generation.abstention_reason

        if not response.should_answer:
            response.status = "abstained"
            response.answer = (
                generation.abstention_reason
                or "I couldn't find enough relevant information to answer that reliably."
            )

        return response

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_abstain_response(
        self,
        response: RAGResponse,
        query: str,
        guard_result: GuardrailResult,
        decisions: list,
        t_rag_start: float,
    ) -> RAGResponse:
        response.timings.total_rag_ms = round((time.monotonic() - t_rag_start) * 1000, 2)
        response.should_answer = False
        response.status = "abstained"
        response.guardrail_decisions = decisions
        response.abstention_reason = guard_result.reason

        if guard_result.guardrail_type in ("unsafe_content", "prompt_injection"):
            response.answer = "I'm unable to process that request."
        elif guard_result.guardrail_type == "off_topic":
            response.answer = "I couldn't find relevant information for that question in the knowledge base."
        elif guard_result.guardrail_type in ("retrieval_empty", "low_retrieval_confidence"):
            response.answer = "I couldn't find enough relevant information in the knowledge base to answer that reliably."
        else:
            response.answer = "I couldn't process that request. Please try again."

        return response

    def _store_trace(self, response: RAGResponse) -> None:
        global _trace_store
        if len(_trace_store) >= _TRACE_LIMIT:
            _trace_store.popitem(last=False)
        _trace_store[response.request_id] = response.model_dump()

    @staticmethod
    def get_trace(request_id: str) -> Optional[dict]:
        return _trace_store.get(request_id)
