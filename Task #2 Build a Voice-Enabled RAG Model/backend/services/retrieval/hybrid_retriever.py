"""
Hybrid retrieval service — dense + BM25 + RRF + reranker.
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Tuple

import numpy as np
import structlog

from backend.config import settings
from backend.schemas.models import RetrievalCandidate, RetrievalResult
from backend.services.retrieval.index_manager import IndexManager

logger = structlog.get_logger(__name__)


def _rrf_score(ranks: List[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)


class HybridRetriever:
    """
    Dense + BM25 hybrid retrieval with RRF fusion and optional reranking.
    """

    def __init__(
        self,
        top_k_dense: int = settings.top_k_dense,
        top_k_bm25: int = settings.top_k_bm25,
        top_k_fused: int = settings.top_k_fused,
        top_k_rerank: int = settings.top_k_rerank,
        rrf_k: int = settings.rrf_k,
        use_reranker: bool = settings.reranker_enabled,
    ):
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_fused = top_k_fused
        self.top_k_rerank = top_k_rerank
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker

    async def retrieve(
        self,
        query: str,
        embedding: np.ndarray,
    ) -> Tuple[RetrievalResult, dict]:
        """
        Run hybrid retrieval pipeline.
        Returns (RetrievalResult, timing_dict).
        """
        mgr = await IndexManager.get_instance()
        timings = {}

        dense_results: List[Tuple[str, float]] = []
        bm25_results: List[Tuple[str, float]] = []

        loop = asyncio.get_event_loop()

        # ── Dense retrieval ───────────────────────────────────────────────────
        if mgr.faiss_index is not None:
            t = time.monotonic()
            dense_results = await loop.run_in_executor(
                None, self._dense_search, mgr, embedding
            )
            timings["dense_search_ms"] = round((time.monotonic() - t) * 1000, 2)
            logger.debug("Dense search", results=len(dense_results), ms=timings["dense_search_ms"])

        # ── BM25 retrieval ────────────────────────────────────────────────────
        if mgr.bm25_index is not None:
            t = time.monotonic()
            bm25_results = await loop.run_in_executor(
                None, self._bm25_search, mgr, query
            )
            timings["bm25_ms"] = round((time.monotonic() - t) * 1000, 2)
            logger.debug("BM25 search", results=len(bm25_results), ms=timings["bm25_ms"])

        # ── RRF Fusion ────────────────────────────────────────────────────────
        t = time.monotonic()
        fused = self._rrf_fuse(dense_results, bm25_results)
        timings["fusion_ms"] = round((time.monotonic() - t) * 1000, 2)

        # ── Reranking ─────────────────────────────────────────────────────────
        rerank_ms = 0.0
        if self.use_reranker and mgr.reranker is not None and fused:
            t = time.monotonic()
            fused = await loop.run_in_executor(
                None, self._rerank, mgr, query, fused
            )
            rerank_ms = round((time.monotonic() - t) * 1000, 2)
            timings["rerank_ms"] = rerank_ms

        # ── Build candidates ──────────────────────────────────────────────────
        candidates = self._build_candidates(mgr, fused, dense_results, bm25_results)

        # ── Confidence ────────────────────────────────────────────────────────
        confidence, confidence_reason = self._calc_confidence(candidates)

        result = RetrievalResult(
            candidates=candidates,
            confidence=confidence,
            confidence_reason=confidence_reason,
            total_dense=len(dense_results),
            total_bm25=len(bm25_results),
            total_fused=len(fused),
            total_reranked=len(candidates),
        )

        return result, timings

    # ── Dense ─────────────────────────────────────────────────────────────────

    def _dense_search(
        self, mgr: IndexManager, embedding: np.ndarray
    ) -> List[Tuple[str, float]]:
        vec = embedding[0:1]  # (1, D)
        scores, indices = mgr.faiss_index.search(vec, self.top_k_dense)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(mgr.chunk_ids):
                continue
            chunk_id = mgr.chunk_ids[idx]
            results.append((chunk_id, float(score)))
        return results

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _bm25_search(
        self, mgr: IndexManager, query: str
    ) -> List[Tuple[str, float]]:
        tokens = query.lower().split()
        scores = mgr.bm25_index.get_scores(tokens)
        top_indices = scores.argsort()[::-1][: self.top_k_bm25]
        results = []
        for idx in top_indices:
            if idx >= len(mgr.chunk_ids):
                continue
            chunk_id = mgr.chunk_ids[idx]
            score = float(scores[idx])
            if score > 0:
                results.append((chunk_id, score))
        return results

    # ── RRF ───────────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: List[Tuple[str, float]],
        bm25: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion: RRF(d) = Σ 1/(k + rank_i(d))"""
        rrf_scores: dict[str, float] = {}
        dense_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(dense)}
        bm25_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25)}
        all_ids = set(dense_ranks) | set(bm25_ranks)

        for cid in all_ids:
            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[cid])
            rrf_scores[cid] = score

        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return sorted_results[: self.top_k_fused]

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _rerank(
        self,
        mgr: IndexManager,
        query: str,
        fused: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        if not fused:
            return fused

        pairs = []
        valid_ids = []
        for cid, _ in fused[: self.top_k_rerank * 2]:
            text = mgr.chunk_texts.get(cid, "")
            if text:
                pairs.append([query, text])
                valid_ids.append(cid)

        if not pairs:
            return fused

        scores = mgr.reranker.predict(pairs)
        ranked = sorted(zip(valid_ids, scores), key=lambda x: -x[1])
        return [(cid, float(s)) for cid, s in ranked[: self.top_k_rerank]]

    # ── Build candidates ──────────────────────────────────────────────────────

    def _build_candidates(
        self,
        mgr: IndexManager,
        fused: List[Tuple[str, float]],
        dense: List[Tuple[str, float]],
        bm25: List[Tuple[str, float]],
    ) -> List[RetrievalCandidate]:
        dense_map = dict(dense)
        bm25_map = dict(bm25)
        candidates = []

        for rank, (cid, final_score) in enumerate(fused):
            meta = mgr.metadata.get(cid, {})
            text = mgr.chunk_texts.get(cid, "")
            candidates.append(
                RetrievalCandidate(
                    chunk_id=cid,
                    document_id=meta.get("document_id", cid),
                    language=meta.get("language", "unknown"),
                    strategy=meta.get("strategy", "unknown"),
                    text=text,
                    dense_score=dense_map.get(cid),
                    bm25_score=bm25_map.get(cid),
                    rrf_score=final_score,
                    rerank_score=final_score if mgr.reranker else None,
                    final_score=final_score,
                    rank=rank + 1,
                )
            )
        return candidates

    # ── Confidence ────────────────────────────────────────────────────────────

    def _calc_confidence(
        self, candidates: List[RetrievalCandidate]
    ) -> Tuple[float, str]:
        if not candidates:
            return 0.0, "No candidates retrieved"

        top_score = candidates[0].final_score
        if len(candidates) >= 2:
            score_gap = top_score - candidates[1].final_score
        else:
            score_gap = top_score

        # Heuristic: normalize RRF scores to [0,1] range
        max_possible_rrf = 2.0 / (settings.rrf_k + 1)
        normalized = min(top_score / max_possible_rrf, 1.0)

        confidence = (normalized * 0.7) + (min(score_gap / max_possible_rrf, 1.0) * 0.3)
        confidence = round(min(confidence, 1.0), 4)

        if confidence < 0.2:
            reason = "Very low retrieval confidence"
        elif confidence < 0.4:
            reason = "Low retrieval confidence"
        elif confidence < 0.7:
            reason = "Moderate retrieval confidence"
        else:
            reason = "High retrieval confidence"

        return confidence, reason
