"""Retrieval confidence guardrail — abstains if evidence is insufficient."""
from __future__ import annotations

from backend.config import settings
from backend.schemas.models import GuardrailResult, RetrievalResult


class RetrievalGuard:
    """Checks whether retrieval confidence exceeds threshold."""

    def __init__(self, threshold: float = settings.retrieval_confidence_threshold):
        self.threshold = threshold

    def check(self, retrieval_result: RetrievalResult) -> GuardrailResult:
        if not retrieval_result.candidates:
            return GuardrailResult(
                passed=False,
                guardrail_type="retrieval_empty",
                reason="No documents retrieved from knowledge base",
                action="abstain",
            )

        if retrieval_result.confidence < self.threshold:
            return GuardrailResult(
                passed=False,
                guardrail_type="low_retrieval_confidence",
                reason=(
                    f"Retrieval confidence {retrieval_result.confidence:.2f} "
                    f"< threshold {self.threshold:.2f}. "
                    f"{retrieval_result.confidence_reason}"
                ),
                action="abstain",
            )

        return GuardrailResult(
            passed=True,
            guardrail_type="retrieval_confidence",
            reason=f"Confidence {retrieval_result.confidence:.2f} above threshold",
            action="allow",
        )
