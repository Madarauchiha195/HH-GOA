"""Output / grounding guardrail — validates LLM answer against retrieved context."""
from __future__ import annotations

import re
from typing import List

import structlog

from backend.schemas.models import GenerationResult, GroundingResult, GuardrailResult, RetrievalCandidate

logger = structlog.get_logger(__name__)


def _jaccard_similarity(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity for grounding check."""
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


class OutputGuard:
    """
    Post-generation grounding validator.
    Checks whether the answer is supported by the retrieved context.
    """

    def __init__(self, grounding_threshold: float = 0.05):
        self.grounding_threshold = grounding_threshold

    def check_grounding(
        self,
        answer: str,
        candidates: List[RetrievalCandidate],
        generation: GenerationResult,
    ) -> Tuple_GroundingResult:
        """
        Validate that the answer is supported by retrieved passages.
        Uses word-overlap heuristic (fast, no external API needed).
        """
        if not answer or not answer.strip():
            return GroundingResult(
                grounded=True,  # Empty answer = safe abstention
                confidence=1.0,
                reason="Empty answer — safe abstention",
            )

        if not candidates:
            return GroundingResult(
                grounded=False,
                confidence=0.0,
                reason="No context to verify against",
            )

        # Check answer similarity against top context chunks
        context_combined = " ".join(c.text for c in candidates[:5])
        similarity = _jaccard_similarity(answer, context_combined)

        grounded = similarity >= self.grounding_threshold
        confidence = round(min(similarity * 10, 1.0), 4)  # scale 0.05 → 0.5

        reason = (
            f"Word overlap with context: {similarity:.3f} "
            f"({'≥' if grounded else '<'} threshold {self.grounding_threshold})"
        )

        return GroundingResult(
            grounded=grounded,
            confidence=confidence,
            reason=reason,
        )

    def check_output(self, generation: GenerationResult) -> GuardrailResult:
        """Check that generation result is structurally valid."""
        if not generation.should_answer:
            return GuardrailResult(
                passed=True,
                guardrail_type="output_abstention",
                reason="Model correctly abstained",
                action="allow",
            )

        if not generation.answer or len(generation.answer.strip()) < 5:
            return GuardrailResult(
                passed=False,
                guardrail_type="empty_output",
                reason="LLM returned an empty or trivially short answer",
                action="abstain",
            )

        return GuardrailResult(
            passed=True,
            guardrail_type="output_validation",
            reason="Output validated successfully",
            action="allow",
        )


# Type alias for cleaner signature
Tuple_GroundingResult = GroundingResult
