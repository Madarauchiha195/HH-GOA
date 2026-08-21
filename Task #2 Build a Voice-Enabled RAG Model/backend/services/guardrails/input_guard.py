"""Guardrails — input validation, off-topic detection, injection protection."""
from __future__ import annotations

import re
from typing import Tuple

import structlog

from backend.schemas.models import GuardrailResult

logger = structlog.get_logger(__name__)

# ── Unsafe / adversarial patterns ────────────────────────────────────────────
_UNSAFE_PATTERNS = [
    r"(?i)(bomb|weapon|kill|murder|terrorist|explosiv)",
    r"(?i)(hack|malware|ransomware|phish)",
    r"(?i)(child abuse|csam|pornograph)",
]

# ── Off-topic heuristic keywords (non-QA tasks) ───────────────────────────────
_OFFTOPIC_PATTERNS = [
    r"(?i)\b(write\s+(?:code|a\s+program|script|function|class))",
    r"(?i)\b(generate\s+(?:image|picture|photo|video))",
    r"(?i)\b(play\s+(?:game|music))",
    r"(?i)\b(translate\s+this\s+(?:text|document))",
    r"(?i)\b(create\s+(?:a\s+website|an\s+app|a\s+game))",
]

# ── Prompt injection heuristics ───────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"(?i)(ignore\s+(all\s+)?previous\s+instructions)",
    r"(?i)(you\s+are\s+now\s+(?:a|an)\s+\w+)",
    r"(?i)(disregard\s+(your|all)\s+(?:previous|prior)\s+instructions)",
    r"(?i)(act\s+as\s+(?:a|an)\s+\w+\s+without\s+restrictions)",
    r"(?i)(jailbreak|DAN\s+mode|developer\s+mode)",
    r"(?i)(system\s*:\s*you\s+are)",
    r"(?i)(print\s+your\s+(?:system\s+)?prompt)",
]

MAX_QUERY_LENGTH = 2000


class InputGuard:
    """Validates and screens user input before retrieval."""

    def check(self, query: str) -> GuardrailResult:
        """
        Run all input guardrails.
        Returns GuardrailResult with passed=True if query is safe.
        """

        # 1. Empty query
        if not query or not query.strip():
            return GuardrailResult(
                passed=False,
                guardrail_type="empty_query",
                reason="Query is empty",
                action="reject",
            )

        # 2. Too long
        if len(query) > MAX_QUERY_LENGTH:
            return GuardrailResult(
                passed=False,
                guardrail_type="query_too_long",
                reason=f"Query exceeds {MAX_QUERY_LENGTH} characters",
                action="reject",
            )

        # 3. Prompt injection
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, query):
                logger.warning("Prompt injection attempt detected", query=query[:100])
                return GuardrailResult(
                    passed=False,
                    guardrail_type="prompt_injection",
                    reason="Potential prompt injection detected",
                    action="reject",
                )

        # 4. Unsafe content
        for pattern in _UNSAFE_PATTERNS:
            if re.search(pattern, query):
                logger.warning("Unsafe query detected", query=query[:100])
                return GuardrailResult(
                    passed=False,
                    guardrail_type="unsafe_content",
                    reason="Query contains potentially unsafe content",
                    action="reject",
                )

        # 5. Off-topic (non-QA request)
        for pattern in _OFFTOPIC_PATTERNS:
            if re.search(pattern, query):
                logger.info("Off-topic query detected", query=query[:100])
                return GuardrailResult(
                    passed=False,
                    guardrail_type="off_topic",
                    reason="Query appears to be outside the knowledge base scope",
                    action="abstain",
                )

        return GuardrailResult(
            passed=True,
            guardrail_type="input",
            reason="Passed all input guardrails",
            action="allow",
        )
