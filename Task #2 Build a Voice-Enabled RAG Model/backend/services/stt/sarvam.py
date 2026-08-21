"""
Sarvam AI Speech-to-Text service.

Uses Saaras v3 (recommended) via the Sarvam REST API.
Implements exponential backoff retry for transient failures.
"""
from __future__ import annotations

import asyncio
import io
import time
from typing import Optional

import httpx
import structlog

from backend.config import settings
from backend.schemas.models import STTResult

logger = structlog.get_logger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# Errors that should NOT be retried
NON_RETRYABLE_STATUS = {400, 401, 403, 415, 422}


class STTError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class SarvamSTT:
    """
    Sarvam Saaras v3 STT provider.

    Supports all major Indic languages + English + code-mixed speech.
    """

    def __init__(self):
        self._api_key = settings.sarvam_api_key
        self._model = settings.sarvam_stt_model
        self._max_retries = settings.sarvam_stt_max_retries
        self._timeout = settings.sarvam_stt_timeout

    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_content_type: str = "audio/wav",
        language_code: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe audio bytes using Sarvam Saaras v3.

        Args:
            audio_bytes: Raw audio bytes
            audio_content_type: MIME type (audio/wav, audio/webm, etc.)
            language_code: Optional ISO 639-1 language code hint

        Returns:
            STTResult with transcript, detected language, confidence, latency
        """
        if not self._api_key:
            raise STTError("SARVAM_API_KEY is not configured.", retryable=False)

        t_start = time.monotonic()

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                result = await self._call_api(audio_bytes, audio_content_type, language_code)
                latency_ms = (time.monotonic() - t_start) * 1000
                result.latency_ms = round(latency_ms, 2)
                logger.info(
                    "STT success",
                    attempt=attempt + 1,
                    latency_ms=result.latency_ms,
                    language=result.detected_language,
                    transcript_len=len(result.transcript),
                )
                return result

            except STTError as exc:
                if not exc.retryable:
                    raise

                last_error = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "STT transient failure, retrying",
                    attempt=attempt + 1,
                    wait_s=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)

            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "STT unexpected error, retrying",
                    attempt=attempt + 1,
                    wait_s=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)

        raise STTError(
            f"Sarvam STT failed after {self._max_retries} attempts: {last_error}",
            retryable=False,
        )

    async def _call_api(
        self,
        audio_bytes: bytes,
        content_type: str,
        language_code: Optional[str],
    ) -> STTResult:
        """Single API call to Sarvam STT endpoint."""

        # Determine file extension for MIME
        ext_map = {
            "audio/wav": "wav",
            "audio/wave": "wav",
            "audio/x-wav": "wav",
            "audio/webm": "webm",
            "audio/ogg": "ogg",
            "audio/mp4": "mp4",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/flac": "flac",
        }
        ext = ext_map.get(content_type.split(";")[0].strip(), "wav")
        filename = f"audio.{ext}"

        # Build form data
        files = {"file": (filename, io.BytesIO(audio_bytes), content_type)}
        data: dict = {"model": self._model}
        if language_code:
            data["language_code"] = language_code

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": self._api_key},
                files=files,
                data=data,
            )

        if response.status_code in NON_RETRYABLE_STATUS:
            raise STTError(
                f"Sarvam API non-retryable error {response.status_code}: {response.text}",
                status_code=response.status_code,
                retryable=False,
            )

        if response.status_code != 200:
            raise STTError(
                f"Sarvam API error {response.status_code}: {response.text}",
                status_code=response.status_code,
                retryable=True,
            )

        body = response.json()

        transcript = body.get("transcript", "").strip()
        if not transcript:
            raise STTError("Empty transcript returned by Sarvam.", retryable=False)

        # Detect code-mixing heuristically (mixed script characters)
        is_code_mixed = _detect_code_mixed(transcript)

        return STTResult(
            transcript=transcript,
            detected_language=body.get("language_code") or language_code,
            confidence=body.get("confidence"),
            provider="sarvam",
            provider_request_id=body.get("request_id"),
            is_code_mixed=is_code_mixed,
        )


def _detect_code_mixed(text: str) -> bool:
    """Heuristic: True if text contains both Latin and non-Latin script characters."""
    has_latin = any("\u0041" <= c <= "\u007a" for c in text)
    has_indic = any("\u0900" <= c <= "\u0DFF" for c in text)  # Covers Devanagari + South Indian scripts
    return has_latin and has_indic


# Singleton
_sarvam_stt: Optional[SarvamSTT] = None


def get_sarvam_stt() -> SarvamSTT:
    global _sarvam_stt
    if _sarvam_stt is None:
        _sarvam_stt = SarvamSTT()
    return _sarvam_stt
