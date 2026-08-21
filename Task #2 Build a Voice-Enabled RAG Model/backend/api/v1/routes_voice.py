"""Voice query endpoint — accepts audio file, runs STT + RAG pipeline."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.schemas.models import RAGResponse
from backend.services.orchestrator import Orchestrator

logger = structlog.get_logger(__name__)
router = APIRouter()

ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "application/octet-stream",  # Some browsers send this for .webm
}

MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/voice/query", response_model=RAGResponse)
async def voice_query(
    audio: UploadFile = File(..., description="Audio file (WAV/WebM/MP3)"),
    language: str = Form(default="", description="Optional language hint (e.g. 'hi')"),
) -> RAGResponse:
    """
    Full voice pipeline:
    1. Validate audio file
    2. Sarvam STT → transcript
    3. Full RAG pipeline on transcript
    4. Return structured response
    """
    # ── Validate content type ─────────────────────────────────────────────────
    content_type = (audio.content_type or "").lower()
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {content_type}. Supported: WAV, WebM, MP3, OGG.",
        )

    # ── Read & validate size ──────────────────────────────────────────────────
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty. Please try recording again.")
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_bytes) // 1024} KB). Maximum: 10 MB.",
        )

    try:
        orchestrator = Orchestrator()
        response = await orchestrator.run_voice_pipeline(
            audio_bytes=audio_bytes,
            audio_content_type=content_type,
            language_hint=language.strip() or None,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Voice query failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="I couldn't process the audio. Please try recording again.",
        ) from exc
