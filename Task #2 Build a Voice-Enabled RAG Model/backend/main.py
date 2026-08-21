"""
HH Goa 2026 — Voice-Enabled RAG System
Backend entrypoint (FastAPI)
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.v1 import routes_health, routes_text, routes_voice
from backend.config import settings
from backend.middleware.logging import configure_logging
from backend.middleware.metrics import MetricsCollector
from backend.services.retrieval.index_manager import IndexManager

logger = structlog.get_logger(__name__)


# ── Lifespan (warm startup / teardown) ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources asynchronously without blocking port binding."""
    configure_logging()
    logger.info("🚀 HH Goa Voice RAG starting up…")

    # Warm up the index manager in background
    asyncio.create_task(IndexManager.get_instance())
    logger.info("✅ Server startup complete - listening for requests")

    yield  # ── Application runs ──────────────────────────────

    logger.info("🛑 Shutting down…")
    IndexManager.reset()


# ── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="HH Goa 2026 — Voice-Enabled Multilingual RAG",
        description=(
            "Production-quality voice-enabled RAG over ai4bharat/MSMARCO-XI "
            "with Sarvam STT, hybrid retrieval, reranking, and grounding guardrails."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(routes_health.router, prefix="/api/v1", tags=["health"])
    app.include_router(routes_text.router,   prefix="/api/v1", tags=["query"])
    app.include_router(routes_voice.router,  prefix="/api/v1", tags=["voice"])

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({"status": "ok", "docs": "/api/docs"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level="info",
    )
