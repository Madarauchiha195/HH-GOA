"""
Application configuration — reads from .env / environment variables.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────
    sarvam_api_key: str = Field(default="", description="Sarvam AI API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")

    # ── Embedding & Vector DB ─────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "data/indexes/faiss.index"
    bm25_index_path: str = "data/indexes/bm25.pkl"
    metadata_path: str = "data/indexes/metadata.json"
    chunk_text_path: str = "data/indexes/chunks.json"

    # ── Reranker ──────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True

    # ── Retrieval ─────────────────────────────────────────────
    top_k_dense: int = 30
    top_k_bm25: int = 30
    top_k_fused: int = 15
    top_k_rerank: int = 5
    rrf_k: int = 60
    retrieval_confidence_threshold: float = 0.3

    # ── LLM ───────────────────────────────────────────────────
    llm_provider: str = "gemini"
    llm_model: str = "gemini-1.5-flash"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1

    # ── Performance ───────────────────────────────────────────
    fast_mode: bool = False
    request_timeout_seconds: int = 30
    embedding_batch_size: int = 32

    # ── Dataset ───────────────────────────────────────────────
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_sample_size: int = 10_000
    dataset_languages: str = "en,hi,kn,mr"

    # ── Chunking ──────────────────────────────────────────────
    chunk_strategy: str = "sliding_window"
    chunk_size: int = 256
    chunk_overlap: int = 64

    # ── Application ───────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ── Sarvam STT ────────────────────────────────────────────
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_max_retries: int = 3
    sarvam_stt_timeout: int = 15

    @property
    def dataset_language_list(self) -> List[str]:
        return [l.strip() for l in self.dataset_languages.split(",") if l.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
