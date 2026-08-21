// ── API types matching backend Pydantic schemas ──────────────────────────────

export interface LatencyBreakdown {
  stt_ms?: number;
  query_processing_ms: number;
  embedding_ms: number;
  dense_search_ms: number;
  bm25_ms: number;
  fusion_ms: number;
  rerank_ms: number;
  generation_ms: number;
  grounding_ms: number;
  validation_ms: number;
  total_rag_ms: number;
  total_voice_ms?: number;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  language: string;
  strategy: string;
  text_excerpt: string;
  score: number;
  rank: number;
}

export interface GuardrailResult {
  passed: boolean;
  guardrail_type: string;
  reason: string;
  action: string;
}

export interface RAGResponse {
  request_id: string;
  timestamp: string;
  query: string;
  transcript?: string;
  detected_language?: string;
  answer: string;
  should_answer: boolean;
  confidence: number;
  grounded: boolean;
  abstention_reason?: string;
  sources: Citation[];
  timings: LatencyBreakdown;
  guardrail_decisions: GuardrailResult[];
  status: "success" | "abstained" | "error";
  error?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  indexes_loaded: boolean;
  embedding_model_loaded: boolean;
  reranker_loaded: boolean;
  llm_configured: boolean;
  sarvam_configured: boolean;
  index_doc_count: number;
  uptime_seconds: number;
}

export interface PercentileStats {
  p50: number;
  p70: number;
  p95: number;
  p100: number;
  mean: number;
  min: number;
  max: number;
  count: number;
}

export interface MetricsResponse {
  retrieval_latency: PercentileStats;
  rag_latency: PercentileStats;
  voice_latency?: PercentileStats;
  grounding_rate: number;
  abstention_rate: number;
  total_requests: number;
  uptime_seconds: number;
}

export interface ConfigResponse {
  embedding_model: string;
  reranker_model: string;
  reranker_enabled: boolean;
  top_k_dense: number;
  top_k_bm25: number;
  top_k_fused: number;
  top_k_rerank: number;
  rrf_k: number;
  retrieval_confidence_threshold: number;
  fast_mode: boolean;
  llm_model: string;
  loaded_languages: string[];
  chunk_count: number;
}
