// ── API client ────────────────────────────────────────────────────────────────

import type { RAGResponse, HealthResponse, MetricsResponse, ConfigResponse } from "./types";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export async function queryText(query: string): Promise<RAGResponse> {
  const res = await fetch(`${BACKEND_URL}/api/v1/text/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return handleResponse<RAGResponse>(res);
}

export async function queryVoice(audioBlob: Blob): Promise<RAGResponse> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  const res = await fetch(`${BACKEND_URL}/api/v1/voice/query`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<RAGResponse>(res);
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BACKEND_URL}/api/v1/health`);
  return handleResponse<HealthResponse>(res);
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${BACKEND_URL}/api/v1/metrics`);
  return handleResponse<MetricsResponse>(res);
}

export async function fetchConfig(): Promise<ConfigResponse> {
  const res = await fetch(`${BACKEND_URL}/api/v1/config`);
  return handleResponse<ConfigResponse>(res);
}
