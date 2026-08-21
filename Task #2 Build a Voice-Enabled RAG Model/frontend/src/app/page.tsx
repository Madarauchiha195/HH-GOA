"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import VoiceRecorder from "@/components/VoiceRecorder";
import TranscriptCard from "@/components/TranscriptCard";
import AnswerCard from "@/components/AnswerCard";
import LatencyPanel from "@/components/LatencyPanel";
import ExampleQueries from "@/components/ExampleQueries";
import TextInput from "@/components/TextInput";
import type { RAGResponse } from "@/lib/types";
import { queryText, queryVoice } from "@/lib/api";

export default function Home() {
  const [mode, setMode] = useState<"voice" | "text">("voice");
  const [response, setResponse] = useState<RAGResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState("");
  const answerRef = useRef<HTMLDivElement>(null);

  const handleTextQuery = useCallback(async (query: string) => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setCurrentQuery(query);
    try {
      const result = await queryText(query);
      setResponse(result);
      setTimeout(() => answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (err: any) {
      setError(err.message || "Failed to process query. Please try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleVoiceQuery = useCallback(async (audioBlob: Blob) => {
    setLoading(true);
    setError(null);
    setCurrentQuery("");
    try {
      const result = await queryVoice(audioBlob);
      setResponse(result);
      setCurrentQuery(result.transcript || "");
      setTimeout(() => answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (err: any) {
      setError(err.message || "I couldn't process the audio. Please try recording again.");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <main style={{ minHeight: "100vh" }}>
      {/* ── Nav ───────────────────────────────────────────────────── */}
      <nav style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        padding: "12px 24px",
        borderBottom: "1px solid var(--border-subtle)",
        background: "rgba(8, 10, 18, 0.85)",
        backdropFilter: "blur(20px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32,
            background: "var(--gradient-button)",
            borderRadius: 8,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>🎙️</div>
          <span style={{ fontWeight: 700, fontSize: "1rem", letterSpacing: "-0.01em" }}>
            HH Goa Voice RAG
          </span>
          <span style={{
            padding: "2px 8px",
            borderRadius: 6,
            fontSize: "0.7rem",
            fontWeight: 600,
            background: "rgba(99,102,241,0.15)",
            color: "var(--accent-primary)",
            border: "1px solid rgba(99,102,241,0.3)",
          }}>2026</span>
        </div>

        <div style={{ display: "flex", gap: 4 }}>
          <Link href="/" className={`nav-item ${true ? "active" : ""}`}>Home</Link>
          <Link href="/dashboard" className="nav-item">Dashboard</Link>
        </div>
      </nav>

      {/* ── Hero ──────────────────────────────────────────────────── */}
      <section style={{
        textAlign: "center",
        padding: "64px 24px 40px",
        maxWidth: 800,
        margin: "0 auto",
      }}>
        <div style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 8,
          padding: "4px 14px",
          borderRadius: 100,
          background: "rgba(99,102,241,0.1)",
          border: "1px solid rgba(99,102,241,0.2)",
          marginBottom: 24,
          fontSize: "0.8rem",
          color: "#818cf8",
          fontWeight: 500,
        }}>
          <span>●</span> MSMARCO-XI · Sarvam STT · Hybrid Retrieval
        </div>

        <h1 style={{ fontSize: "clamp(2rem, 5vw, 3.5rem)", fontWeight: 800, lineHeight: 1.1, marginBottom: 16 }}>
          Ask in{" "}
          <span className="gradient-text">any language</span>
          <br />Get grounded answers
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto 32px" }}>
          Voice-powered RAG over MSMARCO-XI with dense + BM25 hybrid retrieval, 
          reranking, guardrails, and real latency instrumentation.
        </p>

        {/* ── Mode Toggle ────────────────────────────────────────── */}
        <div style={{
          display: "inline-flex",
          gap: 0,
          background: "var(--bg-glass)",
          border: "1px solid var(--border-subtle)",
          borderRadius: 12,
          padding: 4,
          marginBottom: 48,
        }}>
          {(["voice", "text"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                padding: "8px 20px",
                borderRadius: 8,
                border: "none",
                cursor: "pointer",
                fontWeight: 600,
                fontSize: "0.875rem",
                transition: "all 0.2s ease",
                background: mode === m ? "var(--gradient-button)" : "transparent",
                color: mode === m ? "white" : "var(--text-secondary)",
                boxShadow: mode === m ? "0 2px 8px rgba(99,102,241,0.3)" : "none",
              }}
            >
              {m === "voice" ? "🎙️ Voice" : "⌨️ Text"}
            </button>
          ))}
        </div>
      </section>

      {/* ── Main interaction area ──────────────────────────────────── */}
      <section style={{ maxWidth: 720, margin: "0 auto", padding: "0 24px 80px" }}>

        {mode === "voice" ? (
          <VoiceRecorder onAudioReady={handleVoiceQuery} isLoading={loading} />
        ) : (
          <TextInput onSubmit={handleTextQuery} isLoading={loading} />
        )}

        {/* ── Example queries ──────────────────────────────────── */}
        {!response && !loading && (
          <ExampleQueries onSelect={handleTextQuery} onSwitchToText={() => setMode("text")} />
        )}

        {/* ── Error state ──────────────────────────────────────── */}
        {error && (
          <div className="glass-card fade-in" style={{
            marginTop: 24,
            padding: 20,
            borderColor: "rgba(244, 63, 94, 0.3)",
            background: "rgba(244, 63, 94, 0.05)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 20 }}>⚠️</span>
              <div>
                <div style={{ fontWeight: 600, color: "#fb7185", marginBottom: 4 }}>Error</div>
                <div style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>{error}</div>
              </div>
            </div>
          </div>
        )}

        {/* ── Loading skeleton ──────────────────────────────────── */}
        {loading && (
          <div style={{ marginTop: 32 }}>
            {currentQuery && (
              <div className="glass-card" style={{ padding: 20, marginBottom: 16 }}>
                <div style={{ color: "var(--text-secondary)", fontSize: "0.8rem", marginBottom: 8 }}>Processing query…</div>
                <div style={{ fontWeight: 500 }}>"{currentQuery}"</div>
              </div>
            )}
            <div className="glass-card" style={{ padding: 24 }}>
              <div className="skeleton" style={{ height: 16, width: "40%", marginBottom: 16 }} />
              <div className="skeleton" style={{ height: 12, width: "100%", marginBottom: 8 }} />
              <div className="skeleton" style={{ height: 12, width: "90%", marginBottom: 8 }} />
              <div className="skeleton" style={{ height: 12, width: "70%" }} />
            </div>
          </div>
        )}

        {/* ── Results ──────────────────────────────────────────── */}
        {response && !loading && (
          <div ref={answerRef} className="fade-in" style={{ marginTop: 32 }}>
            {response.transcript && (
              <TranscriptCard
                transcript={response.transcript}
                language={response.detected_language}
              />
            )}
            <AnswerCard response={response} />
            <LatencyPanel timings={response.timings} />
          </div>
        )}
      </section>
    </main>
  );
}
