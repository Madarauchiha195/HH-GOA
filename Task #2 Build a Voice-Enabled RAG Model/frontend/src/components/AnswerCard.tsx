"use client";

import type { RAGResponse } from "@/lib/types";

const LANGUAGE_NAMES: Record<string, string> = {
  en: "🇬🇧 English", hi: "🇮🇳 Hindi", kn: "🇮🇳 Kannada",
  mr: "🇮🇳 Marathi", ta: "🇮🇳 Tamil", te: "🇮🇳 Telugu",
  bn: "🇮🇳 Bengali", gu: "🇮🇳 Gujarati", ml: "🇮🇳 Malayalam",
  pa: "🇮🇳 Punjabi", ur: "🇮🇳 Urdu", codemix: "🔀 Code-mixed",
};

interface AnswerCardProps {
  response: RAGResponse;
}

export default function AnswerCard({ response }: AnswerCardProps) {
  const { answer, should_answer, confidence, grounded, sources, guardrail_decisions, status, abstention_reason } = response;
  const isAbstained = status === "abstained" || !should_answer;

  return (
    <div className="glass-card fade-in" style={{ padding: 24, marginBottom: 16 }}>
      {/* ── Header ───────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: "1.1rem" }}>{isAbstained ? "🤔" : "✨"}</span>
          <span style={{ fontWeight: 700, fontSize: "1rem" }}>
            {isAbstained ? "Unable to Answer" : "Answer"}
          </span>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {/* Grounded badge */}
          {!isAbstained && (
            <span className={`badge ${grounded ? "badge-grounded" : "badge-ungrounded"}`}>
              {grounded ? "✓ Grounded" : "⚠ Low confidence"}
            </span>
          )}

          {/* Status badge */}
          {isAbstained && (
            <span className="badge badge-abstained">
              ○ Abstained
            </span>
          )}

          {/* Confidence */}
          {!isAbstained && (
            <span style={{
              padding: "2px 10px",
              borderRadius: 100,
              fontSize: "0.75rem",
              fontWeight: 600,
              background: "rgba(99,102,241,0.15)",
              color: "#818cf8",
              border: "1px solid rgba(99,102,241,0.3)",
            }}>
              {Math.round(confidence * 100)}% conf
            </span>
          )}
        </div>
      </div>

      {/* ── Confidence bar ────────────────────────────────────── */}
      {!isAbstained && (
        <div style={{
          height: 4,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 4,
          marginBottom: 20,
          overflow: "hidden",
        }}>
          <div
            className="confidence-fill"
            style={{
              width: `${Math.round(confidence * 100)}%`,
              background: grounded
                ? "linear-gradient(90deg, #10b981, #34d399)"
                : "linear-gradient(90deg, #f59e0b, #fbbf24)",
            }}
          />
        </div>
      )}

      {/* ── Answer text ───────────────────────────────────────── */}
      <div style={{
        color: isAbstained ? "var(--text-secondary)" : "var(--text-primary)",
        lineHeight: 1.7,
        fontSize: "1.0rem",
        fontStyle: isAbstained ? "italic" : "normal",
        marginBottom: sources.length > 0 ? 20 : 0,
      }}>
        {answer}
      </div>

      {/* ── Guardrail info ────────────────────────────────────── */}
      {isAbstained && guardrail_decisions.length > 0 && (
        <div style={{
          marginTop: 12,
          padding: "10px 14px",
          background: "rgba(148,163,184,0.05)",
          borderRadius: 8,
          border: "1px solid rgba(148,163,184,0.1)",
          fontSize: "0.8rem",
          color: "var(--text-muted)",
        }}>
          {guardrail_decisions.filter(g => !g.passed).map((g, i) => (
            <div key={i}>🛡 {g.guardrail_type.replace(/_/g, " ")}: {g.reason}</div>
          ))}
        </div>
      )}

      {/* ── Sources ───────────────────────────────────────────── */}
      {sources.length > 0 && !isAbstained && (
        <div>
          <div style={{
            fontWeight: 600,
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
            marginBottom: 10,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}>
            Retrieved Sources ({sources.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {sources.slice(0, 3).map((src, i) => (
              <div key={src.chunk_id} style={{
                padding: "10px 14px",
                background: "rgba(255,255,255,0.03)",
                borderRadius: 10,
                border: "1px solid var(--border-subtle)",
                fontSize: "0.825rem",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, flexWrap: "wrap", gap: 4 }}>
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <span style={{ fontWeight: 600, color: "#818cf8" }}>#{i + 1}</span>
                    <span className="badge badge-language">{LANGUAGE_NAMES[src.language] || src.language}</span>
                    <span style={{
                      padding: "1px 8px", borderRadius: 100, fontSize: "0.7rem",
                      background: "rgba(255,255,255,0.05)", color: "var(--text-muted)",
                      border: "1px solid var(--border-subtle)",
                    }}>
                      {src.strategy}
                    </span>
                  </div>
                  <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                    score: {src.score.toFixed(4)}
                  </span>
                </div>
                <div style={{
                  color: "var(--text-secondary)",
                  lineHeight: 1.5,
                  display: "-webkit-box",
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: "vertical" as const,
                  overflow: "hidden",
                }}>
                  {src.text_excerpt}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
