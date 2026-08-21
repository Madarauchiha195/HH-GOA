import React from "react";

interface TranscriptCardProps {
  transcript: string;
  language: string;
}

export default function TranscriptCard({ transcript, language }: TranscriptCardProps) {
  return (
    <div className="glass-card fade-in" style={{ padding: 24, marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <div style={{
          width: 28, height: 28,
          background: "rgba(99,102,241,0.15)",
          color: "#818cf8",
          borderRadius: 8,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "0.85rem",
          fontWeight: 600,
        }}>
          STT
        </div>
        <div style={{ color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 500 }}>
          Transcribed Query
        </div>
        <div style={{ flex: 1 }} />
        <div style={{
          padding: "4px 8px",
          background: "rgba(255,255,255,0.05)",
          border: "1px solid var(--border-subtle)",
          borderRadius: 6,
          fontSize: "0.75rem",
          color: "var(--text-secondary)",
          textTransform: "uppercase",
        }}>
          {language}
        </div>
      </div>
      <div style={{ fontSize: "1.1rem", fontWeight: 500, lineHeight: 1.5 }}>
        "{transcript}"
      </div>
    </div>
  );
}
