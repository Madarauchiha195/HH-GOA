import React from "react";

interface LatencyPanelProps {
  timings: Record<string, number>;
}

export default function LatencyPanel({ timings }: LatencyPanelProps) {
  const total = timings.total || Object.values(timings).reduce((a, b) => a + b, 0);

  const STAGES = [
    { key: "stt", label: "STT", color: "#fb7185" }, // rose
    { key: "embedding", label: "Embed", color: "#f59e0b" }, // amber
    { key: "dense", label: "Dense", color: "#34d399" }, // emerald
    { key: "bm25", label: "BM25", color: "#38bdf8" }, // sky
    { key: "fusion", label: "Fuse", color: "#818cf8" }, // indigo
    { key: "rerank", label: "Rerank", color: "#c084fc" }, // purple
    { key: "generation", label: "Gen", color: "#f472b6" }, // pink
    { key: "validation", label: "Guard", color: "#94a3b8" }, // slate
  ];

  return (
    <div className="glass-card fade-in" style={{ padding: 20, marginTop: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: 500 }}>
          Pipeline Latency
        </div>
        <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
          {total.toFixed(2)}s Total
        </div>
      </div>

      <div style={{
        display: "flex",
        height: 8,
        borderRadius: 4,
        overflow: "hidden",
        background: "rgba(255,255,255,0.05)",
        marginBottom: 16,
      }}>
        {STAGES.map(({ key, color }) => {
          const val = timings[key] || 0;
          if (val <= 0) return null;
          const pct = Math.max((val / total) * 100, 1);
          return (
            <div
              key={key}
              style={{
                width: `${pct}%`,
                background: color,
                height: "100%",
                transition: "width 1s cubic-bezier(0.4, 0, 0.2, 1)",
              }}
              title={`${val.toFixed(2)}s`}
            />
          );
        })}
      </div>

      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "12px 16px",
      }}>
        {STAGES.map(({ key, label, color }) => {
          const val = timings[key];
          if (val === undefined || val <= 0) return null;
          return (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                {label} <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{val.toFixed(2)}s</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
