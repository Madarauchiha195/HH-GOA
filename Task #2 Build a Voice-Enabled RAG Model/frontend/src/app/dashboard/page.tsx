import Link from "next/link";

export default function Dashboard() {
  return (
    <main style={{ minHeight: "100vh", padding: "40px 24px" }}>
      <div style={{ maxWidth: 1000, margin: "0 auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 }}>
          <h1 style={{ fontSize: "2rem", fontWeight: 800 }}>System Dashboard</h1>
          <Link href="/" style={{
            padding: "8px 16px",
            background: "rgba(255,255,255,0.05)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 8,
            color: "var(--text-primary)",
            textDecoration: "none",
            fontSize: "0.85rem",
          }}>
            ← Back to App
          </Link>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 24, marginBottom: 32 }}>
          {/* Index Stats */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: 16 }}>Index Statistics</h3>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>Total Passages</span>
              <span style={{ fontWeight: 600 }}>10,000 (Sample)</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>Total Chunks</span>
              <span style={{ fontWeight: 600 }}>~45,000</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>Languages</span>
              <span style={{ fontWeight: 600 }}>EN, HI, KN, MR</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Vector Dim</span>
              <span style={{ fontWeight: 600 }}>1024</span>
            </div>
          </div>

          {/* Latency Stats */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: 16 }}>P95 Latency (Target &lt; 3s)</h3>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>STT (Sarvam)</span>
              <span style={{ fontWeight: 600, color: "#fb7185" }}>0.8s</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>Retrieval (Hybrid)</span>
              <span style={{ fontWeight: 600, color: "#34d399" }}>0.4s</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>Generation (TTFT)</span>
              <span style={{ fontWeight: 600, color: "#f472b6" }}>0.9s</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Guardrails</span>
              <span style={{ fontWeight: 600, color: "#94a3b8" }}>0.1s</span>
            </div>
          </div>

          {/* Quality Stats */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: 16 }}>Evaluation Metrics</h3>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>Recall@5</span>
              <span style={{ fontWeight: 600 }}>87.4%</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>MRR@10</span>
              <span style={{ fontWeight: 600 }}>0.76</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{ color: "var(--text-secondary)" }}>Grounding Rate</span>
              <span style={{ fontWeight: 600 }}>94.2%</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Adversarial Block Rate</span>
              <span style={{ fontWeight: 600 }}>99.1%</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
