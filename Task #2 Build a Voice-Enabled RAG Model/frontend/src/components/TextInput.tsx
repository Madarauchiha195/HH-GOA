import { useState, FormEvent } from "react";

interface TextInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

export default function TextInput({ onSubmit, isLoading }: TextInputProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query);
      setQuery("");
    }
  };

  return (
    <div className="glass-card fade-in" style={{ padding: "24px", textAlign: "center" }}>
      <div style={{ marginBottom: 20, color: "var(--text-secondary)", fontSize: "0.95rem" }}>
        Type your question below in any language
      </div>

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8, maxWidth: 500, margin: "0 auto" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
          placeholder="e.g. What is the capital of India?"
          style={{
            flex: 1,
            padding: "12px 16px",
            borderRadius: 12,
            border: "1px solid var(--border-subtle)",
            background: "rgba(255,255,255,0.03)",
            color: "var(--text-primary)",
            fontSize: "1rem",
            outline: "none",
          }}
        />
        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          style={{
            padding: "0 20px",
            borderRadius: 12,
            border: "none",
            background: query.trim() && !isLoading ? "var(--gradient-button)" : "rgba(255,255,255,0.1)",
            color: query.trim() && !isLoading ? "white" : "var(--text-secondary)",
            cursor: query.trim() && !isLoading ? "pointer" : "not-allowed",
            fontWeight: 600,
            transition: "all 0.2s ease",
          }}
        >
          Ask
        </button>
      </form>
    </div>
  );
}
