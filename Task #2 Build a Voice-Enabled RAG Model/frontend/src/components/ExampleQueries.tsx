import React from "react";

interface ExampleQueriesProps {
  onSelect: (query: string) => void;
  onSwitchToText: () => void;
}

const QUERIES = [
  { text: "What are the rules for H-1B visa?", lang: "en" },
  { text: "भारत के संविधान में कितने अनुच्छेद हैं?", lang: "hi" },
  { text: "कर्नाटक की राजधानी क्या है?", lang: "hi" },
  { text: "Rules for driving license renewal in Maharashtra kya hai?", lang: "mix" }
];

export default function ExampleQueries({ onSelect, onSwitchToText }: ExampleQueriesProps) {
  return (
    <div className="fade-in" style={{ marginTop: 40, textAlign: "center" }}>
      <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 16, fontWeight: 500 }}>
        OR TRY AN EXAMPLE
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
        {QUERIES.map((q, idx) => (
          <button
            key={idx}
            onClick={() => {
              onSwitchToText();
              onSelect(q.text);
            }}
            style={{
              padding: "10px 16px",
              borderRadius: 100,
              border: "1px solid var(--border-subtle)",
              background: "rgba(255,255,255,0.02)",
              color: "var(--text-primary)",
              fontSize: "0.9rem",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
            onMouseOver={(e) => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--accent-primary)";
              (e.currentTarget as HTMLButtonElement).style.background = "rgba(99,102,241,0.05)";
            }}
            onMouseOut={(e) => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border-subtle)";
              (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.02)";
            }}
          >
            {q.text}
          </button>
        ))}
      </div>
    </div>
  );
}
