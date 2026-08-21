import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HH Goa 2026 — Voice RAG | Multilingual AI Knowledge Assistant",
  description:
    "Production-quality voice-enabled RAG system with Sarvam STT, hybrid retrieval over MSMARCO-XI, and multilingual Indic language support.",
  keywords: ["RAG", "voice AI", "multilingual", "Indic languages", "Sarvam", "MSMARCO"],
  openGraph: {
    title: "HH Goa 2026 — Voice RAG",
    description: "Ask anything in any Indian language. Voice-powered, retrieval-grounded AI.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        {/* Ambient background orbs */}
        <div className="orb orb-1" aria-hidden="true" />
        <div className="orb orb-2" aria-hidden="true" />
        <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
      </body>
    </html>
  );
}
