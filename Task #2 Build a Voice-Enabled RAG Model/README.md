# HH Goa 2026 — Voice-Enabled Multilingual RAG System

A production-quality voice-enabled RAG system using `ai4bharat/MSMARCO-XI`, Sarvam AI STT, hybrid retrieval, and an advanced Guardrail framework.

## Features

- **Voice & Text Queries**: Support for 10 Indic languages via Sarvam API.
- **Hybrid Retrieval**: Dense (E5) + BM25 with Reciprocal Rank Fusion (RRF).
- **Advanced Guardrails**: Input/Output filtering and grounding checks.
- **Modern Frontend**: Next.js 14 App Router, Glassmorphism UI, Framer animations.
- **Evaluation Suite**: Extensive latency, retrieval, and guardrail evaluation frameworks.

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 20+
- Docker & Docker Compose

### Environment Variables
Copy `.env.example` to `.env` and fill in:
```bash
SARVAM_API_KEY=your_key
OPENAI_API_KEY=your_key  # Or GEMINI_API_KEY
```

### Running Locally

1. Start Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

2. Start Frontend:
```bash
cd frontend
npm install
npm run dev
```

### Using Docker
```bash
docker-compose up --build
```
