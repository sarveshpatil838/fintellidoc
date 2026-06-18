# FintelliDoc — Financial Document Intelligence Platform

> **AI-powered extraction, classification, and analysis of financial documents using Anthropic Claude API + RAG**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Anthropic Claude](https://img.shields.io/badge/Claude-API-orange.svg)](https://www.anthropic.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-green.svg)](.github/workflows/ci.yml)

---

## What It Does

FintelliDoc is a production-grade REST API that ingests unstructured financial documents (10-Ks, earnings releases, analyst reports, contracts) and returns structured, validated intelligence:

- **Entity Extraction** — companies, executives, financial figures, dates, risk factors
- **Document Classification** — earnings call, regulatory filing, contract, research report, etc.
- **Semantic Q&A** — ask natural language questions against a document corpus via RAG
- **Risk Signal Detection** — surface material risk language, covenant violations, going-concern flags
- **Comparative Analysis** — diff two documents to identify what changed between periods

The system never silently returns malformed data. Every LLM output passes through a Pydantic validation layer with structured retry logic. If the model fails after retries, the API returns an explicit error — not corrupted output.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
│                                                         │
│  POST /extract    POST /classify    POST /query         │
│  POST /risks      POST /compare     GET  /health        │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐
    │  Extraction │ │    RAG     │ │ Validation │
    │   Service  │ │  Service   │ │   Layer    │
    │            │ │            │ │            │
    │ Claude API │ │ FAISS +    │ │ Pydantic + │
    │ structured │ │ Embeddings │ │ Retry +    │
    │  prompting │ │            │ │ Logging    │
    └─────┬──────┘ └─────┬──────┘ └────┬───────┘
          │              │              │
          └──────────────▼──────────────┘
                         │
                  ┌──────▼──────┐
                  │  PostgreSQL │
                  │  (results + │
                  │  doc store) │
                  └─────────────┘
```

---

## Key Engineering Decisions

### 1. Validation-First AI Pipeline
LLM outputs are unpredictable. FintelliDoc enforces a strict contract: every extraction passes through a Pydantic schema. If the model returns malformed JSON or missing required fields, the system retries with exponential backoff (up to 3 attempts) before surfacing an explicit failure. This is the critical difference between a demo and a production system.

### 2. RAG with FAISS for Long Documents
Financial documents routinely exceed Claude's context window. FintelliDoc chunks documents, generates embeddings, stores them in a FAISS vector index, and retrieves the top-k most relevant chunks before constructing the prompt. This allows accurate Q&A over 200-page 10-K filings.

### 3. Streaming for Long Extractions
For large documents, the `/extract/stream` endpoint uses FastAPI's `StreamingResponse` with Claude's streaming API, so clients receive partial results immediately rather than waiting for the full extraction.

### 4. Tool Use for Structured Extraction
Rather than asking Claude to return JSON in free text (fragile), FintelliDoc uses Claude's tool use / function calling API. The extraction schema is defined as a tool, and Claude is forced to call it — producing structured output that's trivially parseable.

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/sarveshpatil838/fintellidoc
cd fintellidoc
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 2. Run with Docker
docker-compose up --build

# 3. Test the API
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple Inc. reported Q3 2024 revenue of $85.8 billion...", "doc_type": "earnings"}'
```

API docs available at: `http://localhost:8000/docs`

---

## API Reference

### `POST /api/v1/extract`
Extract structured entities from a financial document.

**Request:**
```json
{
  "text": "string",
  "doc_type": "earnings | 10k | contract | report | unknown",
  "extract_fields": ["entities", "financials", "risks", "dates"]
}
```

**Response:**
```json
{
  "doc_id": "uuid",
  "doc_type": "earnings",
  "entities": [{"name": "Apple Inc.", "type": "company", "confidence": 0.98}],
  "financials": [{"metric": "revenue", "value": 85.8, "unit": "billion USD", "period": "Q3 2024"}],
  "risks": [],
  "processing_time_ms": 1243,
  "model": "claude-sonnet-4-6",
  "retry_count": 0
}
```

### `POST /api/v1/query`
Ask a natural language question against indexed documents (RAG).

### `POST /api/v1/risks`
Detect material risk signals: going-concern language, covenant violations, litigation flags.

### `POST /api/v1/compare`
Diff two documents and return a structured change summary.

---

## Project Structure

```
fintellidoc/
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── api/
│   │   └── routes.py           # All API endpoints
│   ├── core/
│   │   ├── config.py           # Settings via pydantic-settings
│   │   └── logging.py          # Structured JSON logging
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   └── services/
│       ├── extraction.py       # Claude API + tool use extraction
│       ├── rag.py              # FAISS vector store + retrieval
│       ├── classification.py   # Document type classification
│       ├── risk_detection.py   # Risk signal extraction
│       └── validation.py       # Retry + validation layer
├── tests/
│   ├── test_extraction.py
│   ├── test_rag.py
│   └── test_validation.py
├── scripts/
│   └── seed_index.py           # Seed FAISS index with sample docs
├── .github/
│   └── workflows/ci.yml        # Lint + test on push
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI 0.111 |
| AI Model | Anthropic Claude (claude-sonnet-4-6) |
| Vector Store | FAISS |
| Embeddings | sentence-transformers |
| Validation | Pydantic v2 |
| Retry Logic | tenacity |
| Database | PostgreSQL + asyncpg |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-asyncio |
| Logging | structlog (JSON) |

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v --asyncio-mode=auto
```

---

## Author

**Sarvesh Patil** — M.S. Computer Science, University of Dayton  
[github.com/sarveshpatil838](https://github.com/sarveshpatil838) · [linkedin.com/in/sarveshpatil838](https://linkedin.com/in/sarveshpatil838)
