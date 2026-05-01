# DocMind — Adaptive RAG Document Assistant

> Upload documents. Ask anything. See exactly how every answer was built.

DocMind is a production-grade **Adaptive Retrieval-Augmented Generation** system that routes each question to the right knowledge source, verifies its own answers for hallucinations, and exposes every step of the reasoning to the user — in real time.

---

## Quick Start

### Option A — One command (local Python)

```bash
cd adaptive_rag
cp .env.example .env        # fill in GROQ_API_KEY and TAVILY_API_KEY
pip install -r requirements.txt
python start.py             # starts backend + frontend + opens browser
```

### Option B — Docker (runs everything, no Python install needed)

```bash
cd adaptive_rag
cp .env.example .env        # fill in GROQ_API_KEY and TAVILY_API_KEY
docker compose up --build
```

Then open **http://localhost:8501**.

---

## Architecture

```
 Browser
   │
   ▼
┌─────────────────────────────────┐
│     Streamlit Frontend          │  app.py — upload, chat, pipeline animation
│     http://localhost:8501       │  real-time SSE token streaming
└──────────────┬──────────────────┘
               │ HTTP + SSE
               ▼
┌─────────────────────────────────┐
│     FastAPI Backend             │  main.py — uvicorn, CORS, lifespan hooks
│     http://localhost:8080       │
│                                 │
│  ┌───────────────────────────┐  │
│  │   LangGraph Pipeline      │  │
│  │                           │  │
│  │  route_question           │  │  Groq LLM classifies: index / web / general
│  │       │                   │  │
│  │   ┌───┴───┐               │  │
│  │ index   search  general   │  │
│  │   │       │        │      │  │
│  │ retrieve  Tavily   │      │  │  sentence-transformers cosine search
│  │   │       │        │      │  │
│  │ grade_docs│        │      │  │  batch LLM relevance scoring
│  │   │       │        │      │  │
│  │   └───────┴────────┘      │  │
│  │           │               │  │
│  │       generate            │  │  Groq token streaming
│  │     + grounding check     │  │  sentence-level hallucination detection
│  │     + knowledge gap alert │  │  completeness analysis
│  └───────────────────────────┘  │
│                                 │
└──────┬──────────────┬───────────┘
       │              │
       ▼              ▼
  ┌─────────┐   ┌──────────┐
  │  Qdrant │   │ MongoDB  │
  │ vectors │   │ history  │
  └─────────┘   └──────────┘
```

---

## Features

| Feature | What it does |
|---|---|
| **Adaptive routing** | Each question is classified into `index` (document), `web` (Tavily search), or `general` (LLM knowledge) — zero hardcoding |
| **Parent-child chunking** | 400-char child vectors are searched; the LLM always receives the full 1500-char parent context |
| **Low-match fallback** | If the best chunk similarity is < 0.3 on the first attempt, skips the rewrite loop and goes straight to web search |
| **Query rewriting** | When retrieved chunks are irrelevant, rewrites the question and retries (up to 2 loops, then web search) |
| **Real token streaming** | LLM tokens stream to the browser via SSE as they're generated — not post-hoc word splitting |
| **Grounding check** | Every sentence in the answer is verified against retrieved chunks: Verified / Inferred / Unsupported |
| **Knowledge gap alerts** | Flags when the document doesn't have enough to answer confidently; suggests what to upload |
| **Answer evolution** | Shows how the answer improved across retrieval attempts when query rewriting runs |
| **Multi-document support** | Upload up to 5 documents per session; all are searched together on every query |
| **Source excerpts** | Shows the exact document chunks used to build each answer, with source filename labels |
| **Dynamic suggestions** | After upload, generates 4 document-specific starter questions by reading actual content |
| **Pipeline animation** | Live step-by-step visualization of routing → retrieval → grading → generation |
| **Cost & token tracking** | Per-query and session-level LLM cost estimation, visible in the sidebar |
| **Persistent storage** | Qdrant falls back to local disk; MongoDB stores chat history — both survive restarts |
| **Session isolation** | Each browser session gets its own Qdrant collection namespace — uploads never bleed |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Get free at [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | ✅ | Get free at [tavily.com](https://tavily.com) — used for web search fallback |
| `GROQ_MODEL` | | Default: `llama-3.3-70b-versatile` |
| `QDRANT_URL` | | Default: `http://localhost:6333` (falls back to local disk if unreachable) |
| `MONGO_URI` | | Default: `mongodb://localhost:27017` |
| `RELEVANCE_THRESHOLD` | | Chunk relevance cutoff (default `0.6`) |
| `LOW_MATCH_THRESHOLD` | | Fast-path to web search below this score (default `0.3`) |
| `MAX_LOOP_COUNT` | | Max query-rewrite retries before web search (default `2`) |
| `API_PORT` | | Backend port (default `8080`) |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Run a question, return full answer + telemetry |
| `POST` | `/api/chat/stream` | Same, streamed as server-sent events |
| `POST` | `/api/upload` | Ingest a document (PDF, TXT, DOCX, MD, CSV — max 10 MB) |
| `GET` | `/api/sessions/{id}` | Last 20 messages for a session |
| `GET` | `/api/suggestions/{id}` | 4 AI-generated starter questions for the session's document |
| `GET` | `/health` | Liveness probe |

Interactive docs at **http://localhost:8080/docs**.

---

## Project Structure

```
adaptive_rag/
├── src/
│   ├── agents/
│   │   ├── graph.py        # LangGraph StateGraph wiring
│   │   ├── nodes.py        # route_question, retrieve, grade, generate
│   │   ├── edges.py        # routing decisions (decide_after_grading)
│   │   └── state.py        # GraphState TypedDict
│   ├── api/
│   │   ├── main.py         # FastAPI app, CORS, lifespan
│   │   ├── schemas.py      # Pydantic request/response models
│   │   └── routers/
│   │       ├── chat.py     # /api/chat + /api/chat/stream (SSE)
│   │       ├── upload.py   # /api/upload + /api/sessions/{id}
│   │       └── suggestions.py  # /api/suggestions/{id}
│   ├── core/
│   │   ├── config.py       # Pydantic Settings from .env
│   │   ├── database.py     # Qdrant + MongoDB singletons
│   │   └── logging.py      # structlog + OpenTelemetry
│   └── services/
│       ├── ingestion.py    # parent-child chunking + Qdrant upsert
│       ├── retrieval.py    # cosine search, retrieve_with_metadata()
│       ├── grounding_checker.py  # sentence-level hallucination check
│       ├── knowledge_gap_analyzer.py  # completeness scoring
│       ├── search.py       # Tavily web search
│       └── cost_tracker.py # tiktoken-based USD estimation
├── eval/
│   └── ragas_eval.py       # offline faithfulness + relevancy scoring
├── app.py                  # Streamlit frontend
├── main.py                 # Uvicorn entry point
├── start.py                # one-command local launcher
├── Dockerfile
├── docker-compose.yml
├── WALKTHROUGH.md          # plain-English UI guide (every element explained)
└── requirements.txt
```

---

## Testing

```bash
# Unit tests — all LLM calls are mocked
pytest src/tests -v

# Offline Ragas evaluation (requires the API running and a document uploaded)
python -m eval.ragas_eval
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Agent graph | LangGraph |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| Vector DB | Qdrant (Docker or local disk fallback) |
| Chat history | MongoDB |
| Web search | Tavily |
| Backend | FastAPI + Pydantic v2 + Uvicorn |
| Frontend | Streamlit |
| Observability | structlog + OpenTelemetry |
| Evaluation | Ragas |
