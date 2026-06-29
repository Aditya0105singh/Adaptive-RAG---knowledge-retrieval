# Adaptive RAG Knowledge Retrieval System
### Technical Documentation & Performance Report

## What Is This System?

Adaptive RAG is an intelligent document Q&A system that goes far beyond a
standard chatbot. Unlike basic RAG systems that always retrieve from a vector
database regardless of query type, this system uses a LangGraph agentic pipeline
with tri-route classification to decide the optimal approach for each query.
The intended audience is engineers and researchers building production
question-answering systems.

## The Three Routes

**Index Route:** When a document is uploaded and the question is
document-specific, the system retrieves relevant chunks from the Qdrant vector
database using semantic search, grades them for relevance, and generates a
grounded answer.

**Web Search Route:** When a question requires real-time information (news,
prices, recent events), the system routes to the Tavily web search API,
retrieves current information, and generates a sourced answer.

**General Route:** For general-knowledge questions unrelated to any uploaded
document, the system answers directly from LLaMA 3.3's training data without any
retrieval overhead.

## Novel Features

**Feature 1: Hallucination Grounding Score.** Every sentence in the generated
answer is classified as GROUNDED (directly supported by retrieved chunks),
INFERRED (logically follows from chunks), or UNGROUNDED (possible
hallucination). A trust score from 0 to 100% is computed and displayed. This
runs using a parallel ThreadPoolExecutor for efficiency.

**Feature 2: Answer Versioning.** When a query rewrite occurs (relevance scores
below the 0.60 threshold), the system stores both the original and refined
answer attempts, showing exactly how retrieval quality improved across loops.

**Feature 3: Knowledge Gap Alerts.** After answering, the system analyzes
whether the document contained sufficient information. If not, it identifies
exactly what is missing and suggests which document types to upload next.
Results are cached for 10 minutes to avoid redundant LLM calls.

## Technical Architecture

- Backend: FastAPI with async SSE streaming (port 8080)
- Frontend: Streamlit with real-time token display (port 8501)
- LLM: Groq LLaMA 3.3 70B Versatile (temperature=0)
- Vector DB: Qdrant Cloud (cosine similarity, 384-dim vectors)
- Embeddings: Cohere embed-english-light-v3.0
- Web Search: Tavily API (RAG-optimized results)
- Workflow: LangGraph StateGraph with MemorySaver

## Chunking Strategy

The system uses parent-child chunking. Child chunks of 400 characters are
embedded for precise search, while parent chunks of 1500 characters are returned
for rich context. This improves retrieval precision while preserving answer
quality.

## Performance Benchmarks (RAGAS Evaluation)

Tested against a Naive RAG baseline (no routing, no grading, no retry):

| Metric | Naive RAG | Adaptive RAG | Improvement |
|--------|-----------|--------------|-------------|
| Faithfulness | 0.759 | 0.816 | +7.5% |
| Context Precision | 0.875 | 1.000 | +14.3% |
| Routing Accuracy | N/A | 100% (25/25) | — |

## Key Design Decisions

**Relevance threshold 0.60:** Chunks below this score are filtered out before
generation, ensuring only relevant context reaches the LLM.

**Low match threshold 0.30:** If the best chunk score is below 0.30 on the first
attempt, query rewriting is skipped and the system immediately falls back to web
search, avoiding a wasted retry loop.

**Max 2 retry loops:** Balances quality improvement with latency. Most
improvement happens in the first rewrite, with diminishing returns after two
attempts.

**Session-scoped collections:** Each user gets an isolated Qdrant collection
(session_{id}) preventing data leakage between users.

## Limitations

The grading and rewrite steps add language-model calls, increasing latency and
token cost relative to single-shot RAG. Free-tier rate limits on the
language-model provider cap throughput, and the quality of web-search answers
depends on the external provider.

## Conclusion

By adding routing, relevance grading, self-correcting query rewrites, and
sentence-level grounding on top of retrieval, the system converts a naive RAG
pipeline into an adaptive, transparent one — more faithful answers, fewer wasted
retrievals, and a glass-box view of how each answer was produced.
