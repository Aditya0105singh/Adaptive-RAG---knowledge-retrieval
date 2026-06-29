# Adaptive RAG vs Naive RAG — Benchmark Results

Both systems share the same Qdrant collection, Cohere embeddings, Groq
pipeline model, and the same 8-question test set. The only
difference is the adaptive control flow (routing, grading, self-correcting
rewrites). RAGAS scores are reference-free, judged by `llama-3.1-8b-instant`.
Adaptive RAGAS is computed over the 7 questions that retrieved from
the document (others routed to web search and are excluded from context metrics).

## Performance Comparison

| Metric | Naive RAG | Adaptive RAG | Improvement |
|--------|-----------|--------------|-------------|
| Faithfulness | 0.759 | 0.816 | +7% |
| Answer Relevancy | 0.528 | 0.500 | -5% |
| Context Precision | 0.875 | 1.000 | +14% |
| Avg Latency (ms) | 1089 | 4773 | +338% |
| Unnecessary Retrievals | always retrieves | routed | eliminated on general/search |

> Note: the adaptive system trades some latency for the grading and
> optional rewrite steps; the payoff is higher faithfulness and the
> elimination of retrievals on questions that need none.

## Feature Comparison

| Feature | Naive RAG | Adaptive RAG |
|---------|-----------|--------------|
| Query Routing (index/search/general) | No | Yes |
| Relevance Grading | No | Yes (0.6 threshold) |
| Query Rewriting | No | Yes (max 2 loops) |
| Web Search Fallback | No | Yes (Tavily) |
| Hallucination Detection | No | Yes (sentence-level) |
| Answer Versioning | No | Yes |
| Knowledge Gap Alerts | No | Yes |
| Session Isolation | No | Yes |

_RAGAS judge: project Groq model. Generated from `ragas_results.json` and `naive_rag_results.json`._
