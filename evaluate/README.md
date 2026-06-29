# Evaluation Suite

Quantitative benchmarks for the Adaptive RAG system. Every script talks to the
**running FastAPI backend** over HTTP (except `naive_rag.py`, which reuses the
project's retrieval/embedding/LLM services directly for the baseline).

## Prerequisites

1. A `.env` with valid keys (`GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
   `COHERE_API_KEY`, `TAVILY_API_KEY`).
2. The backend running:

   ```bash
   python main.py          # serves the API on http://localhost:8080
   ```

   Point at a deployed instance instead with `API_BASE_URL=https://...`.
   If you enable auth (`ENABLE_AUTH=true`), set `EVAL_API_KEY` to the same key.

## Run order

```bash
python evaluate/ragas_eval.py         # 1. uploads sample_doc.md, RAGAS-scores the adaptive system
python evaluate/naive_rag.py          # 2. RAGAS-scores the naive baseline on the same index
python evaluate/compare.py            # 3. writes results/comparison_report.md
python evaluate/test_routing.py       # 4. routing accuracy on 25 labelled cases
python evaluate/benchmark_latency.py  # 5. per-route latency (P50/P95/P99)
```

## What each script does

| Script | Output | Notes |
|--------|--------|-------|
| `ragas_eval.py` | `results/ragas_results.json`, `results/ragas_results.md` | faithfulness, answer_relevancy, context_precision (reference-free) |
| `naive_rag.py` | `results/naive_rag_results.json` | same metrics, no adaptive features |
| `compare.py` | `results/comparison_report.md` | side-by-side table + feature matrix |
| `test_routing.py` | `results/routing_results.json` | index/search/general accuracy |
| `benchmark_latency.py` | `results/benchmark_results.json` | 30 queries, 10 per route |

## Design notes

- **No hidden OpenAI judge.** RAGAS metrics use the project's own Groq model as
  the judge LLM and Cohere embeddings — wired in `_common.py`. This keeps the
  whole benchmark runnable on the free-tier keys the project already uses.
- **Reference-free.** The 15 generic questions need no hand-written ground
  truth: faithfulness and answer-relevancy are reference-free, and context
  precision uses `LLMContextPrecisionWithoutReference`.
- **Rate limits (important).** Groq's free tier caps **~100k tokens/day** and
  ~6k–12k/min. The adaptive pipeline spends ~5 LLM calls per question, so the
  full suite does **not** fit in one free day. Two mitigations are built in:
  the RAGAS judge runs on `llama-3.1-8b-instant` (a *separate* token bucket from
  the 70B pipeline), and the question set defaults to **8** (`EVAL_NUM_QUESTIONS`
  to change). Even so, run the scripts across **two days** on free tier — e.g.
  `ragas_eval` + `naive_rag` + `compare` one day; `test_routing` + `benchmark_latency`
  the next. For a single sitting, upgrade to Groq's pay-as-you-go Dev tier.
- **Routing caveat.** An `index` route can legitimately fall back to `search`
  when the document lacks the answer. `test_routing.py` counts these separately
  so the headline number is honest.
