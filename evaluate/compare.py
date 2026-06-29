"""TASK 3 — Adaptive RAG vs Naive RAG comparison report.

Loads the two result files produced by Tasks 1 and 2 and emits a Markdown
comparison report (metrics table + feature matrix) ready to paste into the
README.

Usage:
    python evaluate/ragas_eval.py   # produces ragas_results.json
    python evaluate/naive_rag.py    # produces naive_rag_results.json
    python evaluate/compare.py
"""
from __future__ import annotations

import json

from _common import RESULTS_DIR, banner

ADAPTIVE_JSON = RESULTS_DIR / "ragas_results.json"
NAIVE_JSON = RESULTS_DIR / "naive_rag_results.json"
REPORT_MD = RESULTS_DIR / "comparison_report.md"

METRIC_KEYS = [
    ("faithfulness", "Faithfulness", "higher"),
    ("answer_relevancy", "Answer Relevancy", "higher"),
    ("context_precision", "Context Precision", "higher"),
]

FEATURE_MATRIX = [
    ("Query Routing (index/search/general)", False, "Yes"),
    ("Relevance Grading", False, "Yes (0.6 threshold)"),
    ("Query Rewriting", False, "Yes (max 2 loops)"),
    ("Web Search Fallback", False, "Yes (Tavily)"),
    ("Hallucination Detection", False, "Yes (sentence-level)"),
    ("Answer Versioning", False, "Yes"),
    ("Knowledge Gap Alerts", False, "Yes"),
    ("Session Isolation", False, "Yes"),
]


def _load(path) -> dict | None:
    if not path.exists():
        print(f"[!] Missing {path.name} — run the corresponding script first.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_change(naive: float, adaptive: float, direction: str) -> str:
    """Format the relative improvement of adaptive over naive."""
    if naive == 0:
        return "n/a"
    delta = (adaptive - naive) / abs(naive) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.0f}%"


def _fmt(val) -> str:
    return f"{val:.3f}" if isinstance(val, (int, float)) else "n/a"


def main() -> None:
    banner("TASK 3 — COMPARISON REPORT")
    adaptive = _load(ADAPTIVE_JSON)
    naive = _load(NAIVE_JSON)
    if adaptive is None or naive is None:
        return

    a_metrics = adaptive.get("metrics", {})
    n_metrics = naive.get("metrics", {})
    a_latency = adaptive.get("summary", {}).get("avg_latency_ms", 0)
    n_latency = naive.get("summary", {}).get("avg_latency_ms", 0)

    n_adaptive = adaptive.get("summary", {}).get("total_questions", 0)
    n_scored = adaptive.get("summary", {}).get("questions_with_context", n_adaptive)
    lines: list[str] = [
        "# Adaptive RAG vs Naive RAG — Benchmark Results",
        "",
        "Both systems share the same Qdrant collection, Cohere embeddings, Groq",
        f"pipeline model, and the same {n_adaptive}-question test set. The only",
        "difference is the adaptive control flow (routing, grading, self-correcting",
        "rewrites). RAGAS scores are reference-free, judged by `llama-3.1-8b-instant`.",
        f"Adaptive RAGAS is computed over the {n_scored} questions that retrieved from",
        "the document (others routed to web search and are excluded from context metrics).",
        "",
        "## Performance Comparison",
        "",
        "| Metric | Naive RAG | Adaptive RAG | Improvement |",
        "|--------|-----------|--------------|-------------|",
    ]

    for key, label, direction in METRIC_KEYS:
        nv, av = n_metrics.get(key), a_metrics.get(key)
        improvement = (
            _pct_change(nv, av, direction)
            if isinstance(nv, (int, float)) and isinstance(av, (int, float))
            else "n/a"
        )
        lines.append(f"| {label} | {_fmt(nv)} | {_fmt(av)} | {improvement} |")

    # Latency — lower is better, so phrase the change accordingly.
    if n_latency and a_latency:
        lat_delta = (a_latency - n_latency) / n_latency * 100
        sign = "+" if lat_delta >= 0 else ""
        lat_note = f"{sign}{lat_delta:.0f}%"
    else:
        lat_note = "n/a"
    lines.append(f"| Avg Latency (ms) | {n_latency} | {a_latency} | {lat_note} |")
    lines.append("| Unnecessary Retrievals | always retrieves | routed | eliminated on general/search |")

    lines += [
        "",
        "> Note: the adaptive system trades some latency for the grading and",
        "> optional rewrite steps; the payoff is higher faithfulness and the",
        "> elimination of retrievals on questions that need none.",
        "",
        "## Feature Comparison",
        "",
        "| Feature | Naive RAG | Adaptive RAG |",
        "|---------|-----------|--------------|",
    ]
    for feature, naive_has, adaptive_has in FEATURE_MATRIX:
        lines.append(f"| {feature} | {'Yes' if naive_has else 'No'} | {adaptive_has} |")

    lines += [
        "",
        f"_RAGAS judge: project Groq model. Generated from "
        f"`{ADAPTIVE_JSON.name}` and `{NAIVE_JSON.name}`._",
        "",
    ]

    report = "\n".join(lines)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved: {REPORT_MD}")


if __name__ == "__main__":
    main()
