"""TASK 5 — Latency benchmark per route.

Runs questions across all three routes (index / general / search) against the
live backend and reports avg / min / max / P50 / P95 / P99 wall-clock latency
per route. Index questions hit the indexed sample document; general questions
use a doc-free session; search questions need real-time info.

Usage:
    python evaluate/ragas_eval.py        # ensures the sample doc is indexed
    python evaluate/benchmark_latency.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from _common import (
    EVAL_SESSION_ID,
    RESULTS_DIR,
    banner,
    chat,
    percentiles,
    require_backend,
    upload_document,
)

RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"
NODOC_SESSION = f"{EVAL_SESSION_ID}-nodoc"
INTER_CALL_DELAY = 1.5

LATENCY_TEST_CASES = {
    "index": [
        "What are the key findings or conclusions?",
        "Summarize the methodology section",
        "What problem does this document address?",
        "What technologies or tools are referenced?",
        "What are the limitations mentioned?",
        "What future work is suggested?",
        "What are the main contributions?",
        "What does the document say about evaluation criteria?",
        "What are the main components or sections?",
        "What is the overall conclusion?",
    ],
    "general": [
        "What is machine learning?",
        "Explain neural networks in simple terms",
        "What is the difference between a list and a tuple in Python?",
        "What is 17 multiplied by 23?",
        "Tell me a short joke",
        "What is the capital of Japan?",
        "Explain what recursion is",
        "What is a hash table?",
        "Write a one-line motivational quote",
        "What does HTTP stand for?",
    ],
    "search": [
        "What is the latest GPT model in 2026?",
        "What is the current stock price of Apple?",
        "What is the weather in Mumbai today?",
        "What are today's top technology headlines?",
        "Who is the current CEO of OpenAI?",
        "What is the latest news about artificial intelligence?",
        "What is the most recent SpaceX launch?",
        "What is trending on the internet right now?",
        "What is the latest version of Python released?",
        "What recent breakthroughs happened in AI this year?",
    ],
}


def _stats(latencies: list[int]) -> dict:
    if not latencies:
        return {"count": 0}
    return {
        "count": len(latencies),
        "avg_ms": int(sum(latencies) / len(latencies)),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": int(percentiles([float(x) for x in latencies], 50)),
        "p95_ms": int(percentiles([float(x) for x in latencies], 95)),
        "p99_ms": int(percentiles([float(x) for x in latencies], 99)),
    }


def main() -> None:
    banner("TASK 5 — LATENCY BENCHMARK")
    require_backend()

    print("Ensuring sample document is indexed...")
    try:
        upload_document(session_id=EVAL_SESSION_ID)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] upload failed ({exc}); index latencies may be unrepresentative")

    per_route_latencies: dict[str, list[int]] = {}
    raw_records: list[dict] = []

    for route, questions in LATENCY_TEST_CASES.items():
        doc_available = route == "index"
        session = EVAL_SESSION_ID if doc_available else NODOC_SESSION
        per_route_latencies[route] = []
        print(f"\n--- {route} ({len(questions)} queries) ---")
        for i, question in enumerate(questions, 1):
            t0 = time.perf_counter()
            try:
                resp = chat(
                    question,
                    session_id=session,
                    doc_available=doc_available,
                    doc_filename="sample_doc.md" if doc_available else "",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i:2d}] ! failed: {exc}")
                time.sleep(INTER_CALL_DELAY)
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)
            per_route_latencies[route].append(latency_ms)
            raw_records.append(
                {
                    "intended_route": route,
                    "actual_route": resp.get("route_taken", "unknown"),
                    "question": question,
                    "latency_ms": latency_ms,
                    "server_processing_ms": resp.get("processing_ms", 0),
                }
            )
            print(f"  [{i:2d}] {latency_ms:5d}ms  (route={resp.get('route_taken')})")
            time.sleep(INTER_CALL_DELAY)

    route_stats = {route: _stats(lats) for route, lats in per_route_latencies.items()}
    all_latencies = [l for lats in per_route_latencies.values() for l in lats]
    route_stats["overall"] = _stats(all_latencies)

    # --- table -----------------------------------------------------------
    print("\nLatency Benchmark Results")
    print("=" * 56)
    header = f"{'Route':<10}|{'Avg':>8}|{'P50':>8}|{'P95':>8}|{'P99':>8}"
    print(header)
    print("-" * len(header))
    for route in ["index", "general", "search", "overall"]:
        s = route_stats.get(route, {})
        if not s.get("count"):
            print(f"{route:<10}|{'n/a':>8}|{'n/a':>8}|{'n/a':>8}|{'n/a':>8}")
            continue
        print(
            f"{route:<10}|{s['avg_ms']:>7}m|{s['p50_ms']:>7}m|"
            f"{s['p95_ms']:>7}m|{s['p99_ms']:>7}m"
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "route_stats": route_stats,
        "records": raw_records,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
