"""TASK 4 — Routing accuracy on a labelled test set.

Sends 25 labelled questions through /api/chat and compares the response's
``route_taken`` against the expected route. ``index`` questions are asked
against the indexed sample document; ``general`` questions use a separate
session with no document so the router cannot route to index.

Caveat: an ``index`` route can legitimately fall back to ``search`` when the
document does not contain the answer (the low-match fast path). Such cases are
counted as mismatches but flagged separately so the number is honest.

Usage:
    python evaluate/ragas_eval.py   # ensures the sample doc is indexed
    python evaluate/test_routing.py
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
    require_backend,
    upload_document,
)

RESULTS_JSON = RESULTS_DIR / "routing_results.json"
NODOC_SESSION = f"{EVAL_SESSION_ID}-nodoc"
INTER_CALL_DELAY = 1.5

# (question, doc_available, expected_route, reason)
ROUTING_TEST_CASES = [
    # --- index: answerable from the uploaded document --------------------- #
    ("What is the main topic of this document?", True, "index", "doc question"),
    ("What methodology or approach is described?", True, "index", "doc question"),
    ("What are the limitations mentioned?", True, "index", "doc question"),
    ("Summarize the introduction section", True, "index", "doc question"),
    ("What technologies or tools are referenced?", True, "index", "doc question"),
    ("What future work is suggested?", True, "index", "doc question"),
    ("What are the main contributions?", True, "index", "doc question"),
    ("What problem does this document address?", True, "index", "doc question"),
    ("What does the document say about evaluation?", True, "index", "explicit doc ref"),
    ("What are the main components or sections?", True, "index", "doc question"),
    ("According to the document, what is naive RAG?", True, "index", "explicit doc ref"),
    ("What is the overall conclusion of the document?", True, "index", "doc question"),
    # --- search: needs real-time info even with a doc present ------------- #
    ("What is the weather in Jaipur today?", True, "search", "real-time info"),
    ("What is the latest AI news in 2026?", True, "search", "current events"),
    ("What is the current stock price of Apple?", True, "search", "live price"),
    ("Who won the most recent ICC cricket match?", True, "search", "recent event"),
    ("What are today's top news headlines?", True, "search", "real-time info"),
    ("What is the newest GPT model released this month?", True, "search", "current events"),
    # --- general: no document, off-topic --------------------------------- #
    ("What is 2 + 2?", False, "general", "pure math, no doc"),
    ("Tell me a joke", False, "general", "casual, no doc"),
    ("What is Python?", False, "general", "general knowledge, no doc"),
    ("Hello, how are you?", False, "general", "greeting, no doc"),
    ("Explain what a binary search tree is", False, "general", "general knowledge, no doc"),
    ("What is the capital of France?", False, "general", "general knowledge, no doc"),
    ("Write a haiku about the ocean", False, "general", "creative, no doc"),
]


def main() -> None:
    banner("TASK 4 — ROUTING ACCURACY")
    require_backend()

    # Ensure the doc is present for index cases (idempotent re-upload is fine).
    print("Ensuring sample document is indexed...")
    try:
        upload_document(session_id=EVAL_SESSION_ID)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] upload failed ({exc}); index cases may misroute")

    results = []
    correct = 0
    fallback_misses = 0

    for i, (question, doc_available, expected, reason) in enumerate(ROUTING_TEST_CASES, 1):
        session = EVAL_SESSION_ID if doc_available else NODOC_SESSION
        print(f"[{i:2d}/{len(ROUTING_TEST_CASES)}] ({expected:7s}) {question}")
        try:
            resp = chat(
                question,
                session_id=session,
                doc_available=doc_available,
                doc_filename="sample_doc.md" if doc_available else "",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    ! failed: {exc}")
            results.append(
                {"question": question, "expected": expected, "got": "error",
                 "correct": False, "reason": reason, "error": str(exc)}
            )
            time.sleep(INTER_CALL_DELAY)
            continue

        got = resp.get("route_taken", "unknown")
        is_correct = got == expected
        # An index->search fallback is a retrieval miss, not a router miss.
        is_fallback = expected == "index" and got == "search"
        if is_correct:
            correct += 1
        elif is_fallback:
            fallback_misses += 1
        results.append(
            {"question": question, "expected": expected, "got": got,
             "correct": is_correct, "fallback": is_fallback, "reason": reason}
        )
        flag = "OK " if is_correct else ("~fb" if is_fallback else "X  ")
        print(f"    {flag} got={got}")
        time.sleep(INTER_CALL_DELAY)

    total = len(ROUTING_TEST_CASES)
    accuracy = correct / total * 100 if total else 0.0
    lenient = (correct + fallback_misses) / total * 100 if total else 0.0

    print("\n" + "-" * 50)
    print(f"Routing Accuracy: {correct}/{total} = {accuracy:.1f}%")
    if fallback_misses:
        print(
            f"  (+{fallback_misses} index->search retrieval fallbacks; "
            f"router-decision accuracy ~{lenient:.1f}%)"
        )
    misses = [r for r in results if not r["correct"]]
    if misses:
        print("\nMismatches:")
        for r in misses:
            note = " [index->search fallback]" if r.get("fallback") else ""
            print(f"  - \"{r['question']}\" → got '{r['got']}', expected '{r['expected']}'{note}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
        "router_decision_accuracy_pct": round(lenient, 1),
        "index_to_search_fallbacks": fallback_misses,
        "cases": results,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
