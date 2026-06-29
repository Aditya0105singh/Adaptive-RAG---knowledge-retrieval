"""TASK 1 — RAGAS evaluation of the Adaptive RAG system.

Uploads a reference document, runs a fixed question set through the live
/api/chat endpoint, then scores the answers with RAGAS:

    * faithfulness      — is the answer supported by the retrieved chunks?
    * answer_relevancy  — does the answer actually address the question?
    * context_precision — are the retrieved chunks relevant? (reference-free)

The judge LLM is the project's own Groq model and the embeddings are the
project's own Cohere embeddings — there is no hidden OpenAI judge. Results are
written to evaluate/results/ragas_results.json and a Markdown summary.

Usage:
    python main.py                 # start backend on :8080 (separate terminal)
    python evaluate/ragas_eval.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from _common import (
    RESULTS_DIR,
    TEST_QUESTIONS,
    banner,
    build_ragas_embeddings,
    build_ragas_llm,
    chat,
    extract_contexts,
    require_backend,
    upload_document,
)

RESULTS_JSON = RESULTS_DIR / "ragas_results.json"
RESULTS_MD = RESULTS_DIR / "ragas_results.md"

# Seconds between chat calls — keeps us under the Groq free-tier 6k TPM budget.
INTER_CALL_DELAY = 2.0


def collect_samples() -> list[dict]:
    """Run every test question through the backend and gather RAGAS inputs."""
    samples: list[dict] = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i:2d}/{len(TEST_QUESTIONS)}] {question}")
        t0 = time.perf_counter()
        try:
            resp = chat(question)
        except Exception as exc:  # noqa: BLE001 — log and skip a single bad question
            print(f"    ! failed: {exc}")
            continue
        latency_ms = int((time.perf_counter() - t0) * 1000)
        contexts = extract_contexts(resp)
        samples.append(
            {
                "question": question,
                "answer": resp.get("answer", ""),
                "contexts": contexts,
                "route_taken": resp.get("route_taken", "unknown"),
                "latency_ms": latency_ms,
                "server_processing_ms": resp.get("processing_ms", 0),
                "cost_usd": resp.get("estimated_cost_usd", 0.0),
                "num_contexts": len(contexts),
            }
        )
        print(
            f"    route={resp.get('route_taken')} "
            f"contexts={len(contexts)} latency={latency_ms}ms"
        )
        time.sleep(INTER_CALL_DELAY)
    return samples


def run_ragas(samples: list[dict]) -> dict:
    """Score collected samples with RAGAS; return {metric: float}."""
    from ragas import RunConfig, evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    # Only evaluate samples that actually retrieved context — faithfulness and
    # context_precision are meaningless with an empty context list.
    usable = [s for s in samples if s["contexts"]]
    if not usable:
        print("[warn] no samples had retrieved contexts; skipping RAGAS scoring")
        return {}

    eval_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
        )
        for s in usable
    ]
    dataset = EvaluationDataset(samples=eval_samples)

    llm = build_ragas_llm()
    embeddings = build_ragas_embeddings()

    metrics = [Faithfulness(), LLMContextPrecisionWithoutReference()]
    if embeddings is not None:
        metrics.append(ResponseRelevancy())

    # Low worker count + generous retries keeps us inside free-tier rate limits.
    run_config = RunConfig(max_workers=2, max_retries=5, max_wait=90, timeout=240)

    print(f"\nScoring {len(usable)} samples with RAGAS (this calls the judge LLM many times)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    # Aggregate via the pandas frame — robust across RAGAS point releases.
    df = result.to_pandas()
    scores: dict[str, float] = {}
    rename = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "llm_context_precision_without_reference": "context_precision",
    }
    for raw_col, nice in rename.items():
        if raw_col in df.columns:
            series = df[raw_col].dropna()
            if len(series):
                scores[nice] = round(float(series.mean()), 4)
    return scores


def print_summary(scores: dict, samples: list[dict]) -> None:
    """Print the boxed results table."""
    n = len(samples)
    avg_latency = int(sum(s["latency_ms"] for s in samples) / n) if n else 0
    total_cost = sum(s["cost_usd"] for s in samples)
    overall = round(sum(scores.values()) / len(scores), 4) if scores else 0.0

    rows = [
        ("Faithfulness", scores.get("faithfulness")),
        ("Answer Relevancy", scores.get("answer_relevancy")),
        ("Context Precision", scores.get("context_precision")),
        ("Overall Score", overall),
    ]
    print("\n+" + "-" * 44 + "+")
    print("|{:^44}|".format("RAGAS EVALUATION RESULTS"))
    print("+" + "-" * 44 + "+")
    for label, val in rows:
        shown = f"{val:.3f}" if isinstance(val, float) else "n/a"
        print("|  {:<28}{:>14}|".format(label + ":", shown))
    print("+" + "-" * 44 + "+")
    print("|  {:<28}{:>14}|".format("Questions Tested:", n))
    print("|  {:<28}{:>14}|".format("Avg Latency:", f"{avg_latency}ms"))
    print("|  {:<28}{:>14}|".format("Total Cost:", f"${total_cost:.4f}"))
    print("+" + "-" * 44 + "+")


def write_outputs(scores: dict, samples: list[dict]) -> None:
    """Persist JSON results and a Markdown summary table."""
    n = len(samples)
    avg_latency = int(sum(s["latency_ms"] for s in samples) / n) if n else 0
    total_cost = round(sum(s["cost_usd"] for s in samples), 6)
    overall = round(sum(scores.values()) / len(scores), 4) if scores else 0.0

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": scores,
        "overall_score": overall,
        "per_question_results": samples,
        "summary": {
            "total_questions": n,
            "questions_with_context": sum(1 for s in samples if s["contexts"]),
            "avg_latency_ms": avg_latency,
            "total_cost_usd": total_cost,
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# RAGAS Evaluation — Adaptive RAG",
        "",
        f"_Generated {payload['timestamp']}_",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Faithfulness | {scores.get('faithfulness', 'n/a')} |",
        f"| Answer Relevancy | {scores.get('answer_relevancy', 'n/a')} |",
        f"| Context Precision | {scores.get('context_precision', 'n/a')} |",
        f"| **Overall** | **{overall}** |",
        "",
        f"- Questions tested: {n}",
        f"- Avg latency: {avg_latency} ms",
        f"- Total cost: ${total_cost}",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {RESULTS_JSON}")
    print(f"Saved: {RESULTS_MD}")


def main() -> None:
    banner("TASK 1 — RAGAS EVALUATION")
    require_backend()

    print("Uploading reference document...")
    info = upload_document()
    print(f"  Indexed {info['parent_count']} parent / {info['child_count']} child chunks\n")

    samples = collect_samples()
    if not samples:
        print("[!] No samples collected — aborting.")
        return

    try:
        scores = run_ragas(samples)
    except Exception as exc:  # noqa: BLE001 — never lose the collected data
        print(f"[!] RAGAS scoring failed: {exc}")
        scores = {}

    print_summary(scores, samples)
    write_outputs(scores, samples)


if __name__ == "__main__":
    main()
