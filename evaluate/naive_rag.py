"""TASK 2 — Naive RAG baseline.

A deliberately minimal RAG pipeline with NONE of the adaptive features:
no routing, no relevance grading, no query rewriting, no retry loop, no
grounding check, no knowledge-gap analysis. It always retrieves the top-3
chunks and generates an answer in a single shot.

It reuses the *same* Qdrant collection, Cohere embeddings, and Groq model as the
main system, so the only variable in the comparison is the adaptive control
flow itself. Run ragas_eval.py first so the document is already indexed in the
``session_eval-suite`` collection.

Outputs evaluate/results/naive_rag_results.json (same shape as ragas_results
plus RAGAS scores) for compare.py to consume.

Usage:
    python evaluate/ragas_eval.py   # ensures the doc is indexed
    python evaluate/naive_rag.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from _common import (
    EVAL_SESSION_ID,
    RESULTS_DIR,
    TEST_QUESTIONS,
    banner,
    build_ragas_embeddings,
    build_ragas_llm,
    pick_working_groq_key,
)

# Imported from the project so the baseline shares the production retrieval +
# embedding + LLM stack — the comparison stays apples-to-apples.
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.core.config import settings
from src.services.retrieval import RetrievalService

RESULTS_JSON = RESULTS_DIR / "naive_rag_results.json"
INTER_CALL_DELAY = 2.0
TOP_K = 3


class NaiveRAG:
    """Textbook RAG: embed -> retrieve top-k -> stuff -> generate. No guardrails."""

    def __init__(self) -> None:
        self._retrieval = RetrievalService()
        self._llm = ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0,
            api_key=pick_working_groq_key(settings.GROQ_MODEL),
            max_tokens=1024,
        )

    def query(self, question: str, collection_name: str) -> dict:
        """Answer a question with no routing, grading, or retries."""
        t0 = time.perf_counter()
        # 1. Embed + retrieve top-k parent chunks. Use every chunk, ungraded.
        contexts = self._retrieval.retrieve(
            question, top_k=TOP_K, collection_name=collection_name
        )
        # 2. Concatenate all retrieved chunks as context — no relevance filter.
        context_block = "\n\n---\n\n".join(contexts) if contexts else "No context available."
        # 3. Single-shot generation.
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Answer the question using only the "
                    "provided context. If the context does not contain the answer, "
                    "say so honestly."
                )
            ),
            HumanMessage(content=f"Context:\n{context_block}\n\nQuestion: {question}"),
        ]
        response = self._llm.invoke(messages)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "answer": (response.content or "").strip(),
            "contexts": contexts,
            "latency_ms": latency_ms,
            "num_contexts": len(contexts),
            "token_usage": {
                "prompt": usage.get("input_tokens", 0),
                "completion": usage.get("output_tokens", 0),
            },
        }


def collect_samples(rag: NaiveRAG, collection: str) -> list[dict]:
    """Run every test question through the naive pipeline."""
    samples: list[dict] = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i:2d}/{len(TEST_QUESTIONS)}] {question}")
        try:
            out = rag.query(question, collection)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! failed: {exc}")
            continue
        samples.append(
            {
                "question": question,
                "answer": out["answer"],
                "contexts": out["contexts"],
                "route_taken": "naive (always retrieve)",
                "latency_ms": out["latency_ms"],
                "num_contexts": out["num_contexts"],
            }
        )
        print(f"    contexts={out['num_contexts']} latency={out['latency_ms']}ms")
        time.sleep(INTER_CALL_DELAY)
    return samples


def run_ragas(samples: list[dict]) -> dict:
    """Score the naive pipeline with the same RAGAS metrics as the adaptive one."""
    from ragas import RunConfig, evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    usable = [s for s in samples if s["contexts"]]
    if not usable:
        print("[warn] no samples had contexts; skipping RAGAS scoring")
        return {}

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
            )
            for s in usable
        ]
    )
    embeddings = build_ragas_embeddings()
    metrics = [Faithfulness(), LLMContextPrecisionWithoutReference()]
    if embeddings is not None:
        metrics.append(ResponseRelevancy())

    print(f"\nScoring {len(usable)} naive samples with RAGAS...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=build_ragas_llm(),
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, max_retries=5, max_wait=90, timeout=240),
        raise_exceptions=False,
        show_progress=True,
    )
    df = result.to_pandas()
    rename = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "llm_context_precision_without_reference": "context_precision",
    }
    scores: dict[str, float] = {}
    for raw_col, nice in rename.items():
        if raw_col in df.columns:
            series = df[raw_col].dropna()
            if len(series):
                scores[nice] = round(float(series.mean()), 4)
    return scores


def main() -> None:
    banner("TASK 2 — NAIVE RAG BASELINE")
    collection = f"session_{EVAL_SESSION_ID}"

    rag = NaiveRAG()
    # Fail fast with a helpful message if the document was never indexed.
    if not rag._retrieval.has_documents(collection):
        print(
            f"[!] Collection '{collection}' has no documents.\n"
            f"    Run 'python evaluate/ragas_eval.py' first to upload + index the doc."
        )
        return

    samples = collect_samples(rag, collection)
    if not samples:
        print("[!] No samples collected — aborting.")
        return

    try:
        scores = run_ragas(samples)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] RAGAS scoring failed: {exc}")
        scores = {}

    n = len(samples)
    avg_latency = int(sum(s["latency_ms"] for s in samples) / n) if n else 0
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "naive_rag",
        "metrics": scores,
        "per_question_results": samples,
        "summary": {"total_questions": n, "avg_latency_ms": avg_latency},
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n+" + "-" * 44 + "+")
    print("|{:^44}|".format("NAIVE RAG RESULTS"))
    print("+" + "-" * 44 + "+")
    for label, key in [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Precision", "context_precision"),
    ]:
        val = scores.get(key)
        shown = f"{val:.3f}" if isinstance(val, float) else "n/a"
        print("|  {:<28}{:>14}|".format(label + ":", shown))
    print("|  {:<28}{:>14}|".format("Avg Latency:", f"{avg_latency}ms"))
    print("+" + "-" * 44 + "+")
    print(f"\nSaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
