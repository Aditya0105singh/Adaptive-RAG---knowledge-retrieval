import json
import os
import time
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from src.core.config import settings
from src.core.database import get_qdrant_client
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["system"])


@router.get("/benchmarks")
async def get_benchmarks():
    """Return actual RAGAS evaluation results for the benchmarks tab."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    results_dir = os.path.join(base_dir, "evaluate", "results")

    ragas_path = os.path.join(results_dir, "ragas_results.json")
    naive_path = os.path.join(results_dir, "naive_rag_results.json")

    adaptive_metrics = {}
    naive_metrics = {}

    try:
        if os.path.exists(ragas_path):
            with open(ragas_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                adaptive_metrics = data.get("metrics", {})

        if os.path.exists(naive_path):
            with open(naive_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                naive_metrics = data.get("metrics", {})
    except Exception as e:
        logger.error("failed_to_load_benchmarks", error=str(e))

    return {"adaptive": adaptive_metrics, "naive": naive_metrics}


@router.get("/system-status")
async def get_system_status():
    """Return health status of FastAPI, Qdrant, and Groq."""
    status = {"fastapi": "online", "qdrant": "offline", "groq": "offline"}

    try:
        client = get_qdrant_client()
        client.get_collections()
        status["qdrant"] = "online"
    except Exception as e:
        logger.warning(f"qdrant_status_check_failed: {e}")

    if settings.GROQ_API_KEY and len(settings.GROQ_API_KEY) > 10:
        status["groq"] = "online"

    return status


# ---------------------------------------------------------------------------
# Grounding comparison endpoint
# ---------------------------------------------------------------------------

class GroundingCompareRequest(BaseModel):
    answer: str
    chunks: List[str]
    llm_results: List[dict]  # already computed per-sentence LLM grounding


@router.post("/grounding/compare")
async def grounding_compare(req: GroundingCompareRequest):
    """Run embedding-based grounding on the same answer+chunks, compare to LLM results.

    The LLM grounding already ran during the pipeline; this adds the embedding
    ablation so the front-end can show the comparison table.
    """
    from src.services.grounding_checker_embedding import check_grounding_embedding

    t0 = time.time()
    emb_result = check_grounding_embedding(req.answer, req.chunks)
    latency_ms = int((time.time() - t0) * 1000)

    # Map embedding results by sentence for O(1) lookup
    emb_by_sentence: dict = {}
    for r in emb_result.get("results") or []:
        emb_by_sentence[r["sentence"]] = r

    comparison = []
    agree = 0
    for llm_r in req.llm_results:
        sentence = llm_r.get("sentence", "")
        # Best-effort fuzzy match: exact first, then first-80-chars key
        emb_r = emb_by_sentence.get(sentence) or emb_by_sentence.get(sentence[:80])
        llm_label = llm_r.get("label", "?")
        emb_label = (emb_r or {}).get("label", "UNGROUNDED")
        similarity = round((emb_r or {}).get("similarity", 0.0), 3)
        match = llm_label == emb_label
        if match:
            agree += 1
        comparison.append({
            "sentence": sentence,
            "llm": llm_label,
            "embedding": emb_label,
            "similarity": similarity,
            "match": match,
        })

    total = len(comparison)
    emb_summary = emb_result.get("summary") or {}

    return {
        "embedding_summary": emb_summary,
        "embedding_latency_ms": latency_ms,
        "comparison": comparison,
        "agreement_rate": round(agree / total, 3) if total > 0 else 0.0,
        "agreement_count": agree,
        "total_sentences": total,
    }
