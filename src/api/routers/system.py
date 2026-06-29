import json
import os
from fastapi import APIRouter
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
        logger.error(f"failed_to_load_benchmarks", error=str(e))
        
    return {
        "adaptive": adaptive_metrics,
        "naive": naive_metrics
    }

@router.get("/system-status")
async def get_system_status():
    """Return health status of FastAPI, Qdrant, and Groq."""
    status = {
        "fastapi": "online",
        "qdrant": "offline",
        "groq": "offline"
    }
    
    # Check Qdrant
    try:
        client = get_qdrant_client()
        client.get_collections()
        status["qdrant"] = "online"
    except Exception as e:
        logger.warning(f"qdrant_status_check_failed: {e}")
        
    # Check Groq API Key
    if settings.GROQ_API_KEY and len(settings.GROQ_API_KEY) > 10:
        status["groq"] = "online"
        
    return status
