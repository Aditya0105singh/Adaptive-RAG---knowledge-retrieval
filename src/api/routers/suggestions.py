"""Endpoint that generates suggested questions from the session's indexed document."""
import json

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.core.config import settings
from src.core.database import get_qdrant_client
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["suggestions"])

_FALLBACK_PROMPTS = [
    "Summarize this document's key points",
    "What are the main topics covered?",
    "What conclusions does this document reach?",
    "What information is missing from this document?",
]

_llm = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=settings.GROQ_MODEL, temperature=0.3, api_key=settings.GROQ_API_KEY)
    return _llm


def _sample_doc_text(collection_name: str, max_chars: int = 1200) -> str:
    """Pull a few stored chunks from Qdrant to give the LLM document context."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists(collection_name):
            return ""
        results = client.scroll(collection_name=collection_name, limit=4, with_payload=True)[0]
        texts = [p.payload.get("parent_text", "") for p in results if p.payload]
        combined = " ".join(texts)
        return combined[:max_chars]
    except Exception as exc:
        logger.warning("suggestions_sample_failed", collection=collection_name, error=str(exc))
        return ""


@router.get("/insight/{session_id}")
async def get_document_insight(session_id: str) -> dict:
    """Return a 2-sentence summary + 5-8 key topic chips for the session's document.

    One LLM call covers both so the frontend can fetch a single endpoint post-upload.
    """
    collection_name = f"session_{session_id}"
    sample_text = _sample_doc_text(collection_name, max_chars=2500)

    if not sample_text.strip():
        return {"summary": "", "topics": []}

    try:
        response = _get_llm().invoke([
            SystemMessage(
                content=(
                    "You analyse documents for a Q&A assistant. "
                    "Given a document excerpt, output ONLY a JSON object with exactly "
                    "two keys:\n"
                    '- "summary": a 2-sentence plain-English description of what this '
                    "document is about (who wrote it / what it covers / what it concludes). "
                    "No fluff — be specific.\n"
                    '- "topics": an array of 5-8 short key concepts or entities '
                    "(1-4 words each) that a user would want to ask about. "
                    "Pick the most unique/technical terms from the document.\n"
                    "No other text, no markdown, no backticks.\n"
                    'Example: {"summary": "This paper presents a causal DAG-based '
                    'approach to process mining using SHAP values.", '
                    '"topics": ["SHAP values", "Causal DAG", "Process Mining"]}'
                )
            ),
            HumanMessage(content=f"Document excerpt:\n{sample_text}"),
        ])
        raw = response.content.strip()
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end + 1]) if start != -1 else {}
        return {
            "summary": str(parsed.get("summary", "")).strip(),
            "topics": [str(t).strip() for t in parsed.get("topics", []) if str(t).strip()][:8],
        }
    except Exception as exc:
        logger.warning("insight_llm_failed", session_id=session_id, error=str(exc))
        return {"summary": "", "topics": []}


@router.get("/suggestions/{session_id}")
async def get_suggestions(session_id: str) -> dict:
    """Return 4 suggested questions tailored to the session's uploaded document."""
    collection_name = f"session_{session_id}"
    sample_text = _sample_doc_text(collection_name)

    if not sample_text.strip():
        return {"prompts": _FALLBACK_PROMPTS}

    try:
        response = _get_llm().invoke([
            SystemMessage(
                content=(
                    "You generate helpful question prompts for a document Q&A app. "
                    "Given a document excerpt, output ONLY a JSON array of exactly 4 short, "
                    "specific questions a user would genuinely want to ask about this document. "
                    "Each question should be under 60 characters. No other text."
                )
            ),
            HumanMessage(content=f"Document excerpt:\n{sample_text}"),
        ])
        raw = response.content.strip()
        start, end = raw.find("["), raw.rfind("]")
        prompts = json.loads(raw[start:end+1]) if start != -1 else []
        prompts = [str(p).strip() for p in prompts if str(p).strip()][:4]
        if len(prompts) < 2:
            return {"prompts": _FALLBACK_PROMPTS}
        return {"prompts": prompts}
    except Exception as exc:
        logger.warning("suggestions_llm_failed", session_id=session_id, error=str(exc))
        return {"prompts": _FALLBACK_PROMPTS}
