"""Knowledge gap analysis: identify what documents are missing to improve answers."""
import hashlib
import json
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

_gap_cache: dict[str, tuple[dict, float]] = {}

KNOWLEDGE_GAP_PROMPT = """You are a knowledge gap analyzer for a RAG (Retrieval Augmented Generation) system.

A user asked: "{question}"

The retrieved context was:
---
{retrieved_chunks_text}
---

The final answer generated was:
---
{final_answer}
---

Analyze the gap between what was asked and what was available. Identify:
1. What specific information was MISSING that would have made the answer more complete?
2. What TYPE of document would likely contain that missing information?
3. Rate the completeness of the answer from 1-10 (10 = fully answered, 1 = barely answered).
4. Write a one-sentence gap summary.

Rules:
- Be specific. Not "more information" but "employment dates and company names for each role".
- If the answer was actually complete, missing_info should be an empty list and score should be 8-10.
- Maximum 4 items in missing_info and suggested_documents.
- suggested_documents should be practical things a user could realistically upload.

Respond ONLY with valid JSON. No preamble, no markdown backticks, no explanation. Exactly this structure:
{{"missing_info": ["specific thing 1"], "suggested_documents": ["Document type 1"], "completeness_score": 7, "gap_summary": "One sentence describing the main gap."}}"""


def _cache_key(question: str, retrieved_chunks: list[str] | None = None) -> str:
    # Sorted chunk digests make the key order-independent across retrieval runs
    # while still invalidating when the underlying documents change.
    chunk_digests = sorted(
        hashlib.md5(c[:200].encode()).hexdigest() for c in (retrieved_chunks or [])
    )
    content = question.lower()[:100] + "|" + "|".join(chunk_digests)
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached(key: str, ttl: int) -> Optional[dict]:
    if key in _gap_cache:
        result, timestamp = _gap_cache[key]
        if time.time() - timestamp < ttl:
            return result
        del _gap_cache[key]
    return None


def _set_cached(key: str, result: dict) -> None:
    _gap_cache[key] = (result, time.time())


def analyze_knowledge_gaps(
    question: str,
    retrieved_chunks: list[str],
    final_answer: str,
    llm: ChatGroq,
    ttl: int = 600,
) -> Optional[dict]:
    """Run knowledge gap analysis synchronously using Groq. Returns parsed dict or None on failure."""
    cache_key = _cache_key(question, retrieved_chunks)
    cached = _get_cached(cache_key, ttl)
    if cached:
        # Cache hits consume no tokens regardless of what the original call cost.
        return {**cached, "from_cache": True, "token_usage": {"prompt": 0, "completion": 0}}

    chunks_text = "\n---\n".join(retrieved_chunks)[:2000]
    if len("\n---\n".join(retrieved_chunks)) > 2000:
        chunks_text += "\n[... truncated ...]"

    prompt = KNOWLEDGE_GAP_PROMPT.format(
        question=question,
        retrieved_chunks_text=chunks_text,
        final_answer=final_answer[:1500],
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        assert "missing_info" in result and isinstance(result["missing_info"], list)
        assert "suggested_documents" in result and isinstance(result["suggested_documents"], list)
        assert "completeness_score" in result and isinstance(result["completeness_score"], (int, float))
        assert "gap_summary" in result and isinstance(result["gap_summary"], str)

        result["completeness_score"] = max(1, min(10, int(result["completeness_score"])))
        result["missing_info"] = [str(i) for i in result["missing_info"][:4]]
        result["suggested_documents"] = [str(d) for d in result["suggested_documents"][:4]]
        result["from_cache"] = False

        usage = getattr(response, "usage_metadata", None) or {}
        result["token_usage"] = {
            "prompt": usage.get("input_tokens", 0),
            "completion": usage.get("output_tokens", 0),
        }

        _set_cached(cache_key, result)
        return result

    except json.JSONDecodeError as e:
        print(f"[KnowledgeGap] JSON parse error: {e}. Raw: {raw[:200] if 'raw' in dir() else 'N/A'}")
        return None
    except Exception as e:
        print(f"[KnowledgeGap] Analysis failed: {e}")
        return None
