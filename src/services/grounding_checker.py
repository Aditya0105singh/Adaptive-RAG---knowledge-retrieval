"""Hallucination grounding checker: classify each answer sentence as GROUNDED/INFERRED/UNGROUNDED."""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

GROUNDING_PROMPT = """You are a grounding checker for an AI answer.

Retrieved document chunks:
---
{chunks_text}
---

Sentence to classify:
"{sentence}"

Classify this sentence with exactly one word:
- GROUNDED: The sentence states a fact directly and explicitly present in the retrieved chunks.
- INFERRED: The sentence makes a claim that logically follows from the chunks but is not explicitly stated.
- UNGROUNDED: The sentence makes a claim that has no basis in the chunks (possible hallucination).

Respond with exactly one word: GROUNDED, INFERRED, or UNGROUNDED. Nothing else."""

VALID_LABELS = {"GROUNDED", "INFERRED", "UNGROUNDED"}


def split_into_sentences(text: str, min_len: int = 20) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    result = []
    for s in sentences:
        parts = [p.strip() for p in s.split('\n') if p.strip()]
        result.extend(parts)
    return [s for s in result if len(s) >= min_len]


def _check_single_sentence(sentence: str, chunks_text: str, llm: ChatGroq) -> dict:
    try:
        prompt = GROUNDING_PROMPT.format(chunks_text=chunks_text[:1500], sentence=sentence)
        response = llm.invoke([HumanMessage(content=prompt)])
        label = response.content.strip().upper()
        if label not in VALID_LABELS:
            for valid in VALID_LABELS:
                if valid in label:
                    label = valid
                    break
            else:
                label = "INFERRED"
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "sentence": sentence,
            "label": label,
            "error": None,
            "_usage": {
                "prompt": usage.get("input_tokens", 0),
                "completion": usage.get("output_tokens", 0),
            },
        }
    except Exception as e:
        return {
            "sentence": sentence,
            "label": "INFERRED",
            "error": str(e),
            "_usage": {"prompt": 0, "completion": 0},
        }


def check_answer_grounding(
    answer: str,
    retrieved_chunks: list[str],
    llm: ChatGroq,
    max_sentences: int = 12,
    min_sentence_len: int = 20,
) -> Optional[dict]:
    """Run grounding checks in parallel using a thread pool (Groq calls are synchronous)."""
    sentences = split_into_sentences(answer, min_len=min_sentence_len)
    if not sentences:
        return None

    if len(sentences) > max_sentences:
        return {
            "skipped": True,
            "reason": f"Answer too long ({len(sentences)} sentences > max {max_sentences})",
            "results": [],
            "summary": None,
        }

    chunks_text = "\n---\n".join(retrieved_chunks)[:2000]

    results = []
    with ThreadPoolExecutor(max_workers=min(len(sentences), 6)) as executor:
        futures = {
            executor.submit(_check_single_sentence, s, chunks_text, llm): s
            for s in sentences
        }
        # Preserve sentence order
        ordered = {s: None for s in sentences}
        for future in as_completed(futures):
            sentence = futures[future]
            ordered[sentence] = future.result()
        results = [ordered[s] for s in sentences if ordered[s] is not None]

    # Roll up token usage and strip the internal field from each result
    total_usage = {"prompt": 0, "completion": 0}
    for r in results:
        usage = r.pop("_usage", {})
        total_usage["prompt"] += usage.get("prompt", 0)
        total_usage["completion"] += usage.get("completion", 0)

    grounded_count = sum(1 for r in results if r["label"] == "GROUNDED")
    inferred_count = sum(1 for r in results if r["label"] == "INFERRED")
    ungrounded_count = sum(1 for r in results if r["label"] == "UNGROUNDED")
    total = len(results)
    trust_score = round(grounded_count / total, 2) if total > 0 else 0.0

    if trust_score >= 0.75:
        trust_level = "HIGH"
    elif trust_score >= 0.45:
        trust_level = "MODERATE"
    else:
        trust_level = "LOW"

    return {
        "skipped": False,
        "results": results,
        "summary": {
            "total_sentences": total,
            "grounded_count": grounded_count,
            "inferred_count": inferred_count,
            "ungrounded_count": ungrounded_count,
            "trust_score": trust_score,
            "trust_level": trust_level,
        },
        "token_usage": total_usage,
    }
