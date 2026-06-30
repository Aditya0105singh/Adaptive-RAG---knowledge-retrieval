"""Embedding-based grounding checker — zero LLM calls, pure cosine similarity.

An ALTERNATIVE to the LLM-based checker in grounding_checker.py, for an ablation
study comparing the two approaches (accuracy vs. latency/cost). Each answer
sentence is embedded and compared to the retrieved chunk embeddings:

  similarity >= 0.75 -> GROUNDED   (strongly present in chunks)
  similarity >= 0.50 -> INFERRED   (related but not explicit)
  similarity <  0.50 -> UNGROUNDED (no basis in chunks)

Returns the same schema as check_answer_grounding() (plus a ``similarity`` per
sentence and ``llm_calls``) so the two can be compared field-for-field.
"""
import math
import re
import time
from typing import List, Optional

from src.services.embeddings import get_embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors (0.0 on degenerate input)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def check_grounding_embedding(
    answer: str,
    retrieved_chunks: List[str],
    grounded_threshold: float = 0.75,
    inferred_threshold: float = 0.50,
    min_sentence_len: int = 20,
) -> Optional[dict]:
    """Grade each answer sentence against retrieved chunks via cosine similarity.

    Mirrors check_answer_grounding()'s return schema for direct comparison.
    """
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', answer)
        if len(s.strip()) >= min_sentence_len
    ]
    if not sentences:
        return {"skipped": True, "reason": "No sentences found", "results": [], "summary": None}

    if not retrieved_chunks:
        return {
            "skipped": False,
            "results": [
                {"sentence": s, "label": "UNGROUNDED", "similarity": 0.0} for s in sentences
            ],
            "summary": {
                "trust_score": 0.0, "trust_level": "LOW",
                "grounded_count": 0, "inferred_count": 0,
                "ungrounded_count": len(sentences),
                "total_latency_ms": 0, "llm_calls": 0,
            },
            "token_usage": {"prompt": 0, "completion": 0},
        }

    t_start = time.time()
    embeddings = get_embeddings()
    chunk_vecs = embeddings.embed_documents(retrieved_chunks)  # one batch call

    results = []
    grounded = inferred = ungrounded = 0
    for sentence in sentences:
        sent_vec = embeddings.embed_query(sentence)
        max_sim = max((cosine_similarity(sent_vec, cv) for cv in chunk_vecs), default=0.0)
        if max_sim >= grounded_threshold:
            label = "GROUNDED"
            grounded += 1
        elif max_sim >= inferred_threshold:
            label = "INFERRED"
            inferred += 1
        else:
            label = "UNGROUNDED"
            ungrounded += 1
        results.append({"sentence": sentence, "label": label, "similarity": round(max_sim, 3)})

    total = len(results)
    trust_score = round(grounded / total, 2) if total > 0 else 0.0
    trust_level = "HIGH" if trust_score >= 0.75 else "MODERATE" if trust_score >= 0.45 else "LOW"

    return {
        "skipped": False,
        "results": results,
        "summary": {
            "trust_score": trust_score,
            "trust_level": trust_level,
            "grounded_count": grounded,
            "inferred_count": inferred,
            "ungrounded_count": ungrounded,
            "total_latency_ms": int((time.time() - t_start) * 1000),
            "llm_calls": 0,
        },
        "token_usage": {"prompt": 0, "completion": 0},
    }
