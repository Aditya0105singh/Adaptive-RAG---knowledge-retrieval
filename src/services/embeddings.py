"""Embedding singleton using the HuggingFace Inference API.

No local model is loaded — embeddings are computed on HF servers via HTTP.
This keeps Render free-tier RAM well under 512 MB (saves ~150 MB vs fastembed).

Set HF_TOKEN env var for higher rate limits; works unauthenticated for demos.
"""
import os
from typing import List, Optional

import httpx

_instance: Optional["_HFEmbeddings"] = None

_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_URL = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{_HF_MODEL}"
)


class _HFEmbeddings:
    """Thin wrapper around the HF Inference API feature-extraction endpoint."""

    def __init__(self) -> None:
        token = os.environ.get("HF_TOKEN", "")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(timeout=60.0)

    def _post(self, payload: dict) -> list:
        resp = self._client.post(_HF_URL, headers=self._headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._post({"inputs": texts, "options": {"wait_for_model": True}})
        # HF returns List[List[float]] for batch inputs
        return result

    def embed_query(self, text: str) -> List[float]:
        result = self._post({"inputs": text, "options": {"wait_for_model": True}})
        # HF returns List[float] for single string input
        if result and isinstance(result[0], list):
            return result[0]
        return result


def get_embeddings() -> _HFEmbeddings:
    """Return the process-wide embeddings client (lazy, created on first call)."""
    global _instance
    if _instance is None:
        _instance = _HFEmbeddings()
    return _instance
