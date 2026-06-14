"""Embedding singleton — HuggingFace Hub InferenceClient (no local model).

Uses huggingface_hub.InferenceClient which routes through
router.huggingface.co (different hostname from the broken
api-inference.huggingface.co). huggingface_hub is already installed
as a transitive dependency so no new packages are needed.

Set HF_TOKEN env var for higher rate limits.
"""
import os
from typing import List, Optional

_instance: Optional["_HFEmbeddings"] = None
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _HFEmbeddings:
    def __init__(self) -> None:
        from huggingface_hub import InferenceClient
        token = os.environ.get("HF_TOKEN") or None
        self._client = InferenceClient(provider="hf-inference", api_key=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._client.feature_extraction(texts, model=_MODEL)
        # result is a numpy array of shape (n, dim) for list input
        return result.tolist()

    def embed_query(self, text: str) -> List[float]:
        result = self._client.feature_extraction(text, model=_MODEL)
        # result is a numpy array of shape (dim,) for string input
        arr = result.tolist()
        # flatten if nested
        if arr and isinstance(arr[0], list):
            return arr[0]
        return arr


def get_embeddings() -> _HFEmbeddings:
    global _instance
    if _instance is None:
        _instance = _HFEmbeddings()
    return _instance
