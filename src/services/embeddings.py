"""Embedding singleton using the HuggingFace Inference API.

No local model is loaded — embeddings are computed on HF servers via HTTP.
Uses requests (synchronous, already a dep) with retry on transient errors.
Set HF_TOKEN env var for higher rate limits; works unauthenticated for demos.
"""
import os
import time
from typing import List, Optional

import requests as _requests

_instance: Optional["_HFEmbeddings"] = None

_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_URL = f"https://api-inference.huggingface.co/models/{_HF_MODEL}"


class _HFEmbeddings:
    def __init__(self) -> None:
        token = os.environ.get("HF_TOKEN", "")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._session = _requests.Session()
        self._session.headers.update(self._headers)

    def _post(self, payload: dict, retries: int = 3) -> list:
        for attempt in range(retries):
            try:
                resp = self._session.post(_HF_URL, json=payload, timeout=60)
                # 503 means model is loading — wait and retry
                if resp.status_code == 503:
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (_requests.ConnectionError, _requests.Timeout) as exc:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("HF Inference API unavailable after retries")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._post({
            "inputs": texts,
            "options": {"wait_for_model": True},
        })
        return result

    def embed_query(self, text: str) -> List[float]:
        result = self._post({
            "inputs": text,
            "options": {"wait_for_model": True},
        })
        if result and isinstance(result[0], list):
            return result[0]
        return result


def get_embeddings() -> _HFEmbeddings:
    global _instance
    if _instance is None:
        _instance = _HFEmbeddings()
    return _instance
