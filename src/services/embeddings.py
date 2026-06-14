"""Embedding singleton using the Cohere Embed API.

Uses cohere's embed-english-light-v3.0 which outputs 384-dim vectors —
identical dimension to all-MiniLM-L6-v2. No local model, no RAM overhead.
Free tier: 1000 calls/month (plenty for a demo).

Requires COHERE_API_KEY environment variable (free from dashboard.cohere.com).
"""
import os
from typing import List, Optional

import requests as _requests

_instance: Optional["_CohereEmbeddings"] = None
_COHERE_URL = "https://api.cohere.com/v1/embed"
_MODEL = "embed-english-light-v3.0"


class _CohereEmbeddings:
    def __init__(self) -> None:
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY env var is not set. "
                "Get a free key at dashboard.cohere.com and add it to Render."
            )
        self._session = _requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _post(self, payload: dict) -> dict:
        resp = self._session.post(_COHERE_URL, json=payload, timeout=60)
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:300]
            raise RuntimeError(f"Cohere embed {resp.status_code}: {detail}")
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = self._post({
            "model": _MODEL,
            "texts": texts,
            "input_type": "search_document",
            "embedding_types": ["float"],
        })
        return data["embeddings"]["float"]

    def embed_query(self, text: str) -> List[float]:
        data = self._post({
            "model": _MODEL,
            "texts": [text],
            "input_type": "search_query",
            "embedding_types": ["float"],
        })
        return data["embeddings"]["float"][0]


def get_embeddings() -> _CohereEmbeddings:
    global _instance
    if _instance is None:
        _instance = _CohereEmbeddings()
    return _instance
