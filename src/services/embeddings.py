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
        self._api_key = os.environ.get("COHERE_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "COHERE_API_KEY env var is not set. "
                "Get a free key at dashboard.cohere.com and add it to Render."
            )
        self._session = self._make_session()

    def _make_session(self) -> "_requests.Session":
        s = _requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })
        return s

    def _post(self, payload: dict) -> dict:
        """POST to Cohere, retrying once on stale-connection errors.

        requests.Session reuses TCP connections; Cohere closes idle sockets
        server-side, producing RemoteDisconnected on the next use. Discard
        the session and retry once with a fresh connection on that error.
        """
        for attempt in range(2):
            try:
                resp = self._session.post(_COHERE_URL, json=payload, timeout=15)
                break
            except _requests.exceptions.ConnectionError:
                if attempt == 1:
                    raise
                self._session.close()
                self._session = self._make_session()
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
