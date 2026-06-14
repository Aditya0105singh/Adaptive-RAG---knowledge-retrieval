"""Process-wide singleton for the embedding model.

Uses fastembed (ONNX runtime, no PyTorch) so the model fits in 512 MB RAM.
BAAI/bge-small-en-v1.5 outputs 384-dim vectors — same as all-MiniLM-L6-v2.
"""
from typing import List, Optional

_instance = None
_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _load():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=_FASTEMBED_MODEL)


def _get_model():
    global _instance
    if _instance is None:
        _instance = _load()
    return _instance


class _EmbeddingsAdapter:
    """Thin adapter so ingestion.py and retrieval.py can call .embed_documents()
    and .embed_query() exactly as they did with HuggingFaceEmbeddings."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = _get_model()
        return [v.tolist() for v in model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        model = _get_model()
        return next(model.embed([text])).tolist()


_adapter: Optional[_EmbeddingsAdapter] = None


def get_embeddings() -> _EmbeddingsAdapter:
    """Return the process-wide embeddings adapter, loading the model on first call."""
    global _adapter
    if _adapter is None:
        _get_model()          # pre-load the fastembed model
        _adapter = _EmbeddingsAdapter()
    return _adapter
