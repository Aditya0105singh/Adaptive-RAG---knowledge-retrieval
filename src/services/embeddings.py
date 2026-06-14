"""Process-wide singleton for the sentence-transformers embedding model.

Both DocumentIngestionService and RetrievalService import get_embeddings()
from here so the model is loaded exactly once, keeping RAM under 512 MB.
"""
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import settings

_instance: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the process-wide embeddings instance, loading it on first call."""
    global _instance
    if _instance is None:
        _instance = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)
    return _instance
