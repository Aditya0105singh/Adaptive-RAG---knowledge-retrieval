"""Synchronous semantic search over Qdrant child vectors returning parent contexts."""
from typing import List

from src.core.config import settings
from src.core.database import get_qdrant_client
from src.core.logging import get_logger
from src.services.embeddings import get_embeddings

logger = get_logger(__name__)


class RetrievalService:
    """Embeds queries with local sentence-transformers and fetches parent chunks from Qdrant."""

    def has_documents(self, collection_name: str = "") -> bool:
        """True if the collection exists and holds at least one indexed point."""
        cname = collection_name or settings.QDRANT_COLLECTION
        try:
            client = get_qdrant_client()
            if not client.collection_exists(cname):
                return False
            return client.count(cname).count > 0
        except Exception as exc:
            logger.error("document_count_failed", collection=cname, error=str(exc))
            return False

    def retrieve(self, query: str, top_k: int = 3, collection_name: str = "") -> List[str]:
        """Search child vectors by cosine similarity; return unique parent_text payloads."""
        cname = collection_name or settings.QDRANT_COLLECTION
        try:
            query_vector = get_embeddings().embed_query(query)
            client = get_qdrant_client()
            results = client.query_points(
                collection_name=cname,
                query=query_vector,
                limit=top_k * 3,
            ).points
        except Exception as exc:
            logger.error("retrieval_search_failed", query=query, collection=cname, error=str(exc))
            return []

        seen_parents: set = set()
        contexts: List[str] = []
        for hit in results:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            contexts.append(payload.get("parent_text", ""))
            if len(contexts) >= top_k:
                break
        logger.info("retrieval_completed", query=query, collection=cname, results=len(contexts))
        return contexts

    def retrieve_with_metadata(
        self, query: str, top_k: int = 3, collection_name: str = ""
    ) -> List[dict]:
        """Like retrieve() but returns {text, filename} dicts for each parent chunk."""
        cname = collection_name or settings.QDRANT_COLLECTION
        try:
            query_vector = get_embeddings().embed_query(query)
            client = get_qdrant_client()
            results = client.query_points(
                collection_name=cname,
                query=query_vector,
                limit=top_k * 3,
            ).points
        except Exception as exc:
            logger.error("retrieval_search_failed", query=query, collection=cname, error=str(exc))
            return []

        seen_parents: set = set()
        contexts: List[dict] = []
        for hit in results:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            contexts.append({
                "text": payload.get("parent_text", ""),
                "filename": payload.get("filename", ""),
            })
            if len(contexts) >= top_k:
                break
        logger.info("retrieval_with_metadata_completed", query=query, collection=cname, results=len(contexts))
        return contexts
