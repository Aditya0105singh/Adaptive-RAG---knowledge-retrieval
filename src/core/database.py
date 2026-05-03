"""Singleton clients for Qdrant (vectors) and MongoDB (chat history)."""
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from qdrant_client import QdrantClient

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

_qdrant_client: Optional[QdrantClient] = None
_mongo_client: Optional[MongoClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return the process-wide QdrantClient singleton, creating it on first use.

    Falls back to an in-memory instance when the configured URL is unreachable
    so the server runs without Docker / a remote Qdrant deployment.
    """
    global _qdrant_client
    if _qdrant_client is None:
        try:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=3,
            )
            # Verify connectivity — raises if Qdrant is not actually running.
            client.get_collections()
            _qdrant_client = client
            logger.info("qdrant_client_initialized", url=settings.QDRANT_URL)
        except Exception:
            logger.warning(
                "qdrant_remote_unavailable",
                url=settings.QDRANT_URL,
                fallback="local-disk",
            )
            # Persist to disk so indexed documents survive backend restarts.
            _qdrant_client = QdrantClient(path="./qdrant_storage")
            logger.info("qdrant_client_initialized", url="./qdrant_storage")
    return _qdrant_client


def get_mongo_client() -> MongoClient:
    """Return the process-wide MongoClient singleton, creating it on first use."""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            logger.info("mongo_client_initialized")
        except Exception as exc:
            logger.error("mongo_client_init_failed", error=str(exc))
            raise
    return _mongo_client


def get_chat_collection() -> Collection:
    """Return the 'chat_history' collection in the 'adaptive_rag' database."""
    try:
        return get_mongo_client()["adaptive_rag"]["chat_history"]
    except Exception as exc:
        logger.error("mongo_collection_access_failed", error=str(exc))
        raise


def close_connections() -> None:
    """Close both client singletons (called from FastAPI lifespan teardown)."""
    global _qdrant_client, _mongo_client
    try:
        if _qdrant_client is not None:
            _qdrant_client.close()
            _qdrant_client = None
        if _mongo_client is not None:
            _mongo_client.close()
            _mongo_client = None
        logger.info("database_connections_closed")
    except Exception as exc:
        logger.error("database_close_failed", error=str(exc))
