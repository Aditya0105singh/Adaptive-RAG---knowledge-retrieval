"""Singleton clients for Qdrant (vectors) and MongoDB (chat history)."""
from typing import Optional

from qdrant_client import QdrantClient

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

_qdrant_client: Optional[QdrantClient] = None
_mongo_client = None          # Optional[MongoClient] — None when MongoDB is unavailable
_mongo_available: bool = True  # flipped to False on first failed attempt


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
                timeout=30,
            )
            # Verify connectivity — raises if Qdrant is not actually running.
            client.get_collections()
            _qdrant_client = client
            logger.info("qdrant_client_initialized", url=settings.QDRANT_URL)
        except Exception:
            logger.warning(
                "qdrant_remote_unavailable",
                url=settings.QDRANT_URL,
                fallback="in-memory",
            )
            # In-memory fallback — no persistence but app stays alive.
            _qdrant_client = QdrantClient(":memory:")
            logger.info("qdrant_client_initialized", url=":memory:")
    return _qdrant_client


def get_mongo_client():
    """Return the process-wide MongoClient singleton, or None if MongoDB is unavailable.

    Never raises — callers must handle a None return value gracefully.
    """
    global _mongo_client, _mongo_available

    if not _mongo_available:
        return None  # already failed once; don't retry on every request

    if _mongo_client is None:
        mongo_uri = settings.MONGO_URI or ""
        if not mongo_uri or mongo_uri in ("mongodb://localhost:27017", ""):
            # No usable URI configured — skip silently.
            logger.warning("mongo_skipped", reason="no remote URI configured")
            _mongo_available = False
            return None
        try:
            from pymongo import MongoClient
            _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Ping to confirm the connection is actually alive.
            _mongo_client.admin.command("ping")
            logger.info("mongo_client_initialized")
        except Exception as exc:
            logger.warning("mongo_client_unavailable", error=str(exc))
            _mongo_client = None
            _mongo_available = False

    return _mongo_client


def get_chat_collection():
    """Return the 'chat_history' collection, or None if MongoDB is unavailable."""
    try:
        client = get_mongo_client()
        if client is None:
            return None
        return client["adaptive_rag"]["chat_history"]
    except Exception as exc:
        logger.warning("mongo_collection_access_failed", error=str(exc))
        return None


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
