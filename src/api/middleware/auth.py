"""API-key authentication dependency.

Protected endpoints declare ``dependencies=[Depends(verify_api_key)]``. When
``settings.ENABLE_AUTH`` is False (the default) the check is a no-op so local
development and the existing test suite need no header. In production, set
ENABLE_AUTH=true and a strong API_KEY, then send the key in the ``X-API-Key``
request header.
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# auto_error=False so a missing header yields our structured 403 rather than
# FastAPI's default 403 with a plain message.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the X-API-Key header unless auth is disabled."""
    if not settings.ENABLE_AUTH:
        return "auth-disabled"
    if not api_key or api_key != settings.API_KEY:
        logger.warning("api_key_rejected", has_key=bool(api_key))
        raise HTTPException(
            status_code=403,
            detail={
                "error": "invalid_api_key",
                "message": "A valid X-API-Key header is required.",
            },
        )
    return api_key
