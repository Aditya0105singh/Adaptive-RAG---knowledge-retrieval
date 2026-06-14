"""Tavily web search wrapper."""
from typing import List, Optional

from tavily import TavilyClient

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class TavilySearchService:
    """Thin wrapper around the Tavily Search API returning plain-text results."""

    def __init__(self) -> None:
        """Defer Tavily client creation until first use (keeps imports key-free)."""
        self._client: Optional[TavilyClient] = None

    def _get_client(self) -> TavilyClient:
        """Lazily create and cache the Tavily client from settings."""
        if self._client is None:
            self._client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        return self._client

    def search(self, query: str, max_results: int = 5) -> List[str]:
        """Run a Tavily search and return results as formatted strings."""
        try:
            response = self._get_client().search(query=query, max_results=max_results)
        except Exception as exc:
            logger.error("tavily_search_failed", query=query, error=str(exc))
            return []
        results = [
            f"{item.get('title', '')}\n{item.get('content', '')}\n(Source: {item.get('url', '')})"
            for item in response.get("results", [])
        ]
        logger.info("tavily_search_completed", query=query, result_count=len(results))
        return results
