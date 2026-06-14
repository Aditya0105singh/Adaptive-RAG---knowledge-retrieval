"""Token counting and USD cost estimation via tiktoken."""
from typing import List

import tiktoken

from src.core.logging import get_logger

logger = get_logger(__name__)


class CostTracker:
    """Calculates per-call and per-session USD cost from token usage."""

    PRICE_PER_1K = {
        # Groq pricing (per 1K tokens, USD)
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
        "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
        "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
        # Local embeddings — free
        "sentence-transformers/all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},
    }

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        """Count tokens in text with tiktoken for the given model."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    @classmethod
    def calculate(cls, model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
        """Return total USD cost for a single API call."""
        prices = cls.PRICE_PER_1K.get(model)
        if prices is None:
            logger.warning("cost_unknown_model", model=model)
            return 0.0
        cost = (prompt_tokens / 1000) * prices.get("input", 0.0)
        cost += (completion_tokens / 1000) * prices.get("output", 0.0)
        return round(cost, 6)

    @classmethod
    def format_session_cost(cls, turns: List[dict]) -> str:
        """Format accumulated session cost, e.g. 'Session: $0.0034 (1,240 tokens)'."""
        total_cost = 0.0
        total_tokens = 0
        for turn in turns:
            prompt = turn.get("prompt", 0)
            completion = turn.get("completion", 0)
            model = turn.get("model", "gpt-4o")
            total_cost += cls.calculate(model, prompt, completion)
            total_tokens += prompt + completion
        return f"Session: ${total_cost:.4f} ({total_tokens:,} tokens)"
