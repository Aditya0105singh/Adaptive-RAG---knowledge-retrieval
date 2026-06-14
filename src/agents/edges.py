"""Conditional edge functions for the adaptive RAG graph."""
from src.agents.state import GraphState
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def decide_route(state: GraphState) -> str:
    """Route to the node matching the classifier's decision."""
    route = state.get("route_taken", "index")
    logger.info("edge_decide_route", route_taken=route)
    return route


def decide_after_grading(state: GraphState) -> str:
    """After grading: generate if docs passed; web search if none passed or loop guardrail hit.

    Fast-path: if the best relevance score is below LOW_MATCH_THRESHOLD on the
    first attempt, skip the query-rewrite retry and fall straight to web search.
    This avoids a wasted LLM rewrite + retrieval round when the document clearly
    doesn't contain the answer.
    """
    if state.get("web_search", False):
        scores = state.get("relevance_scores") or []
        best = max(scores) if scores else 0.0
        loop_count = state.get("loop_count", 0)

        if loop_count >= settings.MAX_LOOP_COUNT:
            logger.warning("loop_guardrail_triggered", loop_count=loop_count)
            return "web_search"

        if best < settings.LOW_MATCH_THRESHOLD and loop_count == 0:
            logger.info("low_match_fast_path_to_web", best_score=best)
            return "web_search"

        logger.info("edge_no_relevant_docs", loop_count=loop_count)
        return "transform_query"
    logger.info("edge_docs_relevant")
    return "generate"
