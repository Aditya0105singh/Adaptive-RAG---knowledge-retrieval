"""GraphState TypedDict shared across all LangGraph nodes."""
from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """State carried through the adaptive RAG graph."""

    question: str
    generation: str
    documents: List[str]
    web_search: bool
    loop_count: int            # increments on each query rewrite
    route_taken: str           # "index" | "general" | "search"
    relevance_scores: List[float]
    token_usage: dict          # {"prompt": int, "completion": int}
    processing_ms: int         # elapsed ms from graph start
    start_time: float          # internal: perf_counter at graph entry

    # Feature 2: Answer Versioning
    answer_versions: List[Dict[str, Any]]   # per-loop retrieval metadata + draft
    answer_improvement: Optional[Dict[str, Any]]

    # Feature 3: Knowledge Gap Alerts
    knowledge_gaps: Optional[Dict[str, Any]]

    # Feature 1: Hallucination Grounding Score
    grounding: Optional[Dict[str, Any]]

    # Bug 2 fix: router document-awareness
    doc_available: bool        # True when the vector store holds indexed documents
    doc_filename: str          # most recently uploaded filename (for router context)

    # Internal: query used in the current retrieval attempt
    current_query: str
    # Internal: raw chunk texts for the final retrieval (for gap/grounding features)
    retrieved_chunk_texts: List[str]
    # Per-chunk metadata (text + source filename) for the "Source excerpts" UI panel
    retrieved_chunk_metadata: List[dict]

    # Session-scoped Qdrant collection — set from session_id so each user's
    # uploads live in their own namespace and never bleed into other sessions.
    collection_name: str

    # Conversation history from the client: [{role, content}, ...].
    # Passed to generate_answer so follow-up questions resolve correctly.
    conversation_history: List[dict]
