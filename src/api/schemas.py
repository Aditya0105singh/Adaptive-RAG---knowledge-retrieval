"""Pydantic v2 request/response models for the API."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message with its session identifier."""

    question: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    # Router context: whether the client believes a document is uploaded.
    # The backend additionally verifies against the vector store.
    doc_available: bool = False
    doc_filename: str = ""
    # Previous turns sent by the client so the LLM can handle follow-ups.
    # Each entry is {"role": "user"|"assistant", "content": str}.
    conversation_history: List[Dict[str, str]] = []


class AnswerVersion(BaseModel):
    """Metadata for one retrieval loop iteration (Feature 2)."""

    loop_number: int
    query_used: str
    retrieval_quality: float
    chunks_retrieved: int
    chunks_passed: int = 0
    draft_answer: Optional[str] = None
    query_was_rewritten: bool = False


class ChatResponse(BaseModel):
    """Answer plus glass-box telemetry for the frontend."""

    answer: str
    route_taken: str
    relevance_scores: List[float] = []
    loops_executed: int = 0
    processing_ms: int = 0
    estimated_cost_usd: float = 0.0
    token_usage: dict = {}

    # Feature 2: Answer Versioning
    answer_versions: List[AnswerVersion] = []
    answer_improvement: Optional[Dict[str, Any]] = None

    # Feature 3: Knowledge Gap Alerts
    knowledge_gaps: Optional[Dict[str, Any]] = None

    # Feature 1: Hallucination Grounding Score
    grounding: Optional[Dict[str, Any]] = None

    # Source chunks surfaced to the UI for the "Source" expander.
    # Each entry is {text: str, filename: str}.
    source_chunks: List[Dict[str, Any]] = []


class UploadResponse(BaseModel):
    """Result of a document ingestion."""

    filename: str
    parent_count: int
    child_count: int
    status: str


class SessionMessage(BaseModel):
    """A single stored chat turn."""

    role: str
    content: str
    timestamp: datetime
