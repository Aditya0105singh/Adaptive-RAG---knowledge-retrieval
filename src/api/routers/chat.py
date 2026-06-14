"""Chat endpoints: synchronous answer and SSE streaming variant with stage events."""
import json
import queue
import threading
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.agents.graph import app as graph_app
from src.agents.nodes import set_token_callback
from src.api.schemas import ChatRequest, ChatResponse
from src.core.config import settings
from src.core.database import get_chat_collection
from src.core.logging import get_logger
from src.services.cost_tracker import CostTracker
from src.services.retrieval import RetrievalService

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

_retrieval_service = RetrievalService()


def _build_initial_state(request: ChatRequest) -> dict:
    """Build the graph input state, verifying doc availability server-side.

    Each session gets its own Qdrant collection so documents never bleed
    between users. The client's doc_available flag is verified against the
    real collection rather than trusted directly.
    """
    collection_name = f"session_{request.session_id}"
    doc_available = _retrieval_service.has_documents(collection_name)
    if request.doc_available and not doc_available:
        logger.warning("doc_available_mismatch", session_id=request.session_id)
    return {
        "question": request.question,
        "loop_count": 0,
        "web_search": False,
        "token_usage": {"prompt": 0, "completion": 0},
        "start_time": time.perf_counter(),
        "doc_available": doc_available,
        "doc_filename": request.doc_filename or "",
        "collection_name": collection_name,
        "conversation_history": list(request.conversation_history),
    }


def _graph_config(request: ChatRequest) -> dict:
    return {"configurable": {"thread_id": request.session_id, "session_id": request.session_id}}


def _persist_turn(request: ChatRequest, response: ChatResponse) -> None:
    """Save the user and assistant messages for this turn to MongoDB (if available)."""
    try:
        collection = get_chat_collection()
        if collection is None:
            return  # MongoDB not configured — skip silently
        now = datetime.utcnow()
        collection.insert_many(
            [
                {
                    "session_id": request.session_id,
                    "role": "user",
                    "content": request.question,
                    "timestamp": now,
                },
                {
                    "session_id": request.session_id,
                    "role": "assistant",
                    "content": response.answer,
                    "timestamp": now,
                    "metadata": {
                        "route_taken": response.route_taken,
                        "relevance_scores": response.relevance_scores,
                        "loops_executed": response.loops_executed,
                        "processing_ms": response.processing_ms,
                        "estimated_cost_usd": response.estimated_cost_usd,
                        "token_usage": response.token_usage,
                        "answer_improvement": response.answer_improvement,
                        "knowledge_gaps": response.knowledge_gaps,
                        "grounding": response.grounding,
                    },
                },
            ]
        )
    except Exception as exc:
        logger.error("chat_history_save_failed", session_id=request.session_id, error=str(exc))


def _build_response(final_state: dict) -> ChatResponse:
    """Map final graph state onto the API response schema."""
    token_usage = final_state.get("token_usage", {"prompt": 0, "completion": 0})
    cost = CostTracker.calculate(
        settings.GROQ_MODEL, token_usage.get("prompt", 0), token_usage.get("completion", 0)
    )
    raw_meta = final_state.get("retrieved_chunk_metadata") or []
    source_chunks = [
        {"text": m["text"][:1200].strip(), "filename": m.get("filename", "")}
        for m in raw_meta
        if m.get("text", "").strip()
    ][:5]
    return ChatResponse(
        answer=final_state.get("generation", ""),
        route_taken=final_state.get("route_taken", "unknown"),
        relevance_scores=final_state.get("relevance_scores", []),
        loops_executed=final_state.get("loop_count", 0),
        processing_ms=final_state.get("processing_ms", 0),
        estimated_cost_usd=cost,
        token_usage=token_usage,
        answer_versions=final_state.get("answer_versions") or [],
        answer_improvement=final_state.get("answer_improvement"),
        knowledge_gaps=final_state.get("knowledge_gaps"),
        grounding=final_state.get("grounding"),
        source_chunks=source_chunks,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run a question through the adaptive RAG graph and persist the turn."""
    try:
        final_state = graph_app.invoke(_build_initial_state(request), config=_graph_config(request))
    except Exception as exc:
        logger.error("chat_graph_failed", session_id=request.session_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={"error": "graph_execution_failed", "message": str(exc)},
        )
    response = _build_response(final_state)
    _persist_turn(request, response)
    logger.info(
        "chat_completed",
        session_id=request.session_id,
        route_taken=response.route_taken,
        loop_count=response.loops_executed,
        processing_ms=response.processing_ms,
    )
    return response


def _sse(payload: dict) -> str:
    """Format one SSE data event."""
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream pipeline stage events AND real LLM tokens concurrently.

    The graph runs in a background thread; generate_answer streams tokens into
    a queue via the thread-local callback set here.  The SSE generator drains
    that queue between stage events so the user sees words appear as they are
    generated — not as a post-hoc burst after the full answer is ready.
    """

    def event_generator():  # noqa: C901 — intentionally long, it's the SSE protocol
        initial_state = _build_initial_state(request)

        token_q: queue.Queue = queue.Queue()
        stage_q: queue.Queue = queue.Queue()

        # --- graph thread ---------------------------------------------------
        def run_graph():
            # Set callback in THIS thread so generate_answer (same thread) picks it up.
            set_token_callback(lambda t: token_q.put(t))
            local: dict = dict(initial_state)
            try:
                for update in graph_app.stream(
                    initial_state, config=_graph_config(request), stream_mode="updates"
                ):
                    for node_name, delta in update.items():
                        local.update(delta or {})
                        stage_q.put(("update", node_name, dict(local)))
            except Exception as exc:
                logger.error("chat_stream_failed", session_id=request.session_id, error=str(exc))
                stage_q.put(("error", str(exc), {}))
            finally:
                set_token_callback(None)
                token_q.put(None)          # sentinel: LLM finished streaming
                stage_q.put(("done", None, dict(local)))

        graph_thread = threading.Thread(target=run_graph, daemon=True)

        # --- helpers --------------------------------------------------------
        def drain_tokens():
            while True:
                try:
                    tok = token_q.get_nowait()
                    if tok is not None:
                        yield _sse({"type": "token", "content": tok})
                except queue.Empty:
                    break

        # --- SSE loop -------------------------------------------------------
        yield _sse({"type": "stage", "stage": "routing", "message": "Classifying your question..."})
        graph_thread.start()

        final_state: dict = {}
        graph_done = False
        stream_error = None

        while not graph_done:
            # Drain all immediately available tokens first.
            yield from drain_tokens()

            try:
                evt_type, node_name, snap = stage_q.get(timeout=0.02)
            except queue.Empty:
                continue

            route = snap.get("route_taken", "")
            loop = snap.get("loop_count", 0)

            if evt_type == "done":
                final_state = snap
                graph_done = True

            elif evt_type == "error":
                stream_error = node_name  # node_name holds the error string here
                graph_done = True

            elif evt_type == "update":
                if node_name == "route_question":
                    yield _sse({"type": "stage", "stage": "routed", "route": route,
                                "message": f"Route decided: {route}"})
                    if route == "general":
                        yield _sse({"type": "stage", "stage": "generating", "route": route,
                                    "message": "Generating your answer..."})
                    elif route == "search":
                        yield _sse({"type": "stage", "stage": "searching_web", "route": route,
                                    "message": "Searching the web..."})
                    elif route == "index":
                        yield _sse({"type": "stage", "stage": "retrieving", "route": route,
                                    "loop": 1, "message": "Searching your document..."})

                elif node_name == "retrieve":
                    yield _sse({"type": "stage", "stage": "retrieving", "route": route,
                                "loop": loop + 1,
                                "message": f"Searching your document... (attempt {loop + 1})"})

                elif node_name == "grade_documents":
                    scores = snap.get("relevance_scores") or []
                    best = max(scores) if scores else 0.0
                    yield _sse({"type": "stage", "stage": "retrieved", "route": route,
                                "loop": loop + 1, "relevance": best,
                                "message": f"Graded {len(scores)} chunks (best: {int(best * 100)}%)"})
                    if not snap.get("web_search", False):
                        yield _sse({"type": "stage", "stage": "generating", "route": route,
                                    "message": "Generating your answer..."})

                elif node_name == "transform_query":
                    yield _sse({"type": "stage", "stage": "rewriting", "route": route,
                                "loop": loop + 1,
                                "message": "Rewriting query for better retrieval..."})

                elif node_name == "do_web_search":
                    # Detect fallback: relevance_scores is non-empty → came via retrieval
                    prev_scores = snap.get("relevance_scores") or []
                    if prev_scores:
                        best_prev = max(prev_scores)
                        msg = (f"Low match ({int(best_prev * 100)}%) — "
                               "searching the web instead...")
                    else:
                        msg = "Searching the web..."
                    yield _sse({"type": "stage", "stage": "searching_web", "route": "search",
                                "message": msg})
                    yield _sse({"type": "stage", "stage": "generating", "route": "search",
                                "message": "Generating your answer..."})

                elif node_name == "generate":
                    # Drain tokens that arrived while we were processing stage events
                    yield from drain_tokens()
                    yield _sse({"type": "stage", "stage": "done", "route": route,
                                "message": "Answer ready"})

        # Final token drain (covers any tokens queued between last drain and "done")
        yield from drain_tokens()
        graph_thread.join(timeout=5)

        if stream_error:
            yield _sse({"type": "error", "message": stream_error})
            return

        response = _build_response(final_state)
        _persist_turn(request, response)
        metadata_event = {"type": "metadata", **json.loads(response.model_dump_json())}
        yield _sse(metadata_event)
        yield _sse({"type": "done_all"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
