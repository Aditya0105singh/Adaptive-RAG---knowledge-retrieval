"""All LangGraph node functions for the adaptive RAG pipeline."""
import json
import threading
import time
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from opentelemetry import trace

from src.agents.state import GraphState
from src.core.config import settings
from src.core.logging import get_logger
from src.services.retrieval import RetrievalService
from src.services.search import TavilySearchService

logger = get_logger(__name__)
tracer = trace.get_tracer("adaptive_rag.nodes")

# Thread-local storage so each SSE request gets its own token stream.
# The SSE endpoint sets a callback before running the graph; generate_answer
# calls _emit_token() for each streamed chunk — zero overhead when not set.
_token_cb = threading.local()


def set_token_callback(fn) -> None:
    """Register (or clear with None) a per-token callable for the current thread."""
    _token_cb.fn = fn


def _emit_token(t: str) -> None:
    fn = getattr(_token_cb, "fn", None)
    if fn and t:
        fn(t)


_llm: Optional[ChatGroq] = None
_retrieval_service = RetrievalService()
_search_service = TavilySearchService()


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0,
            api_key=settings.GROQ_API_KEY,
            max_tokens=32768,
        )
    return _llm


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "prompt": usage.get("input_tokens", 0),
        "completion": usage.get("output_tokens", 0),
    }


def _merge_usage(a: dict, b: dict) -> dict:
    return {
        "prompt": a.get("prompt", 0) + b.get("prompt", 0),
        "completion": a.get("completion", 0) + b.get("completion", 0),
    }


# Keywords that strongly suggest a question targets the uploaded document.
# These are checked as a safety net: if the LLM routes to "general" but a
# document is available and the question matches any of these, override to index.
_DOC_KEYWORDS = (
    # Personal / resume signals
    "his", "her", "their", "my", "the candidate", "project", "skill",
    "experience", "work", "job", "education", "summary", "summarize",
    "background", "resume", "cv", "describe", "tell me about", "who is",
    "about him", "about her", "portfolio", "achievement", "award",
    "publication", "qualification",
    # Document-reference signals
    "document", "file", "paper", "report", "methodology", "according to",
    "in the document", "in the paper", "in the report", "what does it say",
    # Concept-lookup signals — "what is X" when a document is uploaded
    # means "what is X as described in the document"
    "what is", "what are", "define", "explain", "how does", "how do",
    "describe the", "what does", "overview of", "introduction to",
)


def route_question(state: GraphState) -> dict:
    """Classify the question into 'index', 'general', or 'search' with Groq.

    The router is told whether a document is currently indexed so that
    person/content questions prefer the index route when one exists.
    """
    with tracer.start_as_current_span("route_question"):
        question = state["question"]
        doc_available = state.get("doc_available", False)
        doc_filename = state.get("doc_filename") or "none"
        doc_label = (
            f"YES ({doc_filename})" if doc_available and doc_filename != "none"
            else ("YES" if doc_available else "NO")
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a routing classifier for a RAG document assistant.\n"
                    f"Documents currently uploaded: {doc_label}\n\n"
                    "Classify the user question into exactly one category and respond "
                    "with that single word only:\n\n"
                    "- index: use this when a document IS uploaded and the question "
                    "MIGHT be answerable from it. This includes: questions about "
                    "concepts, terms, methods, or topics that could appear in the "
                    "document; questions about a person's background, skills, or work; "
                    "any question where you are UNCERTAIN whether the document covers "
                    "it. When in doubt and a document is uploaded, ALWAYS choose index.\n\n"
                    "- general: ONLY use this when a document is NOT uploaded, OR the "
                    "question is clearly and unambiguously unrelated to any document "
                    "(e.g. pure math, coding syntax, greetings, casual chitchat, "
                    "jokes). Do NOT choose general just because the question sounds "
                    "like a general-knowledge question — it may still be in the doc.\n\n"
                    "- search: use this when the question explicitly needs real-time or "
                    "current information (today's news, live prices, recent events) "
                    "that could not be in any uploaded document.\n\n"
                    "CRITICAL RULES:\n"
                    "1. If a document is uploaded and you are unsure → choose index.\n"
                    "2. A question about a technical concept (e.g. SHAP, causal DAG, "
                    "process mining) should go to index when a document is uploaded — "
                    "the document likely explains it in context.\n"
                    "3. Only choose general when the question is OBVIOUSLY off-topic "
                    "for any document (greetings, pure math, jokes)."
                )
            ),
            HumanMessage(content=question),
        ]
        response = _get_llm().invoke(messages)
        route = response.content.strip().lower()
        if route not in ("index", "general", "search"):
            route = "index"

        # Safety override: doc uploaded but classified general — if the
        # question smells like a document question, force the index route.
        if doc_available and route == "general":
            question_lower = question.lower()
            if any(kw in question_lower for kw in _DOC_KEYWORDS):
                logger.info("route_overridden_to_index", question=question)
                route = "index"

        # Guard: no document available — never route to index (would retrieve
        # stale embeddings from a previous session and hallucinate content).
        if not doc_available and route == "index":
            logger.info("route_overridden_no_doc", question=question)
            route = "general"

        token_usage = _merge_usage(state.get("token_usage", {}), _extract_usage(response))
        logger.info(
            "question_routed",
            route_taken=route,
            question=question,
            doc_available=doc_available,
        )
        return {
            "route_taken": route,
            "token_usage": token_usage,
            "current_query": question,
            "answer_versions": [],
        }


def retrieve_docs(state: GraphState) -> dict:
    """Retrieve top-3 parent context chunks from the session-scoped Qdrant collection."""
    with tracer.start_as_current_span("retrieve_docs"):
        query = state.get("current_query") or state["question"]
        collection_name = state.get("collection_name", "")
        docs_meta = _retrieval_service.retrieve_with_metadata(query, top_k=3, collection_name=collection_name)
        documents = [d["text"] for d in docs_meta]
        logger.info("documents_retrieved", count=len(documents))
        return {"documents": documents, "retrieved_chunk_metadata": docs_meta}


def grade_documents(state: GraphState) -> dict:
    """Score all retrieved documents in one batch LLM call and filter below threshold.

    One call for all docs instead of N individual calls — the main performance win.
    """
    with tracer.start_as_current_span("grade_documents"):
        question = state["question"]
        documents = state.get("documents", [])
        token_usage = state.get("token_usage", {"prompt": 0, "completion": 0})
        loop_count = state.get("loop_count", 0)
        query_used = state.get("current_query") or question

        scores: List[float] = []
        filtered: List[str] = []

        if documents:
            numbered = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(documents))
            messages = [
                SystemMessage(
                    content=(
                        "You grade document relevance for a retrieval system. "
                        f"Given a question and {len(documents)} numbered documents, "
                        "output ONLY a JSON array of floats between 0.0 and 1.0 — "
                        f"one score per document in order. Example for 3 docs: [0.9, 0.3, 0.7]. "
                        "No other text."
                    )
                ),
                HumanMessage(content=f"Question: {question}\n\nDocuments:\n{numbered}"),
            ]
            response = _get_llm().invoke(messages)
            token_usage = _merge_usage(token_usage, _extract_usage(response))
            try:
                raw = response.content.strip()
                # Extract JSON array even if the model wraps it in backticks
                start, end = raw.find("["), raw.rfind("]")
                parsed = json.loads(raw[start:end+1]) if start != -1 else []
                scores = [max(0.0, min(1.0, float(s))) for s in parsed]
            except (ValueError, json.JSONDecodeError):
                scores = [0.0] * len(documents)
            # Pad/trim in case the model returned wrong count
            while len(scores) < len(documents):
                scores.append(0.0)
            scores = scores[:len(documents)]
            filtered = [doc for doc, s in zip(documents, scores) if s >= settings.RELEVANCE_THRESHOLD]

        web_search = len(filtered) == 0
        max_quality = max(scores) if scores else 0.0

        # Filter metadata to match the filtered documents (for source chunk display)
        meta = state.get("retrieved_chunk_metadata") or []
        filtered_meta = [m for m, s in zip(meta, scores) if s >= settings.RELEVANCE_THRESHOLD]

        # Feature 2: record this loop's retrieval attempt
        existing_versions = list(state.get("answer_versions") or [])
        existing_versions.append({
            "loop_number": loop_count + 1,
            "query_used": query_used,
            "retrieval_quality": float(max_quality),
            "chunks_retrieved": len(documents),
            "chunks_passed": len(filtered),
            "query_was_rewritten": loop_count > 0,
            # draft_answer filled in by generate_answer for the final loop
            "draft_answer": None,
        })

        logger.info(
            "documents_graded",
            total=len(documents),
            passed=len(filtered),
            scores=scores,
            web_search=web_search,
        )
        return {
            "documents": filtered,
            "relevance_scores": scores,
            "web_search": web_search,
            "token_usage": token_usage,
            "answer_versions": existing_versions,
            # Keep raw chunks for grounding/gap analysis
            "retrieved_chunk_texts": documents,
            # Filtered metadata for the "Source excerpts" UI panel
            "retrieved_chunk_metadata": filtered_meta,
        }


def transform_query(state: GraphState) -> dict:
    """Rewrite the question for better retrieval and increment loop_count."""
    with tracer.start_as_current_span("transform_query"):
        question = state["question"]
        loop_count = state.get("loop_count", 0) + 1
        messages = [
            SystemMessage(
                content=(
                    "You rewrite search queries to improve vector retrieval. Rewrite the "
                    "question to be more specific and keyword-rich while preserving its "
                    "intent. Output only the rewritten question."
                )
            ),
            HumanMessage(content=question),
        ]
        response = _get_llm().invoke(messages)
        rewritten = response.content.strip()
        token_usage = _merge_usage(state.get("token_usage", {}), _extract_usage(response))
        logger.info("query_transformed", new_query=rewritten, loop_count=loop_count)
        return {
            "question": rewritten,
            "current_query": rewritten,
            "loop_count": loop_count,
            "token_usage": token_usage,
        }


def web_search_node(state: GraphState) -> dict:
    """Fetch web results from Tavily and use them as documents."""
    with tracer.start_as_current_span("web_search"):
        results = _search_service.search(state["question"])
        logger.info("web_search_completed", result_count=len(results))
        return {
            "documents": results,
            "web_search": False,
            "retrieved_chunk_texts": results,
            "retrieved_chunk_metadata": [],  # web results have no file attribution
            "route_taken": "search",
        }


def _compute_answer_improvement(versions: list) -> dict:
    """Determine if the answer meaningfully improved across loop iterations."""
    real_versions = [v for v in versions if v.get("draft_answer")]
    if len(real_versions) <= 1:
        quality_versions = [v for v in versions if v.get("retrieval_quality") is not None]
        if len(quality_versions) > 1:
            delta = quality_versions[-1]["retrieval_quality"] - quality_versions[0]["retrieval_quality"]
            improved = delta > 0.1
            return {
                "improved": improved,
                "reason": f"retrieval quality {quality_versions[0]['retrieval_quality']:.2f} → {quality_versions[-1]['retrieval_quality']:.2f}" if improved else "retrieval quality unchanged",
                "quality_delta": delta,
            }
        return {"improved": False, "reason": "single_loop", "quality_delta": 0.0}

    first_draft = real_versions[0]["draft_answer"]
    final_draft = real_versions[-1]["draft_answer"]
    len_ratio = abs(len(final_draft) - len(first_draft)) / max(len(first_draft), 1)
    length_improved = len_ratio > 0.15
    quality_improved = real_versions[-1]["retrieval_quality"] > real_versions[0]["retrieval_quality"]
    more_content = len(final_draft) > len(first_draft) * 1.1
    improved = (length_improved or quality_improved) and more_content
    reason = []
    if quality_improved:
        reason.append(f"retrieval quality {real_versions[0]['retrieval_quality']:.2f} → {real_versions[-1]['retrieval_quality']:.2f}")
    if length_improved:
        reason.append(f"answer length changed by {len_ratio*100:.0f}%")
    return {
        "improved": improved,
        "reason": ", ".join(reason) if reason else "no meaningful change",
        "quality_delta": real_versions[-1]["retrieval_quality"] - real_versions[0]["retrieval_quality"],
    }


def generate_answer(state: GraphState) -> dict:
    """Generate the final answer from context, with a hallucination self-check."""
    with tracer.start_as_current_span("generate_answer"):
        question = state["question"]
        documents = state.get("documents", [])
        token_usage = state.get("token_usage", {"prompt": 0, "completion": 0})
        route_taken_now = state.get("route_taken", "")
        # Number chunks so the LLM can cite them with [N] markers
        if documents and route_taken_now == "index":
            context = "\n\n---\n\n".join(
                f"[{i + 1}] {doc}" for i, doc in enumerate(documents)
            )
            citation_note = (
                " When you use information from a context chunk, add its number "
                "in brackets right after the relevant sentence, e.g. [1] or [2]. "
                "Only cite chunks you actually used. Do not cite chunks you did not use."
            )
        else:
            context = "\n\n---\n\n".join(documents) if documents else "No context available."
            citation_note = ""
        history = state.get("conversation_history") or []

        # Build history turn messages — cap at last 6 messages (3 exchanges) to
        # stay within token budget. Older context is less relevant anyway.
        history_msgs = []
        for turn in history[-20:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                history_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                history_msgs.append(AIMessage(content=content))

        system_msg = SystemMessage(
            content=(
                "You are a helpful assistant. Answer the question using the provided "
                "context and the conversation history. When the user refers to 'that', "
                "'it', 'the above', or any pronoun referencing a prior answer, use the "
                "conversation history to resolve what they mean. "
                "If the context does not contain the answer, say so honestly."
                + citation_note
            )
        )
        current_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        gen_messages = [system_msg] + history_msgs + [current_msg]
        parts: List[str] = []
        last_chunk = None
        for chunk in _get_llm().stream(gen_messages):
            t = chunk.content or ""
            parts.append(t)
            _emit_token(t)
            last_chunk = chunk
        answer = "".join(parts).strip()
        token_usage = _merge_usage(token_usage, _extract_usage(last_chunk) if last_chunk else {})

        # Feature 2: attach draft_answer to the last version entry
        versions = list(state.get("answer_versions") or [])
        if versions:
            versions[-1] = {**versions[-1], "draft_answer": answer}
        else:
            # No loop tracking happened (general/search route) — create single version
            versions = [{
                "loop_number": 1,
                "query_used": question,
                "retrieval_quality": max(state.get("relevance_scores") or [0.0]) if state.get("relevance_scores") else 0.0,
                "chunks_retrieved": len(documents),
                "chunks_passed": len(documents),
                "query_was_rewritten": False,
                "draft_answer": answer,
            }]

        answer_improvement = _compute_answer_improvement(versions)

        # Feature 3: Knowledge Gap Alerts
        knowledge_gaps = None
        route_taken = state.get("route_taken", "")
        if settings.ENABLE_KNOWLEDGE_GAPS and route_taken == "index":
            try:
                from src.services.knowledge_gap_analyzer import analyze_knowledge_gaps
                knowledge_gaps = analyze_knowledge_gaps(
                    question=state.get("current_query") or question,
                    retrieved_chunks=state.get("retrieved_chunk_texts") or documents,
                    final_answer=answer,
                    llm=_get_llm(),
                    ttl=settings.KNOWLEDGE_GAP_CACHE_TTL,
                )
                if knowledge_gaps and knowledge_gaps.get("token_usage"):
                    token_usage = _merge_usage(token_usage, knowledge_gaps["token_usage"])
            except Exception as e:
                logger.warning("knowledge_gap_skipped", error=str(e))

        # Feature 1: Hallucination Grounding Score
        grounding_data = None
        if settings.ENABLE_GROUNDING_CHECK and route_taken == "index" and documents:
            try:
                from src.services.grounding_checker import check_answer_grounding
                grounding_data = check_answer_grounding(
                    answer=answer,
                    retrieved_chunks=state.get("retrieved_chunk_texts") or documents,
                    llm=_get_llm(),
                    max_sentences=settings.GROUNDING_MAX_SENTENCES,
                    min_sentence_len=settings.GROUNDING_MIN_SENTENCE_LEN,
                )
                if grounding_data and grounding_data.get("token_usage"):
                    token_usage = _merge_usage(token_usage, grounding_data["token_usage"])
            except Exception as e:
                logger.warning("grounding_check_skipped", error=str(e))

        # Measured after the feature calls so the reported latency is honest.
        start_time = state.get("start_time", time.perf_counter())
        processing_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "answer_generated",
            processing_ms=processing_ms,
            loop_count=state.get("loop_count", 0),
            route_taken=route_taken,
        )
        return {
            "generation": answer,
            "token_usage": token_usage,
            "processing_ms": processing_ms,
            "answer_versions": versions,
            "answer_improvement": answer_improvement,
            "knowledge_gaps": knowledge_gaps,
            "grounding": grounding_data,
        }
