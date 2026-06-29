"""Shared helpers for the evaluation suite.

Every evaluation script talks to the *running* FastAPI backend over HTTP rather
than importing the LangGraph app directly — the graph initialises Qdrant /
Cohere / Groq singletons that need live API keys, and going through the real
endpoint is exactly what we want to benchmark.

Run all scripts from the project root (``adaptive_rag/``) with the backend up:

    python main.py                 # terminal 1 — starts the API on :8080
    python evaluate/ragas_eval.py  # terminal 2

Override the backend location with the API_BASE_URL env var if needed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Ensure non-ASCII output (em-dashes, box drawing) prints cleanly on Windows'
# default console codepage instead of raising/mojibaking.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

# Make ``src`` importable when a script is launched as ``python evaluate/foo.py``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import settings  # noqa: E402  (after sys.path tweak)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
SAMPLE_DOC = EVAL_DIR / "sample_doc.md"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# A fixed session id keeps every eval run pointed at the same Qdrant collection
# (``session_eval-suite``) so we upload the document once and reuse it.
EVAL_SESSION_ID = os.environ.get("EVAL_SESSION_ID", "eval-suite")


def api_base() -> str:
    """Base URL of the running backend (defaults to the configured API port)."""
    return os.environ.get("API_BASE_URL", f"http://localhost:{settings.API_PORT}").rstrip("/")


def _auth_headers() -> Dict[str, str]:
    """X-API-Key header when an eval key is configured (only needed if ENABLE_AUTH)."""
    key = os.environ.get("EVAL_API_KEY") or (settings.API_KEY if settings.ENABLE_AUTH else "")
    return {"X-API-Key": key} if key else {}


# --------------------------------------------------------------------------- #
# Backend interaction
# --------------------------------------------------------------------------- #
class BackendError(RuntimeError):
    """Raised when the backend is unreachable or returns an unrecoverable error."""


def check_backend() -> bool:
    """Return True if the backend answers its /health probe."""
    try:
        resp = requests.get(f"{api_base()}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except requests.RequestException:
        return False


def require_backend() -> None:
    """Exit with a clear message if the backend is not running."""
    if not check_backend():
        sys.exit(
            f"\n[!] Backend not reachable at {api_base()}\n"
            f"    Start it first:  python main.py\n"
            f"    (or set API_BASE_URL to point at a deployed instance)\n"
        )


def upload_document(path: Path = SAMPLE_DOC, session_id: str = EVAL_SESSION_ID) -> dict:
    """Upload a document into the session-scoped Qdrant collection.

    Returns the ingestion summary (parent/child counts). Raises BackendError on
    failure so callers can abort early rather than evaluating an empty index.
    """
    if not path.exists():
        raise BackendError(f"Sample document not found: {path}")
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "text/markdown")}
        data = {"session_id": session_id}
        resp = requests.post(
            f"{api_base()}/api/upload", files=files, data=data,
            headers=_auth_headers(), timeout=120,
        )
    if not resp.ok:
        raise BackendError(f"Upload failed ({resp.status_code}): {resp.text[:300]}")
    body = resp.json()
    if body.get("child_count", 0) == 0:
        raise BackendError(f"Document ingested but produced 0 chunks: {body}")
    return body


def chat(
    question: str,
    session_id: str = EVAL_SESSION_ID,
    doc_available: bool = True,
    doc_filename: str = "",
    *,
    timeout: int = 180,
    max_retries: int = 4,
) -> dict:
    """POST one question to /api/chat and return the parsed response.

    Retries with exponential backoff on rate limits (429 / Groq quota) and
    transient 5xx errors so a single hiccup does not abort a long eval run.
    """
    payload = {
        "question": question,
        "session_id": session_id,
        "doc_available": doc_available,
        "doc_filename": doc_filename or (SAMPLE_DOC.name if doc_available else ""),
        "conversation_history": [],
    }
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{api_base()}/api/chat", json=payload,
                headers=_auth_headers(), timeout=timeout,
            )
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        body = resp.text.lower()
        rate_limited = resp.status_code == 429 or "rate" in body or "quota" in body
        if rate_limited or resp.status_code >= 500:
            wait = min(60, 5 * (2 ** attempt))
            last_err = f"{resp.status_code}: {resp.text[:200]}"
            print(f"    ... transient error ({resp.status_code}); retrying in {wait}s")
            time.sleep(wait)
            continue
        raise BackendError(f"/api/chat failed ({resp.status_code}): {resp.text[:300]}")
    raise BackendError(f"/api/chat exhausted retries. Last error: {last_err}")


def extract_contexts(response: dict) -> List[str]:
    """Pull the retrieved chunk texts out of a chat response for RAGAS."""
    return [c.get("text", "") for c in response.get("source_chunks", []) if c.get("text", "").strip()]


# --------------------------------------------------------------------------- #
# RAGAS judge model + embeddings
# --------------------------------------------------------------------------- #
# These power the LLM-as-judge metrics. We deliberately reuse the *same* Groq
# model and Cohere embeddings the system under test uses, so there is no
# proprietary OpenAI judge hidden in the benchmark — everything runs on the
# project's own free-tier keys. Override the judge with RAGAS_JUDGE_MODEL.
# --------------------------------------------------------------------------- #
# Groq API-key rotation
# --------------------------------------------------------------------------- #
# Groq meters tokens per ORGANIZATION, so a second key on the same account does
# not help — but a key from a different account has its own daily budget. Set
# GROQ_API_KEYS to a comma-separated list and the suite probes them in order,
# using the first one that still has budget. When that one is exhausted on a
# later run, it automatically falls through to the next.
_picked_key: Optional[str] = None


def _candidate_keys() -> List[str]:
    raw = os.environ.get("GROQ_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys and settings.GROQ_API_KEY:
        keys = [settings.GROQ_API_KEY]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def pick_working_groq_key(model: str = "llama-3.3-70b-versatile") -> str:
    """Return the first configured Groq key that still has daily budget.

    Probes each key with a real-sized request (a tiny ping can pass on an almost-
    exhausted org, so we request enough tokens to actually trip the daily cap).
    The choice is cached for the life of the process.
    """
    global _picked_key
    if _picked_key:
        return _picked_key
    forced = os.environ.get("GROQ_API_KEY_FORCE")
    if forced:
        _picked_key = forced
        return forced
    keys = _candidate_keys()
    if not keys:
        raise BackendError("No Groq keys configured (set GROQ_API_KEYS or GROQ_API_KEY).")
    # Request enough tokens (~2k prompt + 2k completion) that a nearly-exhausted
    # org trips its daily cap — a tiny ping can pass on an org with only a few
    # hundred tokens left and give a false positive.
    probe = {"model": model, "messages": [{"role": "user", "content": "ping " * 1500}],
             "max_tokens": 2000}
    for i, key in enumerate(keys, 1):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=probe, timeout=30,
            )
        except requests.RequestException as exc:
            print(f"[keys] key #{i}/{len(keys)} probe error ({exc}); trying next")
            continue
        if r.status_code == 200:
            print(f"[keys] using Groq key #{i}/{len(keys)} (has budget)")
            _picked_key = key
            return key
        reason = "rate-limited" if r.status_code == 429 else f"HTTP {r.status_code}"
        print(f"[keys] key #{i}/{len(keys)} unavailable ({reason}); trying next")
    raise BackendError(
        "All Groq keys are exhausted/rate-limited. Wait for the daily reset, "
        "add another account's key to GROQ_API_KEYS, or upgrade to Groq Dev tier."
    )


def build_ragas_llm():
    """Wrap ChatGroq as a RAGAS-compatible judge LLM (on the first working key)."""
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    # Default the judge to the 8B model: it lives in a SEPARATE Groq rate-limit
    # bucket from the 70B pipeline under test, so judging does not compete with
    # the product for the same daily token budget. Override with RAGAS_JUDGE_MODEL.
    model = os.environ.get("RAGAS_JUDGE_MODEL", "llama-3.1-8b-instant")
    llm = ChatGroq(model=model, temperature=0, api_key=pick_working_groq_key(), max_tokens=1024)
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings():
    """Wrap the project's Cohere embeddings for RAGAS answer_relevancy.

    Returns None if embeddings cannot be initialised (e.g. COHERE_API_KEY is
    unset locally); callers then skip the embedding-dependent metric instead of
    crashing the whole run.
    """
    try:
        from langchain_core.embeddings import Embeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        from src.services.embeddings import get_embeddings

        project_embed = get_embeddings()

        class _ProjectEmbeddingsAdapter(Embeddings):
            """Adapt the project's duck-typed embeddings to LangChain's interface."""

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return project_embed.embed_documents(texts)

            def embed_query(self, text: str) -> List[float]:
                return project_embed.embed_query(text)

        return LangchainEmbeddingsWrapper(_ProjectEmbeddingsAdapter())
    except Exception as exc:  # noqa: BLE001 — any init failure → graceful skip
        print(f"[warn] embeddings unavailable ({exc}); answer_relevancy will be skipped")
        return None


# --------------------------------------------------------------------------- #
# Shared test set
# --------------------------------------------------------------------------- #
# Generic, document-agnostic questions — they work against whatever file is
# uploaded. Paired with evaluate/sample_doc.md (a description of this project)
# they all have real, retrievable answers so RAGAS scores are meaningful.
# Trimmed to 8 diverse questions so a full RAGAS run fits inside Groq's free
# daily token budget; set EVAL_NUM_QUESTIONS to use fewer (e.g. 5).
_ALL_TEST_QUESTIONS: List[str] = [
    "What is the main topic of this document?",
    "What methodology or approach is described?",
    "What problem does this document address?",
    "What are the limitations mentioned?",
    "What technologies or tools are referenced?",
    "What future work or improvements are suggested?",
    "What are the main contributions?",
    "What is the overall conclusion?",
    # Extras — included only if EVAL_NUM_QUESTIONS asks for more.
    "What are the key findings or conclusions?",
    "What are the main components or sections?",
    "Who are the intended users or audience?",
    "Summarize the introduction section",
    "What datasets or data sources are used?",
    "What are the evaluation criteria mentioned?",
    "What are the key results or metrics?",
]
_N_QUESTIONS = int(os.environ.get("EVAL_NUM_QUESTIONS", "8"))
TEST_QUESTIONS: List[str] = _ALL_TEST_QUESTIONS[:max(1, _N_QUESTIONS)]


def banner(title: str) -> None:
    """Print a labelled section banner."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def percentiles(values: List[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[k]
