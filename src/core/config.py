"""Application configuration via Pydantic BaseSettings loaded from .env."""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _inject_streamlit_secrets() -> None:
    """
    On Streamlit Cloud, secrets live in st.secrets but the FastAPI backend
    (running in the same process) reads them via os.environ / .env.
    This bridges the gap by copying every st.secrets key into os.environ
    so that Pydantic BaseSettings can see them.
    Must be called before `settings` is instantiated.
    """
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
    except Exception:
        pass  # Not running inside Streamlit, or secrets not configured yet


_inject_streamlit_secrets()


class Settings(BaseSettings):
    """Central application settings; all secrets and tunables live here."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str = ""
    # Optional comma-separated rotation pool — each key has its own TPM bucket.
    GROQ_API_KEYS: str = ""
    # Fallback providers — add any key to activate that provider automatically.
    GOOGLE_API_KEY: str = ""    # Gemini 2.0 Flash — 1M tokens/day free
    CEREBRAS_API_KEY: str = ""  # Cerebras Llama 3.3 70B — very fast, generous free tier
    COHERE_API_KEY: str = ""
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    MONGO_URI: str = "mongodb://localhost:27017"
    TAVILY_API_KEY: str = ""

    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384

    QDRANT_COLLECTION: str = "documents"
    PARENT_CHUNK_SIZE: int = 1500
    CHILD_CHUNK_SIZE: int = 400
    RELEVANCE_THRESHOLD: float = 0.6
    MAX_LOOP_COUNT: int = 2
    # If the best chunk score is below this on the first attempt, skip the
    # retry/rewrite loop and fall straight through to web search.
    LOW_MATCH_THRESHOLD: float = 0.3
    LOG_LEVEL: str = "INFO"
    # Comma-separated list of allowed CORS origins (frontend URLs)
    CORS_ORIGINS: str = "http://localhost:8501,http://localhost:8502"
    # Port for the FastAPI backend (override when 8080 is taken by another app)
    API_PORT: int = 8080

    # API-key authentication. Disabled by default so local dev needs no header;
    # set ENABLE_AUTH=true and a strong API_KEY in production.
    API_KEY: str = "dev-key-change-in-production"
    ENABLE_AUTH: bool = False

    # Feature 3: Knowledge Gap Alerts
    ENABLE_KNOWLEDGE_GAPS: bool = True
    KNOWLEDGE_GAP_MODEL: str = "llama-3.3-70b-versatile"
    KNOWLEDGE_GAP_COMPLETENESS_THRESHOLD: int = 8
    KNOWLEDGE_GAP_CACHE_TTL: int = 600

    # Feature 1: Hallucination Grounding Score
    ENABLE_GROUNDING_CHECK: bool = True
    GROUNDING_MODEL: str = "llama-3.3-70b-versatile"
    GROUNDING_MAX_SENTENCES: int = 12
    GROUNDING_MIN_SENTENCE_LEN: int = 20


settings = Settings()

# embeddings.py reads COHERE_API_KEY from os.environ directly (not via Settings).
# When the key comes from .env it lands on the Settings object but not os.environ,
# so bridge it across for local runs. On Render the var is already in os.environ.
if settings.COHERE_API_KEY and "COHERE_API_KEY" not in os.environ:
    os.environ["COHERE_API_KEY"] = settings.COHERE_API_KEY

# Bridge the rotation pool into os.environ so the eval helpers (which read it
# from the environment) can see a value supplied via .env.
if settings.GROQ_API_KEYS and "GROQ_API_KEYS" not in os.environ:
    os.environ["GROQ_API_KEYS"] = settings.GROQ_API_KEYS

# Bridge fallback provider keys so nodes.py can read them via os.environ.
if settings.GOOGLE_API_KEY and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
if settings.CEREBRAS_API_KEY and "CEREBRAS_API_KEY" not in os.environ:
    os.environ["CEREBRAS_API_KEY"] = settings.CEREBRAS_API_KEY
