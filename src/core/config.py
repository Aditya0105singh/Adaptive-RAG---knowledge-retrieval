"""Application configuration via Pydantic BaseSettings loaded from .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings; all secrets and tunables live here."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str = ""
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
