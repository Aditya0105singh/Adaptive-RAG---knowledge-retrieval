"""FastAPI application: CORS, lifespan, routers, error handling, request logging."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.api.routers import chat, suggestions, upload
from src.core.config import settings
from src.core.database import close_connections, get_mongo_client, get_qdrant_client
from src.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database clients on startup; close them on shutdown."""
    try:
        get_qdrant_client()
        logger.info("qdrant_ready")
    except Exception as exc:
        logger.error("qdrant_init_failed", error=str(exc))

    # MongoDB is optional — log warning but never block startup
    try:
        from src.core.database import get_mongo_client
        client = get_mongo_client()
        if client is None:
            logger.warning("mongo_unavailable_startup", reason="skipping — app runs without chat history persistence")
        else:
            logger.info("mongo_ready")
    except Exception as exc:
        logger.warning("mongo_init_warning", error=str(exc))

    logger.info("startup_complete")
    yield
    close_connections()
    logger.info("shutdown_complete")


app = FastAPI(title="Adaptive RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log method, path, status code, and processing time for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    processing_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        processing_ms=processing_ms,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return any unhandled exception as a structured JSON error."""
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "message": str(exc)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return HTTPExceptions with their structured detail payload."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/", include_in_schema=False)
async def root():
    """Redirect browser visits to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> dict:
    """Liveness probe for the Streamlit backend-status indicator."""
    return {"status": "ok"}


app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(suggestions.router)
