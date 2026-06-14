"""Uvicorn entry point for the Adaptive RAG API."""
import os
import sys

# Force transformers to use PyTorch only — avoids Keras 3 / TensorFlow conflicts.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# torch must be imported before langchain-core / transformers to avoid an
# OSError on Windows when the DLLs are loaded inside an already-large process.
# We import it eagerly here so it is in sys.modules before any other import
# triggers the langchain_core → transformers → torch chain.
try:
    import torch  # noqa: F401
except OSError as _torch_err:
    # GPU-build DLL init failure — insert a lightweight mock so transformers
    # skips its torch-dependent code paths (langchain-core sets
    # _HAS_TRANSFORMERS = False and falls back to tiktoken).
    from unittest.mock import MagicMock as _MagicMock
    _mock = _MagicMock()
    _mock.__version__ = "0.0.0+mock"
    sys.modules["torch"] = _mock

import uvicorn

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def run() -> None:
    """Start the FastAPI app with uvicorn (single-process, no reload)."""
    # Import the app object here — torch is already in sys.modules so the
    # langchain-core / pydantic import chain succeeds without memory pressure.
    from src.api.main import app  # noqa: PLC0415

    # Render injects a dynamic PORT env variable — respect it so the health
    # check passes. Fall back to settings.API_PORT for local development.
    port = int(os.environ.get("PORT", settings.API_PORT))
    logger.info("server_starting", log_level=settings.LOG_LEVEL, port=port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
