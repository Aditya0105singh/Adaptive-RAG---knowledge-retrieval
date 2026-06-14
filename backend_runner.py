"""
backend_runner.py
=================
Starts the FastAPI backend in a background daemon thread so it runs
alongside Streamlit in the same Streamlit Cloud process.

Import this module at the very top of app.py (before any API calls).
It is safe to import multiple times — the server only starts once.
"""
import os
import socket
import threading
import time

_started = False
_lock = threading.Lock()


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """Return True if something is already listening on the given port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run_server(port: int) -> None:
    """Target function for the daemon thread — runs uvicorn forever."""
    import uvicorn
    # Inject env vars so the FastAPI app reads the right port / CORS
    os.environ.setdefault("API_PORT", str(port))
    # Allow all origins when running inside Streamlit Cloud
    os.environ.setdefault(
        "CORS_ORIGINS",
        "http://localhost:8501,https://*.streamlit.app,*",
    )
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",   # keep logs quiet in Streamlit output
        loop="asyncio",
    )


def ensure_backend_running(port: int = 8080, wait_seconds: float = 8.0) -> None:
    """
    Start the FastAPI backend in a daemon thread if it isn't already running.
    Blocks up to *wait_seconds* until the server is accepting connections.

    Call this once from app.py before the first API request.
    """
    global _started

    with _lock:
        if _started:
            return  # Already launched this interpreter session
        if _port_open(port):
            _started = True
            return  # Already running (e.g. local dev where you ran start.py)

        t = threading.Thread(target=_run_server, args=(port,), daemon=True)
        t.start()
        _started = True

    # Wait for the server to be ready
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _port_open(port):
            return
        time.sleep(0.25)
    # If still not up, return anyway — the first request will get a friendly error
