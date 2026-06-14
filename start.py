"""One-command launcher: starts the FastAPI backend and Streamlit frontend together.

Usage:
    python start.py

Both processes are started as children of this script. Press Ctrl+C to stop both.
"""
import os
import signal
import subprocess
import sys
import time
import webbrowser

import requests


BACKEND_URL = "http://localhost:8080"
FRONTEND_URL = "http://localhost:8501"
HEALTH_TIMEOUT = 40  # seconds to wait for backend to be ready


def _ensure_secrets_toml() -> None:
    """Create an empty secrets.toml so Streamlit doesn't show the red banner."""
    os.makedirs(".streamlit", exist_ok=True)
    path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(path):
        open(path, "w").close()


def _wait_for_backend(timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{BACKEND_URL}/health", timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def main() -> None:
    _ensure_secrets_toml()

    print("\n  Adaptive RAG — starting up\n")
    print(f"  Backend  → {BACKEND_URL}")
    print(f"  Frontend → {FRONTEND_URL}")
    print()

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    backend = subprocess.Popen([sys.executable, "main.py"], **kwargs)

    print("  Waiting for backend...", end="", flush=True)
    if not _wait_for_backend(HEALTH_TIMEOUT):
        print(" FAILED")
        print("\n  ERROR: Backend did not start within "
              f"{HEALTH_TIMEOUT}s. Check for errors above.")
        backend.terminate()
        sys.exit(1)
    print(" ready")

    frontend = subprocess.Popen(
        [
            "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        **kwargs,
    )

    time.sleep(2)
    webbrowser.open(FRONTEND_URL)
    print(f"\n  App running at {FRONTEND_URL}")
    print("  Press Ctrl+C to stop both servers.\n")

    def _shutdown(sig, frame):
        print("\n  Stopping...")
        if sys.platform == "win32":
            backend.send_signal(signal.CTRL_BREAK_EVENT)
            frontend.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            backend.terminate()
            frontend.terminate()
        backend.wait()
        frontend.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep alive — both children log to their own stdout
    while True:
        if backend.poll() is not None:
            print("  Backend exited unexpectedly.")
            frontend.terminate()
            sys.exit(1)
        if frontend.poll() is not None:
            print("  Frontend exited unexpectedly.")
            backend.terminate()
            sys.exit(1)
        time.sleep(2)


if __name__ == "__main__":
    main()
