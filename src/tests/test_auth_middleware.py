"""Tests for the API-key authentication middleware.

Covers three scenarios that must hold for the auth layer to be correct:
  1. ENABLE_AUTH=false  → any request passes through regardless of key
  2. ENABLE_AUTH=true   → missing or wrong key → 403 with structured error
  3. ENABLE_AUTH=true   → correct key → request succeeds (200)

Uses FastAPI's TestClient so no live backend is needed.
The graph and all LLM/embedding singletons are mocked out so the test
runs offline and without real API keys.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal app factory — avoids importing main.py which mounts the static
# frontend directory (might not exist in CI) and starts up DB connections.
# ---------------------------------------------------------------------------

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.auth import verify_api_key
from src.core.config import settings


def _make_app(enable_auth: bool, correct_key: str = "test-secret") -> FastAPI:
    """Return a minimal FastAPI app with the auth dependency wired up."""
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(verify_api_key)])
    async def probe():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_off(monkeypatch):
    """TestClient with ENABLE_AUTH=False (the default dev setting)."""
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    monkeypatch.setattr(settings, "API_KEY", "real-secret")
    app = _make_app(enable_auth=False)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def auth_on(monkeypatch):
    """TestClient with ENABLE_AUTH=True and API_KEY='real-secret'."""
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEY", "real-secret")
    app = _make_app(enable_auth=True, correct_key="real-secret")
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Tests — Scenario 1: auth disabled
# ---------------------------------------------------------------------------

class TestAuthDisabled:
    def test_no_key_passes(self, auth_off):
        """With auth off, a request with no key must succeed."""
        r = auth_off.get("/probe")
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True}

    def test_wrong_key_still_passes(self, auth_off):
        """With auth off, even a wrong key is ignored."""
        r = auth_off.get("/probe", headers={"X-API-Key": "garbage"})
        assert r.status_code == 200, r.text

    def test_correct_key_passes(self, auth_off):
        """With auth off, a correct key still works (sanity check)."""
        r = auth_off.get("/probe", headers={"X-API-Key": "real-secret"})
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Tests — Scenario 2: auth enabled, bad / missing key → 403
# ---------------------------------------------------------------------------

class TestAuthEnabledRejections:
    def test_missing_key_returns_403(self, auth_on):
        """No X-API-Key header → 403 with structured error body."""
        r = auth_on.get("/probe")
        assert r.status_code == 403, r.text
        body = r.json()
        assert body["detail"]["error"] == "invalid_api_key"

    def test_wrong_key_returns_403(self, auth_on):
        """Wrong X-API-Key → 403."""
        r = auth_on.get("/probe", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 403, r.text
        body = r.json()
        assert body["detail"]["error"] == "invalid_api_key"

    def test_empty_key_returns_403(self, auth_on):
        """Empty string X-API-Key → 403."""
        r = auth_on.get("/probe", headers={"X-API-Key": ""})
        assert r.status_code == 403, r.text

    def test_error_body_has_message(self, auth_on):
        """The 403 body must include a human-readable 'message' field."""
        r = auth_on.get("/probe")
        assert "message" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Tests — Scenario 3: auth enabled, correct key → 200
# ---------------------------------------------------------------------------

class TestAuthEnabledAccepted:
    def test_correct_key_returns_200(self, auth_on):
        """Correct X-API-Key → request passes through and returns 200."""
        r = auth_on.get("/probe", headers={"X-API-Key": "real-secret"})
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True}
