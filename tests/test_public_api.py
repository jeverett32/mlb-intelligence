"""Smoke test for the public (no-auth) dashboard endpoints.

Requires the database to be reachable; skipped otherwise.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

try:
    from dashboard.app import app
    from fastapi.testclient import TestClient
except Exception as e:  # pragma: no cover
    pytest.skip(f"app import failed (likely no DB): {e}", allow_module_level=True)


client = TestClient(app)


def test_public_page_renders():
    r = client.get("/public")
    assert r.status_code == 200
    assert "MLB" in r.text


def test_public_performance_shape():
    r = client.get("/api/public/performance")
    assert r.status_code == 200
    body = r.json()
    for key in ("total_bets", "wins", "losses", "accuracy", "roi_pct", "calibration"):
        assert key in body


def test_public_model_accuracy_shape():
    r = client.get("/api/public/model-accuracy")
    assert r.status_code == 200
    body = r.json()
    for key in ("total", "correct", "incorrect", "accuracy", "calibration"):
        assert key in body


def test_private_api_requires_auth():
    r = client.get("/api/balance")
    assert r.status_code in (401, 403)
