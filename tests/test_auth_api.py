import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

try:
    import dashboard.app as APP
    from fastapi.testclient import TestClient
except Exception as e:  # pragma: no cover
    pytest.skip(f"app import failed (likely no DB): {e}", allow_module_level=True)


client = TestClient(APP.app)


@pytest.fixture(autouse=True)
def clear_session_cookie():
    client.cookies.clear()
    yield
    client.cookies.clear()


def test_pending_user_cannot_login(monkeypatch):
    password = "test-password"
    hashed = APP._hash_password(password)

    monkeypatch.setattr(APP.DB, "get_user_hash", lambda email: hashed)
    monkeypatch.setattr(
        APP.DB,
        "get_user",
        lambda email: {
            "email": email,
            "approval_status": APP.DB.USER_STATUS_PENDING,
            "is_admin": False,
        },
    )

    r = client.post(
        "/login",
        data={"email": "pending@example.com", "password": password},
        follow_redirects=False,
    )
    assert r.status_code == 302
    assert "pending+admin+approval" in r.headers["location"]


def test_non_admin_cannot_access_db_browser(monkeypatch):
    monkeypatch.setattr(
        APP.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": APP.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    client.cookies.set(APP.COOKIE_NAME, "fake-session")
    r = client.get("/api/db/settings")
    assert r.status_code == 403


def test_user_settings_include_personal_and_effective_live(monkeypatch):
    monkeypatch.setattr(
        APP.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": APP.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    monkeypatch.setattr(APP.DB, "is_user_live_betting", lambda email: True)
    monkeypatch.setattr(APP.DB, "is_global_live_betting", lambda: False)
    monkeypatch.setattr(APP.DB, "get_user_setting", lambda email, key, default="": "America/Denver")
    client.cookies.set(APP.COOKIE_NAME, "fake-session")

    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["live_betting"] is True
    assert body["global_live_betting"] is None
    assert body["effective_live_betting"] is False


def test_admin_can_toggle_global_live(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        APP.DB,
        "get_session_user",
        lambda session_id: {
            "email": "admin@example.com",
            "approval_status": APP.DB.USER_STATUS_APPROVED,
            "is_admin": True,
        },
    )
    monkeypatch.setattr(
        APP.DB,
        "set_setting",
        lambda key, value: seen.update({"key": key, "value": value}),
    )
    client.cookies.set(APP.COOKIE_NAME, "fake-session")

    r = client.post("/api/admin/settings/global_live_betting", json={"value": "false"})
    assert r.status_code == 200
    assert seen == {"key": "global_live_betting", "value": "false"}
