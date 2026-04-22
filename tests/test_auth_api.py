import pytest


def test_pending_user_cannot_login(monkeypatch, app_module, client):
    password = "test-password"
    hashed = app_module._hash_password(password)

    monkeypatch.setattr(app_module.DB, "get_user_hash", lambda email: hashed)
    monkeypatch.setattr(
        app_module.DB,
        "get_user",
        lambda email: {
            "email": email,
            "approval_status": app_module.DB.USER_STATUS_PENDING,
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


def test_non_admin_cannot_access_db_browser(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/db/settings")
    assert r.status_code == 403


def test_user_settings_include_personal_and_effective_live(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    monkeypatch.setattr(app_module.DB, "is_user_live_betting", lambda email: True)
    monkeypatch.setattr(app_module.DB, "is_global_live_betting", lambda: False)
    monkeypatch.setattr(
        app_module.DB,
        "get_user_setting",
        lambda email, key, default="": "America/Denver",
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["live_betting"] is True
    assert body["global_live_betting"] is None
    assert body["effective_live_betting"] is False


def test_admin_can_toggle_global_live(monkeypatch, app_module, client):
    seen = {}
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "admin@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": True,
        },
    )
    monkeypatch.setattr(
        app_module.DB,
        "set_setting",
        lambda key, value: seen.update({"key": key, "value": value}),
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.post("/api/admin/settings/global_live_betting", json={"value": "false"})
    assert r.status_code == 200
    assert seen == {"key": "global_live_betting", "value": "false"}


def test_root_returns_landing_page_for_unauthenticated_user(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "MLB Betting Intelligence" in r.text
    assert "Track the model. Understand the edge. See the results in public." in r.text
    assert "MLB Model Performance" not in r.text
    assert "MLB Betting Dashboard" not in r.text


def test_root_returns_private_dashboard_for_approved_user(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/")
    assert r.status_code == 200
    assert "MLB Betting Dashboard" in r.text
    assert "MLB Betting Intelligence" not in r.text


@pytest.mark.parametrize(
    "approval_status",
    ["pending", "rejected", "disabled"],
)
def test_root_denies_private_dashboard_to_unapproved_users(monkeypatch, approval_status, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": approval_status,
            "is_admin": False,
        },
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/")
    assert r.status_code == 200
    assert "MLB Betting Intelligence" in r.text
    assert "MLB Betting Dashboard" not in r.text
