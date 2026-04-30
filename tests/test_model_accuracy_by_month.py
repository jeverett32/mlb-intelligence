def _admin_session(monkeypatch, app_module):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "admin@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": True,
        },
    )


def _approved_non_admin(monkeypatch, app_module):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )


def test_accuracy_by_month_requires_admin(monkeypatch, app_module, client):
    _approved_non_admin(monkeypatch, app_module)
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month")
    assert r.status_code == 403


def test_accuracy_by_month_returns_months_and_years(monkeypatch, app_module, client):
    _admin_session(monkeypatch, app_module)

    sample_months = [
        {
            "year": 2025, "month": 4, "year_month": "2025-04",
            "count": 120, "brier": 0.21, "accuracy": 0.55,
            "market_brier": 0.22, "market_accuracy": 0.54,
        },
        {
            "year": 2025, "month": 5, "year_month": "2025-05",
            "count": 150, "brier": 0.20, "accuracy": 0.58,
            "market_brier": 0.21, "market_accuracy": 0.55,
        },
    ]
    calls = {}

    def fake_months(artifact_id=None, year=None):
        calls["months_args"] = (artifact_id, year)
        return sample_months

    def fake_years(artifact_id=None):
        calls["years_args"] = (artifact_id,)
        return [2024, 2025]

    monkeypatch.setattr(app_module.DB, "get_model_accuracy_by_month", fake_months)
    monkeypatch.setattr(app_module.DB, "get_model_accuracy_available_years", fake_years)

    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month")
    assert r.status_code == 200
    body = r.json()
    assert body["months"] == sample_months
    assert body["available_years"] == [2024, 2025]
    assert calls["months_args"] == (None, None)
    assert calls["years_args"] == (None,)


def test_accuracy_by_month_passes_filters(monkeypatch, app_module, client):
    _admin_session(monkeypatch, app_module)

    captured = {}

    def fake_months(artifact_id=None, year=None):
        captured["months"] = (artifact_id, year)
        return []

    def fake_years(artifact_id=None):
        captured["years"] = (artifact_id,)
        return []

    monkeypatch.setattr(app_module.DB, "get_model_accuracy_by_month", fake_months)
    monkeypatch.setattr(app_module.DB, "get_model_accuracy_available_years", fake_years)

    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month?artifact_id=42&year=2024")
    assert r.status_code == 200
    assert captured["months"] == (42, 2024)
    assert captured["years"] == (42,)


def test_accuracy_by_month_empty(monkeypatch, app_module, client):
    _admin_session(monkeypatch, app_module)
    monkeypatch.setattr(app_module.DB, "get_model_accuracy_by_month", lambda **kw: [])
    monkeypatch.setattr(app_module.DB, "get_model_accuracy_available_years", lambda **kw: [])
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month")
    assert r.status_code == 200
    assert r.json() == {"months": [], "available_years": []}


def test_accuracy_by_month_db_aggregation(monkeypatch):
    """Unit-test the DB SQL builder by capturing executed SQL + params."""
    import db

    captured = {}

    class FakeCursor:
        def __init__(self):
            self._rows = [
                {
                    "year": 2025, "month": 4, "n": 100,
                    "brier": 0.215, "accuracy": 0.55,
                    "market_brier": 0.221, "market_accuracy": 0.54,
                },
                {
                    "year": 2025, "month": 5, "n": 80,
                    "brier": None, "accuracy": None,
                    "market_brier": None, "market_accuracy": None,
                },
            ]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return self._rows

    class FakeConn:
        def cursor(self, cursor_factory=None):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(db, "get_connection", lambda: FakeConn())

    rows = db.get_model_accuracy_by_month(artifact_id=7, year=2025)
    assert captured["params"] == (7, 2025)
    assert "model_artifact_id = %s" in captured["sql"]
    assert "EXTRACT(YEAR FROM b.game_date) = %s" in captured["sql"]
    assert rows == [
        {
            "year": 2025, "month": 4, "year_month": "2025-04",
            "count": 100, "brier": 0.215, "accuracy": 0.55,
            "market_brier": 0.221, "market_accuracy": 0.54,
        },
        {
            "year": 2025, "month": 5, "year_month": "2025-05",
            "count": 80, "brier": None, "accuracy": None,
            "market_brier": None, "market_accuracy": None,
        },
    ]


def test_accuracy_by_month_db_no_filters(monkeypatch):
    import db

    captured = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return []

    class FakeConn:
        def cursor(self, cursor_factory=None):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(db, "get_connection", lambda: FakeConn())

    rows = db.get_model_accuracy_by_month()
    assert rows == []
    assert captured["params"] == ()
    assert "model_artifact_id" not in captured["sql"]
    assert "EXTRACT(YEAR FROM b.game_date) =" not in captured["sql"]


def test_available_years_db(monkeypatch):
    import db

    captured = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return [(2024,), (2025,), (None,)]

    class FakeConn:
        def cursor(self, cursor_factory=None):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(db, "get_connection", lambda: FakeConn())

    years = db.get_model_accuracy_available_years(artifact_id=11)
    assert years == [2024, 2025]
    assert captured["params"] == (11,)
    assert "model_artifact_id = %s" in captured["sql"]
