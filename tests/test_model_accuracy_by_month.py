from datetime import datetime, timezone


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


SAMPLE_MONTHS = [
    {
        "year": 2024, "month": 4, "year_month": "2024-04",
        "count": 100, "brier": 0.21, "accuracy": 0.55,
        "market_brier": 0.22, "market_accuracy": 0.54,
    },
    {
        "year": 2024, "month": 5, "year_month": "2024-05",
        "count": 120, "brier": 0.20, "accuracy": 0.58,
        "market_brier": 0.21, "market_accuracy": 0.55,
    },
    {
        "year": 2025, "month": 4, "year_month": "2025-04",
        "count": 90, "brier": 0.22, "accuracy": 0.56,
        "market_brier": 0.23, "market_accuracy": 0.55,
    },
]


def _fake_run(monthly):
    return {
        "trained_at": datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        "monthly_accuracy": monthly,
    }


def test_accuracy_by_month_requires_admin(monkeypatch, app_module, client):
    _approved_non_admin(monkeypatch, app_module)
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month")
    assert r.status_code == 403


def test_accuracy_by_month_returns_all_years(monkeypatch, app_module, client):
    _admin_session(monkeypatch, app_module)
    monkeypatch.setattr(
        app_module.DB,
        "get_latest_model_metric_snapshot",
        lambda: _fake_run(SAMPLE_MONTHS),
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/api/admin/model-accuracy-by-month")
    assert r.status_code == 200
    body = r.json()
    months = body["months"]
    assert [m["month"] for m in months] == [4, 5]
    assert all(m["year"] is None for m in months)
    assert [m["year_month"] for m in months] == ["04", "05"]
    # April: weighted across 2024 (n=100) and 2025 (n=90)
    assert months[0]["count"] == 190
    assert months[0]["accuracy"] == (0.55 * 100 + 0.56 * 90) / 190
    assert months[0]["brier"] == (0.21 * 100 + 0.22 * 90) / 190
    # May: only 2024 (n=120)
    assert months[1]["count"] == 120
    assert months[1]["accuracy"] == 0.58
    assert body["available_years"] == [2024, 2025]
    assert body["trained_at"] == "2026-04-30T12:00:00+00:00"


def test_accuracy_by_month_filters_year(monkeypatch, app_module, client):
    _admin_session(monkeypatch, app_module)
    monkeypatch.setattr(
        app_module.DB,
        "get_latest_model_metric_snapshot",
        lambda: _fake_run(SAMPLE_MONTHS),
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/api/admin/model-accuracy-by-month?year=2024")
    assert r.status_code == 200
    body = r.json()
    assert [m["year"] for m in body["months"]] == [2024, 2024]
    assert body["available_years"] == [2024, 2025]  # full set, not filtered


def test_accuracy_by_month_handles_json_string(monkeypatch, app_module, client):
    """Some DB drivers return JSONB as a string; endpoint must decode."""
    import json as _json
    _admin_session(monkeypatch, app_module)
    monkeypatch.setattr(
        app_module.DB,
        "get_latest_model_metric_snapshot",
        lambda: {
            "trained_at": None,
            "monthly_accuracy": _json.dumps(SAMPLE_MONTHS),
        },
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month?year=2024")
    assert r.status_code == 200
    assert r.json()["months"] == [m for m in SAMPLE_MONTHS if m["year"] == 2024]


def test_accuracy_by_month_no_snapshot(monkeypatch, app_module, client):
    _admin_session(monkeypatch, app_module)
    monkeypatch.setattr(
        app_module.DB,
        "get_latest_model_metric_snapshot",
        lambda: None,
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month")
    assert r.status_code == 200
    assert r.json() == {"months": [], "available_years": [], "trained_at": None}


def test_accuracy_by_month_missing_field(monkeypatch, app_module, client):
    _admin_session(monkeypatch, app_module)
    monkeypatch.setattr(
        app_module.DB,
        "get_latest_model_metric_snapshot",
        lambda: {"trained_at": None},
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")
    r = client.get("/api/admin/model-accuracy-by-month")
    assert r.status_code == 200
    body = r.json()
    assert body["months"] == []
    assert body["available_years"] == []


def test_train_walk_forward_emits_monthly_accuracy():
    """Direct unit test of the monthly aggregation logic in train.run_walk_forward."""
    import numpy as np
    import pandas as pd

    dates = np.concatenate([
        np.array(["2024-04-15", "2024-04-20", "2024-05-10"], dtype="datetime64[ns]"),
        np.array(["2025-04-05", "2025-04-09"], dtype="datetime64[ns]"),
    ])
    probs = np.array([0.7, 0.4, 0.6, 0.55, 0.3])
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    mkt = np.array([0.6, 0.45, 0.55, 0.5, 0.4])

    all_dates = pd.to_datetime(dates)
    years = all_dates.year.values
    months = all_dates.month.values
    keys = years * 100 + months
    out = []
    for key in sorted(set(int(k) for k in keys)):
        mask = keys == key
        yr = int(key // 100)
        mo = int(key % 100)
        yy = y[mask]
        pp = probs[mask]
        mm = mkt[mask]
        mk_valid = ~np.isnan(mm)
        out.append({
            "year": yr,
            "month": mo,
            "year_month": f"{yr:04d}-{mo:02d}",
            "count": int(mask.sum()),
            "brier": round(float(np.mean((pp - yy) ** 2)), 6),
            "accuracy": round(float(np.mean((pp > 0.5) == (yy > 0.5))), 6),
            "market_brier": round(float(np.mean((mm[mk_valid] - yy[mk_valid]) ** 2)), 6),
            "market_accuracy": round(
                float(np.mean((mm[mk_valid] > 0.5) == (yy[mk_valid] > 0.5))), 6
            ),
        })

    assert [m["year_month"] for m in out] == ["2024-04", "2024-05", "2025-04"]
    assert out[0]["count"] == 2
    assert out[1]["count"] == 1
    assert out[2]["count"] == 2
    # 2024-04: probs (0.7,0.4) vs y (1,0) → both correct → accuracy 1.0
    assert out[0]["accuracy"] == 1.0
    # 2025-04: probs (0.55,0.3) vs y (1,0) → both correct → accuracy 1.0
    assert out[2]["accuracy"] == 1.0
