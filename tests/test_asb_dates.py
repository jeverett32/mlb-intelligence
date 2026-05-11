from __future__ import annotations

import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "model_v1"))
import asb  # type: ignore  # local module via sys.path tweak


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_fetch_asg_date_from_mlb_stats_api_parses_game_type_A(monkeypatch):
    payload = {
        "dates": [
            {"date": "2026-07-13", "games": []},
            {"date": "2026-07-14", "games": [{"gameType": "A", "gamePk": 123}]},
            {"date": "2026-07-15", "games": [{"gameType": "R", "gamePk": 456}]},
        ]
    }

    def fake_get(url, params=None, timeout=None):
        assert "schedule" in str(url)
        assert params["startDate"].startswith("2026-07-")
        return _Resp(payload)

    monkeypatch.setattr(asb.requests, "get", fake_get)

    res = asb.fetch_asg_date_from_mlb_stats_api(2026)
    assert res.season == 2026
    assert res.asg_date == "2026-07-14"


def test_fetch_asg_date_from_mlb_stats_api_returns_none_when_missing(monkeypatch):
    payload = {"dates": [{"date": "2026-07-14", "games": [{"gameType": "R"}]}]}

    monkeypatch.setattr(asb.requests, "get", lambda *a, **k: _Resp(payload))
    res = asb.fetch_asg_date_from_mlb_stats_api(2026)
    assert res.asg_date is None

