import pandas as pd

import db
from bet import place_bet
from fetch.fetch_data import _align_odds_api_dates


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, markets):
        self._markets = markets

    def get(self, url, params=None, timeout=None):
        if url.endswith("/markets"):
            return _FakeResponse({"markets": self._markets, "cursor": None})
        ticker = url.rsplit("/", 1)[-1]
        for market in self._markets:
            if market["ticker"] == ticker:
                return _FakeResponse({"market": market})
        return _FakeResponse({"market": {}}, status_code=404)


def test_kalshi_market_matching_fails_closed_without_team_suffix(monkeypatch):
    markets = [
        {
            "event_ticker": "KXMLBGAME-26APR271905BOSNYY",
            "ticker": "KXMLBGAME-26APR271905BOSNYY-BOS",
            "status": "open",
            "yes_ask_dollars": "0.44",
        }
    ]
    monkeypatch.setattr(place_bet, "_retry_session", lambda: _FakeSession(markets))

    ticker, market = place_bet.find_kalshi_market(
        "NYY",
        "BOS",
        "2026-04-27",
        "home",
        game_time_utc="2026-04-27 23:05",
        base_url="https://example.test",
    )

    assert ticker is None
    assert market is None


def test_kalshi_market_matching_uses_exact_team_suffix(monkeypatch):
    markets = [
        {
            "event_ticker": "KXMLBGAME-26APR271905BOSNYY",
            "ticker": "KXMLBGAME-26APR271905BOSNYY-BOS",
            "status": "open",
            "yes_ask_dollars": "0.44",
        },
        {
            "event_ticker": "KXMLBGAME-26APR271905BOSNYY",
            "ticker": "KXMLBGAME-26APR271905BOSNYY-NYY",
            "status": "open",
            "yes_ask_dollars": "0.57",
        },
    ]
    monkeypatch.setattr(place_bet, "_retry_session", lambda: _FakeSession(markets))

    ticker, market = place_bet.find_kalshi_market(
        "NYY",
        "BOS",
        "2026-04-27",
        "home",
        game_time_utc="2026-04-27 23:05",
        base_url="https://example.test",
    )

    assert ticker.endswith("-NYY")
    assert market["yes_ask_dollars"] == "0.57"


def test_kalshi_price_parser_accepts_cents_fields():
    assert place_bet._market_yes_price({"yes_ask": 57}) == 0.57
    assert place_bet._market_yes_price({"yes_ask_cents": "42"}) == 0.42
    assert place_bet._market_yes_price({"yes_ask_dollars": "0.61"}) == 0.61


def test_place_user_bet_persists_no_market_status(monkeypatch):
    recorded = {}
    row = {
        "game_pk": 123,
        "game_date": "2026-04-27",
        "home_team": "NYY",
        "away_team": "BOS",
        "predicted_prob": 0.7,
        "market_implied_prob": 0.55,
        "edge": 0.15,
        "bet_side": "home",
        "bet_frac": 0.1,
    }

    monkeypatch.setattr(place_bet.DB, "get_user", lambda email: {"approval_status": place_bet.DB.USER_STATUS_APPROVED})
    monkeypatch.setattr(place_bet.DB, "get_kalshi_account", lambda email: {"is_active": True, "key_id": "kid", "key_path": "k.pem", "kalshi_env": "demo"})
    monkeypatch.setattr(place_bet.DB, "get_bet", lambda game_pk: row)
    monkeypatch.setattr(place_bet.DB, "get_user_order", lambda email, game_pk: None)
    monkeypatch.setattr(place_bet, "fetch_balance_for_account", lambda **kwargs: 10000)
    monkeypatch.setattr(place_bet, "_execute_bet_row", lambda *args, **kwargs: (_ for _ in ()).throw(place_bet.PlaceBetError("No open Kalshi market found")))
    monkeypatch.setattr(place_bet.DB, "upsert_user_order", lambda email, game_pk, **kwargs: recorded.update(kwargs))

    result = place_bet.place_user_bet("User@Example.com", "123")

    assert result["status"] == "skipped_no_market"
    assert recorded["status"] == "skipped_no_market"
    assert recorded["bet_dollars"] is None


def test_missing_user_live_betting_setting_defaults_off(monkeypatch):
    monkeypatch.setattr(db, "get_user_setting", lambda email, key, default="": default)

    assert db.is_user_live_betting("user@example.com") is False


def test_retry_query_includes_retryable_user_order_statuses():
    assert "user_orders uo" in db.get_upcoming_needing_prediction.__doc__ or db.RETRYABLE_ORDER_STATUSES
    assert "skipped_no_market" in db.RETRYABLE_ORDER_STATUSES
    assert "error" in db.RETRYABLE_ORDER_STATUSES


def test_odds_api_dates_align_to_mlb_schedule_date():
    schedule_df = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2026-04-27"),
                "game_time_utc": "2026-04-28 02:05",
                "home_team": "LAD",
                "away_team": "SFG",
            }
        ]
    )
    api_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-28",
                "commence_time_utc": "2026-04-28 02:05",
                "home_team": "LAD",
                "away_team": "SFG",
            }
        ]
    )

    aligned = _align_odds_api_dates(api_df, schedule_df)

    assert pd.to_datetime(aligned.iloc[0]["game_date"]).date().isoformat() == "2026-04-27"


def test_dashboard_open_positions_exclude_skipped_and_error_statuses(app_module):
    df = pd.DataFrame(
        [
            {"game_pk": 1, "result": None, "bet_dollars": 12.0, "status": "filled"},
            {"game_pk": 2, "result": None, "bet_dollars": 12.0, "status": "error"},
            {"game_pk": 3, "result": None, "bet_dollars": 0.0, "status": "skipped_no_live_edge"},
            {"game_pk": 4, "result": True, "bet_dollars": 12.0, "status": "filled"},
            {"game_pk": 5, "result": None, "bet_dollars": 12.0, "status": "dry_run"},
        ]
    )

    open_rows = df[app_module._is_open_position_frame(df)]

    assert open_rows["game_pk"].tolist() == [1, 5]
