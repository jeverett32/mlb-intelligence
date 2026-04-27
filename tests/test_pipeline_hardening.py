from datetime import datetime, timezone

import pandas as pd

import db
import run_pipeline
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


class _FakeCursor:
    def __init__(self):
        self.calls = []
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if "FROM app_users" in sql:
            self._rows = [{"email": "user@example.com"}]
        elif "FROM bets b" in sql:
            self._rows = [
                {
                    "game_pk": 1,
                    "game_date": "2026-04-01",
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "predicted_prob": 0.65,
                    "market_implied_prob": 0.55,
                    "edge": 0.10,
                    "bet_side": "home",
                    "bet_frac": 0.05,
                    "home_win": True,
                },
                {
                    "game_pk": 2,
                    "game_date": "2026-04-02",
                    "home_team": "LAD",
                    "away_team": "SFG",
                    "predicted_prob": 0.40,
                    "market_implied_prob": 0.50,
                    "edge": 0.10,
                    "bet_side": "away",
                    "bet_frac": 0.05,
                    "home_win": False,
                },
            ]
        elif "FROM paper_orders" in sql:
            self._rows = []
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.committed = False
        self.closed = False

    def cursor(self, cursor_factory=None):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        now = cls(2026, 4, 27, 19, 55, tzinfo=timezone.utc)
        if tz is None:
            return now.replace(tzinfo=None)
        return now.astimezone(tz)


def _upcoming_games(*times: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": idx,
                "game_date": "2026-04-27",
                "game_time_utc": game_time,
                "home_team": f"HOME{idx}",
                "away_team": f"AWAY{idx}",
            }
            for idx, game_time in enumerate(times, start=1)
        ]
    )


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


def test_next_batch_does_not_pull_distinct_start_slot_too_early(monkeypatch):
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_upcoming_needing_prediction",
        lambda season: _upcoming_games("2026-04-27 20:10", "2026-04-27 20:25"),
    )

    batch, run_at = run_pipeline.get_next_batch()

    assert [game["game_pk"] for game in batch] == [1]
    assert run_at == datetime(2026, 4, 27, 19, 55, tzinfo=timezone.utc)


def test_next_batch_keeps_same_start_slot_together(monkeypatch):
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_upcoming_needing_prediction",
        lambda season: _upcoming_games("2026-04-27 20:10", "2026-04-27 20:10"),
    )

    batch, run_at = run_pipeline.get_next_batch()

    assert [game["game_pk"] for game in batch] == [1, 2]
    assert run_at == datetime(2026, 4, 27, 19, 55, tzinfo=timezone.utc)


def test_next_batch_runs_late_game_before_first_pitch(monkeypatch):
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_upcoming_needing_prediction",
        lambda season: _upcoming_games("2026-04-27 20:00"),
    )

    batch, run_at = run_pipeline.get_next_batch()

    assert [game["game_pk"] for game in batch] == [1]
    assert run_at == datetime(2026, 4, 27, 19, 45, tzinfo=timezone.utc)


def test_backfill_paper_orders_uses_rolling_bankroll(monkeypatch):
    fake_cursor = _FakeCursor()
    fake_conn = _FakeConnection(fake_cursor)
    inserted = []

    monkeypatch.setattr(db, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(
        db,
        "execute_values",
        lambda cur, sql, rows, template=None: inserted.extend(rows),
    )

    count = db.backfill_paper_orders_from_bets("user@example.com")

    assert count == 2
    assert inserted[0][10] == 500.0
    assert inserted[0][16] == 409.09
    assert inserted[0][17] == 10000.0
    assert inserted[0][18] == 10409.09
    assert inserted[1][10] == 520.45
    assert inserted[1][17] == 10409.09


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
    recorded_paper = {}
    recorded_live = {}
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
    monkeypatch.setattr(place_bet.DB, "get_paper_bankroll_dollars", lambda email: 10000.0)
    monkeypatch.setattr(place_bet, "fetch_balance_for_account", lambda **kwargs: 10000)
    monkeypatch.setattr(place_bet, "_execute_bet_row", lambda *args, **kwargs: (_ for _ in ()).throw(place_bet.PlaceBetError("No open Kalshi market found")))
    monkeypatch.setattr(place_bet.DB, "upsert_paper_order", lambda email, game_pk, **kwargs: recorded_paper.update(kwargs))
    monkeypatch.setattr(place_bet.DB, "upsert_user_order", lambda email, game_pk, **kwargs: recorded_live.update(kwargs))

    result = place_bet.place_user_bet("User@Example.com", "123")

    assert result["status"] == "skipped_no_market"
    assert result["mode"] == "paper"
    assert recorded_paper["status"] == "skipped_no_market"
    assert recorded_paper["bet_dollars"] is None
    assert recorded_live == {}


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
