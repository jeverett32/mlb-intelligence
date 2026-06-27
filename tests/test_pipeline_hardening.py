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


class _SingleCursor:
    def __init__(self, *, rows=None, one=None):
        self.calls = []
        self.rows = rows or []
        self.one = one

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.one


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
    assert run_at == datetime(2026, 4, 27, 20, 0, tzinfo=timezone.utc)


def test_next_batch_keeps_same_start_slot_together(monkeypatch):
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_upcoming_needing_prediction",
        lambda season: _upcoming_games("2026-04-27 20:10", "2026-04-27 20:10"),
    )

    batch, run_at = run_pipeline.get_next_batch()

    assert [game["game_pk"] for game in batch] == [1, 2]
    assert run_at == datetime(2026, 4, 27, 20, 0, tzinfo=timezone.utc)


def test_next_batch_runs_late_game_before_first_pitch(monkeypatch):
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_upcoming_needing_prediction",
        lambda season: _upcoming_games("2026-04-27 20:00"),
    )

    batch, run_at = run_pipeline.get_next_batch()

    assert [game["game_pk"] for game in batch] == [1]
    assert run_at == datetime(2026, 4, 27, 19, 50, tzinfo=timezone.utc)


def test_get_pending_payout_user_orders_filters_incomplete_refresh_jobs(monkeypatch):
    rows = [
        {
            "game_pk": 42,
            "bet_cents": 5000,
            "profit_loss_cents": 7500,
            "n_contracts": 125,
            "balance_refresh_status": "pending",
        }
    ]
    cursor = _SingleCursor(rows=rows)
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(db, "get_connection", lambda: conn)

    df = db.get_pending_payout_user_orders("user@example.com")

    sql, params = cursor.calls[0]
    assert params == ("user@example.com", db.KALSHI_BALANCE_REFRESH_PENDING_LOOKBACK_DAYS)
    assert "kalshi_balance_refresh_jobs" in sql
    assert "LEFT JOIN kalshi_balance_refresh_jobs" in sql
    assert "j.status <> 'completed'" not in sql
    assert "uo.profit_loss_cents > 0" in sql
    assert "ub.balance_cents - before_balance.balance_cents" in sql
    assert len(df) == 1
    assert df.iloc[0]["bet_dollars"] == 50.0
    assert df.iloc[0]["profit_loss"] == 75.0


def test_complete_stale_kalshi_balance_refresh_jobs_only_closes_old_games(monkeypatch):
    cursor = _SingleCursor(rows=[])
    cursor.rowcount = 1
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(db, "get_connection", lambda: conn)

    count = db.complete_stale_kalshi_balance_refresh_jobs()

    sql, params = cursor.calls[0]
    assert count == 1
    assert params == (db.KALSHI_BALANCE_REFRESH_STALE_DAYS,)
    assert "g.game_date < CURRENT_DATE - (%s * INTERVAL '1 day')" in sql
    assert conn.committed


def test_reopen_uncredited_kalshi_balance_refresh_jobs_requeues_completed(monkeypatch):
    cursor = _SingleCursor(rows=[])
    cursor.rowcount = 2
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(db, "get_connection", lambda: conn)

    count = db.reopen_uncredited_kalshi_balance_refresh_jobs()

    sql, params = cursor.calls[0]
    assert count == 2
    assert params == (db.KALSHI_BALANCE_REFRESH_STALE_DAYS,)
    assert "SET status = 'pending'" in sql
    assert "Reopened: payout not yet reflected in balance sync" in sql
    assert conn.committed


def test_complete_credited_kalshi_balance_refresh_jobs_marks_synced_credit(monkeypatch):
    cursor = _SingleCursor(rows=[])
    cursor.rowcount = 2
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(db, "get_connection", lambda: conn)

    count = db.complete_credited_kalshi_balance_refresh_jobs()

    sql, params = cursor.calls[0]
    assert count == 2
    assert params is None
    assert "UPDATE kalshi_balance_refresh_jobs" in sql
    assert "ub.balance_cents - before_balance.balance_cents" in sql
    assert "Completed after Kalshi balance sync showed payout credited" in sql
    assert conn.committed


def test_enqueue_kalshi_balance_refresh_job_dedupes_with_upsert(monkeypatch):
    cursor = _SingleCursor(one=[123])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(db, "get_connection", lambda: conn)

    job_id = db.enqueue_kalshi_balance_refresh_job(
        "User@Example.com",
        99,
        delay_minutes=10,
    )

    sql, params = cursor.calls[0]
    assert job_id == 123
    assert "ON CONFLICT (email, game_pk) DO UPDATE" in sql
    assert "WHERE kalshi_balance_refresh_jobs.status <> 'completed'" in sql
    assert params[0] == "user@example.com"
    assert params[1] == 99
    assert conn.committed


def test_backfill_user_order_results_queues_only_winning_live_orders(monkeypatch):
    rows = [
        {
            "email": "winner@example.com",
            "game_pk": 1,
            "status": "filled",
            "dry_run": False,
            "profit_loss_cents": 750,
        },
        {
            "email": "loser@example.com",
            "game_pk": 2,
            "status": "filled",
            "dry_run": False,
            "profit_loss_cents": -500,
        },
        {
            "email": "paper@example.com",
            "game_pk": 3,
            "status": "filled",
            "dry_run": True,
            "profit_loss_cents": 900,
        },
    ]
    cursor = _SingleCursor(rows=rows)
    conn = _FakeConnection(cursor)
    queued = []
    monkeypatch.setattr(db, "get_connection", lambda: conn)
    monkeypatch.setattr(
        db,
        "enqueue_kalshi_balance_refresh_job",
        lambda email, game_pk: queued.append((email, game_pk)),
    )

    count = db.backfill_user_order_results()

    sql, params = cursor.calls[0]
    assert count == 3
    assert "RETURNING uo.email, uo.game_pk, uo.status, uo.dry_run, uo.profit_loss_cents" in sql
    assert "uo.status <> %s" in sql
    assert params == (db.SOLD_STOP_LOSS_STATUS,)
    assert queued == [("winner@example.com", 1)]


def test_backfill_user_order_results_skips_stop_loss_orders(monkeypatch):
    cursor = _SingleCursor(rows=[])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(db, "get_connection", lambda: conn)

    count = db.backfill_user_order_results()

    sql, params = cursor.calls[0]
    assert count == 0
    assert "uo.status <> %s" in sql
    assert params == (db.SOLD_STOP_LOSS_STATUS,)


def test_process_due_kalshi_balance_refresh_jobs_marks_success(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "reconcile_kalshi_balance_refresh_jobs",
        lambda: {"reopened": 0, "credited": 0, "stale": 0},
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "claim_due_kalshi_balance_refresh_jobs",
        lambda limit: [{"id": 7, "email": "user@example.com", "game_pk": 42}],
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_kalshi_account",
        lambda email: {
            "is_active": True,
            "key_id": "kid",
            "key_path": "key.pem",
            "kalshi_env": "demo",
            "private_key_pem": "pem",
        },
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_user_order",
        lambda email, game_pk: {"kalshi_ticker": "KTEST-YES"},
    )
    monkeypatch.setattr(run_pipeline, "_retry_session", lambda: object())
    monkeypatch.setattr(
        run_pipeline,
        "_fetch_market",
        lambda session, base_url, ticker: {
            "ticker": ticker,
            "status": "finalized",
            "close_time": "2026-04-27T19:50:00Z",
            "settlement_timer_seconds": 60,
        },
    )
    monkeypatch.setattr(
        run_pipeline,
        "fetch_balance_for_account",
        lambda **kwargs: calls.append(("fetch", kwargs)) or 1000,
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "is_user_order_payout_credited",
        lambda email, game_pk: calls.append(("credited", email, game_pk)) or True,
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "mark_kalshi_balance_refresh_job_succeeded",
        lambda job_id: calls.append(("success", job_id)),
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "mark_kalshi_balance_refresh_job_failed",
        lambda job_id, error: calls.append(("failed", job_id, error)),
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "reschedule_kalshi_balance_refresh_job",
        lambda job_id, run_after, reason="": calls.append(("reschedule", job_id)),
    )

    run_pipeline.process_due_kalshi_balance_refresh_jobs()

    assert calls[0][0] == "fetch"
    assert calls[0][1]["email"] == "user@example.com"
    assert calls[0][1]["allow_stale_fallback"] is False
    assert calls[1] == ("credited", "user@example.com", 42)
    assert calls[2] == ("success", 7)


def test_process_due_kalshi_balance_refresh_jobs_reschedules_when_not_credited(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "reconcile_kalshi_balance_refresh_jobs",
        lambda: {"reopened": 0, "credited": 0, "stale": 0},
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "claim_due_kalshi_balance_refresh_jobs",
        lambda limit: [{"id": 9, "email": "user@example.com", "game_pk": 44}],
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_kalshi_account",
        lambda email: {
            "is_active": True,
            "key_id": "kid",
            "key_path": "key.pem",
            "kalshi_env": "demo",
            "private_key_pem": "pem",
        },
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_user_order",
        lambda email, game_pk: {"kalshi_ticker": "KTEST-YES"},
    )
    monkeypatch.setattr(run_pipeline, "_retry_session", lambda: object())
    monkeypatch.setattr(
        run_pipeline,
        "_fetch_market",
        lambda session, base_url, ticker: {
            "ticker": ticker,
            "status": "finalized",
            "close_time": "2026-04-27T19:50:00Z",
            "settlement_timer_seconds": 60,
        },
    )
    monkeypatch.setattr(
        run_pipeline,
        "fetch_balance_for_account",
        lambda **kwargs: calls.append(("fetch", kwargs)) or 1000,
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "is_user_order_payout_credited",
        lambda email, game_pk: calls.append(("credited", email, game_pk)) or False,
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "mark_kalshi_balance_refresh_job_succeeded",
        lambda job_id: calls.append(("success", job_id)),
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "reschedule_kalshi_balance_refresh_job",
        lambda job_id, run_after, reason="": calls.append(("reschedule", job_id, reason)),
    )

    run_pipeline.process_due_kalshi_balance_refresh_jobs()

    assert ("fetch",) == calls[0][:1]
    assert calls[1] == ("credited", "user@example.com", 44)
    assert calls[2][0] == "reschedule"
    assert calls[2][1] == 9
    assert "payout not credited yet" in calls[2][2]


def test_process_due_kalshi_balance_refresh_jobs_marks_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "reconcile_kalshi_balance_refresh_jobs",
        lambda: {"reopened": 0, "credited": 0, "stale": 0},
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "claim_due_kalshi_balance_refresh_jobs",
        lambda limit: [{"id": 8, "email": "user@example.com", "game_pk": 43}],
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_kalshi_account",
        lambda email: {
            "is_active": True,
            "key_id": "kid",
            "key_path": "key.pem",
            "kalshi_env": "demo",
        },
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_user_order",
        lambda email, game_pk: {"kalshi_ticker": "KTEST-YES"},
    )
    monkeypatch.setattr(run_pipeline, "_retry_session", lambda: object())
    monkeypatch.setattr(
        run_pipeline,
        "_fetch_market",
        lambda session, base_url, ticker: {
            "ticker": ticker,
            "status": "finalized",
            "close_time": "2026-04-27T19:50:00Z",
            "settlement_timer_seconds": 60,
        },
    )

    def fail_fetch(**kwargs):
        raise RuntimeError("temporary Kalshi outage")

    monkeypatch.setattr(run_pipeline, "fetch_balance_for_account", fail_fetch)
    monkeypatch.setattr(
        run_pipeline.DB,
        "mark_kalshi_balance_refresh_job_succeeded",
        lambda job_id: calls.append(("success", job_id)),
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "mark_kalshi_balance_refresh_job_failed",
        lambda job_id, error: calls.append(("failed", job_id, error)),
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "reschedule_kalshi_balance_refresh_job",
        lambda job_id, run_after, reason="": calls.append(("reschedule", job_id)),
    )

    run_pipeline.process_due_kalshi_balance_refresh_jobs()

    assert calls == [("failed", 8, "temporary Kalshi outage")]


def test_balance_refresh_job_reschedules_when_market_active(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "reconcile_kalshi_balance_refresh_jobs",
        lambda: {"reopened": 0, "credited": 0, "stale": 0},
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "claim_due_kalshi_balance_refresh_jobs",
        lambda limit: [{"id": 9, "email": "user@example.com", "game_pk": 44}],
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_kalshi_account",
        lambda email: {"is_active": True, "kalshi_env": "prod"},
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_user_order",
        lambda email, game_pk: {"kalshi_ticker": "KTEST-YES"},
    )
    monkeypatch.setattr(run_pipeline, "_retry_session", lambda: object())
    monkeypatch.setattr(
        run_pipeline,
        "_fetch_market",
        lambda session, base_url, ticker: {
            "ticker": ticker,
            "status": "active",
            "close_time": "2026-04-27T19:54:00Z",
            "settlement_timer_seconds": 180,
        },
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "reschedule_kalshi_balance_refresh_job",
        lambda job_id, run_after, reason="": calls.append(("reschedule", job_id, run_after, reason)),
    )
    monkeypatch.setattr(
        run_pipeline,
        "fetch_balance_for_account",
        lambda **kwargs: calls.append(("fetch", kwargs)),
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "mark_kalshi_balance_refresh_job_succeeded",
        lambda job_id: calls.append(("success", job_id)),
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "mark_kalshi_balance_refresh_job_failed",
        lambda job_id, error: calls.append(("failed", job_id, error)),
    )

    run_pipeline.process_due_kalshi_balance_refresh_jobs()

    assert calls[0][0] == "reschedule"
    assert calls[0][1] == 9
    assert calls[0][2] == datetime(2026, 4, 27, 19, 58, tzinfo=timezone.utc)
    assert not any(call[0] in {"fetch", "success", "failed"} for call in calls)


def test_balance_refresh_job_reschedules_until_settlement_timer_elapsed(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "reconcile_kalshi_balance_refresh_jobs",
        lambda: {"reopened": 0, "credited": 0, "stale": 0},
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "claim_due_kalshi_balance_refresh_jobs",
        lambda limit: [{"id": 10, "email": "user@example.com", "game_pk": 45}],
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_kalshi_account",
        lambda email: {"is_active": True, "kalshi_env": "prod"},
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_user_order",
        lambda email, game_pk: {"kalshi_ticker": "KTEST-YES"},
    )
    monkeypatch.setattr(run_pipeline, "_retry_session", lambda: object())
    monkeypatch.setattr(
        run_pipeline,
        "_fetch_market",
        lambda session, base_url, ticker: {
            "ticker": ticker,
            "status": "finalized",
            "close_time": "2026-04-27T19:54:00Z",
            "settlement_timer_seconds": 180,
        },
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "reschedule_kalshi_balance_refresh_job",
        lambda job_id, run_after, reason="": calls.append(("reschedule", job_id, run_after)),
    )
    monkeypatch.setattr(run_pipeline, "fetch_balance_for_account", lambda **kwargs: calls.append(("fetch", kwargs)))
    monkeypatch.setattr(run_pipeline.DB, "mark_kalshi_balance_refresh_job_succeeded", lambda job_id: calls.append(("success", job_id)))
    monkeypatch.setattr(run_pipeline.DB, "mark_kalshi_balance_refresh_job_failed", lambda job_id, error: calls.append(("failed", job_id, error)))

    run_pipeline.process_due_kalshi_balance_refresh_jobs()

    assert calls == [("reschedule", 10, datetime(2026, 4, 27, 19, 58, tzinfo=timezone.utc))]


def test_repair_requeues_completed_job_before_kalshi_payout_cutoff(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_recent_completed_kalshi_balance_refresh_jobs_for_winners",
        lambda lookback_hours: [
            {
                "id": 11,
                "email": "user@example.com",
                "game_pk": 46,
                "kalshi_ticker": "KTEST-YES",
                "completed_at": datetime(2026, 4, 27, 19, 55, tzinfo=timezone.utc),
            }
        ],
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "get_kalshi_account",
        lambda email: {"is_active": True, "kalshi_env": "prod"},
    )
    monkeypatch.setattr(run_pipeline, "_retry_session", lambda: object())
    monkeypatch.setattr(
        run_pipeline,
        "_fetch_market",
        lambda session, base_url, ticker: {
            "ticker": ticker,
            "status": "finalized",
            "close_time": "2026-04-27T19:54:00Z",
            "settlement_timer_seconds": 180,
        },
    )
    monkeypatch.setattr(
        run_pipeline.DB,
        "reschedule_kalshi_balance_refresh_job",
        lambda job_id, run_after, reason="": calls.append((job_id, run_after, reason)),
    )

    run_pipeline.repair_completed_kalshi_balance_refresh_jobs()

    assert calls == [
        (
            11,
            datetime(2026, 4, 27, 19, 55, tzinfo=timezone.utc),
            "Repair: completed before Kalshi payout window for KTEST-YES",
        )
    ]


def test_model_training_fingerprint_tracks_feature_inputs(monkeypatch):
    from models.model_v1 import predict

    monkeypatch.setattr(predict, "_git_commit", lambda: "abc123")
    df = pd.DataFrame(
        [
            {
                "game_pk": 2,
                "game_date": "2026-04-02",
                "home_win": False,
                "market_implied_prob": 0.45,
                "home_games_played": 20,
                "away_games_played": 22,
                "feature_a": 1.0,
            },
            {
                "game_pk": 1,
                "game_date": "2026-04-01",
                "home_win": True,
                "market_implied_prob": 0.55,
                "home_games_played": 30,
                "away_games_played": 31,
                "feature_a": 2.0,
            },
        ]
    )

    fp1, meta1 = predict._training_fingerprint(df, ["feature_a"], ["market_implied_prob"])
    fp2, _ = predict._training_fingerprint(df.iloc[::-1], ["feature_a"], ["market_implied_prob"])
    changed = df.copy()
    changed.loc[changed["game_pk"] == 1, "feature_a"] = 2.5
    fp3, _ = predict._training_fingerprint(changed, ["feature_a"], ["market_implied_prob"])

    assert fp1 == fp2
    assert fp1 != fp3
    assert meta1["settled_row_count"] == 2
    assert meta1["max_settled_game_date"] == "2026-04-02"


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
    assert inserted[0][15] == 50000
    assert inserted[0][16] == 40909
    assert inserted[0][17] == 1000000
    assert inserted[0][18] == 1040909
    assert inserted[1][15] == 52045
    assert inserted[1][17] == 1040909


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
    monkeypatch.setattr(place_bet.DB, "get_active_live_model_version", lambda: "v1")
    monkeypatch.setattr(place_bet.DB, "get_bet_for_live_pipeline", lambda game_pk, pv: row if str(game_pk) == "123" and pv == "v1" else None)
    monkeypatch.setattr(place_bet.DB, "get_user_order", lambda email, game_pk: None)
    monkeypatch.setattr(place_bet.DB, "get_paper_bankroll_dollars", lambda email, version="v1": 10000.0)
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
            {"game_pk": 6, "result": None, "bet_dollars": 12.0, "status": "sold_stop_loss"},
        ]
    )

    open_rows = df[app_module._is_open_position_frame(df)]

    assert open_rows["game_pk"].tolist() == [1, 5]


def test_dashboard_open_positions_exclude_voided_status(app_module):
    df = pd.DataFrame(
        [
            {"game_pk": 1, "result": None, "bet_dollars": 12.0, "status": "filled"},
            {"game_pk": 2, "result": None, "bet_dollars": 12.0, "status": "voided"},
        ]
    )

    open_rows = df[app_module._is_open_position_frame(df)]

    assert open_rows["game_pk"].tolist() == [1]


def test_void_orders_for_inactive_games_closes_open_live_orders(monkeypatch):
    executed = []

    class FakeCursor:
        def __init__(self):
            self.rowcount = 0

        def execute(self, sql, params=None):
            executed.append((sql.strip(), params))
            if "UPDATE user_orders" in sql:
                self.rowcount = 1
            elif "UPDATE paper_orders po" in sql:
                self.rowcount = 0
            elif "UPDATE paper_orders_v2" in sql:
                self.rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            executed.append(("commit", None))

        def close(self):
            pass

    monkeypatch.setattr(db, "get_connection", lambda: FakeConn())
    monkeypatch.setattr(db, "recompute_paper_order_v2_financials", lambda: 0)

    counts = db.void_orders_for_inactive_games([777001])

    assert counts["user_orders"] == 1
    assert any("postponed" in sql for sql, _ in executed if isinstance(sql, str))
    user_update = next(item for item in executed if "UPDATE user_orders" in item[0])
    assert user_update[1][0] == db.VOIDED_ORDER_STATUS
    assert user_update[1][1] == [777001]


def test_apply_settlements_voids_orders_for_postponed_games(monkeypatch):
    calls = []

    monkeypatch.setattr(db, "void_orders_for_inactive_games", lambda pks: calls.append(pks) or {})
    monkeypatch.setattr(db, "backfill_bet_results", lambda: calls.append("backfill"))
    monkeypatch.setattr(db, "execute_values", lambda *args, **kwargs: None)

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(db, "pooled_connection", lambda: FakeConn())

    db.apply_settlements([], [12345])

    assert calls == [[12345]]


def test_run_fetch_data_builds_scoped_argv(monkeypatch):
    calls = []

    def fake_fetch_data_main():
        calls.append(run_pipeline.sys.argv[:])

    monkeypatch.setattr(run_pipeline, "fetch_data_main", fake_fetch_data_main)

    run_pipeline.run_fetch_data(skip_odds=True, odds_game_pks=["1", "2"])

    assert calls == [["fetch/fetch_data.py", "--skip-odds", "--odds-game-pks", "1,2"]]


def test_run_fetch_data_builds_odds_only_argv(monkeypatch):
    calls = []

    def fake_fetch_data_main():
        calls.append(run_pipeline.sys.argv[:])

    monkeypatch.setattr(run_pipeline, "fetch_data_main", fake_fetch_data_main)

    run_pipeline.run_fetch_data(odds_only=True, odds_game_pks=["1"])

    assert calls == [["fetch/fetch_data.py", "--odds-only", "--odds-game-pks", "1"]]


def test_run_fetch_data_uses_in_memory_odds_games(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "fetch_data_main", lambda: calls.append("main"))
    monkeypatch.setattr(
        run_pipeline,
        "refresh_odds_only",
        lambda df: calls.append(df["game_pk"].tolist()),
    )

    run_pipeline.run_fetch_data(odds_only=True, odds_games=[{"game_pk": 1}])

    assert calls == [[1]]
