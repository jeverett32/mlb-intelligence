import pytest
import pandas as pd


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
    assert "MLB Intelligence" in r.text
    assert "Public MLB model results, updated daily." in r.text
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
    assert "MLB Intelligence" in r.text
    assert "MLB Betting Dashboard" not in r.text
    assert 'data-page="teams"' in r.text
    assert 'id="page-teams"' in r.text


def test_team_overview_computes_division_standings_and_streaks(app_module):
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-04-01",
                "game_time_utc": "2026-04-01T17:00:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "home_score": 5,
                "away_score": 3,
                "home_win": True,
            },
            {
                "game_pk": 2,
                "game_date": "2026-04-02",
                "game_time_utc": "2026-04-02T17:00:00Z",
                "home_team": "TOR",
                "away_team": "NYY",
                "home_score": 2,
                "away_score": 4,
                "home_win": False,
            },
            {
                "game_pk": 3,
                "game_date": "2026-04-03",
                "game_time_utc": "2026-04-03T17:00:00Z",
                "home_team": "BOS",
                "away_team": "TOR",
                "home_score": None,
                "away_score": None,
                "home_win": None,
            },
        ]
    )

    overview = app_module._build_team_overview(games, season=2026)
    al_east = next(d for d in overview["divisions"] if d["name"] == "AL East")
    yankees = next(t for t in al_east["teams"] if t["abbr"] == "NYY")
    blue_jays = next(t for t in al_east["teams"] if t["abbr"] == "TOR")

    assert overview["summary"]["completed_games"] == 2
    assert yankees["division_rank"] == 1
    assert yankees["wins"] == 2
    assert yankees["streak"] == "W2"
    assert yankees["run_diff"] == 4
    assert blue_jays["games_back"] == 1.5
    assert blue_jays["next_game"]["opponent"] == "BOS"


def test_teams_api_requires_auth(client):
    r = client.get("/api/teams")
    assert r.status_code in (401, 403)


def test_teams_api_returns_overview_for_approved_user(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    monkeypatch.setattr(
        app_module.DB,
        "get_games_df",
        lambda season=2026, upcoming_only=False: pd.DataFrame(
            [
                {
                    "game_pk": 1,
                    "game_date": "2026-04-01",
                    "game_time_utc": "2026-04-01T17:00:00Z",
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "home_score": 5,
                    "away_score": 3,
                    "home_win": True,
                }
            ]
        ),
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/api/teams")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"season", "last_updated_utc", "summary", "divisions", "teams"}
    assert body["summary"]["teams"] == 30
    assert body["summary"]["completed_games"] == 1


def test_paper_bankroll_endpoint_includes_history(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    monkeypatch.setattr(app_module.DB, "backfill_paper_orders_from_bets", lambda email: 0)
    monkeypatch.setattr(app_module.DB, "get_paper_bankroll_dollars", lambda email: 10409.09)
    monkeypatch.setattr(
        app_module.DB,
        "get_paper_bankroll_history",
        lambda email: [
            {"recorded_at": None, "balance_dollars": 10000.0, "source": "paper"},
            {"recorded_at": "2026-04-01", "balance_dollars": 10409.09, "source": "paper"},
        ],
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/api/paper-bankroll")
    assert r.status_code == 200
    body = r.json()
    assert body["current_dollars"] == 10409.09
    assert body["history"][-1]["balance_dollars"] == 10409.09


def test_upcoming_keeps_started_same_day_games_until_settled(monkeypatch, app_module, client):
    now_utc = pd.Timestamp.now(tz="UTC")
    today_started = now_utc - pd.Timedelta(hours=2)
    tomorrow_game = now_utc + pd.Timedelta(hours=6)

    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    monkeypatch.setattr(
        app_module.DB,
        "get_user_setting",
        lambda email, key, default="": "UTC" if key == "dashboard_timezone" else default,
    )
    monkeypatch.setattr(
        app_module.DB,
        "get_games_df",
        lambda season=2026, upcoming_only=True: pd.DataFrame(
            [
                {
                    "game_pk": "started-today",
                    "game_date": today_started.date().isoformat(),
                    "game_time_utc": today_started.isoformat(),
                    "home_team": "COL",
                    "away_team": "LAD",
                    "home_implied_prob": 0.46,
                    "away_implied_prob": 0.54,
                    "close_home_ml": 118,
                    "close_away_ml": -128,
                    "home_win": None,
                },
                {
                    "game_pk": "tomorrow",
                    "game_date": tomorrow_game.date().isoformat(),
                    "game_time_utc": tomorrow_game.isoformat(),
                    "home_team": "SEA",
                    "away_team": "TEX",
                    "home_implied_prob": 0.51,
                    "away_implied_prob": 0.49,
                    "close_home_ml": -104,
                    "close_away_ml": 102,
                    "home_win": None,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        app_module.DB,
        "get_all_bets",
        lambda: pd.DataFrame(
            [
                {
                    "game_pk": "started-today",
                    "predicted_prob": 0.49,
                    "edge": 0.03,
                    "bet_side": "none",
                    "bet_frac": 0.0,
                    "market_implied_prob": 0.46,
                    "bet_dollars": None,
                    "result": None,
                    "kalshi_order_id": None,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        app_module.DB,
        "get_user_orders",
        lambda email: pd.DataFrame(),
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/api/upcoming")
    assert r.status_code == 200
    body = r.json()
    game_pks = {g["game_pk"] for g in body["games"]}
    assert "started-today" in game_pks
    started = next(g for g in body["games"] if g["game_pk"] == "started-today")
    assert started["predicted_prob"] == 0.49
    assert started["bet_side"] == "none"


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
    assert "MLB Intelligence" in r.text
    assert "MLB Betting Dashboard" not in r.text
