from datetime import timedelta

import pandas as pd

import fetch.fetch_data as F


def test_split_odds_refresh_games_freezes_started_and_predicted(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    games = pd.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "game_time_utc": [
                (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M"),
                (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                (now + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
            ],
            "is_completed": [False, False, False],
        }
    )
    monkeypatch.setattr(F, "_predicted_game_pks", lambda game_pks: {2})

    refresh, frozen = F._split_odds_refresh_games(games)

    assert frozen == {1, 2}
    assert refresh["game_pk"].tolist() == [3]


def test_preserve_frozen_odds_keeps_existing_values():
    new_rows = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "close_home_ml": [-140, -120],
            "close_away_ml": [120, 100],
            "odds_source": ["odds_api", "sbr"],
        }
    )
    existing = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "close_home_ml": [-110, -125],
            "close_away_ml": [-110, 105],
            "odds_source": ["sbr", "sbr"],
        }
    )

    out = F._preserve_frozen_odds(new_rows, existing, {1})

    frozen = out[out["game_pk"] == 1].iloc[0]
    refreshable = out[out["game_pk"] == 2].iloc[0]
    assert frozen["close_home_ml"] == -110
    assert frozen["close_away_ml"] == -110
    assert frozen["odds_source"] == "sbr"
    assert refreshable["close_home_ml"] == -120
    assert refreshable["odds_source"] == "sbr"


def test_fetch_odds_scopes_sbr_rows_to_requested_games(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(F, "SEASON", 2026)

    async def fake_scrape_range_async(*args, **kwargs):
        return []

    monkeypatch.setattr("fetch.scraper.scrape_range_async", fake_scrape_range_async)
    monkeypatch.setattr(
        F,
        "_parse_scraped_odds",
        lambda raw: pd.DataFrame(
            [
                {
                    "game_date": "2026-04-01",
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "open_home_ml": -120,
                    "open_away_ml": 110,
                    "close_home_ml": -130,
                    "close_away_ml": 120,
                    "open_total": 8.5,
                    "close_total": 8.0,
                    "odds_source": "pinnacle",
                },
                {
                    "game_date": "2026-04-01",
                    "home_team": "LAD",
                    "away_team": "SFG",
                    "open_home_ml": -150,
                    "open_away_ml": 140,
                    "close_home_ml": -160,
                    "close_away_ml": 150,
                    "open_total": 7.5,
                    "close_total": 7.0,
                    "odds_source": "pinnacle",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "fetch.odds_api.fetch_odds_api",
        lambda: (_ for _ in ()).throw(AssertionError("Odds API should not run")),
    )

    schedule = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2026-04-01")],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
        }
    )

    out = F.fetch_odds(schedule)

    assert out[["home_team", "away_team"]].values.tolist() == [["NYY", "BOS"]]


def test_build_odds_update_rows_is_minimal():
    schedule = pd.DataFrame(
        {
            "game_pk": [1],
            "game_date": [pd.Timestamp("2026-04-01")],
            "season": [2026],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
            "temp_c": [20.0],
        }
    )
    odds = pd.DataFrame(
        {
            "game_date": ["2026-04-01"],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
            "open_home_ml": [-120],
            "open_away_ml": [110],
            "close_home_ml": [-130],
            "close_away_ml": [120],
            "close_total": [8.0],
            "odds_source": ["pinnacle"],
        }
    )

    rows = F.build_odds_update_rows(schedule, odds)

    assert rows.iloc[0]["game_pk"] == 1
    assert rows.iloc[0]["over_under"] == 8.0
    assert "temp_c" not in rows.columns
    assert round(rows.iloc[0]["home_implied_prob"] + rows.iloc[0]["away_implied_prob"], 12) == 1
