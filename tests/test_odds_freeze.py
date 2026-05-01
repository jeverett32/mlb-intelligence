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
