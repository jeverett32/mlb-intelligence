import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import model.train as T
from fetch.fetch_data import _parse_fg_wrc_plus


def test_rookie_diff_is_engineered_from_starter_history():
    df = pd.DataFrame(
        {
            "game_id": range(1, 8),
            "game_date": pd.date_range("2024-04-01", periods=7, freq="D"),
            "home_starter_id": [10] * 7,
            "away_starter_id": range(100, 107),
        }
    )

    features = T.build_pitcher_features(df)

    assert "rookie_DIFF" in features.columns
    assert features.loc[6, "home_is_rookie"] == 0
    assert features.loc[6, "away_is_rookie"] == 1
    assert features.loc[6, "rookie_DIFF"] == 1


def test_training_feature_columns_do_not_enable_missing_starter_fip_features():
    assert "sp_fip_DIFF" not in T.FEATURE_COLUMNS
    assert "sharp_x_fip" not in T.FEATURE_COLUMNS


def test_war_diff_uses_lagged_fangraphs_war(monkeypatch):
    sample = pd.DataFrame(
        {
            "game_id": [1, 2],
            "game_pk": [1, 2],
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
            "game_time_utc": pd.to_datetime(["2024-04-01T23:00:00Z", "2024-04-02T23:00:00Z"]),
            "season": [2024, 2024],
            "home_team": ["NYY", "NYY"],
            "away_team": ["BOS", "BOS"],
            "home_score": [5, 4],
            "away_score": [3, 6],
            "home_win": [1.0, 0.0],
            "open_home_ml": [-120, -125],
            "open_away_ml": [110, 115],
            "close_home_ml": [-130, -135],
            "close_away_ml": [120, 125],
            "open_total": [8.5, 8.5],
            "close_total": [8.0, 8.0],
            "is_night_game": [1.0, 1.0],
            "wind_dir_deg": [90.0, 90.0],
            "wind_speed_kmh": [10.0, 10.0],
            "temp_c": [20.0, 20.0],
            "home_starter_id": [11, 12],
            "away_starter_id": [21, 22],
            "home_pitcher_is_lefty": [0.0, 0.0],
            "away_pitcher_is_lefty": [1.0, 1.0],
            "home_sp_era": [3.5, 3.6],
            "away_sp_era": [4.5, 4.6],
            "home_sp_whip": [1.1, 1.2],
            "away_sp_whip": [1.3, 1.4],
            "home_sp_k9": [8.0, 8.1],
            "away_sp_k9": [7.0, 7.1],
            "home_sp_bb9": [2.0, 2.1],
            "away_sp_bb9": [3.0, 3.1],
            "home_war": [1.25, 1.50],
            "away_war": [2.50, 2.75],
            "home_wrc_plus": [100.0, 101.0],
            "away_wrc_plus": [95.0, 96.0],
        }
    )
    monkeypatch.setattr(T, "load_training_frame", lambda: sample.copy())

    features = T.load_and_engineer_features()
    row = features.loc[features["game_id"] == 2].iloc[0]

    assert row["war_DIFF"] == pytest.approx(-1.25)


def test_fangraphs_wrc_plus_parser_rejects_rate_scale_values():
    assert _parse_fg_wrc_plus(112) == 112
    assert pd.isna(_parse_fg_wrc_plus(0.392))
    assert pd.isna(_parse_fg_wrc_plus(400))
