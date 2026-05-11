from __future__ import annotations

import pandas as pd

from models.model_v2.sandbox.model_lab import features
from models.model_v2.sandbox.model_lab import planned_features


def test_sandbox_contract_excludes_2026_rows():
    df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "game_date": ["2025-09-28", "2026-03-27"],
            "home_win": [1, 0],
        }
    )

    out = features.apply_sandbox_contract(df)

    assert list(out["game_id"]) == [1]
    assert out["game_date"].max() < pd.Timestamp("2026-01-01")


def test_sandbox_contract_requires_game_date():
    df = pd.DataFrame({"game_id": [1], "home_win": [1]})

    try:
        features.apply_sandbox_contract(df)
    except ValueError as exc:
        assert "game_date" in str(exc)
    else:
        raise AssertionError("expected missing game_date to fail")


def test_planned_feature_schema_adds_all_catalog_columns():
    df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "game_pk": [1, 2],
            "season": [2025, 2025],
            "game_date": pd.to_datetime(["2025-04-01", "2025-04-02"]),
            "home_team": ["MIL", "CHC"],
            "away_team": ["CHC", "MIL"],
            "temp_c": [20.0, 10.0],
            "wind_speed_kmh": [12.0, 8.0],
            "wind_dir_deg": [180.0, 90.0],
            "park_factor": [1.02, 0.98],
        }
    )

    out = planned_features.add_planned_features(df)

    missing = [c for c in planned_features.PLANNED_FEATURE_COLUMNS if c not in out.columns]
    assert missing == []
    assert out["roof_possible_flag"].isna().all()
    assert out["temp_c_game_time"].isna().all()
    assert out["travel_miles"].notna().all()
