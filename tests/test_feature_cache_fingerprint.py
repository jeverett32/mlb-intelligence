import pandas as pd

from model.predict import _engineered_history_fingerprint


def test_engineered_history_fingerprint_changes_for_mid_table_value_change():
    df = pd.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "game_date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
            "home_win": [1.0, None, 0.0],
            "market_implied_prob": [0.51, 0.52, 0.53],
        }
    )
    changed = df.copy()
    changed.loc[1, "market_implied_prob"] = 0.57

    assert _engineered_history_fingerprint(df) != _engineered_history_fingerprint(changed)


def test_engineered_history_fingerprint_ignores_row_order_when_game_pk_exists():
    df = pd.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "game_date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
            "home_win": [1.0, None, 0.0],
        }
    )

    assert _engineered_history_fingerprint(df) == _engineered_history_fingerprint(
        df.iloc[[2, 0, 1]].reset_index(drop=True)
    )


def test_engineered_history_fingerprint_changes_for_dtype_change():
    numeric = pd.DataFrame({"game_pk": [1, 2], "home_win": [1.0, 0.0]})
    object_typed = numeric.astype({"home_win": "object"})

    assert _engineered_history_fingerprint(numeric) != _engineered_history_fingerprint(object_typed)
