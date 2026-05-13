"""V2 market odds resolution (games_v2 row + prod games fallback)."""

import pandas as pd
import pytest

from models.model_v2.predict import (
    PredictV2Error,
    _resolve_market_inputs_v2,
)


def test_resolve_from_frame_moneylines() -> None:
    row = pd.DataFrame([{"close_home_ml": -110.0, "close_away_ml": 100.0}])
    h_dec, a_dec, m = _resolve_market_inputs_v2("1", row)
    assert 0.0 < m < 1.0
    assert h_dec > 1.0 and a_dec > 1.0


def test_resolve_fallback_to_games_table(monkeypatch: pytest.MonkeyPatch) -> None:
    row = pd.DataFrame([{}])

    def _fb(_pk: int) -> dict:
        return {
            "close_home_ml": None,
            "close_away_ml": None,
            "home_implied_prob": 0.52,
        }

    monkeypatch.setattr(
        "models.model_v2.predict.DB.get_game_market_odds_for_v2_fallback",
        _fb,
    )
    h_dec, a_dec, m = _resolve_market_inputs_v2("999", row)
    assert abs(m - 0.52) < 1e-9


def test_resolve_invalid_home_implied_then_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    row = pd.DataFrame([{}])

    def _fb(_pk: int) -> dict:
        return {
            "close_home_ml": None,
            "close_away_ml": None,
            "home_implied_prob": 1.0,
        }

    monkeypatch.setattr(
        "models.model_v2.predict.DB.get_game_market_odds_for_v2_fallback",
        _fb,
    )
    with pytest.raises(PredictV2Error, match="Market odds missing"):
        _resolve_market_inputs_v2("1000", row)
