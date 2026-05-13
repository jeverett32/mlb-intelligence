"""Tests for dynamic walk-forward folds in models.model_v2.eval."""

from __future__ import annotations

import pandas as pd

from models.model_v2.eval import build_eval_folds
from models.model_v2.sandbox.model_lab.feature_engineer import FOLDS


def test_build_eval_folds_static_only_through_2025():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2025-04-01", "2025-08-20"]),
            "home_win": [1, 0],
        }
    )
    folds = build_eval_folds(df)
    assert folds == [tuple(f) for f in FOLDS]


def test_build_eval_folds_appends_ytd_2026():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2025-06-01", "2026-05-10"]),
            "home_win": [1, 0],
        }
    )
    folds = build_eval_folds(df)
    assert len(folds) == len(FOLDS) + 1
    assert folds[-1] == ("2026-01-01", "2026-05-11")


def test_build_eval_folds_empty():
    df = pd.DataFrame({"game_date": [], "home_win": []})
    assert build_eval_folds(df) == [tuple(f) for f in FOLDS]
