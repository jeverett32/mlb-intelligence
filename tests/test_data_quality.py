"""Unit tests for data_quality contracts, validators, and audit logic."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data_quality.contracts import (
    CONTRACTS,
    constraint_eligible,
    contract_names,
    get_contract,
)
from data_quality.validators import coerce, coerce_series, violation_mask


def test_contracts_have_unique_names():
    names = contract_names()
    assert len(names) == len(set(names)), "duplicate contract names"


def test_contracts_have_valid_ranges():
    for c in CONTRACTS:
        if c.min_value is not None and c.max_value is not None:
            assert c.min_value < c.max_value, f"{c.name}: min >= max"


def test_critical_columns_have_contracts():
    for must_have in [
        "game_pk", "season", "home_score", "away_score",
        "close_home_ml", "close_away_ml",
        "home_implied_prob", "away_implied_prob",
        "home_wrc_plus", "away_wrc_plus",
    ]:
        assert get_contract(must_have) is not None, f"missing contract for {must_have}"


def test_coerce_in_range_returns_value():
    assert coerce("home_wrc_plus", 105) == 105.0


def test_coerce_below_min_returns_nan():
    assert math.isnan(coerce("home_wrc_plus", 5))


def test_coerce_above_max_returns_nan():
    assert math.isnan(coerce("home_wrc_plus", 999))


def test_coerce_non_numeric_returns_nan():
    assert math.isnan(coerce("home_wrc_plus", "not-a-number"))
    assert math.isnan(coerce("home_wrc_plus", None))


def test_coerce_unknown_column_passes_through():
    assert coerce("nonexistent_col", 7.5) == 7.5
    assert math.isnan(coerce("nonexistent_col", "junk"))


def test_coerce_probability_bounds():
    assert coerce("home_implied_prob", 0.55) == 0.55
    assert math.isnan(coerce("home_implied_prob", 1.5))
    assert math.isnan(coerce("home_implied_prob", -0.1))


def test_coerce_series_clamps_out_of_range_to_nan():
    s = pd.Series([100, 5, 250, "bad", None, 400])
    result = coerce_series("home_wrc_plus", s)
    assert result.iloc[0] == 100.0
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == 250.0
    assert math.isnan(result.iloc[3])
    assert math.isnan(result.iloc[4])
    assert math.isnan(result.iloc[5])


def test_violation_mask_only_flags_non_null_out_of_range():
    s = pd.Series([100, 5, np.nan, 250, 400])
    mask = violation_mask("home_wrc_plus", s)
    assert mask.tolist() == [False, True, False, False, True]


def test_violation_mask_unknown_column_returns_all_false():
    s = pd.Series([1, 2, 3])
    assert not violation_mask("nonexistent_col", s).any()


def test_constraint_eligible_excludes_unbounded_and_unsafe():
    eligible = list(constraint_eligible())
    names = {c.name for c in eligible}
    # game_pk is constraint_safe=False -> excluded
    assert "game_pk" not in names
    # home_wrc_plus is bounded + safe -> included
    assert "home_wrc_plus" in names


def test_audit_report_classifies_critical_columns(monkeypatch):
    """build_report should bucket failures correctly without touching DB."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "_dq_audit",
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_data_quality.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = pd.DataFrame({
        "game_pk": [1, 2, 3],
        "season": [2025, 2025, 2025],
        "home_team": ["LAD", "NYY", "BOS"],
        "away_team": ["SF", "BAL", "TB"],
        # critical: out-of-range close odds
        "close_home_ml": [-150, 99999, -120],
        "close_away_ml": [130, -110, 105],
        "home_implied_prob": [0.6, 0.5, 0.55],
        "away_implied_prob": [0.4, 0.5, 0.45],
        # warning-only: out-of-range wRC+
        "home_wrc_plus": [105, 5, 110],
        "away_wrc_plus": [95, 100, 102],
    })
    report = mod.build_report(df)
    assert "close_home_ml" in report["critical_columns"]
    assert "home_wrc_plus" in report["warning_columns"]
    assert report["critical_issue_count"] >= 1
    assert report["warning_issue_count"] >= 1


def test_migration_generator_emits_check_constraints(tmp_path, monkeypatch):
    """Confirm SQL has expected shape (regenerator stays consistent w/ contracts)."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_dq_migration",
        root / "scripts" / "generate_constraint_migration.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sql = mod.build()
    assert "BEGIN;" in sql
    assert "COMMIT;" in sql
    assert "games_home_wrc_plus_range_chk" in sql
    assert "NOT VALID" in sql
    # constraint_safe=False columns must not appear
    assert "games_game_pk_range_chk" not in sql
