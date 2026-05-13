"""Backfill overall_log_loss from monthly_accuracy in DB snapshot reads."""

from __future__ import annotations

import db


def test_coalesce_overall_log_loss_from_monthly_weighted():
    row = {
        "overall_log_loss": None,
        "monthly_accuracy": [
            {"year": 2024, "month": 4, "count": 100, "log_loss": 0.65},
            {"year": 2024, "month": 5, "count": 50, "log_loss": 0.70},
        ],
    }
    db._coalesce_overall_log_loss_from_monthly(row)
    assert row["overall_log_loss"] == round((0.65 * 100 + 0.70 * 50) / 150, 6)


def test_coalesce_skips_when_overall_present():
    row = {"overall_log_loss": 0.55, "monthly_accuracy": []}
    db._coalesce_overall_log_loss_from_monthly(row)
    assert row["overall_log_loss"] == 0.55


def test_coalesce_parses_json_string_monthly():
    import json

    row = {
        "overall_log_loss": None,
        "monthly_accuracy": json.dumps(
            [{"year": 2024, "month": 4, "count": 10, "log_loss": 0.66}]
        ),
    }
    db._coalesce_overall_log_loss_from_monthly(row)
    assert row["overall_log_loss"] == 0.66
