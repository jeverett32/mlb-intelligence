import pytest

import db


def test_browse_table_order_by_covers_all_browsable_tables():
    assert set(db.BROWSE_TABLE_ORDER_BY) == db.BROWSABLE_TABLES


@pytest.mark.parametrize(
    ("table", "expected_prefix"),
    [
        ("games", "game_date DESC"),
        ("bets", "game_date DESC"),
        ("settings", "updated_at DESC"),
        ("app_users", "created_at DESC"),
        ("app_sessions", "created_at DESC"),
        ("user_settings", "updated_at DESC"),
        ("kalshi_accounts", "updated_at DESC"),
        ("user_balance", "recorded_at DESC"),
        ("user_orders", "created_at DESC"),
        ("paper_orders", "created_at DESC"),
        ("model_metric_snapshots", "trained_at DESC"),
        ("model_artifacts", "created_at DESC"),
        ("admin_notes", "updated_at DESC"),
    ],
)
def test_browse_table_order_by_uses_recent_columns(table, expected_prefix):
    assert db._browse_table_order_by(table).startswith(expected_prefix)


def test_browse_table_order_by_rejects_unknown_table():
    with pytest.raises(ValueError):
        db._browse_table_order_by("not_allowed")
