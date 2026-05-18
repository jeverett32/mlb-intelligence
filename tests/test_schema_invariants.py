import os

import pytest

import db


def _requires_db_env() -> None:
    missing = [name for name in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD") if not os.environ.get(name)]
    if missing:
        pytest.skip("DB env missing: " + ", ".join(missing))


@pytest.fixture()
def live_conn():
    _requires_db_env()
    try:
        conn = db.get_connection()
    except Exception as exc:
        pytest.skip(f"live DB unavailable: {exc}")
    try:
        yield conn
    finally:
        conn.close()


def test_foreign_keys_have_supporting_indexes(live_conn):
    with live_conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.conrelid::regclass::text AS table_name, c.conname
            FROM pg_constraint c
            WHERE c.contype = 'f'
              AND c.connamespace = 'public'::regnamespace
              AND NOT EXISTS (
                  SELECT 1
                    FROM pg_index i
                    WHERE i.indrelid = c.conrelid
                      AND i.indisvalid
                    AND (string_to_array(i.indkey::text, ' ')::smallint[])[1:array_length(c.conkey, 1)] = c.conkey
                )
            ORDER BY table_name, c.conname
            """
        )
        assert cur.fetchall() == []


def test_core_schema_types_and_constraints(live_conn):
    expected_constraints = {
        "bets_model_artifact_id_fkey",
        "paper_orders_game_pk_fkey",
        "user_orders_game_pk_fkey",
        "user_order_snapshots_game_pk_fkey",
        "bets_bet_side_check",
        "bets_predicted_prob_check",
        "user_orders_status_check",
        "paper_orders_status_check",
        "pipeline_runs_status_check",
        "kalshi_market_snapshots_market_status_check",
        "kalshi_accounts_key_id_encrypted_check",
        "kalshi_accounts_key_path_encrypted_check",
        "kalshi_accounts_private_key_pem_encrypted_check",
        "bets_bet_cents_check",
        "user_orders_bet_cents_check",
        "paper_orders_bet_cents_check",
    }
    with live_conn.cursor() as cur:
        cur.execute(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'games'
              AND column_name = 'game_time_utc'
            """
        )
        assert cur.fetchone() == ("timestamp with time zone",)

        cur.execute(
            """
            SELECT conname
            FROM pg_constraint
            WHERE connamespace = 'public'::regnamespace
              AND convalidated
              AND conname = ANY(%s)
            """,
            (list(expected_constraints),),
        )
        found = {row[0] for row in cur.fetchall()}
        assert expected_constraints <= found


def test_kalshi_credentials_are_encrypted(live_conn):
    with live_conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM kalshi_accounts
            WHERE key_id NOT LIKE 'enc:%'
               OR key_path NOT LIKE 'enc:%'
               OR (private_key_pem IS NOT NULL AND private_key_pem NOT LIKE 'enc:%')
            """
        )
        assert cur.fetchone() == (0,)


def test_money_cents_columns_are_present_and_in_sync(live_conn):
    expected_columns = {
        ("bets", "bet_cents"),
        ("bets", "profit_loss_cents"),
        ("user_orders", "bet_cents"),
        ("user_orders", "profit_loss_cents"),
        ("user_orders", "current_value_cents"),
        ("user_orders", "unrealized_pnl_cents"),
        ("paper_orders", "bet_cents"),
        ("paper_orders", "profit_loss_cents"),
        ("paper_orders", "current_value_cents"),
        ("paper_orders", "unrealized_pnl_cents"),
        ("paper_orders", "paper_bankroll_before_cents"),
        ("paper_orders", "paper_bankroll_after_cents"),
        ("user_order_snapshots", "current_value_cents"),
        ("user_order_snapshots", "unrealized_pnl_cents"),
        ("paper_orders_v2", "bet_cents"),
        ("paper_orders_v2", "pnl_cents"),
        ("paper_orders_v2", "paper_bankroll_before_cents"),
        ("paper_orders_v2", "paper_bankroll_after_cents"),
        ("user_balance", "balance_cents"),
    }
    parity_checks = {
        ("bets", "bet_dollars", "bet_cents"),
        ("bets", "profit_loss", "profit_loss_cents"),
        ("user_orders", "bet_dollars", "bet_cents"),
        ("user_orders", "profit_loss", "profit_loss_cents"),
        ("user_orders", "current_value", "current_value_cents"),
        ("user_orders", "unrealized_pnl", "unrealized_pnl_cents"),
        ("paper_orders", "bet_dollars", "bet_cents"),
        ("paper_orders", "profit_loss", "profit_loss_cents"),
        ("paper_orders", "current_value", "current_value_cents"),
        ("paper_orders", "unrealized_pnl", "unrealized_pnl_cents"),
        ("paper_orders", "paper_bankroll_before", "paper_bankroll_before_cents"),
        ("paper_orders", "paper_bankroll_after", "paper_bankroll_after_cents"),
        ("user_order_snapshots", "current_value", "current_value_cents"),
        ("user_order_snapshots", "unrealized_pnl", "unrealized_pnl_cents"),
        ("paper_orders_v2", "bet_dollars", "bet_cents"),
        ("paper_orders_v2", "pnl", "pnl_cents"),
        ("paper_orders_v2", "paper_bankroll_before", "paper_bankroll_before_cents"),
        ("paper_orders_v2", "paper_bankroll_after", "paper_bankroll_after_cents"),
        ("user_balance", "balance_dollars", "balance_cents"),
    }
    with live_conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = ANY(%s)
            """,
            (sorted({table for table, _ in expected_columns}),),
        )
        found = {(row[0], row[1]) for row in cur.fetchall()}
        assert expected_columns <= found

        mismatches = {}
        for table, legacy_col, cents_col in parity_checks:
            if (table, legacy_col) not in found:
                continue
            name = f"{table}.{cents_col}"
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {table}
                WHERE {legacy_col} IS NOT NULL
                  AND {cents_col} IS DISTINCT FROM ROUND({legacy_col} * 100)::bigint
                """
            )
            count = cur.fetchone()[0]
            if count:
                mismatches[name] = count
        assert mismatches == {}
