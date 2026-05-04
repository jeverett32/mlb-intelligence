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
    }
    parity_checks = {
        "bets.bet_cents": "SELECT COUNT(*) FROM bets WHERE bet_dollars IS NOT NULL AND bet_cents IS DISTINCT FROM ROUND(bet_dollars * 100)::bigint",
        "bets.profit_loss_cents": "SELECT COUNT(*) FROM bets WHERE profit_loss IS NOT NULL AND profit_loss_cents IS DISTINCT FROM ROUND(profit_loss * 100)::bigint",
        "user_orders.bet_cents": "SELECT COUNT(*) FROM user_orders WHERE bet_dollars IS NOT NULL AND bet_cents IS DISTINCT FROM ROUND(bet_dollars * 100)::bigint",
        "user_orders.profit_loss_cents": "SELECT COUNT(*) FROM user_orders WHERE profit_loss IS NOT NULL AND profit_loss_cents IS DISTINCT FROM ROUND(profit_loss * 100)::bigint",
        "user_orders.current_value_cents": "SELECT COUNT(*) FROM user_orders WHERE current_value IS NOT NULL AND current_value_cents IS DISTINCT FROM ROUND(current_value * 100)::bigint",
        "user_orders.unrealized_pnl_cents": "SELECT COUNT(*) FROM user_orders WHERE unrealized_pnl IS NOT NULL AND unrealized_pnl_cents IS DISTINCT FROM ROUND(unrealized_pnl * 100)::bigint",
        "paper_orders.bet_cents": "SELECT COUNT(*) FROM paper_orders WHERE bet_dollars IS NOT NULL AND bet_cents IS DISTINCT FROM ROUND(bet_dollars * 100)::bigint",
        "paper_orders.profit_loss_cents": "SELECT COUNT(*) FROM paper_orders WHERE profit_loss IS NOT NULL AND profit_loss_cents IS DISTINCT FROM ROUND(profit_loss * 100)::bigint",
        "paper_orders.current_value_cents": "SELECT COUNT(*) FROM paper_orders WHERE current_value IS NOT NULL AND current_value_cents IS DISTINCT FROM ROUND(current_value * 100)::bigint",
        "paper_orders.unrealized_pnl_cents": "SELECT COUNT(*) FROM paper_orders WHERE unrealized_pnl IS NOT NULL AND unrealized_pnl_cents IS DISTINCT FROM ROUND(unrealized_pnl * 100)::bigint",
        "paper_orders.paper_bankroll_before_cents": "SELECT COUNT(*) FROM paper_orders WHERE paper_bankroll_before IS NOT NULL AND paper_bankroll_before_cents IS DISTINCT FROM ROUND(paper_bankroll_before * 100)::bigint",
        "paper_orders.paper_bankroll_after_cents": "SELECT COUNT(*) FROM paper_orders WHERE paper_bankroll_after IS NOT NULL AND paper_bankroll_after_cents IS DISTINCT FROM ROUND(paper_bankroll_after * 100)::bigint",
        "user_order_snapshots.current_value_cents": "SELECT COUNT(*) FROM user_order_snapshots WHERE current_value IS NOT NULL AND current_value_cents IS DISTINCT FROM ROUND(current_value * 100)::bigint",
        "user_order_snapshots.unrealized_pnl_cents": "SELECT COUNT(*) FROM user_order_snapshots WHERE unrealized_pnl IS NOT NULL AND unrealized_pnl_cents IS DISTINCT FROM ROUND(unrealized_pnl * 100)::bigint",
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
        for name, sql in parity_checks.items():
            cur.execute(sql)
            count = cur.fetchone()[0]
            if count:
                mismatches[name] = count
        assert mismatches == {}
