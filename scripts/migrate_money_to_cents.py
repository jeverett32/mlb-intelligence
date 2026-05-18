#!/usr/bin/env python3
"""Add and backfill integer-cent money columns.

Run with a DB owner/superuser connection before dropping legacy dollar columns.
The script is idempotent after the drop; missing legacy columns are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2.errors

import db


TABLE_COLUMNS = {
    "bets": {
        "bet_dollars": "bet_cents",
        "profit_loss": "profit_loss_cents",
    },
    "user_orders": {
        "bet_dollars": "bet_cents",
        "profit_loss": "profit_loss_cents",
        "current_value": "current_value_cents",
        "unrealized_pnl": "unrealized_pnl_cents",
    },
    "paper_orders": {
        "bet_dollars": "bet_cents",
        "profit_loss": "profit_loss_cents",
        "current_value": "current_value_cents",
        "unrealized_pnl": "unrealized_pnl_cents",
        "paper_bankroll_before": "paper_bankroll_before_cents",
        "paper_bankroll_after": "paper_bankroll_after_cents",
    },
    "user_order_snapshots": {
        "current_value": "current_value_cents",
        "unrealized_pnl": "unrealized_pnl_cents",
    },
    "paper_orders_v2": {
        "bet_dollars": "bet_cents",
        "pnl": "pnl_cents",
        "paper_bankroll_before": "paper_bankroll_before_cents",
        "paper_bankroll_after": "paper_bankroll_after_cents",
    },
}

NONNEGATIVE_COLUMNS = {
    ("bets", "bet_cents"),
    ("user_orders", "bet_cents"),
    ("user_orders", "current_value_cents"),
    ("paper_orders", "bet_cents"),
    ("paper_orders", "current_value_cents"),
    ("paper_orders", "paper_bankroll_before_cents"),
    ("paper_orders", "paper_bankroll_after_cents"),
    ("user_order_snapshots", "current_value_cents"),
    ("paper_orders_v2", "bet_cents"),
    ("paper_orders_v2", "paper_bankroll_before_cents"),
    ("paper_orders_v2", "paper_bankroll_after_cents"),
}


def _constraint_exists(cur, table: str, name: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = %s::regclass
          AND conname = %s
        """,
        (table, name),
    )
    return cur.fetchone() is not None


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def _migrate_table(cur, table: str, columns: dict[str, str]) -> None:
    """Add + backfill cent columns and non-negative checks for one table.

    Wrapped in a savepoint so a table the connection role does not own
    (e.g. `bets` is owned by `postgres`) is skipped without aborting the
    rest of the migration.
    """
    cur.execute(f"SAVEPOINT migrate_{table}")
    try:
        for dollars_col, cents_col in columns.items():
            cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {cents_col} BIGINT")
            if not _column_exists(cur, table, dollars_col):
                continue
            cur.execute(
                f"""
                UPDATE {table}
                SET {cents_col} = ROUND({dollars_col} * 100)::bigint
                WHERE {cents_col} IS NULL
                  AND {dollars_col} IS NOT NULL
                """
            )
        for cents_col in columns.values():
            if (table, cents_col) not in NONNEGATIVE_COLUMNS:
                continue
            name = f"{table}_{cents_col}_check"
            if not _constraint_exists(cur, table, name):
                cur.execute(
                    f"""
                    ALTER TABLE {table}
                    ADD CONSTRAINT {name}
                    CHECK ({cents_col} IS NULL OR {cents_col} >= 0)
                    NOT VALID
                    """
                )
            cur.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")
    except psycopg2.errors.InsufficientPrivilege:
        cur.execute(f"ROLLBACK TO SAVEPOINT migrate_{table}")
        print(f"  SKIP {table}: connection role is not the owner (run as table owner).")
        return
    cur.execute(f"RELEASE SAVEPOINT migrate_{table}")
    print(f"  OK   {table}")


def main() -> int:
    with db.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout TO 0")
            for table, columns in TABLE_COLUMNS.items():
                _migrate_table(cur, table, columns)
        conn.commit()

    print("Money cents migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
