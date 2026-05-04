#!/usr/bin/env python3
"""Add and backfill integer-cent money columns.

Run with a DB owner/superuser connection. This is phase A/B migration: keep
legacy dollar columns in place while the app dual-writes cents.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def main() -> int:
    with db.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout TO 0")
            for table, columns in TABLE_COLUMNS.items():
                for dollars_col, cents_col in columns.items():
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {cents_col} BIGINT")
                    cur.execute(
                        f"""
                        UPDATE {table}
                        SET {cents_col} = ROUND({dollars_col} * 100)::bigint
                        WHERE {cents_col} IS NULL
                          AND {dollars_col} IS NOT NULL
                        """
                    )

            for table, cents_col in NONNEGATIVE_COLUMNS:
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

        conn.commit()

    print("Money cents migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
