#!/usr/bin/env python3
"""Drop legacy dollar money columns after cents-only code is deployed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from psycopg2 import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


LEGACY_COLUMNS = {
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
    "user_balance": {
        "balance_dollars": "balance_cents",
    },
}


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


def _preflight(cur) -> dict[str, str]:
    blockers: dict[str, str] = {}
    for table, columns in LEGACY_COLUMNS.items():
        for legacy_col, cents_col in columns.items():
            name = f"{table}.{legacy_col}"
            if not _column_exists(cur, table, legacy_col):
                continue
            if not _column_exists(cur, table, cents_col):
                blockers[name] = f"missing {table}.{cents_col}"
                continue
            cur.execute(
                sql.SQL(
                    """
                    SELECT COUNT(*)
                    FROM {table}
                    WHERE {legacy_col} IS NOT NULL
                      AND {cents_col} IS NULL
                    """
                ).format(
                    table=sql.Identifier(table),
                    legacy_col=sql.Identifier(legacy_col),
                    cents_col=sql.Identifier(cents_col),
                )
            )
            gaps = int(cur.fetchone()[0] or 0)
            if gaps:
                blockers[name] = f"{gaps} rows have legacy value but NULL cents"
    return blockers


def _owner_blockers(cur) -> dict[str, str]:
    tables = sorted(LEGACY_COLUMNS)
    cur.execute(
        """
        SELECT c.relname,
               pg_get_userbyid(c.relowner) AS owner,
               current_user,
               (
                   pg_get_userbyid(c.relowner) = current_user
                   OR pg_has_role(current_user, c.relowner, 'MEMBER')
               ) AS can_alter
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind = 'r'
          AND c.relname = ANY(%s)
        """,
        (tables,),
    )
    blockers = {}
    for table, owner, current_user, can_alter in cur.fetchall():
        legacy_exists = any(_column_exists(cur, table, col) for col in LEGACY_COLUMNS[table])
        if legacy_exists and not can_alter:
            blockers[table] = f"owned by {owner}; current user is {current_user}"
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="drop legacy columns")
    args = parser.parse_args()

    with db.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout TO 0")
            blockers = _preflight(cur)
            if blockers:
                for name, reason in blockers.items():
                    print(f"BLOCK {name}: {reason}")
                conn.rollback()
                return 1
            owner_blockers = _owner_blockers(cur)
            if owner_blockers:
                for table, reason in owner_blockers.items():
                    print(f"OWNER {table}: {reason}")

            if not args.apply:
                print("Preflight OK. Re-run with --apply to drop legacy columns.")
                conn.rollback()
                return 0
            if owner_blockers:
                conn.rollback()
                return 1

            for table, columns in LEGACY_COLUMNS.items():
                for legacy_col in columns:
                    cur.execute(
                        sql.SQL("ALTER TABLE {table} DROP COLUMN IF EXISTS {legacy_col}").format(
                            table=sql.Identifier(table),
                            legacy_col=sql.Identifier(legacy_col),
                        )
                    )
                    print(f"DROP {table}.{legacy_col}")
        conn.commit()

    print("Legacy money columns dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
