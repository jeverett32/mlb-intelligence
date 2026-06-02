#!/usr/bin/env python3
"""Idempotent Postgres schema hardening for the MLB pipeline.

This intentionally stays as a small operational script instead of introducing
Alembic before the existing schema cleanup is done.

Run with a DB owner/superuser connection; the app role does not own every
table this script touches.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


ORDER_STATUSES = (
    "pending",
    "filled",
    "dry_run",
    "unfilled",
    "error",
    "skipped_no_market",
    "skipped_no_live_price",
    "skipped_no_live_edge",
    "skipped_too_small",
    "skipped_below_contract",
    "sold_stop_loss",
)

PIPELINE_STATUSES = ("running", "success", "aborted", "error")
MARKET_STATUSES = ("active", "open", "inactive", "closed", "settled", "finalized")


def _q(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _in(values: tuple[str, ...]) -> str:
    return ", ".join(_q(v) for v in values)


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


def _add_constraint(cur, table: str, name: str, definition: str) -> None:
    if not _constraint_exists(cur, table, name):
        cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} {definition} NOT VALID")
    cur.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")


def _replace_check_constraint(cur, table: str, name: str, definition: str) -> None:
    """Drop and recreate a CHECK constraint so allowed values can be extended."""
    if _constraint_exists(cur, table, name):
        cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")
    cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} {definition} NOT VALID")
    cur.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")


def _set_not_null(cur, table: str, column: str) -> None:
    cur.execute(
        """
        SELECT is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        """,
        (table, column),
    )
    row = cur.fetchone()
    if row and row[0] == "YES":
        cur.execute(f"ALTER TABLE {table} ALTER COLUMN {column} SET NOT NULL")


def _convert_game_time(cur) -> None:
    cur.execute(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'games'
          AND column_name = 'game_time_utc'
        """
    )
    row = cur.fetchone()
    if row and row[0] == "timestamp without time zone":
        cur.execute(
            """
            ALTER TABLE games
            ALTER COLUMN game_time_utc TYPE timestamptz
            USING game_time_utc AT TIME ZONE 'UTC'
            """
        )


def main() -> int:
    with db.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout TO 0")

            cur.execute("UPDATE bets SET bet_side = 'none' WHERE bet_side IS NULL")

            _convert_game_time(cur)

            _add_constraint(
                cur,
                "bets",
                "bets_model_artifact_id_fkey",
                "FOREIGN KEY (model_artifact_id) REFERENCES model_artifacts(id) ON DELETE SET NULL",
            )
            for table in ("user_orders", "paper_orders", "user_order_snapshots"):
                _add_constraint(
                    cur,
                    table,
                    f"{table}_game_pk_fkey",
                    "FOREIGN KEY (game_pk) REFERENCES games(game_pk) ON DELETE RESTRICT",
                )

            for table in ("bets", "user_orders", "paper_orders"):
                _add_constraint(
                    cur,
                    table,
                    f"{table}_bet_side_check",
                    "CHECK (bet_side IS NULL OR bet_side IN ('home', 'away', 'none'))",
                )
                _add_constraint(
                    cur,
                    table,
                    f"{table}_predicted_prob_check",
                    "CHECK (predicted_prob IS NULL OR predicted_prob BETWEEN 0 AND 1)",
                )
                _add_constraint(
                    cur,
                    table,
                    f"{table}_n_contracts_check",
                    "CHECK (n_contracts IS NULL OR n_contracts >= 0)",
                )
                _add_constraint(
                    cur,
                    table,
                    f"{table}_bet_cents_check",
                    "CHECK (bet_cents IS NULL OR bet_cents >= 0)",
                )

            for table in ("user_orders", "paper_orders"):
                _replace_check_constraint(
                    cur,
                    table,
                    f"{table}_status_check",
                    f"CHECK (status IN ({_in(ORDER_STATUSES)}))",
                )

            _add_constraint(
                cur,
                "pipeline_runs",
                "pipeline_runs_status_check",
                f"CHECK (status IS NULL OR status IN ({_in(PIPELINE_STATUSES)}))",
            )
            _add_constraint(
                cur,
                "kalshi_market_snapshots",
                "kalshi_market_snapshots_market_status_check",
                f"CHECK (market_status IS NULL OR market_status IN ({_in(MARKET_STATUSES)}))",
            )

            for column in ("home_team", "away_team", "bet_side", "created_at", "updated_at"):
                _set_not_null(cur, "bets", column)

            cur.execute("DROP TABLE IF EXISTS user_settings_kv_backup")
            if _constraint_exists(cur, "model_metric_snapshots", "model_training_runs_pkey"):
                cur.execute(
                    """
                    ALTER TABLE model_metric_snapshots
                    RENAME CONSTRAINT model_training_runs_pkey TO model_metric_snapshots_pkey
                    """
                )

            cur.execute("DROP INDEX IF EXISTS idx_games_teams")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_orders_pending
                ON user_orders (last_checked_at)
                WHERE status = 'filled'
                  AND dry_run = FALSE
                  AND result IS NULL
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_orders_pending
                ON paper_orders (last_checked_at)
                WHERE status = 'dry_run'
                  AND result IS NULL
                """
            )

        conn.commit()

    print("Schema hardening complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
