"""Helper for recording bulk data-quality repairs against historical rows.

Every script under scripts/ that backfills, normalizes, or otherwise rewrites
historical data should call `record_repair` so we keep an auditable trail of
what changed and when. Pair with `data_quality_runs` to verify a repair
actually moved the needle on the relevant audit metric.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import db


def record_repair(
    *,
    script: str,
    column: str | None,
    rows_affected: int,
    reason: str,
    details: dict[str, Any] | None = None,
) -> int:
    """Insert one row into data_repair_log and return its id."""
    db.init_data_quality_tables()
    payload = json.dumps(details or {}, default=str)
    with db.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO data_repair_log
                    (script, column_name, rows_affected, reason, details, created_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                RETURNING id
                """,
                (
                    script,
                    column,
                    int(rows_affected),
                    reason,
                    payload,
                    datetime.now(timezone.utc),
                ),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
    return int(row_id)


def list_repairs(limit: int = 100) -> list[dict]:
    db.init_data_quality_tables()
    with db.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, script, column_name, rows_affected, reason,
                       details, created_at
                FROM data_repair_log
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
