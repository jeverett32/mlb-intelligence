"""Reset known 2025 live-contaminated close lines back to opening lines.

Dry-run is the default. Pass --apply to update the games table.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db as DB  # noqa: E402
from scripts.backfill_sbr_odds import (  # noqa: E402
    _db_ssh_client,
    _implied_probs,
    _remote_psql,
    _sql_literal,
)

AFFECTED_GAME_PKS = (
    776300,
    776304,
    776771,
    776295,
    776772,
    776294,
    776302,
    776297,
    776308,
    776298,
    776303,
    776770,
    776317,
    776306,
    776301,
    776194,
    776287,
    776199,
    777842,
    776629,
    776305,
    777825,
    776293,
)


def _fetch_rows() -> list[dict[str, Any]]:
    sql = """
        SELECT game_pk, game_date, home_team, away_team,
               open_home_ml, open_away_ml, close_home_ml, close_away_ml
        FROM games
        WHERE season = 2025 AND game_pk = ANY(%s)
        ORDER BY game_date, game_pk
    """
    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (list(AFFECTED_GAME_PKS),))
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _fetch_rows_via_db_ssh() -> list[dict[str, Any]]:
    game_pks = ", ".join(str(game_pk) for game_pk in AFFECTED_GAME_PKS)
    sql = f"""
        COPY (
            SELECT game_pk, game_date, home_team, away_team,
                   open_home_ml, open_away_ml, close_home_ml, close_away_ml
            FROM games
            WHERE season = 2025 AND game_pk IN ({game_pks})
            ORDER BY game_date, game_pk
        ) TO STDOUT WITH CSV HEADER;
    """
    with _db_ssh_client() as client:
        output = _remote_psql(client, sql)

    rows = []
    for row in csv.DictReader(io.StringIO(output)):
        rows.append(
            {
                "game_pk": int(row["game_pk"]),
                "game_date": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "open_home_ml": _parse_float(row["open_home_ml"]),
                "open_away_ml": _parse_float(row["open_away_ml"]),
                "close_home_ml": _parse_float(row["close_home_ml"]),
                "close_away_ml": _parse_float(row["close_away_ml"]),
            }
        )
    return rows


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def build_updates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates = []
    for row in rows:
        home_prob, away_prob = _implied_probs(row["open_home_ml"], row["open_away_ml"])
        updates.append(
            {
                "game_pk": row["game_pk"],
                "close_home_ml": row["open_home_ml"],
                "close_away_ml": row["open_away_ml"],
                "home_implied_prob": home_prob,
                "away_implied_prob": away_prob,
            }
        )
    return updates


def _apply_updates(updates: list[dict[str, Any]]) -> int:
    if not updates:
        return 0
    sql = """
        UPDATE games
        SET close_home_ml = %(close_home_ml)s,
            close_away_ml = %(close_away_ml)s,
            home_implied_prob = %(home_implied_prob)s,
            away_implied_prob = %(away_implied_prob)s,
            updated_at = NOW()
        WHERE game_pk = %(game_pk)s
    """
    conn = DB.get_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, updates)
        conn.commit()
        return len(updates)
    finally:
        conn.close()


def _apply_updates_via_db_ssh(updates: list[dict[str, Any]]) -> int:
    if not updates:
        return 0
    values = []
    cols = ("game_pk", "close_home_ml", "close_away_ml", "home_implied_prob", "away_implied_prob")
    for update in updates:
        values.append("(" + ", ".join(_sql_literal(update.get(col)) for col in cols) + ")")
    sql = f"""
        WITH updates (
            game_pk, close_home_ml, close_away_ml, home_implied_prob, away_implied_prob
        ) AS (
            VALUES
            {",\n".join(values)}
        )
        UPDATE games
        SET close_home_ml = updates.close_home_ml::double precision,
            close_away_ml = updates.close_away_ml::double precision,
            home_implied_prob = updates.home_implied_prob::double precision,
            away_implied_prob = updates.away_implied_prob::double precision,
            updated_at = NOW()
        FROM updates
        WHERE games.game_pk = updates.game_pk::bigint;
    """
    with _db_ssh_client() as client:
        _remote_psql(client, sql)
    return len(updates)


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Repair known 2025 live-line contamination.")
    parser.add_argument("--apply", action="store_true", help="Write updates to the DB")
    parser.add_argument(
        "--via-db-ssh",
        dest="via_db_ssh",
        action="store_true",
        help="Query/apply through the DB LXC using sudo postgres psql",
    )
    args = parser.parse_args()

    try:
        rows = _fetch_rows_via_db_ssh() if args.via_db_ssh else _fetch_rows()
    except psycopg2.OperationalError as exc:
        print(f"DB connection failed: {exc}")
        print("Run this from a host allowed by Postgres pg_hba, or via the DB/app LXC.")
        return 2
    updates = build_updates(rows)
    print(f"Found {len(rows)} of {len(AFFECTED_GAME_PKS)} known affected row(s).")
    print(f"Prepared {len(updates)} close-line reset(s).")
    for row in rows[:10]:
        print(
            "  game_pk={game_pk} {away_team}@{home_team} "
            "open=({open_home_ml}, {open_away_ml}) close=({close_home_ml}, {close_away_ml})".format(
                **row
            )
        )
    if len(rows) > 10:
        print(f"  ... {len(rows) - 10} more")

    if not args.apply:
        print("Dry run only. Re-run with --apply to update the DB.")
        return 0

    updated = _apply_updates_via_db_ssh(updates) if args.via_db_ssh else _apply_updates(updates)
    print(f"Updated {updated} game row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
