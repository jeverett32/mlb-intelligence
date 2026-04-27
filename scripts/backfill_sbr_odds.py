"""Backfill missing 2026 SBR odds for rows that only have fallback odds.

Dry-run is the default. Pass --apply to update the games table.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import os
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extensions import adapt
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
FETCH_DIR = ROOT / "fetch"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FETCH_DIR))

import db as DB  # noqa: E402
from homelab import HomelabClient  # noqa: E402
from fetch_data import _parse_scraped_odds  # noqa: E402
from scraper import scrape_range_async  # noqa: E402

FALLBACK_SOURCE_SQL = "(odds_source LIKE 'odds-api:%%' OR odds_source = 'fanduel')"
ODDS_UPDATE_COLS = (
    "open_home_ml",
    "open_away_ml",
    "close_home_ml",
    "close_away_ml",
    "open_total",
    "close_total",
    "home_implied_prob",
    "away_implied_prob",
    "odds_source",
)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _moneyline_raw_prob(value: Any) -> float | None:
    try:
        ml = float(value)
    except (TypeError, ValueError):
        return None
    if ml <= -100:
        return -ml / (-ml + 100)
    if ml >= 100:
        return 100 / (ml + 100)
    return None


def _implied_probs(home_ml: Any, away_ml: Any) -> tuple[float | None, float | None]:
    home_raw = _moneyline_raw_prob(home_ml)
    away_raw = _moneyline_raw_prob(away_ml)
    if home_raw is None or away_raw is None:
        return None, None
    total = home_raw + away_raw
    if total <= 0:
        return None, None
    return home_raw / total, away_raw / total


def _db_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _valid_real_sbr_odds(row: pd.Series | dict[str, Any]) -> bool:
    vals = {}
    for col in ("open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml"):
        try:
            vals[col] = float(row[col])
        except (KeyError, TypeError, ValueError):
            return False
        if abs(vals[col]) > 500:
            return False
    return (
        vals["open_home_ml"] != vals["close_home_ml"]
        or vals["open_away_ml"] != vals["close_away_ml"]
    )


def _row_key(row: pd.Series | dict[str, Any]) -> tuple[date, str, str]:
    game_date = pd.to_datetime(row["game_date"]).date()
    return game_date, str(row["home_team"]), str(row["away_team"])


def _dedupe_sbr_rows(sbr_df: pd.DataFrame) -> dict[tuple[date, str, str], dict[str, Any]]:
    if sbr_df.empty:
        return {}
    valid = sbr_df[sbr_df.apply(_valid_real_sbr_odds, axis=1)].copy()
    if valid.empty:
        return {}
    valid["game_date"] = pd.to_datetime(valid["game_date"])
    valid = valid.drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="first"
    )

    rows: dict[tuple[date, str, str], dict[str, Any]] = {}
    for _, row in valid.iterrows():
        data = row.to_dict()
        data["home_implied_prob"], data["away_implied_prob"] = _implied_probs(
            data["close_home_ml"], data["close_away_ml"]
        )
        rows[_row_key(data)] = data
    return rows


def build_backfill_updates(
    candidates: list[dict[str, Any]], sbr_df: pd.DataFrame
) -> list[dict[str, Any]]:
    sbr_by_key = _dedupe_sbr_rows(sbr_df)
    key_counts = Counter(_row_key(candidate) for candidate in candidates)
    updates = []

    for candidate in candidates:
        key = _row_key(candidate)
        if key_counts[key] > 1:
            # Doubleheaders cannot be safely matched by date + teams alone.
            continue
        sbr_row = sbr_by_key.get(key)
        if not sbr_row:
            continue
        update = {"game_pk": candidate["game_pk"]}
        for col in ODDS_UPDATE_COLS:
            update[col] = _db_value(sbr_row.get(col))
        updates.append(update)

    return updates


def _fetch_candidates(start: date | None, end: date | None) -> list[dict[str, Any]]:
    conditions = ["season = 2026", FALLBACK_SOURCE_SQL]
    params: list[Any] = []
    if start:
        conditions.append("game_date::date >= %s")
        params.append(start)
    if end:
        conditions.append("game_date::date <= %s")
        params.append(end)

    sql = f"""
        SELECT game_pk, game_date::date AS game_date, home_team, away_team,
               odds_source, open_home_ml, open_away_ml, close_home_ml, close_away_ml
        FROM games
        WHERE {' AND '.join(conditions)}
        ORDER BY game_date, game_pk
    """
    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _homelab_db_client() -> HomelabClient:
    host = _first_env("HOMELAB_DB_SSH_HOST", "DB_HOST")
    if not host:
        raise RuntimeError("Set HOMELAB_DB_SSH_HOST or DB_HOST to use --via-homelab-db")
    return HomelabClient(
        host=host,
        username=_first_env("HOMELAB_DB_SSH_USER", "HOMELAB_USER", default="root") or "root",
        port=int(_first_env("HOMELAB_DB_SSH_PORT", "HOMELAB_PORT", default="22") or "22"),
        password=_first_env("HOMELAB_DB_SSH_PASSWORD", "HOMELAB_PASSWORD"),
        key_path=_first_env("HOMELAB_DB_SSH_KEY_PATH", "HOMELAB_SSH_KEY_PATH"),
        key_passphrase=_first_env(
            "HOMELAB_DB_SSH_KEY_PASSPHRASE", "HOMELAB_SSH_KEY_PASSPHRASE"
        ),
    )


def _remote_psql(client: HomelabClient, sql: str, timeout: int = 120) -> str:
    result = client.run(
        "sudo -u postgres psql -d mlb -v ON_ERROR_STOP=1 -X -q <<'SQL'\n"
        + sql
        + "\nSQL",
        timeout=timeout,
    )
    if result.exit_code != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def _fetch_candidates_via_homelab(
    start: date | None, end: date | None
) -> list[dict[str, Any]]:
    conditions = ["season = 2026", FALLBACK_SOURCE_SQL.replace("%%", "%")]
    if start:
        conditions.append(f"game_date::date >= {_sql_literal(start)}::date")
    if end:
        conditions.append(f"game_date::date <= {_sql_literal(end)}::date")

    sql = f"""
        COPY (
            SELECT game_pk, game_date::date AS game_date, home_team, away_team,
                   odds_source, open_home_ml, open_away_ml, close_home_ml, close_away_ml
            FROM games
            WHERE {' AND '.join(conditions)}
            ORDER BY game_date, game_pk
        ) TO STDOUT WITH CSV HEADER;
    """
    with _homelab_db_client() as client:
        output = _remote_psql(client, sql)

    reader = csv.DictReader(io.StringIO(output))
    rows = []
    for row in reader:
        rows.append(
            {
                "game_pk": int(row["game_pk"]),
                "game_date": _parse_date(row["game_date"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "odds_source": row["odds_source"],
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


def _sql_literal(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    return adapt(value).getquoted().decode("utf-8")


def _values_sql(updates: list[dict[str, Any]]) -> str:
    cols = ("game_pk",) + ODDS_UPDATE_COLS
    values = []
    for update in updates:
        values.append("(" + ", ".join(_sql_literal(update.get(col)) for col in cols) + ")")
    return ",\n".join(values)


def _apply_updates(updates: list[dict[str, Any]]) -> int:
    if not updates:
        return 0
    sql = """
        UPDATE games
        SET open_home_ml = %(open_home_ml)s,
            open_away_ml = %(open_away_ml)s,
            close_home_ml = %(close_home_ml)s,
            close_away_ml = %(close_away_ml)s,
            home_implied_prob = %(home_implied_prob)s,
            away_implied_prob = %(away_implied_prob)s,
            odds_source = %(odds_source)s,
            extra = COALESCE(extra, '{}'::jsonb) || jsonb_strip_nulls(
                jsonb_build_object(
                    'open_total', %(open_total)s,
                    'close_total', %(close_total)s
                )
            ),
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


def _apply_updates_via_homelab(updates: list[dict[str, Any]]) -> int:
    if not updates:
        return 0
    sql = f"""
        WITH updates (
            game_pk, open_home_ml, open_away_ml, close_home_ml, close_away_ml,
            open_total, close_total, home_implied_prob, away_implied_prob, odds_source
        ) AS (
            VALUES
            {_values_sql(updates)}
        )
        UPDATE games
        SET open_home_ml = updates.open_home_ml::double precision,
            open_away_ml = updates.open_away_ml::double precision,
            close_home_ml = updates.close_home_ml::double precision,
            close_away_ml = updates.close_away_ml::double precision,
            home_implied_prob = updates.home_implied_prob::double precision,
            away_implied_prob = updates.away_implied_prob::double precision,
            odds_source = updates.odds_source::text,
            extra = COALESCE(games.extra, '{{}}'::jsonb) || jsonb_strip_nulls(
                jsonb_build_object(
                    'open_total', updates.open_total::double precision,
                    'close_total', updates.close_total::double precision
                )
            ),
            updated_at = NOW()
        FROM updates
        WHERE games.game_pk = updates.game_pk::bigint;
    """
    with _homelab_db_client() as client:
        _remote_psql(client, sql, timeout=300)
    return len(updates)


async def _scrape_sbr(start: date, end: date) -> pd.DataFrame:
    raw = await scrape_range_async(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        fast=False,
        max_concurrent=5,
        odds_types=["moneyline", "totals"],
    )
    return _parse_scraped_odds(raw)


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Backfill 2026 fallback odds with SBR odds.")
    parser.add_argument("--start", default="2026-03-25", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-04-15", help="End date YYYY-MM-DD")
    parser.add_argument("--apply", action="store_true", help="Write updates to the DB")
    parser.add_argument("--limit", type=int, default=None, help="Limit candidate rows for testing")
    parser.add_argument(
        "--via-homelab-db",
        action="store_true",
        help="Query/apply through the DB LXC using sudo postgres psql",
    )
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    try:
        if args.via_homelab_db:
            candidates = _fetch_candidates_via_homelab(start, end)
        else:
            candidates = _fetch_candidates(start, end)
    except psycopg2.OperationalError as exc:
        print(f"DB connection failed: {exc}")
        print("Run this from a host allowed by Postgres pg_hba, or via the DB/app LXC.")
        return 2
    if args.limit is not None:
        candidates = candidates[: args.limit]

    if not candidates:
        print("No fallback odds rows found for the requested range.")
        return 0

    candidate_dates = sorted({_row_key(row)[0] for row in candidates})
    scrape_start = candidate_dates[0]
    scrape_end = candidate_dates[-1]
    print(
        f"Found {len(candidates)} candidate row(s) across "
        f"{len(candidate_dates)} date(s): {scrape_start} -> {scrape_end}"
    )

    sbr_df = asyncio.run(_scrape_sbr(scrape_start, scrape_end))
    updates = build_backfill_updates(candidates, sbr_df)
    print(f"SBR returned {0 if sbr_df.empty else len(sbr_df)} row(s).")
    print(f"Prepared {len(updates)} valid SBR update(s).")

    for update in updates[:10]:
        print(
            "  game_pk={game_pk} open=({open_home_ml}, {open_away_ml}) "
            "close=({close_home_ml}, {close_away_ml}) source={odds_source}".format(**update)
        )
    if len(updates) > 10:
        print(f"  ... {len(updates) - 10} more")

    if not args.apply:
        print("Dry run only. Re-run with --apply to update the DB.")
        return 0

    updated = _apply_updates_via_homelab(updates) if args.via_homelab_db else _apply_updates(updates)
    print(f"Updated {updated} game row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
