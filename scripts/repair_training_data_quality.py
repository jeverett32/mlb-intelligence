#!/usr/bin/env python3
"""Repair known training data quality issues in Postgres.

Dry-run is the default. Pass --apply to write updates.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
FETCH_DIR = ROOT / "fetch"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FETCH_DIR))

import db as DB  # noqa: E402
from fetch_data import _parse_scraped_odds  # noqa: E402
from scraper import scrape_range_async  # noqa: E402


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


def _overround(home_ml: Any, away_ml: Any) -> float | None:
    home_raw = _moneyline_raw_prob(home_ml)
    away_raw = _moneyline_raw_prob(away_ml)
    if home_raw is None or away_raw is None:
        return None
    return home_raw + away_raw


def _parse_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_odds_row(row: pd.Series | dict[str, Any]) -> bool:
    for col in ("open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml"):
        value = _parse_float(row.get(col))
        if value is None or abs(value) > 500:
            return False
        if _moneyline_raw_prob(value) is None:
            return False
    overround = _overround(row.get("close_home_ml"), row.get("close_away_ml"))
    return overround is not None and 0.99 <= overround <= 1.20


def _row_key(row: pd.Series | dict[str, Any]) -> tuple[date, str, str]:
    return (
        pd.to_datetime(row["game_date"]).date(),
        str(row["home_team"]),
        str(row["away_team"]),
    )


def _db_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return value


def fetch_wrc_issue_counts() -> dict[str, int]:
    sql = """
        SELECT
            COUNT(*) FILTER (
                WHERE home_wrc_plus IS NOT NULL
                  AND (home_wrc_plus < 10 OR home_wrc_plus > 300)
            ) AS home_bad,
            COUNT(*) FILTER (
                WHERE away_wrc_plus IS NOT NULL
                  AND (away_wrc_plus < 10 OR away_wrc_plus > 300)
            ) AS away_bad
        FROM games
    """
    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            row = dict(cur.fetchone())
    finally:
        conn.close()
    return {"home_wrc_plus": int(row["home_bad"]), "away_wrc_plus": int(row["away_bad"])}


def apply_wrc_null_repair() -> int:
    sql = """
        UPDATE games
        SET home_wrc_plus = CASE
                WHEN home_wrc_plus IS NOT NULL AND (home_wrc_plus < 10 OR home_wrc_plus > 300)
                THEN NULL ELSE home_wrc_plus END,
            away_wrc_plus = CASE
                WHEN away_wrc_plus IS NOT NULL AND (away_wrc_plus < 10 OR away_wrc_plus > 300)
                THEN NULL ELSE away_wrc_plus END,
            updated_at = NOW()
        WHERE (home_wrc_plus IS NOT NULL AND (home_wrc_plus < 10 OR home_wrc_plus > 300))
           OR (away_wrc_plus IS NOT NULL AND (away_wrc_plus < 10 OR away_wrc_plus > 300))
    """
    conn = DB.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            updated = int(cur.rowcount or 0)
        conn.commit()
    finally:
        conn.close()
    return updated


def fetch_missing_wrc_2025_count() -> int:
    sql = """
        SELECT COUNT(*)
        FROM games
        WHERE season = 2025
          AND (home_wrc_plus IS NULL OR away_wrc_plus IS NULL)
    """
    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row["count"])
    finally:
        conn.close()


def apply_wrc_2025_proxy_from_woba() -> int:
    """Fill missing 2025 wRC+ using a simple wOBA-based proxy.

    FanGraphs wRC+ feed is intermittently missing for 2025 in our DB, while
    team wOBA is present. We approximate wRC+ scale with:

        wrc_plus_proxy = 100 * team_woba / league_woba(season)

    This is not park-adjusted wRC+, but it keeps scale consistent and prevents
    2025 feature-collapse.
    """
    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT AVG(woba) AS league_woba
                FROM (
                    SELECT home_woba AS woba FROM games WHERE season = 2025 AND home_woba IS NOT NULL
                    UNION ALL
                    SELECT away_woba AS woba FROM games WHERE season = 2025 AND away_woba IS NOT NULL
                ) t
                """
            )
            league_woba = cur.fetchone().get("league_woba")
            league_woba = _parse_float(league_woba)
            if not league_woba or league_woba <= 0:
                return 0

        # Update in a separate cursor (avoid mixing fetch + update with dict cursor)
        with conn.cursor() as cur2:
            cur2.execute(
                """
                UPDATE games
                SET home_wrc_plus = CASE
                        WHEN home_wrc_plus IS NULL AND home_woba IS NOT NULL
                        THEN LEAST(300, GREATEST(10, 100.0 * home_woba / %s))
                        ELSE home_wrc_plus
                    END,
                    away_wrc_plus = CASE
                        WHEN away_wrc_plus IS NULL AND away_woba IS NOT NULL
                        THEN LEAST(300, GREATEST(10, 100.0 * away_woba / %s))
                        ELSE away_wrc_plus
                    END,
                    updated_at = NOW()
                WHERE season = 2025
                  AND (
                      (home_wrc_plus IS NULL AND home_woba IS NOT NULL)
                      OR (away_wrc_plus IS NULL AND away_woba IS NOT NULL)
                  )
                """,
                (league_woba, league_woba),
            )
            updated = int(cur2.rowcount or 0)
        conn.commit()
        return updated
    finally:
        conn.close()


def fetch_bad_odds_candidates() -> list[dict[str, Any]]:
    sql = """
        SELECT game_pk, game_date::date AS game_date, home_team, away_team,
               open_home_ml, open_away_ml, close_home_ml, close_away_ml,
               odds_source
        FROM games
        WHERE home_win IS NOT NULL
          AND (
              close_home_ml IS NULL
              OR close_away_ml IS NULL
              OR (
                  close_home_ml <= -100 OR close_home_ml >= 100
              )
              AND (
                  close_away_ml <= -100 OR close_away_ml >= 100
              )
              AND (
                  CASE WHEN close_home_ml < 0
                       THEN abs(close_home_ml) / (abs(close_home_ml) + 100.0)
                       ELSE 100.0 / (close_home_ml + 100.0)
                  END
                  +
                  CASE WHEN close_away_ml < 0
                       THEN abs(close_away_ml) / (abs(close_away_ml) + 100.0)
                       ELSE 100.0 / (close_away_ml + 100.0)
                  END
              ) < 0.99
          )
        ORDER BY game_date, game_pk
    """
    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


async def scrape_sbr_for_dates(dates: list[date]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    frames = []
    for game_date in sorted(dates):
        raw = await scrape_range_async(
            game_date.strftime("%Y-%m-%d"),
            game_date.strftime("%Y-%m-%d"),
            fast=False,
            max_concurrent=3,
            odds_types=["moneyline", "totals"],
        )
        parsed = _parse_scraped_odds(raw)
        if not parsed.empty:
            frames.append(parsed)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_odds_updates(candidates: list[dict[str, Any]], sbr_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sbr_df.empty:
        return [], candidates

    valid = sbr_df[sbr_df.apply(_valid_odds_row, axis=1)].copy()
    if valid.empty:
        return [], candidates

    valid["game_date"] = pd.to_datetime(valid["game_date"])
    valid = valid.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
    by_key = {_row_key(row): row.to_dict() for _, row in valid.iterrows()}

    updates = []
    skipped = []
    for candidate in candidates:
        sbr_row = by_key.get(_row_key(candidate))
        if not sbr_row:
            skipped.append(candidate)
            continue
        home_prob, away_prob = _implied_probs(sbr_row["close_home_ml"], sbr_row["close_away_ml"])
        updates.append(
            {
                "game_pk": candidate["game_pk"],
                "open_home_ml": _db_value(sbr_row.get("open_home_ml")),
                "open_away_ml": _db_value(sbr_row.get("open_away_ml")),
                "close_home_ml": _db_value(sbr_row.get("close_home_ml")),
                "close_away_ml": _db_value(sbr_row.get("close_away_ml")),
                "open_total": _db_value(sbr_row.get("open_total")),
                "close_total": _db_value(sbr_row.get("close_total")),
                "home_implied_prob": home_prob,
                "away_implied_prob": away_prob,
                "odds_source": sbr_row.get("odds_source"),
            }
        )
    return updates, skipped


def apply_odds_updates(updates: list[dict[str, Any]]) -> int:
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
    finally:
        conn.close()
    return len(updates)


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write repairs to DB")
    parser.add_argument("--skip-wrc", action="store_true", help="skip wRC+ null repair")
    parser.add_argument("--repair-2025-wrc-proxy", action="store_true", help="fill missing 2025 wRC+ from wOBA proxy")
    parser.add_argument("--skip-odds", action="store_true", help="skip SBR odds repair")
    args = parser.parse_args()

    if not args.skip_wrc:
        wrc_counts = fetch_wrc_issue_counts()
        print(f"wrc_plus_bad={wrc_counts}")
        if args.apply:
            updated = apply_wrc_null_repair()
            print(f"wrc_plus_rows_updated={updated}")

    if args.repair_2025_wrc_proxy:
        missing = fetch_missing_wrc_2025_count()
        print(f"wrc_plus_missing_2025={missing}")
        if args.apply:
            updated = apply_wrc_2025_proxy_from_woba()
            print(f"wrc_plus_2025_proxy_rows_updated={updated}")

    if not args.skip_odds:
        candidates = fetch_bad_odds_candidates()
        print(f"bad_odds_candidates={len(candidates)}")
        for row in candidates[:10]:
            print(
                "  candidate game_pk={game_pk} {away_team}@{home_team} "
                "{game_date} close=({close_home_ml}, {close_away_ml}) source={odds_source}".format(**row)
            )

        dates = sorted({_row_key(row)[0] for row in candidates})
        sbr_df = asyncio.run(scrape_sbr_for_dates(dates))
        updates, skipped = build_odds_updates(candidates, sbr_df)
        print(f"sbr_rows={0 if sbr_df.empty else len(sbr_df)}")
        print(f"odds_updates_prepared={len(updates)}")
        for update in updates[:10]:
            print(
                "  update game_pk={game_pk} open=({open_home_ml}, {open_away_ml}) "
                "close=({close_home_ml}, {close_away_ml}) source={odds_source}".format(**update)
            )
        if skipped:
            print(f"odds_skipped_no_valid_sbr={len(skipped)}")
            for row in skipped[:10]:
                print(
                    "  skipped game_pk={game_pk} {away_team}@{home_team} {game_date}".format(**row)
                )
        if args.apply:
            updated = apply_odds_updates(updates)
            print(f"odds_rows_updated={updated}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to update DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
