#!/usr/bin/env python3
"""Backfill starter FIP features from MLB StatsAPI game logs.

Dry-run is the default. Pass --apply to update games.extra with home_sp_fip and
away_sp_fip. Historical values use pitcher starts before each game date.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
FETCH_DIR = ROOT / "fetch"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FETCH_DIR))

import db as DB  # noqa: E402
from fetch_data import (  # noqa: E402
    _fetch_league_fip_constant,
    _fetch_pitcher_game_log,
    _fip_from_components,
)


def _fetch_candidate_games(season: int | None, only_missing: bool, limit: int | None) -> pd.DataFrame:
    conditions = ["home_starter_id IS NOT NULL", "away_starter_id IS NOT NULL"]
    params: list[Any] = []
    if season is not None:
        conditions.append("season = %s")
        params.append(season)
    if only_missing:
        conditions.append(
            """
            (
                extra->>'home_sp_fip' IS NULL
                OR extra->>'away_sp_fip' IS NULL
            )
            """
        )
    sql = f"""
        SELECT game_pk, game_date::date AS game_date, season,
               home_team, away_team, home_starter_id, away_starter_id,
               extra->>'home_sp_fip' AS home_sp_fip,
               extra->>'away_sp_fip' AS away_sp_fip
        FROM games
        WHERE {' AND '.join(conditions)}
        ORDER BY season, game_date, game_pk
    """
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return pd.DataFrame(rows)


def _fetch_logs(pairs: list[tuple[int, int]], max_workers: int) -> pd.DataFrame:
    cache_path = ROOT / "data" / "cache" / "starter_fip_game_logs.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    cached_pairs = (
        set(zip(cached["pitcher_id"].astype(int), cached["season"].astype(int)))
        if not cached.empty
        else set()
    )
    missing_pairs = [pair for pair in pairs if pair not in cached_pairs]

    rows = []
    if missing_pairs:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_pitcher_game_log, pid, season): (pid, season) for pid, season in missing_pairs}
            for i, fut in enumerate(as_completed(futs), start=1):
                rows.extend(fut.result())
                if i % 100 == 0:
                    print(f"  fetched_logs={i}/{len(missing_pairs)}", flush=True)
        new_logs = pd.DataFrame(rows)
        if not new_logs.empty:
            cached = pd.concat([cached, new_logs], ignore_index=True).drop_duplicates(
                subset=["pitcher_id", "season", "game_date"]
            )
            cached.to_parquet(cache_path)

    if cached.empty:
        return pd.DataFrame()
    pair_set = set(pairs)
    mask = list(zip(cached["pitcher_id"].astype(int), cached["season"].astype(int)))
    logs = cached[[pair in pair_set for pair in mask]].copy()
    if logs.empty:
        return pd.DataFrame()
    logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.date
    return logs


def _build_prior_fip_lookup(logs: pd.DataFrame) -> dict[tuple[int, int, Any], float]:
    if logs.empty:
        return {}
    lookup = {}
    constants: dict[int, float] = {}
    for (pid, season), grp in logs.groupby(["pitcher_id", "season"]):
        season = int(season)
        if season not in constants:
            try:
                constants[season] = _fetch_league_fip_constant(season)
            except Exception:
                constants[season] = np.nan
        constant = constants[season]
        grp = grp.sort_values("game_date").copy()
        grp["cum_ip"] = grp["ip"].cumsum().shift(1)
        grp["cum_hr"] = grp["hr"].cumsum().shift(1)
        grp["cum_bb"] = grp["bb"].cumsum().shift(1)
        grp["cum_hbp"] = grp["hbp"].cumsum().shift(1)
        grp["cum_so"] = grp["so"].cumsum().shift(1)
        for _, row in grp.iterrows():
            fip = _fip_from_components(
                row["cum_ip"], row["cum_hr"], row["cum_bb"], row["cum_hbp"], row["cum_so"], constant
            )
            if pd.notna(fip):
                lookup[(int(pid), season, row["game_date"])] = float(fip)
    return lookup


def _latest_prior_fip(lookup_dates: dict[tuple[int, int], list[tuple[Any, float]]], pid: int, season: int, game_date: Any) -> float | None:
    values = lookup_dates.get((pid, season), [])
    best = None
    for log_date, fip in values:
        if log_date <= game_date:
            best = fip
        else:
            break
    return best


def _build_updates(games: pd.DataFrame, lookup: dict[tuple[int, int, Any], float]) -> list[dict[str, Any]]:
    lookup_dates: dict[tuple[int, int], list[tuple[Any, float]]] = {}
    for (pid, season, game_date), fip in lookup.items():
        lookup_dates.setdefault((pid, season), []).append((game_date, fip))
    for key in lookup_dates:
        lookup_dates[key].sort()

    updates = []
    for _, row in games.iterrows():
        game_date = pd.to_datetime(row["game_date"]).date()
        season = int(row["season"])
        home_pid = int(row["home_starter_id"])
        away_pid = int(row["away_starter_id"])
        home_fip = _latest_prior_fip(lookup_dates, home_pid, season, game_date)
        away_fip = _latest_prior_fip(lookup_dates, away_pid, season, game_date)
        update = {"game_pk": int(row["game_pk"])}
        if home_fip is not None and pd.isna(pd.to_numeric(row.get("home_sp_fip"), errors="coerce")):
            update["home_sp_fip"] = home_fip
        if away_fip is not None and pd.isna(pd.to_numeric(row.get("away_sp_fip"), errors="coerce")):
            update["away_sp_fip"] = away_fip
        if "home_sp_fip" in update or "away_sp_fip" in update:
            updates.append(update)
    return updates


def _apply_updates(updates: list[dict[str, Any]]) -> int:
    if not updates:
        return 0
    conn = DB.get_connection()
    try:
        with conn.cursor() as cur:
            count = 0
            for update in updates:
                expr = "COALESCE(extra, '{}'::jsonb)"
                params: list[Any] = []
                for key in ("home_sp_fip", "away_sp_fip"):
                    if key not in update:
                        continue
                    expr = f"jsonb_set({expr}, %s, to_jsonb(%s::double precision), true)"
                    params.extend([[key], float(update[key])])
                if not params:
                    continue
                sql = f"UPDATE games SET extra = {expr}, updated_at = NOW() WHERE game_pk = %s"
                params.append(update["game_pk"])
                cur.execute(sql, params)
                count += int(cur.rowcount or 0)
        conn.commit()
    finally:
        conn.close()
    return count


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write updates to DB")
    parser.add_argument("--season", type=int, default=None, help="restrict to one season")
    parser.add_argument("--all", action="store_true", help="recompute even rows that already have FIP")
    parser.add_argument("--limit", type=int, default=None, help="limit game rows for test runs")
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    games = _fetch_candidate_games(args.season, only_missing=not args.all, limit=args.limit)
    print(f"candidate_games={len(games)}", flush=True)
    if games.empty:
        return 0

    pairs = sorted(
        {
            (int(row[f"{side}_starter_id"]), int(row["season"]))
            for _, row in games.iterrows()
            for side in ("home", "away")
            if pd.notna(row[f"{side}_starter_id"])
        }
    )
    print(f"pitcher_season_pairs={len(pairs)}", flush=True)
    logs = _fetch_logs(pairs, args.max_workers)
    print(f"starter_log_rows={len(logs)}", flush=True)
    lookup = _build_prior_fip_lookup(logs)
    updates = _build_updates(games, lookup)
    print(f"updates_prepared={len(updates)}", flush=True)
    for update in updates[:10]:
        print(update, flush=True)
    if len(updates) > 10:
        print(f"... {len(updates) - 10} more", flush=True)

    if not args.apply:
        print("Dry run only. Re-run with --apply to update DB.", flush=True)
        return 0

    updated = _apply_updates(updates)
    print(f"rows_updated={updated}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
