#!/usr/bin/env python3
"""
Backfill V2 predictions for upcoming games that never got bets_v2.predicted_prob.

Typical skips: predict failed (e.g. odds present on prod `games` but absent on the
`games_v2` engineered row). `predict_one` now falls back to `games` for lines.

Examples:
  uv run python scripts/backfill_v2_skipped_predictions.py
  uv run python scripts/backfill_v2_skipped_predictions.py --dry-run
  uv run python scripts/backfill_v2_skipped_predictions.py --horizon-days 14
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

load_dotenv()

import db as DB
from config import ACTIVE_SEASON
from models.model_v2.feature_loader import invalidate_games_v2_frame_cache
from models.model_v2.ingest_features import ingest_features
from models.model_v2.predict import PredictV2Error, prepare_shared, predict_one


def fetch_skipped_rows(season: int, limit: int | None, horizon_days: int) -> pd.DataFrame:
    end_d = datetime.now(timezone.utc).date() + timedelta(days=int(horizon_days))
    sql = """
        SELECT g.game_pk, g.game_date::text AS game_date
        FROM games g
        WHERE g.season = %s
          AND g.home_win IS NULL
          AND COALESCE(g.extra->>'game_status', '') NOT IN ('postponed', 'cancelled')
          AND NOT EXISTS (
              SELECT 1 FROM bets_v2 b
              WHERE b.game_pk = g.game_pk AND b.predicted_prob IS NOT NULL
          )
          AND EXISTS (SELECT 1 FROM games_v2 v WHERE v.game_pk = g.game_pk)
          AND g.game_date::date <= %s
        ORDER BY g.game_date NULLS LAST, g.game_pk
    """
    params: list = [season, end_d.isoformat()]
    lim_sql = sql
    if limit is not None:
        lim_sql += " LIMIT %s"
        params.append(int(limit))
    with DB.pooled_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(lim_sql, params)
            rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["game_pk", "game_date"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, default=ACTIVE_SEASON)
    p.add_argument(
        "--horizon-days",
        type=int,
        default=10,
        help="Only games with game_date <= today+this many days (avoids far-future schedule without games_v2).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max games to process.")
    p.add_argument("--dry-run", action="store_true", help="Predict only; no DB writes.")
    p.add_argument(
        "--refresh-odds",
        action="store_true",
        help="Run odds-only fetch for selected rows (updates `games`), then invalidate V2 frame cache.",
    )
    p.add_argument(
        "--sync-v2-features",
        action="store_true",
        help="After odds refresh, run ingest_features for the min–max game_date window (heavy).",
    )
    args = p.parse_args()

    DB.init_v2_tables()
    df = fetch_skipped_rows(args.season, args.limit, args.horizon_days)
    if df.empty:
        print(f"No skipped upcoming games for season {args.season}.")
        return 0

    df["game_date"] = df["game_date"].astype(str).str[:10]
    by_date: dict[str, list[str]] = defaultdict(list)
    for _, r in df.iterrows():
        gd = str(r["game_date"])
        if not gd or gd.lower() == "nan":
            continue
        by_date[gd].append(str(int(r["game_pk"])))

    pks_all = [pk for pks in by_date.values() for pk in pks]
    print(f"Skipped games: {len(pks_all)} row(s) across {len(by_date)} date(s)")

    if args.refresh_odds:
        from fetch.fetch_data import refresh_odds_only

        with DB.pooled_connection() as conn:
            odds_df = pd.read_sql_query(
                "SELECT * FROM games WHERE game_pk = ANY(%s)",
                conn,
                params=([int(x) for x in pks_all],),
            )
        if odds_df.empty:
            print("  [warn] No rows in games for --refresh-odds; skipping fetch.")
        else:
            refresh_odds_only(odds_df, today_only=False)
        invalidate_games_v2_frame_cache()

    if args.sync_v2_features:
        dates = sorted(by_date.keys())
        start_d, end_d = dates[0], dates[-1]
        print(f"  ingest_features {args.season} {start_d} .. {end_d}")
        ingest_features(args.season, start_date=start_d, end_date=end_d)
        invalidate_games_v2_frame_cache()

    ok = fail = 0
    for gd in sorted(by_date.keys()):
        pks = by_date[gd]
        try:
            shared = prepare_shared(pks, gd)
        except PredictV2Error as e:
            print(f"{gd}: prepare_shared failed ({e}); skipping {len(pks)} pk(s)")
            fail += len(pks)
            continue
        for pk in pks:
            try:
                predict_one(pk, shared, dry_run=args.dry_run)
                ok += 1
                print(f"  ok pk={pk} date={gd}")
            except PredictV2Error as e:
                fail += 1
                print(f"  FAIL pk={pk} date={gd}: {e}")

    print(f"\nDone: success={ok} failed={fail} dry_run={args.dry_run}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
