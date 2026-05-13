#!/usr/bin/env python3
"""
Backfill V2 predictions only for games that have ALREADY STARTED but still have
no bets_v2.predicted_prob (e.g. missed at T-10). Does not touch not-yet-started games.

Premature V2 rows for future first pitch can block the normal T-10 flow; use
--undo-future-v2 once to clear bets_v2 + paper_orders_v2 for games with
game_time_utc > NOW().

Examples:
  uv run python scripts/backfill_v2_skipped_predictions.py --undo-future-v2
  uv run python scripts/backfill_v2_skipped_predictions.py
  uv run python scripts/backfill_v2_skipped_predictions.py --dry-run
  uv run python scripts/backfill_v2_skipped_predictions.py --max-start-age-days 1
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


def _undo_future_counts(season: int | None) -> tuple[int, int]:
    """How many rows --undo-future-v2 would affect (dry-run)."""
    season_clause_paper = "AND g.season = %s" if season is not None else ""
    season_clause_bets = "AND g.season = %s" if season is not None else ""
    params_p = [int(season)] if season is not None else []
    params_b = [int(season)] if season is not None else []
    with DB.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) FROM paper_orders_v2 o
                JOIN games g ON g.game_pk = o.game_pk
                WHERE g.home_win IS NULL
                  AND g.game_time_utc IS NOT NULL
                  AND g.game_time_utc > NOW()
                  {season_clause_paper}
                """,
                params_p,
            )
            n_paper = int(cur.fetchone()[0])
            cur.execute(
                f"""
                SELECT COUNT(*) FROM bets_v2 b
                JOIN games g ON g.game_pk = b.game_pk
                WHERE g.home_win IS NULL
                  AND g.game_time_utc IS NOT NULL
                  AND g.game_time_utc > NOW()
                  AND (
                      b.predicted_prob IS NOT NULL
                      OR b.market_implied_prob IS NOT NULL
                      OR COALESCE(b.bet_frac, 0) <> 0
                      OR COALESCE(b.bet_side, '') NOT IN ('', 'none')
                  )
                  {season_clause_bets}
                """,
                params_b,
            )
            n_bets = int(cur.fetchone()[0])
    return n_paper, n_bets


def fetch_skipped_started_rows(
    season: int,
    limit: int | None,
    max_start_age_days: int,
) -> pd.DataFrame:
    """
    Games: first pitch in the past, within a recent window, unsettled, in games_v2,
    and still missing a stored V2 predicted_prob.
    """
    now = datetime.now(timezone.utc)
    oldest = now - timedelta(days=int(max_start_age_days))
    sql = """
        SELECT g.game_pk, g.game_date::text AS game_date
        FROM games g
        WHERE g.season = %s
          AND g.home_win IS NULL
          AND g.game_time_utc IS NOT NULL
          AND g.game_time_utc <= NOW()
          AND g.game_time_utc >= %s::timestamptz
          AND COALESCE(g.extra->>'game_status', '') NOT IN ('postponed', 'cancelled')
          AND NOT EXISTS (
              SELECT 1 FROM bets_v2 b
              WHERE b.game_pk = g.game_pk AND b.predicted_prob IS NOT NULL
          )
          AND EXISTS (SELECT 1 FROM games_v2 v WHERE v.game_pk = g.game_pk)
        ORDER BY g.game_time_utc ASC, g.game_pk
    """
    params: list = [season, oldest.isoformat()]
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
        "--max-start-age-days",
        type=int,
        default=2,
        help="Only games whose first pitch was at most this many days ago (default 2).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max games to process.")
    p.add_argument("--dry-run", action="store_true", help="Predict only; no DB writes.")
    p.add_argument(
        "--undo-future-v2",
        action="store_true",
        help="Clear bets_v2 + paper_orders_v2 for games that have not started yet (game_time_utc > NOW).",
    )
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

    if args.undo_future_v2:
        np, nb = _undo_future_counts(args.season)
        print(
            f"[undo-future-v2] would affect paper_orders={np} bets_v2={nb} "
            f"(season filter={args.season})",
            flush=True,
        )
        if not args.dry_run:
            pdel, bup = DB.clear_v2_prediction_rows_for_not_started_games(args.season)
            print(
                f"[undo-future-v2] done paper_deleted={pdel} bets_v2_updated={bup}",
                flush=True,
            )
            invalidate_games_v2_frame_cache()

    df = fetch_skipped_started_rows(args.season, args.limit, args.max_start_age_days)
    if df.empty:
        print(f"No skipped started games for season {args.season}.", flush=True)
        return 0

    df["game_date"] = df["game_date"].astype(str).str[:10]
    by_date: dict[str, list[str]] = defaultdict(list)
    for _, r in df.iterrows():
        gd = str(r["game_date"])
        if not gd or gd.lower() == "nan":
            continue
        by_date[gd].append(str(int(r["game_pk"])))

    pks_all = [pk for pks in by_date.values() for pk in pks]
    print(
        f"Started-but-missing-V2-prob: {len(pks_all)} game(s) across {len(by_date)} date(s)",
        flush=True,
    )

    if args.refresh_odds:
        from fetch.fetch_data import refresh_odds_only

        with DB.pooled_connection() as conn:
            odds_df = pd.read_sql_query(
                "SELECT * FROM games WHERE game_pk = ANY(%s)",
                conn,
                params=([int(x) for x in pks_all],),
            )
        if odds_df.empty:
            print("  [warn] No rows in games for --refresh-odds; skipping fetch.", flush=True)
        else:
            refresh_odds_only(odds_df, today_only=False)
        invalidate_games_v2_frame_cache()

    if args.sync_v2_features:
        dates = sorted(by_date.keys())
        start_d, end_d = dates[0], dates[-1]
        print(f"  ingest_features {args.season} {start_d} .. {end_d}", flush=True)
        ingest_features(args.season, start_date=start_d, end_date=end_d)
        invalidate_games_v2_frame_cache()

    ok = fail = 0
    for gd in sorted(by_date.keys()):
        pks = by_date[gd]
        try:
            shared = prepare_shared(pks, gd)
        except PredictV2Error as e:
            print(f"{gd}: prepare_shared failed ({e}); skipping {len(pks)} pk(s)", flush=True)
            fail += len(pks)
            continue
        for pk in pks:
            try:
                predict_one(pk, shared, dry_run=args.dry_run)
                ok += 1
                print(f"  ok pk={pk} date={gd}", flush=True)
            except PredictV2Error as e:
                fail += 1
                print(f"  FAIL pk={pk} date={gd}: {e}", flush=True)

    print(f"\nDone: success={ok} failed={fail} dry_run={args.dry_run}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
