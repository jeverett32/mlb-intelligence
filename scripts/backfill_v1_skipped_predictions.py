#!/usr/bin/env python3
"""
Backfill V1 predictions and paper bets for games that have ALREADY STARTED but
have no bets row (e.g. missed at T-10 due to pipeline outage).

Mirrors scripts/backfill_v1_skipped_predictions's behavior for V1:
  1. Predict (writes bets row).
  2. Run paper-only execution per approved user (writes paper_orders row).
Live execution is forced OFF for safety — settled markets are inappropriate
for live order placement.

Examples:
  uv run python scripts/backfill_v1_skipped_predictions.py
  uv run python scripts/backfill_v1_skipped_predictions.py --dry-run
  uv run python scripts/backfill_v1_skipped_predictions.py --max-start-age-days 1
  uv run python scripts/backfill_v1_skipped_predictions.py --game-pks 824928,823710
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
from models.model_v1 import predict as PREDICT
from bet import place_bet as PLACE_BET


def fetch_skipped_started_rows(
    season: int,
    limit: int | None,
    max_start_age_days: int,
    game_pks: list[int] | None,
) -> pd.DataFrame:
    """Started games missing a V1 bets row."""
    now = datetime.now(timezone.utc)
    oldest = now - timedelta(days=int(max_start_age_days))
    sql = """
        SELECT g.game_pk, g.game_date::text AS game_date
        FROM games g
        WHERE g.season = %s
          AND g.game_time_utc IS NOT NULL
          AND g.game_time_utc <= NOW()
          AND g.game_time_utc >= %s::timestamptz
          AND COALESCE(g.extra->>'game_status', '') NOT IN ('postponed', 'cancelled')
          AND NOT EXISTS (
              SELECT 1 FROM bets b WHERE b.game_pk = g.game_pk
          )
    """
    params: list = [season, oldest.isoformat()]
    if game_pks:
        sql += " AND g.game_pk = ANY(%s)"
        params.append([int(x) for x in game_pks])
    sql += " ORDER BY g.game_time_utc ASC, g.game_pk"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))
    with DB.pooled_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["game_pk", "game_date"])


def _paper_only_place(game_pk: str) -> list[dict]:
    """Run place_user_bet but coerce every user to paper_only for safety."""
    original = PLACE_BET._execution_mode_for_email
    PLACE_BET._execution_mode_for_email = lambda email: "paper_only"
    try:
        results: list[dict] = []
        for user in DB.list_approved_users_with_accounts():
            try:
                results.append(PLACE_BET.place_user_bet(user["email"], game_pk))
            except Exception as exc:
                results.append(
                    {
                        "game_pk": str(game_pk),
                        "email": user["email"],
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results
    finally:
        PLACE_BET._execution_mode_for_email = original


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, default=ACTIVE_SEASON)
    p.add_argument(
        "--max-start-age-days",
        type=int,
        default=2,
        help="Only games whose first pitch was at most this many days ago (default 2).",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Predict only; no DB writes.")
    p.add_argument(
        "--game-pks",
        type=str,
        default=None,
        help="Comma-separated game_pk list to restrict backfill to.",
    )
    args = p.parse_args()

    explicit_pks = None
    if args.game_pks:
        explicit_pks = [int(x.strip()) for x in args.game_pks.split(",") if x.strip()]

    df = fetch_skipped_started_rows(args.season, args.limit, args.max_start_age_days, explicit_pks)
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
        f"Started-but-missing-V1-bets: {len(pks_all)} game(s) across {len(by_date)} date(s)",
        flush=True,
    )
    for gd, pks in sorted(by_date.items()):
        print(f"  {gd}: {pks}", flush=True)

    ok = fail = 0
    paper_ok = 0
    for gd in sorted(by_date.keys()):
        pks = by_date[gd]
        try:
            shared = PREDICT.prepare_shared(pks, gd)
        except PREDICT.PredictError as e:
            print(f"{gd}: prepare_shared failed ({e}); skipping {len(pks)} pk(s)", flush=True)
            fail += len(pks)
            continue
        if not args.dry_run:
            with DB.pooled_connection() as conn:
                meta = pd.read_sql_query(
                    "SELECT game_pk, game_date::text AS game_date, home_team, away_team "
                    "FROM games WHERE game_pk = ANY(%s)",
                    conn,
                    params=([int(x) for x in pks],),
                )
            for _, mrow in meta.iterrows():
                DB.init_bet(int(mrow["game_pk"]), str(mrow["game_date"])[:10],
                            str(mrow["home_team"]), str(mrow["away_team"]))
        for pk in pks:
            try:
                PREDICT.predict_one(pk, shared, dry_run=args.dry_run)
                ok += 1
                print(f"  predict ok pk={pk} date={gd}", flush=True)
            except PREDICT.PredictError as e:
                fail += 1
                print(f"  predict FAIL pk={pk} date={gd}: {e}", flush=True)
                continue

            if args.dry_run:
                continue

            try:
                results = _paper_only_place(pk)
                successes = sum(
                    1 for r in results if str(r.get("status") or "").startswith(("dry_run", "filled", "skipped"))
                )
                paper_ok += successes
                statuses = ",".join(f"{r.get('email','?')}={r.get('status','?')}" for r in results)
                print(f"  paper pk={pk} -> {statuses}", flush=True)
            except Exception as e:
                print(f"  paper FAIL pk={pk}: {e}", flush=True)

    print(f"\nDone: predict_success={ok} predict_failed={fail} paper_user_writes={paper_ok} dry_run={args.dry_run}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
