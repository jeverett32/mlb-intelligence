"""Backfill v2 paper predictions/orders for settled games in a season.

Walks forward through unique game_dates, fitting LGBM with leakage-safe
cutoff per date, predicts every settled game, and writes to bets_v2 +
paper_orders_v2. Settles paper_orders_v2 results/pnl at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import db as DB
from models.model_v2.predict import prepare_shared, predict_one, PredictV2Error


def get_settled_dates(season: int) -> list[tuple[str, list[str]]]:
    sql = """
        SELECT game_date, game_pk
        FROM games
        WHERE season = %s
          AND home_win IS NOT NULL
          AND close_home_ml IS NOT NULL
          AND close_away_ml IS NOT NULL
        ORDER BY game_date, game_pk
    """
    with DB.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (season,))
            rows = cur.fetchall()
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=["game_date", "game_pk"])
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return [
        (date, [str(pk) for pk in grp["game_pk"].tolist()])
        for date, grp in df.groupby("game_date")
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill v2 paper predictions/orders.")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--limit-dates", type=int, default=None,
                   help="Process only first N dates (testing).")
    p.add_argument("--dry-run", action="store_true",
                   help="Predict but skip DB writes.")
    args = p.parse_args()

    DB.init_v2_tables()
    dates = get_settled_dates(args.season)
    if args.limit_dates:
        dates = dates[: args.limit_dates]
    if not dates:
        print(f"No settled games for season {args.season}")
        return 0

    print(f"Backfilling {len(dates)} dates for season {args.season} "
          f"(dry_run={args.dry_run})")
    t_start = time.time()
    total_pred = total_bet = total_fail = 0

    for i, (date, pks) in enumerate(dates, 1):
        t0 = time.time()
        try:
            shared = prepare_shared(pks, date)
        except PredictV2Error as e:
            print(f"[{i}/{len(dates)}] {date}: prepare_shared failed: {e}")
            total_fail += len(pks)
            continue

        bets_today = 0
        for pk in pks:
            try:
                res = predict_one(pk, shared, dry_run=args.dry_run)
                total_pred += 1
                if res.get("bet_side") not in (None, "none"):
                    bets_today += 1
            except PredictV2Error as e:
                print(f"  pk={pk}: {e}")
                total_fail += 1

        total_bet += bets_today
        print(f"[{i}/{len(dates)}] {date} games={len(pks)} "
              f"bets={bets_today} elapsed={time.time()-t0:.1f}s")

    print(f"\nDone: predictions={total_pred} bets_placed={total_bet} "
          f"failures={total_fail} total_elapsed={time.time()-t_start:.1f}s")

    if not args.dry_run:
        print("\nSettling paper_orders_v2 results...")
        try:
            n = DB.backfill_paper_order_v2_results()
            print(f"Settled {n} paper orders.")
        except Exception as e:
            print(f"Settlement error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
