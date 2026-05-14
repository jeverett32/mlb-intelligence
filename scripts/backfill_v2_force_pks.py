#!/usr/bin/env python3
"""
Backfill V2 predictions for a specific list of game_pks (even if already settled),
bypassing the `home_win IS NULL` guard in backfill_v2_skipped_predictions.

Usage:
  uv run python scripts/backfill_v2_force_pks.py --game-pks 824928,823710,...
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

load_dotenv()

import db as DB
from models.model_v2.predict import PredictV2Error, prepare_shared, predict_one


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game-pks", required=True, help="Comma-separated game_pk list")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pks = [int(x.strip()) for x in args.game_pks.split(",") if x.strip()]
    with DB.pooled_connection() as conn:
        df = pd.read_sql_query(
            "SELECT game_pk, game_date::text AS game_date FROM games WHERE game_pk = ANY(%s) ORDER BY game_time_utc",
            conn,
            params=(pks,),
        )
    if df.empty:
        print("No matching games.")
        return 1

    by_date: dict[str, list[str]] = defaultdict(list)
    for _, r in df.iterrows():
        by_date[str(r["game_date"])[:10]].append(str(int(r["game_pk"])))

    ok = fail = 0
    for gd, gpks in sorted(by_date.items()):
        try:
            shared = prepare_shared(gpks, gd)
        except PredictV2Error as e:
            print(f"{gd}: prepare_shared failed: {e}")
            fail += len(gpks)
            continue
        for pk in gpks:
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
