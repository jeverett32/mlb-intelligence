#!/usr/bin/env python3
"""
Backfill explainability payloads for historical bet signals.

Runs from project root:
  uv run scripts/backfill_explanations.py --since 2026-03-01 --limit 200
  uv run scripts/backfill_explanations.py --limit 50 --dry-run
"""

import argparse
from collections import defaultdict

import pandas as pd

import db as DB
from model import predict as P


def _load_candidates(since: str | None, limit: int, only_signals: bool) -> pd.DataFrame:
    """
    Return bets needing explanations.
    We intentionally use a SQL query instead of DB.get_all_bets() to avoid loading full history.
    """
    where = ["predicted_prob IS NOT NULL", "explanation IS NULL"]
    if only_signals:
        where += ["bet_side IN ('home','away')", "COALESCE(bet_frac,0) > 0"]
    if since:
        where.append("game_date >= %s")
        params = [since]
    else:
        params = []
    sql = f"""
        SELECT game_pk, game_date, home_team, away_team,
               predicted_prob, market_implied_prob, edge, bet_side, bet_frac
        FROM bets
        WHERE {' AND '.join(where)}
        ORDER BY game_date ASC NULLS LAST, game_pk ASC
        LIMIT %s
    """
    params.append(int(limit))

    with DB.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill bets.explanation for historical signals.")
    ap.add_argument("--since", type=str, default=None, help="Only backfill bets on/after YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=200, help="Max bets to backfill this run")
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write to DB")
    ap.add_argument("--include-non-signals", action="store_true", help="Also include pass/no-bet rows")
    args = ap.parse_args()

    DB.init_bets_explainability()

    only_signals = not args.include_non_signals
    df = _load_candidates(args.since, max(1, int(args.limit)), only_signals)
    if df.empty:
        print("No bets need backfill.")
        return 0

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    by_date: dict[str, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        d = str(row.get("game_date") or "")[:10]
        if not d or d == "NaT":
            continue
        by_date[d].append(row.to_dict())

    total = 0
    wrote = 0
    for game_date, rows in sorted(by_date.items()):
        pks = [str(r["game_pk"]) for r in rows]
        print(f"\n{game_date}: preparing shared artifacts for {len(pks)} game(s)")
        try:
            shared = P.prepare_shared(pks, game_date)
        except Exception as e:
            print(f"  ERROR: shared prep failed for {game_date}: {e}")
            continue

        for r in rows:
            total += 1
            game_pk = str(r["game_pk"])
            bet_side = str(r.get("bet_side") or "none")
            bet_frac = float(r.get("bet_frac") or 0.0)
            pred = r.get("predicted_prob")
            mkt = r.get("market_implied_prob")
            edge = r.get("edge")
            try:
                explanation = P.explain_one(
                    game_pk,
                    shared,
                    predicted_prob_home=float(pred) if pred is not None else None,
                    market_implied_prob_home=float(mkt) if mkt is not None else None,
                    edge=float(edge) if edge is not None else None,
                    bet_side=bet_side,
                    bet_frac=bet_frac,
                    recomputed=True,
                    recomputed_reason="Backfilled later using current pipeline code/data",
                    write_db=(not args.dry_run),
                )
                if explanation is None:
                    print(f"  skip game_pk={game_pk} (no signal or missing artifacts)")
                    continue
                wrote += 1
                print(f"  ok game_pk={game_pk} ({bet_side} {bet_frac:.4f})")
            except Exception as e:
                print(f"  ERROR game_pk={game_pk}: {e}")

    print(f"\nDone. candidates={len(df)} processed={total} explained={wrote} dry_run={bool(args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

