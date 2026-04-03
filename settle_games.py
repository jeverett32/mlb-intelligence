"""
settle_games.py — Fetch final scores from the MLB Stats API for any
pipeline-processed 2026 game that is still missing a result in the DB.

Designed to be run as a nightly cron job (e.g. 2 AM MT) after all games
have finished. Safe to run multiple times — already-settled rows are skipped.

Usage:
    uv run settle_games.py
"""

import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, ".")
import db as DB


def settle_completed_games(cutoff_hours: int = 4) -> int:
    """
    Settle all unsettled games that started more than `cutoff_hours` ago.
    Returns the number of games successfully settled.
    """
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=cutoff_hours)

    # Find pipeline-processed games with no result yet
    conn = DB.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.game_pk, g.game_date, g.home_team, g.away_team,
                       g.game_time_utc
                FROM games g
                WHERE g.season = 2026
                  AND g.home_win IS NULL
                  AND g.game_time_utc IS NOT NULL
                  AND g.game_time_utc::timestamptz < %s
                ORDER BY g.game_date, g.game_pk
            """,
                (cutoff,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("No unsettled games found.")
        return 0

    print(f"Checking {len(rows)} unsettled game(s)...")
    settled_count = 0

    for game_pk, game_date, home_team, away_team, _ in rows:
        try:
            resp = requests.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "gamePk": game_pk, "hydrate": "linescore"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            dates = data.get("dates", [])
            if not dates:
                print(f"  {away_team} @ {home_team}: no schedule data, skipping.")
                continue

            game_data = dates[0]["games"][0]
            abstract_state = game_data.get("status", {}).get("abstractGameState", "")
            if abstract_state != "Final":
                print(
                    f"  {away_team} @ {home_team}: status={abstract_state!r}, skipping."
                )
                continue

            linescore = game_data.get("linescore", {})
            teams_ls = linescore.get("teams", {})
            h_score = teams_ls.get("home", {}).get("runs")
            a_score = teams_ls.get("away", {}).get("runs")

            if h_score is None or a_score is None:
                print(f"  {away_team} @ {home_team}: no score in linescore, skipping.")
                continue

            home_win = bool(h_score > a_score)

            conn2 = DB.get_connection()
            try:
                with conn2.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE games
                        SET home_score = %s,
                            away_score = %s,
                            home_win   = %s,
                            updated_at = NOW()
                        WHERE game_pk = %s
                    """,
                        (int(h_score), int(a_score), home_win, int(game_pk)),
                    )
                conn2.commit()
            finally:
                conn2.close()

            result = "Home win" if home_win else "Away win"
            print(
                f"  Settled: {away_team} @ {home_team} — {a_score}-{h_score} ({result})"
            )
            settled_count += 1

        except Exception as e:
            print(
                f"  ERROR settling game_pk={game_pk} ({away_team} @ {home_team}): {e}"
            )

    print(f"\nDone. Settled {settled_count}/{len(rows)} game(s).")
    return settled_count


if __name__ == "__main__":
    settle_completed_games()
