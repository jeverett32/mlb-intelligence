"""
run_pipeline.py — MLB betting pipeline orchestrator.
Runs from the project root. Loops automatically, executing the full pipeline
15 minutes before each scheduled game window.

Games starting within 30 minutes of each other are treated as a batch:
  - fetch_data + fetch_balance run ONCE for the whole batch
  - predict + place_bet run IN PARALLEL for each game

Usage:
    uv run run_pipeline.py                     # loop forever (normal operation)
    uv run run_pipeline.py --game_pk 12345     # run once for a specific game right now
    uv run run_pipeline.py --now               # run immediately for the next upcoming game
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure uv is findable when launched from systemd (no user PATH)
_UV = Path(os.environ.get("HOME", "/root")) / ".local/bin/uv"
UV = str(_UV) if _UV.exists() else "uv"

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import db as DB

EASTERN = ZoneInfo("America/New_York")
MLB_CSV = Path("data/mlb_2026.csv")  # local CSV fallback

# How many minutes before game start to trigger the pipeline
LEAD_MINUTES = 15

# Games starting within this window of the earliest game are batched together
BATCH_WINDOW_MINUTES = 30

# Don't process games that start less than this many minutes from now
MIN_GAME_TIME_MINUTES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_step(cmd: list[str]):
    """Run a pipeline step as a subprocess. Raises RuntimeError on non-zero exit."""
    if cmd[0] == "uv":
        cmd = [UV] + cmd[1:]
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed (exit {result.returncode}): {' '.join(cmd)}")


def init_game_row(game: dict):
    """Insert a bare bet row for this game if it doesn't already exist."""
    game_pk = int(game["game_pk"])
    game_date = str(game.get("game_date", ""))[:10]
    home_team = str(game.get("home_team", ""))
    away_team = str(game.get("away_team", ""))
    DB.init_bet(game_pk, game_date, home_team, away_team)
    print(f"  Initialized: {away_team} @ {home_team} ({game_date})")


def get_game_start_utc(game: dict) -> datetime:
    """
    Parse game_time_utc (format: "2026-04-01 17:05") and return a UTC datetime.
    Falls back to 6 PM UTC on game_date if parsing fails.
    """
    game_time_utc = str(game.get("game_time_utc") or "")
    try:
        return datetime.strptime(game_time_utc[:16], "%Y-%m-%d %H:%M").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        date_str = str(game.get("game_date", ""))[:10]
        return datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=18, tzinfo=timezone.utc
        )


def get_all_upcoming_unprocessed() -> list[dict]:
    """
    Return all upcoming 2026 games not yet in the bets table, sorted by start time.
    """
    try:
        df = DB.get_games_df(season=2026, upcoming_only=True)
        if df.empty:
            return []
        processed = DB.get_processed_game_pks()
        df = df[~df["game_pk"].astype(str).isin({str(p) for p in processed})]
    except Exception as e:
        print(f"  WARNING: DB unavailable ({e}), falling back to CSV.")
        if not MLB_CSV.exists():
            return []
        df = pd.read_csv(MLB_CSV, low_memory=False)
        df = df[df["home_win"].isna()].copy()

    if df.empty:
        return []

    sort_col = "game_time_utc" if "game_time_utc" in df.columns else "game_date"
    df = df.sort_values(sort_col)
    return [row.to_dict() for _, row in df.iterrows()]


def get_next_batch() -> tuple[list[dict], datetime | None]:
    """
    Find the next game(s) to process.
    Returns (batch, run_at_utc) where batch is a list of games starting
    within BATCH_WINDOW_MINUTES of each other, and run_at_utc is when to
    kick off the pipeline (LEAD_MINUTES before the earliest game).
    Returns ([], None) if nothing to process.
    """
    now_utc = datetime.now(timezone.utc)
    all_games = get_all_upcoming_unprocessed()

    if not all_games:
        return [], None

    # Find the earliest game that hasn't already started too long ago
    first_game = None
    for game in all_games:
        start = get_game_start_utc(game)
        if start >= now_utc - timedelta(hours=3):
            first_game = game
            break

    if first_game is None:
        return [], None

    first_start = get_game_start_utc(first_game)
    run_at_utc = first_start - timedelta(minutes=LEAD_MINUTES)
    window_end = first_start + timedelta(minutes=BATCH_WINDOW_MINUTES)

    # Collect all games in the batch window
    batch = []
    for game in all_games:
        start = get_game_start_utc(game)
        # Only include games starting within MIN_GAME_TIME_MINUTES to BATCH_WINDOW_MINUTES from now
        if (
            start >= now_utc + timedelta(minutes=MIN_GAME_TIME_MINUTES)
            and start <= window_end
        ):
            batch.append(game)

    return batch, run_at_utc


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_predict_and_bet(game_pk: str):
    """Run predict + place_bet for one game (fetch already done by caller)."""
    try:
        run_step(["uv", "run", "model/predict.py", "--game_pk", game_pk])
        run_step(["uv", "run", "bet/place_bet.py", "--game_pk", game_pk])
        print(f"\n  game_pk={game_pk} complete.")
    except RuntimeError as e:
        print(f"\n  ERROR for game_pk={game_pk}: {e}")


def run_batch(games: list[dict]):
    """
    Full pipeline for a batch of games:
      1. fetch_data + fetch_balance once (shared)
      2. predict + place_bet in parallel for each game
    """
    labels = ", ".join(f"{g.get('away_team')}@{g.get('home_team')}" for g in games)
    print(f"\n{'=' * 60}")
    print(f"BATCH START — {len(games)} game(s): {labels}")
    print(f"{'=' * 60}")

    # Shared fetch steps — run once for the whole batch
    try:
        run_step(["uv", "run", "fetch/fetch_data.py"])
        run_step(["uv", "run", "fetch/fetch_balance.py"])
    except RuntimeError as e:
        print(f"\nERROR in shared fetch step: {e}")
        print("Aborting batch.")
        return

    # Init all bet rows before spawning threads
    for game in games:
        init_game_row(game)

    # Predict + bet in parallel
    pks = [str(g["game_pk"]) for g in games]
    if len(pks) == 1:
        run_predict_and_bet(pks[0])
    else:
        print(f"\n  Running predict+bet for {len(pks)} games in parallel...")
        with ThreadPoolExecutor(max_workers=len(pks)) as executor:
            futures = {executor.submit(run_predict_and_bet, pk): pk for pk in pks}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Thread error for game_pk={futures[future]}: {e}")

    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE — {len(games)} game(s)")
    print(f"{'=' * 60}")


def run_pipeline_for_game(game_pk: str):
    """Single-game pipeline (used for --game_pk one-shot mode)."""
    print(f"\n{'=' * 60}")
    print(f"PIPELINE START — game_pk={game_pk}")
    print(f"{'=' * 60}")
    try:
        run_step(["uv", "run", "fetch/fetch_data.py"])
        run_step(["uv", "run", "fetch/fetch_balance.py"])
        run_step(["uv", "run", "model/predict.py", "--game_pk", game_pk])
        run_step(["uv", "run", "bet/place_bet.py", "--game_pk", game_pk])
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — game_pk={game_pk}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def settle_completed_games():
    """
    Fetch final scores from the MLB Stats API for any 2026 game that:
      - is in the bets table (was processed by the pipeline)
      - still has home_win = NULL in the games table (not yet settled)
      - started more than 4 hours ago (safely complete)
    Writes home_score, away_score, home_win back to the games table.
    """
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=4)

    try:
        conn = DB.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT g.game_pk, g.game_date, g.home_team, g.away_team,
                           g.game_time_utc
                    FROM games g
                    JOIN bets b ON g.game_pk = b.game_pk
                    WHERE g.season = 2026
                      AND g.home_win IS NULL
                      AND g.game_time_utc IS NOT NULL
                      AND g.game_time_utc::timestamptz < %s
                """,
                    (cutoff,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"  Settlement check failed: {e}")
        return

    if not rows:
        return

    print(f"  Settling {len(rows)} completed game(s)...")
    import requests as _requests

    for game_pk, game_date, home_team, away_team, _ in rows:
        try:
            resp = _requests.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "gamePk": game_pk, "hydrate": "linescore"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            dates = data.get("dates", [])
            if not dates:
                continue

            game_data = dates[0]["games"][0]
            abstract_state = game_data.get("status", {}).get("abstractGameState", "")
            if abstract_state != "Final":
                continue

            linescore = game_data.get("linescore", {})
            teams_ls = linescore.get("teams", {})
            h_score = teams_ls.get("home", {}).get("runs")
            a_score = teams_ls.get("away", {}).get("runs")

            if h_score is None or a_score is None:
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
            print(
                f"    Settled: {away_team} @ {home_team} — "
                f"{a_score}-{h_score} ({'Home' if home_win else 'Away'} win)"
            )
        except Exception as e:
            print(f"    Could not settle game_pk={game_pk}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MLB betting pipeline orchestrator.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--game_pk", type=str, help="Run once immediately for a specific game_pk"
    )
    group.add_argument(
        "--now", action="store_true", help="Run immediately for the next batch"
    )
    args = parser.parse_args()

    # ── One-shot: specific game ───────────────────────────────────────────
    if args.game_pk:
        try:
            game_df = DB.get_games_df(season=2026)
            rows = game_df[game_df["game_pk"].astype(str) == args.game_pk]
            if not rows.empty:
                init_game_row(rows.iloc[0].to_dict())
        except Exception:
            DB.init_bet(int(args.game_pk), "", "", "")
        run_pipeline_for_game(args.game_pk)
        return

    # ── Main loop ─────────────────────────────────────────────────────────
    print("MLB pipeline orchestrator started. Press Ctrl+C to stop.\n")

    while True:
        settle_completed_games()

        print("Refreshing MLB schedule...")
        try:
            subprocess.run([UV, "run", "fetch/fetch_data.py"], check=False)
        except Exception as e:
            print(f"  Schedule refresh failed: {e}")

        batch, run_at_utc = get_next_batch()
        now_utc = datetime.now(timezone.utc)

        if not batch:
            print("No upcoming unprocessed games found. Rechecking in 30 minutes...")
            time.sleep(1800)
            continue

        first_start = get_game_start_utc(batch[0])
        labels = ", ".join(f"{g.get('away_team')}@{g.get('home_team')}" for g in batch)
        print(
            f"\nNext batch ({len(batch)} game(s)): {labels}"
            f"\n  First start: {first_start.strftime('%Y-%m-%d %H:%M UTC')}"
            f"  |  Run at: {run_at_utc.strftime('%H:%M UTC')}"
        )

        # Skip games that started more than 3 hours ago
        if first_start < now_utc - timedelta(hours=3):
            print(
                f"  Batch already started too long ago — marking as processed and skipping."
            )
            for game in batch:
                init_game_row(game)
            continue

        if not args.now and run_at_utc > now_utc:
            sleep_secs = (run_at_utc - now_utc).total_seconds()
            print(
                f"Sleeping {sleep_secs / 60:.1f} minutes until {LEAD_MINUTES} min before first pitch..."
            )
            time.sleep(sleep_secs)

        run_batch(batch)

        if args.now:
            print("\n--now flag: exiting after one batch.")
            return


if __name__ == "__main__":
    main()
