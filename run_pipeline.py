"""
run_pipeline.py — MLB betting pipeline orchestrator.
Runs from the project root. Loops automatically, executing the full pipeline
5 minutes before each scheduled game.

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

EASTERN  = ZoneInfo("America/New_York")
MLB_CSV  = Path("data/mlb_2026.csv")   # local CSV fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(cmd: list[str]):
    """Run a pipeline step as a subprocess. Raises RuntimeError on non-zero exit."""
    # Replace bare 'uv' with full path so systemd can find it
    if cmd[0] == "uv":
        cmd = [UV] + cmd[1:]
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed (exit {result.returncode}): {' '.join(cmd)}")


def init_game_row(game: dict):
    """Insert a bare bet row for this game if it doesn't already exist."""
    game_pk   = int(game["game_pk"])
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
        return datetime.strptime(game_time_utc[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        date_str = str(game.get("game_date", ""))[:10]
        return datetime.strptime(date_str, "%Y-%m-%d").replace(hour=18, tzinfo=timezone.utc)


def get_next_game() -> dict | None:
    """
    Return the earliest upcoming 2026 game not yet in the bets table.
    Reads from DB (falls back to local CSV if DB unavailable).
    """
    try:
        df = DB.get_games_df(season=2026, upcoming_only=True)
        if df.empty:
            return None
        processed = DB.get_processed_game_pks()
        df = df[~df["game_pk"].astype(str).isin({str(p) for p in processed})]
    except Exception as e:
        print(f"  WARNING: DB unavailable ({e}), falling back to CSV.")
        if not MLB_CSV.exists():
            return None
        df = pd.read_csv(MLB_CSV, low_memory=False)
        df = df[df["home_win"].isna()].copy()

    if df.empty:
        return None

    sort_col = "game_time_et" if "game_time_et" in df.columns else "game_date"
    df = df.sort_values(sort_col)
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline_for_game(game_pk: str):
    """Execute all four pipeline steps for a specific game."""
    print(f"\n{'='*60}")
    print(f"PIPELINE START — game_pk={game_pk}")
    print(f"{'='*60}")

    try:
        run_step(["uv", "run", "fetch/fetch_data.py"])
        run_step(["uv", "run", "fetch/fetch_balance.py"])
        run_step(["uv", "run", "model/predict.py", "--game_pk", game_pk])
        run_step(["uv", "run", "bet/place_bet.py", "--game_pk", game_pk])
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Pipeline halted for this game. Will continue to next game.")
        return

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — game_pk={game_pk}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLB betting pipeline orchestrator.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--game_pk", type=str, help="Run once immediately for a specific game_pk")
    group.add_argument("--now", action="store_true", help="Run immediately for the next upcoming game")
    args = parser.parse_args()

    # ── One-shot: specific game ───────────────────────────────────────────
    if args.game_pk:
        try:
            game_df = DB.get_games_df(season=2026)
            rows = game_df[game_df["game_pk"].astype(str) == args.game_pk]
            if not rows.empty:
                init_game_row(rows.iloc[0].to_dict())
        except Exception:
            # If DB load fails, init_bet still works
            DB.init_bet(int(args.game_pk), "", "", "")
        run_pipeline_for_game(args.game_pk)
        return

    # ── Main loop ─────────────────────────────────────────────────────────
    print("MLB pipeline orchestrator started. Press Ctrl+C to stop.\n")

    while True:
        print("Refreshing MLB schedule...")
        try:
            subprocess.run([UV, "run", "fetch/fetch_data.py"], check=False)
        except Exception as e:
            print(f"  Schedule refresh failed: {e}")

        game = get_next_game()

        if game is None:
            print("No upcoming unprocessed games found. Rechecking in 30 minutes...")
            time.sleep(1800)
            continue

        start_utc  = get_game_start_utc(game)
        run_at_utc = start_utc - timedelta(minutes=5)
        now_utc    = datetime.now(timezone.utc)

        print(
            f"\nNext: {game.get('away_team')} @ {game.get('home_team')}"
            f"  |  Start: {start_utc.strftime('%Y-%m-%d %H:%M UTC')}"
            f"  |  Run at: {run_at_utc.strftime('%H:%M UTC')}"
        )

        # Skip games that started more than 3 hours ago — too late to bet
        if start_utc < now_utc - timedelta(hours=3):
            print(f"  Game already started {(now_utc - start_utc).seconds // 3600}h ago — skipping.")
            init_game_row(game)  # mark as processed so we don't loop on it
            continue

        if not args.now and run_at_utc > now_utc:
            sleep_secs = (run_at_utc - now_utc).total_seconds()
            print(f"Sleeping {sleep_secs / 60:.1f} minutes...")
            time.sleep(sleep_secs)

        init_game_row(game)
        run_pipeline_for_game(str(game["game_pk"]))

        if args.now:
            print("\n--now flag: exiting after one game.")
            return


if __name__ == "__main__":
    main()
