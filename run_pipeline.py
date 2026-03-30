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
import csv
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

EASTERN    = ZoneInfo("America/New_York")
DATA_DIR   = Path("data")
GAMES_CSV  = DATA_DIR / "games.csv"
MLB_CSV    = DATA_DIR / "mlb_2026.csv"

GAMES_CSV_COLS = [
    "game_pk", "game_date", "home_team", "away_team",
    "predicted_prob", "edge", "bet_side", "bet_frac",
    "market_implied_prob", "kalshi_order_id", "bet_dollars", "n_contracts", "result",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(cmd: list[str]):
    """Run a pipeline step as a subprocess. Raises RuntimeError on non-zero exit."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed (exit {result.returncode}): {' '.join(cmd)}")


def ensure_games_csv():
    """Create data/games.csv with headers if it doesn't exist."""
    if not GAMES_CSV.exists():
        GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(GAMES_CSV, "w", newline="") as f:
            csv.writer(f).writerow(GAMES_CSV_COLS)
        print(f"Created {GAMES_CSV}")


def game_already_processed(game_pk: str) -> bool:
    if not GAMES_CSV.exists():
        return False
    games = pd.read_csv(GAMES_CSV, dtype=str)
    return str(game_pk) in games["game_pk"].astype(str).values


def init_game_row(game: dict):
    """Add a row for this game to games.csv if it isn't already there."""
    game_pk = str(game["game_pk"])
    if game_already_processed(game_pk):
        return

    new_row = {col: "" for col in GAMES_CSV_COLS}
    new_row["game_pk"]   = game_pk
    new_row["game_date"] = str(game.get("game_date", ""))[:10]
    new_row["home_team"] = str(game.get("home_team", ""))
    new_row["away_team"] = str(game.get("away_team", ""))

    existing = (
        pd.read_csv(GAMES_CSV, dtype=str)
        if GAMES_CSV.exists()
        else pd.DataFrame(columns=GAMES_CSV_COLS)
    )
    updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    updated.to_csv(GAMES_CSV, index=False)
    print(f"  Initialized: {new_row['away_team']} @ {new_row['home_team']} ({new_row['game_date']})")


def get_game_start_utc(game: dict) -> datetime:
    """
    Parse game_time_et (format: "2026-04-01 13:05") and return a UTC datetime.
    Falls back to noon ET on game_date if parsing fails.
    """
    game_time_et = str(game.get("game_time_et") or "")
    try:
        et_dt = datetime.strptime(game_time_et, "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
        return et_dt.astimezone(timezone.utc)
    except ValueError:
        date_str = str(game.get("game_date", ""))[:10]
        et_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=13, tzinfo=EASTERN  # default to 1 PM ET if time is unknown
        )
        return et_dt.astimezone(timezone.utc)


def get_next_game() -> dict | None:
    """
    Return the earliest upcoming game not yet in games.csv, or None if none found.
    """
    if not MLB_CSV.exists():
        return None

    df = pd.read_csv(MLB_CSV, low_memory=False)
    upcoming = df[df["home_win"].isna()].copy()
    if upcoming.empty:
        return None

    # Remove already-processed games
    if GAMES_CSV.exists():
        done = set(pd.read_csv(GAMES_CSV, dtype=str)["game_pk"].astype(str).tolist())
        upcoming = upcoming[~upcoming["game_pk"].astype(str).isin(done)]

    if upcoming.empty:
        return None

    # Sort by game time and return the soonest
    sort_col = "game_time_et" if "game_time_et" in upcoming.columns else "game_date"
    upcoming = upcoming.sort_values(sort_col)
    return upcoming.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline_for_game(game_pk: str):
    """Execute all four pipeline steps for a specific game."""
    print(f"\n{'='*60}")
    print(f"PIPELINE START — game_pk={game_pk}")
    print(f"{'='*60}")

    try:
        # Step 2a — Fetch MLB data
        run_step(["uv", "run", "fetch/fetch_data.py"])

        # Step 2b — Fetch Kalshi balance
        run_step(["uv", "run", "fetch/fetch_balance.py"])

        # Step 3 — Run model
        run_step(["uv", "run", "model/predict.py", "--game_pk", game_pk])

        # Step 4 — Place bet
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
    group.add_argument(
        "--game_pk", type=str,
        help="Run once immediately for a specific game_pk"
    )
    group.add_argument(
        "--now", action="store_true",
        help="Run immediately for the next upcoming game (no sleep)"
    )
    args = parser.parse_args()

    ensure_games_csv()

    # ── One-shot: specific game ───────────────────────────────────────────
    if args.game_pk:
        if MLB_CSV.exists():
            df   = pd.read_csv(MLB_CSV, dtype=str)
            rows = df[df["game_pk"].astype(str) == args.game_pk]
            if not rows.empty:
                init_game_row(rows.iloc[0].to_dict())
        run_pipeline_for_game(args.game_pk)
        return

    # ── Main loop ─────────────────────────────────────────────────────────
    print("MLB pipeline orchestrator started. Press Ctrl+C to stop.\n")

    while True:
        # Refresh schedule on every loop iteration
        print("Refreshing MLB schedule...")
        try:
            subprocess.run(["uv", "run", "fetch/fetch_data.py"], check=False)
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
