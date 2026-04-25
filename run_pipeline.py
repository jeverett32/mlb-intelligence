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

# Cap BLAS/OpenMP threads before importing numerical libs so concurrent
# prediction workers do not oversubscribe the CPU.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import db as DB
import notify as NOTIFY
sys.path.insert(0, str(Path(__file__).parent / "model"))
sys.path.insert(0, str(Path(__file__).parent / "bet"))
from model import predict as PREDICT  # noqa: E402
from bet import place_bet as PLACE_BET  # noqa: E402

EASTERN = ZoneInfo("America/New_York")
MLB_CSV = Path("data/mlb_2026.csv")  # local CSV fallback

# How many minutes before game start to trigger the pipeline
LEAD_MINUTES = 15

# Games starting within this window of the earliest game are batched together
BATCH_WINDOW_MINUTES = 30

# Don't process games that start less than this many minutes from now
MIN_GAME_TIME_MINUTES = 10

# Skip the pre-batch schedule refresh if we fetched more recently than this
FETCH_STALE_SECONDS = 300

_LAST_FETCH_TS: float = 0.0


def _sd_notify(state: str) -> None:
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    try:
        import socket
        addr = "\0" + sock_path[1:] if sock_path.startswith("@") else sock_path
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(state.encode(), addr)
    except Exception:
        pass


def _watchdog_sleep(total_secs: float, ping_interval: float = 30.0) -> None:
    """Sleep in chunks, pinging the systemd watchdog between chunks."""
    remaining = max(0.0, total_secs)
    while remaining > 0:
        chunk = min(ping_interval, remaining)
        time.sleep(chunk)
        remaining -= chunk
        _sd_notify("WATCHDOG=1")


def _mark_fetched() -> None:
    global _LAST_FETCH_TS
    _LAST_FETCH_TS = time.time()


def _fetch_is_fresh() -> bool:
    return (time.time() - _LAST_FETCH_TS) < FETCH_STALE_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_step(cmd: list[str]):
    """Run a pipeline step as a subprocess. Raises RuntimeError on non-zero exit."""
    if cmd[0] == "uv":
        # Reuse the current interpreter so systemd services do not depend on a
        # separate uv install or a specific HOME-owned binary path.
        if len(cmd) >= 3 and cmd[1] == "run":
            cmd = [sys.executable, cmd[2], *cmd[3:]]
        else:
            raise RuntimeError(f"Unsupported uv command: {' '.join(cmd)}")
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
    Return all upcoming 2026 games that need predictions, sorted by start time.
    Uses a single JOIN query against the DB; falls back to CSV if DB is down.
    """
    try:
        df = DB.get_upcoming_needing_prediction(season=2026)
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

    # Find the earliest game that is still processable:
    #   - not more than 3 hours in the past (safety net for stale games)
    #   - at least MIN_GAME_TIME_MINUTES in the future (enough time to run pipeline)
    first_game = None
    for game in all_games:
        start = get_game_start_utc(game)
        if start < now_utc - timedelta(hours=3):
            continue  # too old, skip
        if start < now_utc + timedelta(minutes=MIN_GAME_TIME_MINUTES):
            continue  # too soon to process, skip
        first_game = game
        break

    if first_game is None:
        return [], None

    first_start = get_game_start_utc(first_game)
    run_at_utc = first_start - timedelta(minutes=LEAD_MINUTES)
    window_end = first_start + timedelta(minutes=BATCH_WINDOW_MINUTES)

    # Collect all games in the batch window
    batch = [
        game
        for game in all_games
        if first_start <= get_game_start_utc(game) <= window_end
    ]

    return batch, run_at_utc


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _predict_and_bet(game_pk: str, shared: dict) -> None:
    """Run predict + per-user execution for one game, in-process."""
    try:
        PREDICT.predict_one(game_pk, shared)
    except Exception as e:
        print(f"  PREDICT ERROR game_pk={game_pk}: {e}")
        NOTIFY.send(f":warning: **Predict error** `game_pk={game_pk}`: `{e}`")
        return
    try:
        PLACE_BET.execute_for_all_users(game_pk)
    except PLACE_BET.PlaceBetError as e:
        print(f"  PLACE_BET ERROR game_pk={game_pk}: {e}")
        NOTIFY.send(f":no_entry: **Bet failed** `game_pk={game_pk}`: `{e}`")
    except Exception as e:
        print(f"  PLACE_BET unexpected error game_pk={game_pk}: {e}")
        NOTIFY.send(f":rotating_light: **Bet crashed** `game_pk={game_pk}`: `{e}`")


def run_batch(games: list[dict]) -> None:
    """
    Full pipeline for a batch of games:
      1. fetch_data once (shared)
      2. prepare shared model once
      3. predict + per-user execution in parallel, in-process
    """
    labels = ", ".join(f"{g.get('away_team')}@{g.get('home_team')}" for g in games)
    print(f"\n{'=' * 60}")
    print(f"BATCH START — {len(games)} game(s): {labels}")
    print(f"{'=' * 60}")

    try:
        run_step(["uv", "run", "fetch/fetch_data.py"])
        _mark_fetched()
    except RuntimeError as e:
        print(f"\nERROR in shared fetch step: {e}")
        print("Aborting batch.")
        NOTIFY.send(f":rotating_light: **Batch aborted** (fetch failed): `{e}`")
        return

    for game in games:
        init_game_row(game)

    pks = [str(g["game_pk"]) for g in games]

    # Use the earliest game's date as the training cutoff so batches spanning
    # UTC midnight still see a consistent training frontier.
    earliest = min(get_game_start_utc(g) for g in games)
    prep_date = earliest.astimezone(timezone.utc).date().isoformat()

    print(f"\n  Preparing shared model for {prep_date} ({len(pks)} game(s))...")
    try:
        shared = PREDICT.prepare_shared(pks, prep_date)
    except PREDICT.PredictError as e:
        print(f"ERROR preparing shared model: {e}")
        print("Aborting batch.")
        NOTIFY.send(f":rotating_light: **Batch aborted** (shared model prep): `{e}`")
        return

    if len(pks) == 1:
        _predict_and_bet(pks[0], shared)
    else:
        print(f"\n  Running predict+bet for {len(pks)} games in parallel...")
        with ThreadPoolExecutor(max_workers=min(3, len(pks))) as executor:
            futures = {executor.submit(_predict_and_bet, pk, shared): pk for pk in pks}
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
        _mark_fetched()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return
    try:
        info = PREDICT.find_target_game(game_pk=str(game_pk))
        shared = PREDICT.prepare_shared([info["game_pk"]], info["game_date"])
        _predict_and_bet(info["game_pk"], shared)
    except PREDICT.PredictError as e:
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
    For each bet whose game started >4h ago and still has no result:
      - fetch the final from MLB Stats API by gamePk
      - accumulate finals / postponements
      - apply all updates + bet backfill in one transaction at the end
    """
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=4)

    try:
        rows = DB.get_settleable_games(2026, cutoff)
    except Exception as e:
        print(f"  Settlement check failed: {e}")
        return
    if not rows:
        return

    print(f"  Settling {len(rows)} completed game(s)...")
    import requests as _requests

    finals: list[dict] = []
    postponed: list[int] = []
    for r in rows:
        game_pk = r["game_pk"]
        home_team = r["home_team"]
        away_team = r["away_team"]
        try:
            resp = _requests.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "gamePk": game_pk, "hydrate": "linescore"},
                timeout=10,
            )
            resp.raise_for_status()
            dates = resp.json().get("dates", [])
            if not dates:
                continue
            game_data = dates[0]["games"][0]
            status = game_data.get("status", {})
            if "postponed" in status.get("detailedState", "").lower():
                postponed.append(int(game_pk))
                print(f"    Marked postponed: {away_team} @ {home_team}")
                continue
            if status.get("abstractGameState", "") != "Final":
                continue
            ls = game_data.get("linescore", {}).get("teams", {})
            h_score = ls.get("home", {}).get("runs")
            a_score = ls.get("away", {}).get("runs")
            if h_score is None or a_score is None:
                continue
            if h_score == a_score:
                print(
                    f"    Tie {a_score}-{h_score} for {away_team} @ {home_team} — skipping settlement."
                )
                continue
            home_win = bool(h_score > a_score)
            finals.append({
                "game_pk":    game_pk,
                "home_score": h_score,
                "away_score": a_score,
                "home_win":   home_win,
            })
            print(
                f"    Settled: {away_team} @ {home_team} — "
                f"{a_score}-{h_score} ({'Home' if home_win else 'Away'} win)"
            )
        except Exception as e:
            print(f"    Could not settle game_pk={game_pk}: {e}")

    try:
        DB.apply_settlements(finals, postponed)
        DB.backfill_user_order_results()
    except Exception as e:
        print(f"  Settlement write failed: {e}")


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
    _sd_notify("READY=1")

    while True:
        _sd_notify("WATCHDOG=1")
        settle_completed_games()

        if _fetch_is_fresh():
            print(
                f"Schedule refresh skipped — last fetch "
                f"{int(time.time() - _LAST_FETCH_TS)}s ago."
            )
        else:
            print("Refreshing MLB schedule...")
            try:
                rc = subprocess.run(
                    [sys.executable, "fetch/fetch_data.py"], check=False
                ).returncode
                if rc == 0:
                    _mark_fetched()
            except Exception as e:
                print(f"  Schedule refresh failed: {e}")

        batch, run_at_utc = get_next_batch()
        now_utc = datetime.now(timezone.utc)

        if not batch:
            print("No upcoming unprocessed games found. Rechecking in 30 minutes...")
            _watchdog_sleep(1800)
            continue

        first_start = get_game_start_utc(batch[0])
        labels = ", ".join(f"{g.get('away_team')}@{g.get('home_team')}" for g in batch)
        print(
            f"\nNext batch ({len(batch)} game(s)): {labels}"
            f"\n  First start: {first_start.strftime('%Y-%m-%d %H:%M UTC')}"
            f"  |  Run at: {run_at_utc.strftime('%H:%M UTC')}"
        )

        if not args.now and run_at_utc > now_utc:
            sleep_secs = (run_at_utc - now_utc).total_seconds()
            print(
                f"Sleeping {sleep_secs / 60:.1f} minutes until {LEAD_MINUTES} min before first pitch..."
            )
            _watchdog_sleep(sleep_secs)

        run_batch(batch)

        if args.now:
            print("\n--now flag: exiting after one batch.")
            return


if __name__ == "__main__":
    main()
