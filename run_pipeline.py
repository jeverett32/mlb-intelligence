"""
run_pipeline.py — MLB betting pipeline orchestrator.
Runs from the project root. Loops automatically, executing the full pipeline
15 minutes before each scheduled game window.

Games sharing the same scheduled start minute are treated as a batch:
  - fetch_data + fetch_balance run ONCE for the whole batch
  - predict + place_bet run IN PARALLEL for each game

Usage:
    uv run run_pipeline.py                     # loop forever (normal operation)
    uv run run_pipeline.py --game_pk 12345     # run once for a specific game right now
    uv run run_pipeline.py --now               # run immediately for the next upcoming game
"""

import argparse
import contextlib
import os
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
from config import ACTIVE_SEASON, CURRENT_CSV
sys.path.insert(0, str(Path(__file__).parent / "bet"))
from models.model_v1 import predict as PREDICT  # noqa: E402
from bet import place_bet as PLACE_BET  # noqa: E402
from fetch.fetch_balance import fetch_balance_for_account  # noqa: E402
from fetch.fetch_data import main as fetch_data_main, refresh_odds_only  # noqa: E402
from fetch.fetch_live_positions import refresh_due_orders  # noqa: E402
from fetch.fetch_live_scores import refresh_scores_for_date  # noqa: E402

EASTERN = ZoneInfo("America/New_York")
MLB_CSV = Path(CURRENT_CSV)  # local CSV fallback

# How many minutes before game start to trigger the pipeline
LEAD_MINUTES = 10

# Skip games once first pitch has passed. If the ideal 15-minute lead was
# missed because an earlier run took too long, run the game immediately.
MIN_GAME_TIME_MINUTES = 0

# Skip the pre-batch schedule refresh if we fetched more recently than this
FETCH_STALE_SECONDS = 300
LIVE_POSITION_REFRESH_SECONDS = 300
LIVE_SCORE_REFRESH_SECONDS = 300

_LAST_FETCH_TS: float = 0.0
_LAST_LIVE_POSITION_REFRESH_TS: float = 0.0
_LAST_LIVE_SCORE_REFRESH_TS: float = 0.0


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


def refresh_live_positions_if_due(force: bool = False) -> None:
    global _LAST_LIVE_POSITION_REFRESH_TS
    now = time.time()
    if not force and now - _LAST_LIVE_POSITION_REFRESH_TS < LIVE_POSITION_REFRESH_SECONDS:
        return
    _LAST_LIVE_POSITION_REFRESH_TS = now
    try:
        stats = refresh_due_orders(stale_seconds=LIVE_POSITION_REFRESH_SECONDS)
        if stats.checked:
            print(
                "Live position refresh: "
                f"checked={stats.checked} updated={stats.updated} "
                f"skipped={stats.skipped} errors={stats.errors}"
            )
    except Exception as e:
        print(f"  Live position refresh failed: {e}")


def refresh_live_scores_if_due(force: bool = False) -> None:
    global _LAST_LIVE_SCORE_REFRESH_TS
    now = time.time()
    if not force and now - _LAST_LIVE_SCORE_REFRESH_TS < LIVE_SCORE_REFRESH_SECONDS:
        return
    _LAST_LIVE_SCORE_REFRESH_TS = now
    try:
        count = refresh_scores_for_date()
        if count:
            print(f"Live score refresh: updated={count}")
    except Exception as e:
        print(f"  Live score refresh failed: {e}")


def _watchdog_sleep(total_secs: float, ping_interval: float = 30.0) -> None:
    """Sleep in chunks, pinging the systemd watchdog between chunks."""
    remaining = max(0.0, total_secs)
    while remaining > 0:
        chunk = min(ping_interval, remaining)
        time.sleep(chunk)
        remaining -= chunk
        _sd_notify("WATCHDOG=1")
        refresh_live_positions_if_due()
        refresh_live_scores_if_due()


def _mark_fetched() -> None:
    global _LAST_FETCH_TS
    _LAST_FETCH_TS = time.time()


def _fetch_is_fresh() -> bool:
    return (time.time() - _LAST_FETCH_TS) < FETCH_STALE_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_fetch_data(
    *,
    skip_odds: bool = False,
    odds_only: bool = False,
    odds_game_pks: list[str] | None = None,
    odds_games: list[dict] | None = None,
) -> None:
    """Run fetch_data in-process while isolating its argparse argv."""
    if odds_only and odds_games is not None:
        pks = ",".join(str(g.get("game_pk")) for g in odds_games)
        print(f"\n>>> fetch.refresh_odds_only({pks})")
        refresh_odds_only(pd.DataFrame(odds_games))
        return

    argv = ["fetch/fetch_data.py"]
    if skip_odds:
        argv.append("--skip-odds")
    if odds_only:
        argv.append("--odds-only")
    if odds_game_pks:
        argv.extend(["--odds-game-pks", ",".join(str(pk) for pk in odds_game_pks)])
    print(f"\n>>> fetch.fetch_data.main() {' '.join(argv[1:])}".rstrip())
    with _patched_argv(argv):
        fetch_data_main()


@contextlib.contextmanager
def _patched_argv(argv: list[str]):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


def refresh_fetch_data_if_stale(
    label: str = "Shared fetch",
    *,
    skip_odds: bool = False,
    odds_only: bool = False,
    odds_game_pks: list[str] | None = None,
    odds_games: list[dict] | None = None,
    force: bool = False,
) -> None:
    if not force and _fetch_is_fresh():
        print(
            f"{label} skipped — last fetch "
            f"{int(time.time() - _LAST_FETCH_TS)}s ago."
        )
        return
    run_fetch_data(
        skip_odds=skip_odds,
        odds_only=odds_only,
        odds_game_pks=odds_game_pks,
        odds_games=odds_games,
    )
    if not odds_only:
        _mark_fetched()


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
    Return all upcoming active-season games that need predictions, sorted by start time.
    Uses a single JOIN query against the DB; falls back to CSV if DB is down.
    """
    try:
        df = DB.get_upcoming_needing_prediction(season=ACTIVE_SEASON)
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
    at the same scheduled start minute, and run_at_utc is when to
    kick off the pipeline (LEAD_MINUTES before the earliest game).
    Returns ([], None) if nothing to process.
    """
    now_utc = datetime.now(timezone.utc)
    all_games = get_all_upcoming_unprocessed()

    if not all_games:
        return [], None

    # Find the earliest game that is still processable. If compute delays make
    # us miss the ideal lead time, process it immediately until first pitch.
    first_game = None
    for game in all_games:
        start = get_game_start_utc(game)
        if start < now_utc + timedelta(minutes=MIN_GAME_TIME_MINUTES):
            continue  # already started, skip
        first_game = game
        break

    if first_game is None:
        return [], None

    first_start = get_game_start_utc(first_game)
    run_at_utc = first_start - timedelta(minutes=LEAD_MINUTES)

    # Batch only games in the exact same first-pitch slot. This handles crowded
    # slates without pulling later standalone games into an earlier run.
    batch = [
        game
        for game in all_games
        if get_game_start_utc(game) == first_start
    ]

    return batch, run_at_utc


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _predict_and_bet(game_pk: str, shared: dict, dry_run: bool = False) -> None:
    """Run predict + per-user execution for one game, in-process."""
    try:
        prediction = PREDICT.predict_one(game_pk, shared, dry_run=dry_run)
    except Exception as e:
        print(f"  PREDICT ERROR game_pk={game_pk}: {e}")
        if not dry_run:
            NOTIFY.send(f":warning: **Predict error** `game_pk={game_pk}`: `{e}`")
        return
    try:
        if dry_run:
            PLACE_BET.preview_for_all_users(prediction)
        else:
            PLACE_BET.execute_for_all_users(game_pk)
    except PLACE_BET.PlaceBetError as e:
        print(f"  PLACE_BET ERROR game_pk={game_pk}: {e}")
        if not dry_run:
            NOTIFY.send(f":no_entry: **Bet failed** `game_pk={game_pk}`: `{e}`")
    except Exception as e:
        print(f"  PLACE_BET unexpected error game_pk={game_pk}: {e}")
        if not dry_run:
            NOTIFY.send(f":rotating_light: **Bet crashed** `game_pk={game_pk}`: `{e}`")


def run_batch(games: list[dict], dry_run: bool = False) -> None:
    """
    Full pipeline for a batch of games:
      1. fetch_data once (shared)
      2. prepare shared model once
      3. predict + per-user execution in parallel, in-process
    """
    labels = ", ".join(f"{g.get('away_team')}@{g.get('home_team')}" for g in games)
    print(f"\n{'=' * 60}")
    print(f"BATCH START{' [DRY RUN]' if dry_run else ''} — {len(games)} game(s): {labels}")
    print(f"{'=' * 60}")

    pks_list = [str(g["game_pk"]) for g in games]
    run_id = None
    if not dry_run:
        try:
            run_id = DB.start_pipeline_run("batch", len(games), ",".join(pks_list))
        except Exception as e:
            print(f"  warn: could not record pipeline run start: {e}")

    t0 = time.monotonic()
    status = "success"
    err_msg: str | None = None

    try:
        try:
            refresh_fetch_data_if_stale(
                "Shared fetch",
                odds_only=_fetch_is_fresh(),
                odds_game_pks=pks_list,
                odds_games=games,
                force=True,
            )
        except Exception as e:
            print(f"\nERROR in shared fetch step: {e}")
            print("Aborting batch.")
            if not dry_run:
                NOTIFY.send(f":rotating_light: **Batch aborted** (fetch failed): `{e}`")
            status = "aborted"
            err_msg = f"fetch failed: {e}"
            return

        if not dry_run:
            for game in games:
                init_game_row(game)

        pks = pks_list

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
            if not dry_run:
                NOTIFY.send(f":rotating_light: **Batch aborted** (shared model prep): `{e}`")
            status = "aborted"
            err_msg = f"shared model prep: {e}"
            return

        if len(pks) == 1:
            _predict_and_bet(pks[0], shared, dry_run=dry_run)
        else:
            print(f"\n  Running predict+bet for {len(pks)} games in parallel...")
            with ThreadPoolExecutor(max_workers=min(3, len(pks))) as executor:
                futures = {executor.submit(_predict_and_bet, pk, shared, dry_run): pk for pk in pks}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  Thread error for game_pk={futures[future]}: {e}")

        print(f"\n{'=' * 60}")
        print(f"BATCH COMPLETE{' [DRY RUN]' if dry_run else ''} — {len(games)} game(s)")
        print(f"{'=' * 60}")
    except Exception as e:
        status = "error"
        err_msg = str(e)
        raise
    finally:
        duration = time.monotonic() - t0
        if run_id is not None:
            try:
                DB.finish_pipeline_run(run_id, status, duration, err_msg)
            except Exception as e:
                print(f"  warn: could not record pipeline run finish: {e}")


def run_pipeline_for_game(game_pk: str, dry_run: bool = False):
    """Single-game pipeline (used for --game_pk one-shot mode)."""
    print(f"\n{'=' * 60}")
    print(f"PIPELINE START{' [DRY RUN]' if dry_run else ''} — game_pk={game_pk}")
    print(f"{'=' * 60}")
    run_id = None
    if not dry_run:
        try:
            run_id = DB.start_pipeline_run("single", 1, str(game_pk))
        except Exception as e:
            print(f"  warn: could not record pipeline run start: {e}")

    t0 = time.monotonic()
    status = "success"
    err_msg: str | None = None
    try:
        try:
            refresh_fetch_data_if_stale(
                "Single-game fetch",
                odds_only=_fetch_is_fresh(),
                odds_game_pks=[str(game_pk)],
                force=True,
            )
        except Exception as e:
            print(f"\nERROR: {e}")
            status = "aborted"
            err_msg = f"fetch failed: {e}"
            return
        try:
            info = PREDICT.find_target_game(game_pk=str(game_pk))
            shared = PREDICT.prepare_shared([info["game_pk"]], info["game_date"])
            _predict_and_bet(info["game_pk"], shared, dry_run=dry_run)
        except PREDICT.PredictError as e:
            print(f"\nERROR: {e}")
            status = "aborted"
            err_msg = str(e)
            return
        print(f"\n{'=' * 60}")
        print(f"PIPELINE COMPLETE{' [DRY RUN]' if dry_run else ''} — game_pk={game_pk}")
        print(f"{'=' * 60}")
    except Exception as e:
        status = "error"
        err_msg = str(e)
        raise
    finally:
        duration = time.monotonic() - t0
        if run_id is not None:
            try:
                DB.finish_pipeline_run(run_id, status, duration, err_msg)
            except Exception as e:
                print(f"  warn: could not record pipeline run finish: {e}")


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def settle_completed_games():
    """
    For each game that started >4h ago and still has no result:
      - fetch the final from MLB Stats API by gamePk
      - accumulate finals / postponements
      - apply all updates + bet backfill in one transaction at the end
    """
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=4)

    try:
        rows = DB.get_settleable_games(ACTIVE_SEASON, cutoff)
    except Exception as e:
        print(f"  Settlement check failed: {e}")
        rows = []

    if rows:
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
        except Exception as e:
            print(f"  Settlement write failed: {e}")

    try:
        DB.backfill_user_order_results()
    except Exception as e:
        print(f"  User order backfill failed: {e}")
    try:
        DB.backfill_paper_order_results()
    except Exception as e:
        print(f"  Paper order backfill failed: {e}")

    for user in DB.list_approved_users_with_accounts():
        try:
            account = DB.get_kalshi_account(user["email"])
            if account and account.get("is_active"):
                fetch_balance_for_account(
                    key_id=account["key_id"],
                    key_path=account["key_path"],
                    kalshi_env=account["kalshi_env"],
                    email=user["email"],
                )
        except Exception as e:
            print(f"  Balance refresh failed for {user['email']}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    try:
        DB.init_pipeline_runs_table()
    except Exception as e:
        print(f"  warn: could not init pipeline_runs table: {e}")

    parser = argparse.ArgumentParser(description="MLB betting pipeline orchestrator.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--game_pk", type=str, help="Run once immediately for a specific game_pk"
    )
    group.add_argument(
        "--now", action="store_true", help="Run immediately for the next batch"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full pipeline timing path without prediction/order/paper writes",
    )
    args = parser.parse_args()

    # ── One-shot: specific game ───────────────────────────────────────────
    if args.game_pk:
        try:
            game_df = DB.get_games_df(season=ACTIVE_SEASON)
            rows = game_df[game_df["game_pk"].astype(str) == args.game_pk]
            if not rows.empty:
                init_game_row(rows.iloc[0].to_dict())
        except Exception:
            DB.init_bet(int(args.game_pk), "", "", "")
        run_pipeline_for_game(args.game_pk, dry_run=args.dry_run)
        return

    # ── Main loop ─────────────────────────────────────────────────────────
    print("MLB pipeline orchestrator started. Press Ctrl+C to stop.\n")
    _sd_notify("READY=1")

    while True:
        _sd_notify("WATCHDOG=1")
        if args.dry_run:
            print("DRY RUN: skipping settlement/live order refresh writes.")
        else:
            settle_completed_games()
            refresh_live_positions_if_due(force=True)
            refresh_live_scores_if_due(force=True)

        if _fetch_is_fresh():
            print(
                f"Schedule refresh skipped — last fetch "
                f"{int(time.time() - _LAST_FETCH_TS)}s ago."
            )
        else:
            print("Refreshing MLB schedule...")
            try:
                refresh_fetch_data_if_stale("Schedule refresh", skip_odds=True)
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

        run_batch(batch, dry_run=args.dry_run)

        if args.now:
            print(f"\n--now{' --dry-run' if args.dry_run else ''}: exiting after one batch.")
            return


if __name__ == "__main__":
    main()
