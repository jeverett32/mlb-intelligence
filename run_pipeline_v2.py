"""
run_pipeline_v2.py — Parallel MLB betting pipeline orchestrator for V2.
Isolated from V1 and NEVER places real bets on Kalshi.
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

# Cap BLAS/OpenMP threads before importing numerical libs
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import db as DB
import notify as NOTIFY
from config import ACTIVE_SEASON, CURRENT_CSV
import model_v2.predict as PREDICT_V2
from model_v2.ingest_features import ingest_features
from fetch.fetch_data import main as fetch_data_main, refresh_odds_only

EASTERN = ZoneInfo("America/New_York")
MLB_CSV = Path(CURRENT_CSV)

# V2 Hardcoded safety
DRY_RUN_KALSHI = True

LEAD_MINUTES = 10
MIN_GAME_TIME_MINUTES = 0
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
    remaining = max(0.0, total_secs)
    while remaining > 0:
        chunk = min(ping_interval, remaining)
        time.sleep(chunk)
        remaining -= chunk
        _sd_notify("WATCHDOG=1")
        # V2 only settles its own paper orders
        settle_v2_paper_orders()

def _mark_fetched() -> None:
    global _LAST_FETCH_TS
    _LAST_FETCH_TS = time.time()

def _fetch_is_fresh() -> bool:
    return (time.time() - _LAST_FETCH_TS) < FETCH_STALE_SECONDS


def ensure_v2_features_ready(games: list[dict], no_ingest: bool = False) -> None:
    """Check if games_v2 has fresh data for these PKs, otherwise ingest."""
    if no_ingest:
        return

    pks = [int(g["game_pk"]) for g in games]
    if not pks:
        return

    # Check for missing or stale rows (>24h)
    with DB.pooled_connection() as conn:
        df = pd.read_sql_query(
            "SELECT game_pk, updated_at FROM games_v2 WHERE game_pk = ANY(%s)",
            conn,
            params=(pks,),
        )

    missing = set(pks) - set(df["game_pk"].tolist())
    stale = []
    now = datetime.now(timezone.utc)
    for _, r in df.iterrows():
        # Ensure updated_at is tz-aware for comparison if it's not already
        ua = r["updated_at"]
        if ua.tzinfo is None:
            ua = ua.replace(tzinfo=timezone.utc)
        if now - ua > timedelta(hours=24):
            stale.append(int(r["game_pk"]))

    if missing or stale:
        print(
            f"[v2] Triggering feature ingest (missing={len(missing)}, stale={len(stale)})..."
        )
        # Ingest for today
        earliest_date = min(
            g.get("game_date", datetime.now().strftime("%Y-%m-%d")) for g in games
        )
        # Use a 1-day window around the games
        start_date = earliest_date
        end_date = (pd.Timestamp(earliest_date) + pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        try:
            ingest_features(
                season=ACTIVE_SEASON, start_date=start_date, end_date=end_date
            )
        except Exception as e:
            print(f"[v2] Feature ingest failed: {e}")


def settle_v2_paper_orders():

    """Backfill results for V2 paper orders from games table."""
    try:
        count = DB.backfill_paper_order_v2_results()
        if count:
            print(f"[v2] Settled {count} paper orders.")
    except Exception as e:
        print(f"[v2] Settlement error: {e}")

@contextlib.contextmanager
def _patched_argv(argv: list[str]):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv

def run_fetch_data(
    *,
    skip_odds: bool = False,
    odds_only: bool = False,
    odds_game_pks: list[str] | None = None,
    odds_games: list[dict] | None = None,
) -> None:
    if odds_only and odds_games is not None:
        pks = ",".join(str(g.get("game_pk")) for g in odds_games)
        print(f"\n[v2] fetch.refresh_odds_only({pks})")
        refresh_odds_only(pd.DataFrame(odds_games))
        return

    argv = ["fetch/fetch_data.py"]
    if skip_odds:
        argv.append("--skip-odds")
    if odds_only:
        argv.append("--odds-only")
    if odds_game_pks:
        argv.extend(["--odds-game-pks", ",".join(str(pk) for pk in odds_game_pks)])
    print(f"\n[v2] fetch.fetch_data.main() {' '.join(argv[1:])}".rstrip())
    with _patched_argv(argv):
        fetch_data_main()

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
        return
    run_fetch_data(
        skip_odds=skip_odds,
        odds_only=odds_only,
        odds_game_pks=odds_game_pks,
        odds_games=odds_games,
    )
    if not odds_only:
        _mark_fetched()

def get_game_start_utc(game: dict) -> datetime:
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
    try:
        df = DB.get_upcoming_needing_prediction_v2(season=ACTIVE_SEASON)
    except Exception as e:
        print(f"  [v2] WARNING: DB unavailable ({e}), falling back to CSV.")
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
    now_utc = datetime.now(timezone.utc)
    all_games = get_all_upcoming_unprocessed()
    if not all_games:
        return [], None

    first_game = None
    for game in all_games:
        start = get_game_start_utc(game)
        if start < now_utc + timedelta(minutes=MIN_GAME_TIME_MINUTES):
            continue
        first_game = game
        break

    if first_game is None:
        return [], None

    first_start = get_game_start_utc(first_game)
    run_at_utc = first_start - timedelta(minutes=LEAD_MINUTES)
    batch = [g for g in all_games if get_game_start_utc(g) == first_start]
    return batch, run_at_utc

def _predict_v2(game_pk: str, shared: dict) -> None:
    try:
        res = PREDICT_V2.predict_one(game_pk, shared, dry_run=False)
        away = shared['target_rows'][game_pk].get('away_team', pd.Series(['?'])).iloc[0]
        home = shared['target_rows'][game_pk].get('home_team', pd.Series(['?'])).iloc[0]
        print(f"[v2] {away}@{home} pk={game_pk} prob={res['prob']:.4f} edge={res['edge']:+.4f} side={res['bet_side']} frac={res['bet_frac']:.4f}")
    except Exception as e:
        print(f"  [v2] PREDICT ERROR game_pk={game_pk}: {e}")

def run_batch(games: list[dict], no_ingest: bool = False) -> None:
    labels = ", ".join(f"{g.get('away_team')}@{g.get('home_team')}" for g in games)
    print(f"\n{'=' * 60}")
    print(f"[v2] BATCH START — {len(games)} game(s): {labels}")
    print(f"{'=' * 60}")

    pks_list = [str(g["game_pk"]) for g in games]

    # Ensure V2 features are in DB before predicting
    ensure_v2_features_ready(games, no_ingest=no_ingest)

    try:
        refresh_fetch_data_if_stale(
            "Shared fetch",
            odds_only=_fetch_is_fresh(),
            odds_game_pks=pks_list,
            odds_games=games,
            force=True,
        )
    except Exception as e:
        print(f"\n[v2] ERROR in shared fetch step: {e}")
        return

    earliest = min(get_game_start_utc(g) for g in games)
    prep_date = earliest.astimezone(timezone.utc).date().isoformat()

    print(f"\n  [v2] Preparing shared model for {prep_date} ({len(pks_list)} game(s))...")
    try:
        shared = PREDICT_V2.prepare_shared(pks_list, prep_date)
    except Exception as e:
        print(f"[v2] ERROR preparing shared model: {e}")
        return

    if len(pks_list) == 1:
        _predict_v2(pks_list[0], shared)
    else:
        print(f"\n  [v2] Running predict for {len(pks_list)} games in parallel...")
        with ThreadPoolExecutor(max_workers=min(3, len(pks_list))) as executor:
            futures = {executor.submit(_predict_v2, pk, shared): pk for pk in pks_list}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  [v2] Thread error for game_pk={futures[future]}: {e}")

    print(f"\n{'=' * 60}")
    print(f"[v2] BATCH COMPLETE — {len(games)} game(s)")
    print(f"{'=' * 60}")

def run_pipeline_for_game(game_pk: str):
    print(f"\n[v2] PIPELINE START — game_pk={game_pk}")
    try:
        refresh_fetch_data_if_stale(
            "Single-game fetch",
            odds_only=_fetch_is_fresh(),
            odds_game_pks=[str(game_pk)],
            force=True,
        )
    except Exception as e:
        print(f"\n[v2] ERROR: {e}")
        return
    
    # We need game_date to prepare shared
    with DB.pooled_connection() as conn:
        df = pd.read_sql_query("SELECT game_date FROM games WHERE game_pk = %s", conn, params=(int(game_pk),))
        if df.empty:
            print(f"[v2] Game {game_pk} not found in DB.")
            return
        game_date = str(df.iloc[0]['game_date'])[:10]

    shared = PREDICT_V2.prepare_shared([str(game_pk)], game_date)
    _predict_v2(str(game_pk), shared)

def main():
    DB.init_v2_tables()
    
    parser = argparse.ArgumentParser(description="V2 MLB betting pipeline orchestrator.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--game_pk", type=str, help="Run once for a specific game_pk")
    group.add_argument("--now", action="store_true", help="Run once for the next batch")
    group.add_argument("--once", action="store_true", help="Run once for the next batch (alias for --now)")
    parser.add_argument("--no-ingest", action="store_true", help="Skip feature ingestion check")
    args = parser.parse_args()

    if args.game_pk:
        run_pipeline_for_game(args.game_pk)
        return

    print("MLB V2 pipeline orchestrator started. Press Ctrl+C to stop.\n")
    _sd_notify("READY=1")

    while True:
        _sd_notify("WATCHDOG=1")
        settle_v2_paper_orders()

        if not _fetch_is_fresh():
            print("[v2] Refreshing MLB schedule...")
            try:
                refresh_fetch_data_if_stale("Schedule refresh", skip_odds=True)
            except Exception as e:
                print(f"  [v2] Schedule refresh failed: {e}")

        batch, run_at_utc = get_next_batch()
        now_utc = datetime.now(timezone.utc)

        if not batch:
            print("[v2] No upcoming unprocessed games found. Rechecking in 30 minutes...")
            _watchdog_sleep(1800)
            continue

        first_start = get_game_start_utc(batch[0])
        labels = ", ".join(f"{g.get('away_team')}@{g.get('home_team')}" for g in batch)
        print(f"\n[v2] Next batch ({len(batch)} game(s)): {labels}")
        print(f"  First start: {first_start.strftime('%Y-%m-%d %H:%M UTC')} | Run at: {run_at_utc.strftime('%H:%M UTC')}")

        if not (args.now or args.once) and run_at_utc > now_utc:
            sleep_secs = (run_at_utc - now_utc).total_seconds()
            print(f"[v2] Sleeping {sleep_secs / 60:.1f} minutes...")
            _watchdog_sleep(sleep_secs)

        run_batch(batch, no_ingest=args.no_ingest)

        if args.now or args.once:
            print("\n[v2] Exiting after one batch.")
            return

if __name__ == "__main__":
    main()
