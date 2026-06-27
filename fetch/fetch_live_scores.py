"""
Refresh live MLB scores/status for dashboard game cards.

Run once:
    uv run python -m fetch.fetch_live_scores --date 2026-04-29
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import sys

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent.parent))
import db as DB


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
EASTERN = ZoneInfo("America/New_York")


def _retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _score(game: dict, side: str):
    team = game.get("teams", {}).get(side, {})
    if "score" not in team:
        return None
    try:
        return int(team.get("score"))
    except (TypeError, ValueError):
        return None


def _live_status_text(status: dict, linescore: dict) -> str:
    detailed = status.get("detailedState") or ""
    abstract = status.get("abstractGameState") or ""
    if detailed.lower() in {"warmup", "pre-game", "scheduled"}:
        return detailed
    if abstract == "Live":
        ordinal = linescore.get("currentInningOrdinal")
        state = linescore.get("inningState")
        outs = linescore.get("outs")
        parts = [p for p in (state, ordinal) if p]
        if outs is not None and state in {"Top", "Bottom"}:
            parts.append(f"{outs} out" if outs == 1 else f"{outs} outs")
        return " ".join(parts) or detailed or "Live"
    return detailed or abstract or ""


def _parse_game(game: dict) -> dict:
    status = game.get("status", {}) or {}
    linescore = game.get("linescore", {}) or {}
    home_score = _score(game, "home")
    away_score = _score(game, "away")
    abstract = status.get("abstractGameState") or ""
    detailed = status.get("detailedState") or ""
    final = abstract == "Final" or detailed == "Final"
    home_win = None
    if final and home_score is not None and away_score is not None and home_score != away_score:
        home_win = home_score > away_score

    return {
        "game_pk": game.get("gamePk"),
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
        "game_status": (detailed or abstract or "").lower().replace(" ", "_"),
        "game_state": abstract,
        "detailed_state": detailed,
        "inning": linescore.get("currentInning"),
        "inning_ordinal": linescore.get("currentInningOrdinal"),
        "inning_state": linescore.get("inningState"),
        "is_top_inning": linescore.get("isTopInning"),
        "outs": linescore.get("outs"),
        "balls": linescore.get("balls"),
        "strikes": linescore.get("strikes"),
        "live_status_text": _live_status_text(status, linescore),
        "live_updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def refresh_scores_for_date(date_str: str | None = None) -> int:
    if not date_str:
        date_str = datetime.now(EASTERN).strftime("%Y-%m-%d")
    session = _retry_session()
    resp = session.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": date_str, "hydrate": "linescore"},
        timeout=15,
    )
    resp.raise_for_status()
    updates = []
    for date_block in resp.json().get("dates", []):
        for game in date_block.get("games", []):
            parsed = _parse_game(game)
            if parsed.get("game_pk"):
                updates.append(parsed)
    count = DB.apply_live_game_updates(updates)
    if count:
        try:
            DB.void_orders_for_inactive_games()
            DB.backfill_user_order_results()
            DB.backfill_paper_order_results()
        except Exception:
            pass
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh MLB live scores.")
    parser.add_argument("--date", help="Game date YYYY-MM-DD. Defaults to current ET date.")
    args = parser.parse_args()
    count = refresh_scores_for_date(args.date)
    print(f"Live score refresh: updated={count}")


if __name__ == "__main__":
    main()
