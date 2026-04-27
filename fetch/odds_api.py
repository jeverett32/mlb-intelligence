"""
Fetch MLB odds from The Odds API (the-odds-api.com).

Returns a DataFrame with the same columns as the SBR scraper output so it
slots directly into the existing pipeline:
    game_date, home_team, away_team,
    open_home_ml, open_away_ml, close_home_ml, close_away_ml,
    open_total, close_total, odds_source

Uses a JSON file cache to avoid burning credits when multiple games start
within a short window.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path("data/cache/odds_api_cache.json")
CACHE_TTL_SECONDS = 300  # 5-minute cache window

PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "betonlineag"]

TEAM_NAME_TO_ABB = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "ATH",
    "Athletics": "ATH",
    "Las Vegas Athletics": "ATH",
    "Sacramento Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def _load_cache():
    """Return (data_list, fetch_timestamp) from cache, or (None, 0)."""
    if not CACHE_PATH.exists():
        return None, 0
    try:
        raw = json.loads(CACHE_PATH.read_text())
        return raw["data"], raw["fetched_at"]
    except Exception:
        return None, 0


def _save_cache(data):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "data": data,
            }
        )
    )


def _fetch_from_api():
    """Call The Odds API and return the raw JSON list (one entry per game)."""
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY not set. Add it to .env "
            "(free key from https://the-odds-api.com)."
        )

    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,totals",
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  Odds API credits: {used} used, {remaining} remaining this month")

    return resp.json()


def _pick_book(bookmakers, market_key):
    """Pick the best available bookmaker for a given market."""
    book_map = {b["key"]: b for b in bookmakers}
    for pref in PREFERRED_BOOKS:
        if pref in book_map:
            for mkt in book_map[pref].get("markets", []):
                if mkt["key"] == market_key:
                    return mkt, pref
    for b in bookmakers:
        for mkt in b.get("markets", []):
            if mkt["key"] == market_key:
                return mkt, b["key"]
    return None, None


def _parse_h2h(market, home_team_name):
    """Extract home/away American moneylines from an h2h market."""
    home_ml = away_ml = None
    for outcome in market.get("outcomes", []):
        if outcome["name"] == home_team_name:
            home_ml = outcome["price"]
        else:
            away_ml = outcome["price"]
    return home_ml, away_ml


def _parse_totals(market):
    """Extract total (point) from a totals market."""
    for outcome in market.get("outcomes", []):
        if outcome["name"] == "Over":
            return outcome.get("point")
    return None


def fetch_odds_api() -> pd.DataFrame:
    """
    Fetch current MLB odds, with 5-minute caching.
    Returns a DataFrame matching the SBR odds schema.
    """
    cached_data, fetched_at = _load_cache()
    age = time.time() - fetched_at

    if cached_data is not None and age < CACHE_TTL_SECONDS:
        print(f"  Odds API: using cached data ({int(age)}s old)")
        data = cached_data
    else:
        print("  Odds API: fetching fresh odds...")
        try:
            data = _fetch_from_api()
            _save_cache(data)
        except Exception as e:
            print(f"  WARNING: Odds API fetch failed: {e}")
            if cached_data is not None:
                print("  Falling back to stale cache.")
                data = cached_data
            else:
                return pd.DataFrame()

    # Only accept games starting within next 3 days to avoid futures markets
    now_utc = datetime.now(timezone.utc)
    max_days_ahead = 3
    cutoff = now_utc + timedelta(days=max_days_ahead)

    rows = []
    for game in data:
        home_name = game.get("home_team", "")
        away_name = game.get("away_team", "")
        home_abb = TEAM_NAME_TO_ABB.get(home_name)
        away_abb = TEAM_NAME_TO_ABB.get(away_name)
        if not home_abb or not away_abb:
            continue

        commence = game.get("commence_time", "")
        try:
            commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            # Skip games too far in the future (likely futures markets)
            if commence_dt > cutoff:
                continue
            game_date = commence_dt.strftime("%Y-%m-%d")
        except Exception:
            continue

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        h2h_mkt, h2h_book = _pick_book(bookmakers, "h2h")
        if h2h_mkt is None:
            continue
        home_ml, away_ml = _parse_h2h(h2h_mkt, home_name)
        if home_ml is None or away_ml is None:
            continue

        # Skip implausible odds (likely futures markets, not game lines)
        # Regular season MLB game lines should be roughly -400 to +400
        if abs(home_ml) > 500 or abs(away_ml) > 500:
            continue

        tot_mkt, _ = _pick_book(bookmakers, "totals")
        total = _parse_totals(tot_mkt) if tot_mkt else None

        rows.append(
            {
                "game_date": game_date,
                "commence_time_utc": commence_dt.strftime("%Y-%m-%d %H:%M"),
                "home_team": home_abb,
                "away_team": away_abb,
                "open_home_ml": home_ml,
                "open_away_ml": away_ml,
                "close_home_ml": home_ml,
                "close_away_ml": away_ml,
                "open_total": total,
                "close_total": total,
                "odds_source": f"odds-api:{h2h_book}",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"  Odds API: {len(df)} games with odds")
    return df
