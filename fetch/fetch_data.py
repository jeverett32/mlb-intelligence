"""
fetch_data.py — Daily data pipeline for the 2026 MLB season.

Fetches schedule, odds, pitcher stats, weather, and FanGraphs team stats,
then assembles mlb_2026.csv in the same format as master_mlb.csv so that
train.py can consume it directly.

Incremental: loads existing CSV, skips complete past games, and re-fetches
fresh schedule data for today and tomorrow on every run to capture time
changes, postponements, and lineup updates. Only creates rows for completed
past games and games within the today/tomorrow window — no full-season
skeleton rows.

Usage:
    uv run fetch/fetch_data.py             # update through tomorrow
    uv run fetch/fetch_data.py --today-only  # only scrape odds for today
"""

import os, sys, json, time, random, asyncio, warnings, argparse, functools
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
import db as DB

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import statsapi

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEASON = 2026
CACHE_DIR = Path("data/cache")
OUTPUT_CSV = "data/mlb_2026.csv"
MLB_API = "https://statsapi.mlb.com/api/v1"
PREFERRED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "bet365"]

# Columns that must be non-null for a completed-game row to be "complete."
# Odds are excluded because SBR may not have data for every date;
# run without --today-only to backfill odds when available.
COMPLETENESS_COLS = [
    "home_score", "away_score", "home_win",
    "home_starter_id", "temp_c", "home_avg",
]

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Team-name mappings
# ---------------------------------------------------------------------------
FULL_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH",
    "Atlanta Braves": "ATL",       "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",       "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",  "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL",     "Detroit Tigers": "DET",
    "Houston Astros": "HOU",       "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",   "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",        "Florida Marlins": "MIA",
    "Milwaukee Brewers": "MIL",    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",        "New York Yankees": "NYY",
    "Oakland Athletics": "ATH",    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",   "San Diego Padres": "SDP",
    "Seattle Mariners": "SEA",     "San Francisco Giants": "SFG",
    "St. Louis Cardinals": "STL",  "Tampa Bay Rays": "TBR",
    "Tampa Bay Devil Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",    "Washington Nationals": "WSN",
    "Montreal Expos": "WSN",
}

TEAM_ID_TO_ABB = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KCR", 119: "LAD", 120: "WSN", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG", 138: "STL",
    139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}

SBR_NAME_TO_ABB = {
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL",
    "baltimore orioles": "BAL",    "boston red sox": "BOS",
    "chicago cubs": "CHC",         "chicago white sox": "CHW",
    "cincinnati reds": "CIN",      "cleveland guardians": "CLE",
    "cleveland indians": "CLE",    "colorado rockies": "COL",
    "detroit tigers": "DET",       "houston astros": "HOU",
    "kansas city royals": "KCR",   "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",  "miami marlins": "MIA",
    "florida marlins": "MIA",      "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",      "new york mets": "NYM",
    "new york yankees": "NYY",     "oakland athletics": "ATH",
    "athletics": "ATH",            "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",   "san diego padres": "SDP",
    "san francisco giants": "SFG", "seattle mariners": "SEA",
    "st louis cardinals": "STL",   "st. louis cardinals": "STL",
    "tampa bay rays": "TBR",       "tampa bay devil rays": "TBR",
    "texas rangers": "TEX",        "toronto blue jays": "TOR",
    "washington nationals": "WSN", "montreal expos": "WSN",
    "anaheim angels": "LAA",
}

FG_TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CWS": "CHW", "CIN": "CIN",
    "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU",
    "KCR": "KCR", "KC":  "KCR", "KCA": "KCR",
    "LAA": "LAA", "LAD": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NYM": "NYM", "NYY": "NYY",
    "OAK": "ATH", "ATH": "ATH",
    "PHI": "PHI", "PIT": "PIT",
    "SDP": "SDP", "SD":  "SDP", "SDG": "SDP",
    "SEA": "SEA",
    "SFG": "SFG", "SF":  "SFG", "SFO": "SFG",
    "STL": "STL",
    "TBR": "TBR", "TB":  "TBR", "TAM": "TBR",
    "TEX": "TEX", "TOR": "TOR",
    "WSN": "WSN", "WSH": "WSN", "WAS": "WSN",
}

PARK_COORDS = {
    "ARI": (33.4453, -112.0667), "ATL": (33.8908, -84.4678),
    "BAL": (39.2838, -76.6216),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CHW": (41.8299, -87.6338),
    "CIN": (39.0974, -84.5082),  "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),  "KCR": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "ATH": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),  "PIT": (40.4469, -80.0057),
    "SDP": (32.7076, -117.1570), "SEA": (47.5914, -122.3325),
    "SFG": (37.7786, -122.3893), "STL": (38.6226, -90.1928),
    "TBR": (27.7683, -82.6534),  "TEX": (32.7512, -97.0832),
    "TOR": (43.6414, -79.3894),  "WSN": (38.8730, -77.0074),
}

# Canonical output columns (always present, NaN when unavailable)
ALL_OUTPUT_COLS = (
    ["game_id", "game_pk", "game_date", "game_time_utc", "season",
     "home_team", "away_team", "is_night_game",
     "home_implied_prob", "away_implied_prob",
     "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml",
     "open_total", "close_total", "odds_source",
     "home_score", "away_score", "home_win",
     "home_starter_id", "away_starter_id",
     "home_pitcher_is_lefty", "away_pitcher_is_lefty", "pitcher_handedness_diff",
     "home_sp_era", "home_sp_whip", "home_sp_k9", "home_sp_bb9",
     "away_sp_era", "away_sp_whip", "away_sp_k9", "away_sp_bb9",
     "home_rolling_era", "home_rolling_whip", "home_rolling_k9",
     "away_rolling_era", "away_rolling_whip", "away_rolling_k9",
     "temp_c", "wind_speed_kmh", "wind_dir_deg"]
    + [f"{s}_{c}" for s in ("home", "away")
       for c in ("avg", "obp", "slg", "woba", "wrc_plus", "war",
                 "k_pct", "bb_pct", "k_per_9", "bb_per_9", "hr_per_9",
                 "era", "fip", "owar")]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def sbr_normalize(name):
    return (name.lower().replace(".", "").replace("'", "")
            .replace("-", " ").replace("&", "and").strip())


def _parse_game_datetime(game_datetime_str):
    """Parse MLB API UTC datetime string. Returns (utc_str, night_flag)."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import timezone
        utc_time = datetime.strptime(
            game_datetime_str, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        et = utc_time.astimezone(ZoneInfo("America/New_York"))
        return utc_time.strftime("%Y-%m-%d %H:%M"), (1 if et.hour >= 17 else 0)
    except Exception:
        return None, 1


def _api_get(url, params=None, timeout=15, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  API error {url}: {e}")
                return None


def _ensure_columns(df):
    """Guarantee every canonical column exists (NaN if missing)."""
    for c in ALL_OUTPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[ALL_OUTPUT_COLS]


# ---------------------------------------------------------------------------
# Existing-CSV helpers
# ---------------------------------------------------------------------------
def _load_existing():
    """Load existing 2026 season rows from PostgreSQL (falls back to CSV if DB unavailable)."""
    try:
        df = DB.get_games_df(season=SEASON)
        if not df.empty:
            df["game_date"] = pd.to_datetime(df["game_date"])
            return df
    except Exception as e:
        print(f"  WARNING: DB unavailable ({e}), falling back to CSV.")
    # CSV fallback
    if not Path(OUTPUT_CSV).exists():
        return pd.DataFrame()
    df = pd.read_csv(OUTPUT_CSV, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _is_row_complete(row):
    """True if this completed-game row has all key feature columns filled."""
    return all(pd.notna(row.get(c)) for c in COMPLETENESS_COLS)


def _split_existing(existing, schedule_pks, today):
    """
    Given the existing CSV and the current schedule, return:
      complete_pks : set of game_pks that are done and fully populated
      incomplete   : existing rows that need re-processing
    """
    if existing.empty:
        return set(), pd.DataFrame()

    # Only consider games on or before today
    past_mask = existing["game_date"] <= today
    past = existing[past_mask]

    complete_pks = set()
    for _, row in past.iterrows():
        if _is_row_complete(row):
            complete_pks.add(row["game_pk"])

    return complete_pks, existing


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Schedule from MLB Stats API
# ═══════════════════════════════════════════════════════════════════════════
def fetch_schedule():
    """Fetch all 2026 regular-season games (completed + upcoming)."""
    print("  Querying MLB Stats API...")
    data = _api_get(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "startDate": f"{SEASON}-03-01",
            "endDate": f"{SEASON}-11-15",
            "hydrate": "probablePitcher,game(content(summary)),linescore",
            "gameType": "R",
        },
        timeout=30,
    )
    if not data:
        print("  WARNING: Could not fetch schedule")
        return pd.DataFrame()

    rows = []
    for date_info in data.get("dates", []):
        for g in date_info.get("games", []):
            if g.get("gameType") != "R":
                continue
            status = g.get("status", {}).get("detailedState", "")
            completed = status == "Final"

            home_team_data = g.get("teams", {}).get("home", {})
            away_team_data = g.get("teams", {}).get("away", {})
            home_tid = home_team_data.get("team", {}).get("id")
            away_tid = away_team_data.get("team", {}).get("id")
            home_abb = TEAM_ID_TO_ABB.get(home_tid, FULL_TO_ABBR.get(
                home_team_data.get("team", {}).get("name", ""), ""))
            away_abb = TEAM_ID_TO_ABB.get(away_tid, FULL_TO_ABBR.get(
                away_team_data.get("team", {}).get("name", ""), ""))

            if not home_abb or not away_abb:
                continue

            home_pp = home_team_data.get("probablePitcher", {})
            away_pp = away_team_data.get("probablePitcher", {})

            game_time_utc, night_flag = _parse_game_datetime(
                g.get("gameDate", ""))

            rows.append({
                "game_pk":   g["gamePk"],
                "game_date": pd.to_datetime(date_info["date"]),
                "game_time_utc": game_time_utc,
                "home_team": home_abb,
                "away_team": away_abb,
                "home_score": home_team_data.get("score") if completed else np.nan,
                "away_score": away_team_data.get("score") if completed else np.nan,
                "is_completed": completed,
                "is_night_game": night_flag,
                "home_probable_id": home_pp.get("id"),
                "away_probable_id": away_pp.get("id"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["game_pk"])
    df["home_win"] = np.where(
        df["is_completed"],
        (df["home_score"] > df["away_score"]).astype(float),
        np.nan,
    )
    df["season"] = SEASON
    df["game_id"] = df["game_pk"]
    print(f"  {len(df)} games ({df['is_completed'].sum()} completed, "
          f"{(~df['is_completed']).sum()} upcoming)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Odds from SBR scraper
# ═══════════════════════════════════════════════════════════════════════════
def _pick_best_book(book_map, open_keys, close_keys):
    for book in PREFERRED_BOOKS:
        if book not in book_map:
            continue
        entry = book_map[book]
        o_line = entry.get("openingLine", {}) or {}
        c_line = entry.get("currentLine", {}) or {}
        o_vals = {k: o_line.get(k) for k in open_keys}
        c_vals = {k: c_line.get(k) for k in close_keys}
        if any(v is not None for v in c_vals.values()):
            return o_vals, c_vals, book
    for entry in book_map.values():
        c_line = entry.get("currentLine", {}) or {}
        o_line = entry.get("openingLine", {}) or {}
        o_vals = {k: o_line.get(k) for k in open_keys}
        c_vals = {k: c_line.get(k) for k in close_keys}
        if any(v is not None for v in c_vals.values()):
            return o_vals, c_vals, entry.get("sportsbook", "unknown")
    return None, None, None


def _parse_scraped_odds(scraped_data):
    rows = []
    for date_str, games in scraped_data.items():
        for game in games:
            gv = game.get("gameView", {})
            home_raw = gv.get("homeTeam", {}).get("fullName", "")
            away_raw = gv.get("awayTeam", {}).get("fullName", "")
            home_abb = SBR_NAME_TO_ABB.get(sbr_normalize(home_raw))
            away_abb = SBR_NAME_TO_ABB.get(sbr_normalize(away_raw))
            if not home_abb or not away_abb:
                continue

            ml_views = game.get("odds", {}).get("moneyline", [])
            ml_bm = {o.get("sportsbook", "").lower(): o for o in ml_views if o}
            ml_open, ml_close, used_book = _pick_best_book(
                ml_bm, ["homeOdds", "awayOdds"], ["homeOdds", "awayOdds"])
            if ml_close is None:
                continue

            tot_views = game.get("odds", {}).get("totals", [])
            tot_bm = {o.get("sportsbook", "").lower(): o for o in tot_views if o}
            tot_open, tot_close, _ = _pick_best_book(
                tot_bm,
                ["total", "overOdds", "underOdds"],
                ["total", "overOdds", "underOdds"],
            )

            rows.append({
                "game_date":     date_str,
                "home_team":     home_abb,
                "away_team":     away_abb,
                "open_home_ml":  ml_open.get("homeOdds") if ml_open else None,
                "open_away_ml":  ml_open.get("awayOdds") if ml_open else None,
                "close_home_ml": ml_close.get("homeOdds"),
                "close_away_ml": ml_close.get("awayOdds"),
                "open_total":    tot_open.get("total") if tot_open else None,
                "close_total":   tot_close.get("total") if tot_close else None,
                "odds_source":   used_book,
            })
    return pd.DataFrame(rows)


def _merge_odds_into_cache(new_df, odds_cache):
    """Merge new odds rows into the CSV cache, deduplicating by game."""
    cached = pd.read_csv(odds_cache) if odds_cache.exists() else pd.DataFrame()
    if not cached.empty:
        cached["game_date"] = pd.to_datetime(cached["game_date"])

    new_df["game_date"] = pd.to_datetime(new_df["game_date"])

    if not cached.empty:
        new_keys = set(
            zip(new_df["game_date"].dt.date, new_df["home_team"], new_df["away_team"])
        )
        cached = cached[
            ~cached.apply(
                lambda r: (r["game_date"].date(), r["home_team"], r["away_team"]) in new_keys,
                axis=1,
            )
        ]

    combined = pd.concat([cached, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last")
    combined = combined.sort_values(["game_date", "home_team"]).reset_index(drop=True)
    combined.to_csv(odds_cache, index=False)
    return combined


def fetch_odds(schedule_df, today_only=False):
    """Fetch odds from SBR, falling back to The Odds API if SBR fails."""
    from scraper import scrape_range_async
    from odds_api import fetch_odds_api

    odds_cache = CACHE_DIR / "odds_2026.csv"
    cached = pd.read_csv(odds_cache) if odds_cache.exists() else pd.DataFrame()
    if not cached.empty:
        cached["game_date"] = pd.to_datetime(cached["game_date"])

    if schedule_df.empty:
        return cached

    today = pd.to_datetime(datetime.today().date())
    tomorrow = (today + timedelta(days=1)).date()
    game_dates = set(schedule_df["game_date"].dt.date.unique())
    cached_dates = (set(cached["game_date"].dt.date.unique())
                    if not cached.empty else set())

    if today_only:
        dates_needed = {today.date()} & game_dates
    else:
        dates_needed = game_dates - cached_dates
        for d in [today.date(), tomorrow]:
            if d in game_dates:
                dates_needed.add(d)

    if not dates_needed:
        print("  Odds already cached for all dates.")
        return cached

    # --- Try SBR first ---
    sorted_dates = sorted(dates_needed)
    start = sorted_dates[0].strftime("%Y-%m-%d")
    end = sorted_dates[-1].strftime("%Y-%m-%d")
    print(f"  Scraping SBR odds: {start} -> {end} ({len(sorted_dates)} dates)...")

    new_df = pd.DataFrame()
    try:
        raw = asyncio.run(scrape_range_async(
            start, end, fast=False, max_concurrent=5,
            odds_types=["moneyline", "totals"],
        ))
        new_df = _parse_scraped_odds(raw)
        # Sanity check: MLB moneylines should never exceed ±1500 for regular games.
        # Values beyond this indicate SBR returned futures/alternate market data.
        if not new_df.empty:
            ml_cols = ["close_home_ml", "close_away_ml"]
            bad_mask = pd.Series(False, index=new_df.index)
            for col in ml_cols:
                if col in new_df.columns:
                    vals = pd.to_numeric(new_df[col], errors="coerce").abs()
                    bad_mask = bad_mask | (vals > 1500)
            n_bad = bad_mask.sum()
            if n_bad > 0:
                print(f"  WARNING: {n_bad} SBR rows have implausible moneylines (|ml| > 1500) — dropping.")
                new_df = new_df[~bad_mask]
    except Exception as e:
        print(f"  WARNING: SBR scraping failed: {e}")

    if new_df.empty:
        print("  No valid odds from SBR. Trying The Odds API...")
        new_df = fetch_odds_api()
        # Apply same sanity check to Odds API data
        if not new_df.empty:
            ml_cols = ["close_home_ml", "close_away_ml"]
            bad_mask = pd.Series(False, index=new_df.index)
            for col in ml_cols:
                if col in new_df.columns:
                    vals = pd.to_numeric(new_df[col], errors="coerce").abs()
                    bad_mask = bad_mask | (vals > 1500)
            n_bad = bad_mask.sum()
            if n_bad > 0:
                print(f"  WARNING: {n_bad} Odds API rows have implausible moneylines (|ml| > 1500) — dropping.")
                new_df = new_df[~bad_mask]

    if new_df.empty:
        print("  No new odds from any source.")
        return cached

    combined = _merge_odds_into_cache(new_df, odds_cache)
    print(f"  Odds cached: {len(combined)} rows total")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Pitcher data from MLB Stats API
# ═══════════════════════════════════════════════════════════════════════════
def _fetch_boxscore_starter(game_pk):
    try:
        r = requests.get(f"{MLB_API}/game/{game_pk}/boxscore", timeout=10)
        r.raise_for_status()
        box = r.json().get("teams", {})
        h_pitchers = box.get("home", {}).get("pitchers", [])
        a_pitchers = box.get("away", {}).get("pitchers", [])
        return game_pk, h_pitchers[0] if h_pitchers else None, \
               a_pitchers[0] if a_pitchers else None
    except Exception:
        return game_pk, None, None


def fetch_starters(schedule_df):
    """Get starter IDs: boxscore for completed games, probable for today."""
    cache_path = CACHE_DIR / "starters.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()

    completed = schedule_df[schedule_df["is_completed"]].copy()
    today_upcoming = schedule_df[~schedule_df["is_completed"]].copy()

    done_pks = set(cached["game_pk"].tolist()) if not cached.empty else set()
    todo_pks = [pk for pk in completed["game_pk"] if pk not in done_pks]

    if todo_pks:
        print(f"  Fetching boxscore starters for {len(todo_pks)} completed games...")
        new_rows = []
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_fetch_boxscore_starter, pk): pk for pk in todo_pks}
            for i, fut in enumerate(as_completed(futs)):
                pk, h, a = fut.result()
                new_rows.append({"game_pk": pk, "home_starter_id": h,
                                 "away_starter_id": a})
                if (i + 1) % 200 == 0:
                    print(f"    {i + 1}/{len(todo_pks)}...")
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            cached = pd.concat([cached, new_df], ignore_index=True).drop_duplicates(
                subset=["game_pk"])
            cached.to_parquet(cache_path)

    # For today's upcoming games, use probable pitcher IDs
    prob = today_upcoming[["game_pk", "home_probable_id", "away_probable_id"]].rename(
        columns={"home_probable_id": "home_starter_id",
                 "away_probable_id": "away_starter_id"})
    result = pd.concat([cached, prob], ignore_index=True).drop_duplicates(
        subset=["game_pk"], keep="first")
    return result


def _fetch_pitcher_handedness(pid):
    try:
        r = requests.get(f"{MLB_API}/people/{pid}", timeout=10)
        r.raise_for_status()
        people = r.json().get("people", [{}])
        if not people:
            return None
        return people[0].get("pitchHand", {}).get("code")
    except Exception:
        return None


def _fetch_pitcher_season_stats(pid, season):
    try:
        r = requests.get(
            f"{MLB_API}/people/{pid}/stats",
            params={"stats": "season", "group": "pitching",
                    "season": season, "sportId": 1},
            timeout=10,
        )
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        s = splits[0]["stat"]
        ip_raw = float(s.get("inningsPitched", 0) or 0)
        ip = int(ip_raw) + (ip_raw % 1) / 0.3 * 0.333
        if ip == 0:
            return {}
        so = int(s.get("strikeOuts", 0) or 0)
        bb = int(s.get("baseOnBalls", 0) or 0)
        h = int(s.get("hits", 0) or 0)
        return {
            "sp_era":  float(s.get("era", np.nan) or np.nan),
            "sp_whip": (h + bb) / ip,
            "sp_k9":   so / ip * 9,
            "sp_bb9":  bb / ip * 9,
        }
    except Exception:
        return {}


def _fetch_pitcher_game_log(pid, season):
    try:
        r = requests.get(
            f"{MLB_API}/people/{pid}/stats",
            params={"stats": "gameLog", "group": "pitching",
                    "season": season, "sportId": 1},
            timeout=10,
        )
        r.raise_for_status()
        rows = []
        for split in r.json().get("stats", [{}])[0].get("splits", []):
            s = split.get("stat", {})
            if int(s.get("gamesStarted", 0) or 0) < 1:
                continue
            rows.append({
                "pitcher_id": pid,
                "season": season,
                "game_date": pd.to_datetime(split.get("date", "")),
                "ip": float(s.get("inningsPitched", 0) or 0),
                "er": int(s.get("earnedRuns", 0) or 0),
                "h":  int(s.get("hits", 0) or 0),
                "bb": int(s.get("baseOnBalls", 0) or 0),
                "so": int(s.get("strikeOuts", 0) or 0),
            })
        return rows
    except Exception:
        return []


def _compute_rolling_pitcher(logs_df, n=3):
    frames = []
    for pid, grp in logs_df.groupby("pitcher_id"):
        grp = grp.sort_values("game_date").copy()
        s_er = grp["er"].shift(1)
        s_ip = grp["ip"].shift(1).replace(0, np.nan)
        s_h  = grp["h"].shift(1)
        s_bb = grp["bb"].shift(1)
        s_so = grp["so"].shift(1)
        r_ip = s_ip.rolling(n, min_periods=1).sum().replace(0, np.nan)
        frames.append(pd.DataFrame({
            "pitcher_id": pid,
            "game_date":  grp["game_date"],
            "rolling_era":  s_er.rolling(n, min_periods=1).sum() / r_ip * 9,
            "rolling_whip": (s_h.rolling(n, min_periods=1).sum() +
                             s_bb.rolling(n, min_periods=1).sum()) / r_ip,
            "rolling_k9":   s_so.rolling(n, min_periods=1).sum() / r_ip * 9,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_pitcher_features(starters_df, schedule_df):
    """Season stats, handedness, rolling stats for starters in schedule_df."""
    merged = starters_df.merge(
        schedule_df[["game_pk", "game_date"]], on="game_pk", how="inner")
    merged["season"] = SEASON

    all_pids = set()
    for col in ["home_starter_id", "away_starter_id"]:
        all_pids |= set(merged[col].dropna().astype(int))
    all_pids = sorted(all_pids)

    if not all_pids:
        return pd.DataFrame(columns=["game_pk", "game_date"])

    # --- Season stats ---
    sp_cache = CACHE_DIR / "pitcher_season_stats.parquet"
    sp_cached = pd.read_parquet(sp_cache) if sp_cache.exists() else pd.DataFrame()
    # Always re-fetch current season (stats change daily)
    if not sp_cached.empty:
        sp_cached = sp_cached[sp_cached["season"] != SEASON]
    done_sp = (set(zip(sp_cached["pitcher_id"], sp_cached["season"]))
               if not sp_cached.empty else set())
    todo_sp = [(pid, SEASON) for pid in all_pids if (pid, SEASON) not in done_sp]

    if todo_sp:
        print(f"  Fetching season stats for {len(todo_sp)} pitchers...")
        new_sp = []
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_fetch_pitcher_season_stats, pid, s): (pid, s)
                    for pid, s in todo_sp}
            for i, fut in enumerate(as_completed(futs)):
                pid, s = futs[fut]
                new_sp.append({"pitcher_id": pid, "season": s, **fut.result()})
                if (i + 1) % 200 == 0:
                    print(f"    {i + 1}/{len(todo_sp)}...")
        sp_stats = pd.concat([sp_cached, pd.DataFrame(new_sp)],
                             ignore_index=True).drop_duplicates(
                             subset=["pitcher_id", "season"])
        sp_stats.to_parquet(sp_cache)
    else:
        sp_stats = sp_cached

    # --- Handedness (permanent, cached once) ---
    hand_cache = CACHE_DIR / "pitcher_handedness.parquet"
    hand_cached = pd.read_parquet(hand_cache) if hand_cache.exists() else pd.DataFrame()
    done_hand = set(hand_cached["pitcher_id"].tolist()) if not hand_cached.empty else set()
    todo_hand = [pid for pid in all_pids if pid not in done_hand]

    if todo_hand:
        print(f"  Fetching handedness for {len(todo_hand)} pitchers...")
        new_hand = []
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_fetch_pitcher_handedness, pid): pid for pid in todo_hand}
            for fut in as_completed(futs):
                pid = futs[fut]
                new_hand.append({"pitcher_id": pid, "pitch_hand": fut.result()})
        hand_df = pd.concat([hand_cached, pd.DataFrame(new_hand)],
                            ignore_index=True).drop_duplicates(subset=["pitcher_id"])
        hand_df.to_parquet(hand_cache)
    else:
        hand_df = hand_cached

    # --- Game logs + rolling (re-fetch current season) ---
    log_cache = CACHE_DIR / "pitcher_game_logs.parquet"
    log_cached = pd.read_parquet(log_cache) if log_cache.exists() else pd.DataFrame()
    if not log_cached.empty:
        log_cached = log_cached[log_cached["season"] != SEASON]
    done_log = (set(zip(log_cached["pitcher_id"], log_cached["season"]))
                if not log_cached.empty else set())
    todo_logs = [(pid, SEASON) for pid in all_pids
                 if (pid, SEASON) not in done_log]
    if todo_logs:
        print(f"  Fetching game logs for {len(todo_logs)} pitchers...")
        new_logs = []
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_fetch_pitcher_game_log, pid, s): (pid, s)
                    for pid, s in todo_logs}
            for fut in as_completed(futs):
                new_logs.extend(fut.result())
        if new_logs:
            logs_df = pd.concat([log_cached, pd.DataFrame(new_logs)],
                                ignore_index=True).drop_duplicates(
                                subset=["pitcher_id", "game_date"])
            logs_df.to_parquet(log_cache)
        else:
            logs_df = log_cached
    else:
        logs_df = log_cached

    rolling_df = (_compute_rolling_pitcher(logs_df, n=3)
                  if not logs_df.empty else pd.DataFrame())

    # --- Assemble pitcher feature table ---
    result = merged[["game_pk", "game_date", "season",
                     "home_starter_id", "away_starter_id"]].copy()

    for side, id_col in [("home", "home_starter_id"), ("away", "away_starter_id")]:
        ss = sp_stats.rename(columns={
            "pitcher_id": id_col,
            "sp_era":  f"{side}_sp_era",  "sp_whip": f"{side}_sp_whip",
            "sp_k9":   f"{side}_sp_k9",   "sp_bb9":  f"{side}_sp_bb9",
        })
        result = result.merge(
            ss[["season", id_col, f"{side}_sp_era", f"{side}_sp_whip",
                f"{side}_sp_k9", f"{side}_sp_bb9"]].drop_duplicates(),
            on=["season", id_col], how="left")

        if not rolling_df.empty:
            rr = rolling_df.rename(columns={
                "pitcher_id": id_col,
                "rolling_era":  f"{side}_rolling_era",
                "rolling_whip": f"{side}_rolling_whip",
                "rolling_k9":   f"{side}_rolling_k9",
            })
            result = result.merge(
                rr[[id_col, "game_date",
                    f"{side}_rolling_era", f"{side}_rolling_whip",
                    f"{side}_rolling_k9"]].drop_duplicates(),
                on=[id_col, "game_date"], how="left")

        hh = hand_df.rename(columns={
            "pitcher_id": id_col,
            "pitch_hand": f"{side}_pitch_hand",
        })
        result = result.merge(
            hh[[id_col, f"{side}_pitch_hand"]].drop_duplicates(),
            on=id_col, how="left")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Weather from Open-Meteo
# ═══════════════════════════════════════════════════════════════════════════
def fetch_weather(schedule_df, refresh_dates=None):
    """Fetch weather for games in schedule_df, forcing re-fetch for refresh_dates."""
    cache_path = CACHE_DIR / "weather.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()

    if not cached.empty:
        cached["game_date"] = pd.to_datetime(cached["game_date"])
        if refresh_dates:
            cached = cached[~cached["game_date"].dt.date.isin(refresh_dates)]
        done = set(zip(cached["game_date"].dt.date, cached["home_team"]))
    else:
        done = set()

    needed = schedule_df[["game_date", "home_team"]].drop_duplicates()
    needed = needed[needed.apply(
        lambda r: (r["game_date"].date(), r["home_team"]) not in done, axis=1)]

    if needed.empty:
        print("  Weather already cached.")
        return cached

    today = datetime.today().date()
    new_rows = []

    for i, (home_team, grp) in enumerate(needed.groupby("home_team")):
        if home_team not in PARK_COORDS:
            continue
        lat, lon = PARK_COORDS[home_team]
        dates = sorted(set(d.strftime("%Y-%m-%d") for d in grp["game_date"]))
        past_dates = [d for d in dates
                      if datetime.strptime(d, "%Y-%m-%d").date() < today]
        future_dates = [d for d in dates
                        if datetime.strptime(d, "%Y-%m-%d").date() >= today]

        if past_dates:
            try:
                r = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat, "longitude": lon,
                        "start_date": past_dates[0], "end_date": past_dates[-1],
                        "daily": ("temperature_2m_max,wind_speed_10m_max,"
                                  "wind_direction_10m_dominant"),
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "kmh", "timezone": "auto",
                    },
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json().get("daily", {})
                dm = {d: j for j, d in enumerate(data.get("time", []))}
                temps = data.get("temperature_2m_max", [])
                winds = data.get("wind_speed_10m_max", [])
                wdirs = data.get("wind_direction_10m_dominant", [])
                for d in past_dates:
                    idx = dm.get(d)
                    if idx is None:
                        continue
                    new_rows.append({
                        "game_date": d, "home_team": home_team,
                        "temp_c": temps[idx] if idx < len(temps) else np.nan,
                        "wind_speed_kmh": winds[idx] if idx < len(winds) else np.nan,
                        "wind_dir_deg": wdirs[idx] if idx < len(wdirs) else np.nan,
                    })
            except Exception as e:
                print(f"  Weather archive error ({home_team}): {e}")

        if future_dates:
            try:
                r = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat, "longitude": lon,
                        "daily": ("temperature_2m_max,wind_speed_10m_max,"
                                  "wind_direction_10m_dominant"),
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "kmh", "timezone": "auto",
                        "forecast_days": 7,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json().get("daily", {})
                dm = {d: j for j, d in enumerate(data.get("time", []))}
                temps = data.get("temperature_2m_max", [])
                winds = data.get("wind_speed_10m_max", [])
                wdirs = data.get("wind_direction_10m_dominant", [])
                for d in future_dates:
                    idx = dm.get(d)
                    if idx is None:
                        continue
                    new_rows.append({
                        "game_date": d, "home_team": home_team,
                        "temp_c": temps[idx] if idx < len(temps) else np.nan,
                        "wind_speed_kmh": winds[idx] if idx < len(winds) else np.nan,
                        "wind_dir_deg": wdirs[idx] if idx < len(wdirs) else np.nan,
                    })
            except Exception as e:
                print(f"  Weather forecast error ({home_team}): {e}")

        time.sleep(0.5)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df["game_date"] = pd.to_datetime(new_df["game_date"])
        weather = pd.concat([cached, new_df], ignore_index=True)
        weather = weather.drop_duplicates(subset=["game_date", "home_team"])
        weather.to_parquet(cache_path)
        print(f"  Weather: {len(weather)} rows cached")
        return weather

    print(f"  Weather: {len(cached)} rows (from cache)")
    return cached


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: FanGraphs team stats
# ═══════════════════════════════════════════════════════════════════════════
def _normalize_fg_team(name):
    if not name:
        return None
    name = str(name).strip()
    if name in FG_TEAM_MAP:
        return FG_TEAM_MAP[name]
    for full, abb in FULL_TO_ABBR.items():
        if name.lower() in full.lower() or full.lower() in name.lower():
            return abb
    return None


def _parse_fg_pct(val):
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return val / 100.0 if abs(val) > 1 else val
    s = str(val).replace("%", "").replace(" ", "").strip()
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def fetch_fangraphs_stats():
    """
    Fetch current season-to-date team batting + pitching stats.
    Caches a daily snapshot; historical game dates use the closest prior snapshot.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    cache_file = CACHE_DIR / f"fg_snapshot_{today_str}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            team_stats = json.load(f)
        print(f"  FanGraphs: loaded cached snapshot for {today_str}")
        return _fg_dict_to_df(team_stats, today_str)

    from pools import USER_AGENTS
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    team_stats = {}

    bat_url = (
        f"https://www.fangraphs.com/api/leaders/major-league/data"
        f"?pos=all&stats=bat&lg=all&qual=0&type=8"
        f"&season={SEASON}&season1={SEASON}&month=0"
        f"&team=0%2Cts&ind=0&rost=0&age=0&players=0"
        f"&pageitems=100&pagenum=1"
    )
    try:
        r = requests.get(bat_url, headers=headers, timeout=30)
        r.raise_for_status()
        bat_data = r.json()
        if isinstance(bat_data, dict):
            bat_data = bat_data.get("data", bat_data.get("Data", []))
        if isinstance(bat_data, list):
            for entry in bat_data:
                team_name = (entry.get("TeamName") or entry.get("Team")
                             or entry.get("teamName") or "")
                abb = _normalize_fg_team(team_name)
                if not abb:
                    continue
                team_stats.setdefault(abb, {})
                team_stats[abb].update({
                    "avg":      entry.get("AVG"),
                    "obp":      entry.get("OBP"),
                    "slg":      entry.get("SLG"),
                    "woba":     entry.get("wOBA"),
                    "wrc_plus": entry.get("wRC+"),
                    "war":      entry.get("WAR"),
                    "k_pct":    entry.get("K%"),
                    "bb_pct":   entry.get("BB%"),
                    "owar":     entry.get("Off"),
                })
            print(f"  FanGraphs batting: {len(bat_data)} teams")
    except Exception as e:
        print(f"  WARNING: FanGraphs batting fetch failed: {e}")

    pit_url = bat_url.replace("stats=bat", "stats=pit")
    try:
        r = requests.get(pit_url, headers=headers, timeout=30)
        r.raise_for_status()
        pit_data = r.json()
        if isinstance(pit_data, dict):
            pit_data = pit_data.get("data", pit_data.get("Data", []))
        if isinstance(pit_data, list):
            for entry in pit_data:
                team_name = (entry.get("TeamName") or entry.get("Team")
                             or entry.get("teamName") or "")
                abb = _normalize_fg_team(team_name)
                if not abb:
                    continue
                team_stats.setdefault(abb, {})
                team_stats[abb].update({
                    "k_per_9":  entry.get("K/9"),
                    "bb_per_9": entry.get("BB/9"),
                    "hr_per_9": entry.get("HR/9"),
                    "era":      entry.get("ERA"),
                    "fip":      entry.get("FIP"),
                })
            print(f"  FanGraphs pitching: {len(pit_data)} teams")
    except Exception as e:
        print(f"  WARNING: FanGraphs pitching fetch failed: {e}")

    if team_stats:
        with open(cache_file, "w") as f:
            json.dump(team_stats, f, indent=2)
        print(f"  FanGraphs snapshot saved for {today_str}")

    return _fg_dict_to_df(team_stats, today_str)


def _fg_dict_to_df(team_stats, snapshot_date):
    rows = []
    for team, stats in team_stats.items():
        row = {"team_abbr": team, "fg_date": snapshot_date}
        for k, v in stats.items():
            if k in ("k_pct", "bb_pct"):
                row[k] = _parse_fg_pct(v)
            else:
                try:
                    row[k] = float(v) if v is not None else np.nan
                except (TypeError, ValueError):
                    row[k] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _get_fg_snapshot_for_date(game_date_str):
    """Return the best FG snapshot dict for a game date (closest prior cached)."""
    snapshots = sorted(CACHE_DIR.glob("fg_snapshot_*.json"))
    best = None
    for s in snapshots:
        snap_date = s.stem.replace("fg_snapshot_", "")
        if snap_date <= game_date_str:
            best = s
    if best is None and snapshots:
        best = snapshots[0]
    if best is None:
        return {}
    with open(best) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Assembly
# ═══════════════════════════════════════════════════════════════════════════
def assemble(games_to_process, odds_df, pitcher_df, weather_df, fg_df):
    """
    Build feature rows ONLY for games_to_process (already filtered to <= today).
    Returns a DataFrame with the canonical column set.
    """
    master = games_to_process.copy()

    # --- Odds ---
    # Match on UTC date derived from game_time_utc + team names.
    # The Odds API returns UTC commence_times, so using the UTC date of the
    # game_time_utc avoids the ±1-day mismatch for late West Coast games and
    # correctly disambiguates back-to-back series games.
    if not odds_df.empty:
        odds_df = odds_df.copy()
        odds_df["game_date"] = pd.to_datetime(odds_df["game_date"])

        # Compute UTC date for each game row
        if "game_time_utc" in master.columns:
            master["_utc_date"] = pd.to_datetime(master["game_time_utc"]).dt.normalize()
        else:
            master["_utc_date"] = pd.to_datetime(master["game_date"])

        odds_cols = ["game_date", "home_team", "away_team",
                     "open_home_ml", "open_away_ml",
                     "close_home_ml", "close_away_ml",
                     "open_total", "close_total", "odds_source"]
        odds_keyed = odds_df[odds_cols].rename(columns={"game_date": "_utc_date"})
        master = master.merge(odds_keyed, on=["_utc_date", "home_team", "away_team"], how="left")
        master = master.drop(columns=["_utc_date"])

    # --- Pitchers ---
    if not pitcher_df.empty:
        ptch_cols = [c for c in pitcher_df.columns if c != "season"]
        master = master.merge(pitcher_df[ptch_cols],
                              on=["game_pk", "game_date"], how="left")

        for side in ["home", "away"]:
            hand_col = f"{side}_pitch_hand"
            id_col = f"{side}_starter_id"
            if hand_col in master.columns:
                has_pitcher = (master[id_col].notna()
                               if id_col in master.columns
                               else pd.Series(True, index=master.index))
                master[f"{side}_pitcher_is_lefty"] = np.where(
                    has_pitcher, (master[hand_col] == "L").astype(float), np.nan)
                master.drop(columns=[hand_col], inplace=True)

        if ("home_pitcher_is_lefty" in master.columns
                and "away_pitcher_is_lefty" in master.columns):
            master["pitcher_handedness_diff"] = (
                master["home_pitcher_is_lefty"] - master["away_pitcher_is_lefty"])

    # --- Weather ---
    if not weather_df.empty:
        weather_df = weather_df.copy()
        weather_df["game_date"] = pd.to_datetime(weather_df["game_date"])
        master = master.merge(
            weather_df[["game_date", "home_team", "temp_c",
                        "wind_speed_kmh", "wind_dir_deg"]],
            on=["game_date", "home_team"], how="left")

    # --- FanGraphs team stats (per-date snapshot lookup) ---
    fg_stat_names = ["avg", "obp", "slg", "woba", "wrc_plus", "war",
                     "k_pct", "bb_pct", "k_per_9", "bb_per_9", "hr_per_9",
                     "era", "fip", "owar"]

    game_dates = master["game_date"].dt.strftime("%Y-%m-%d").unique()
    fg_cache = {}
    for gd_str in game_dates:
        fg_cache[gd_str] = _get_fg_snapshot_for_date(gd_str)

    for side in ["home", "away"]:
        team_col = f"{side}_team"
        for stat in fg_stat_names:
            col_name = f"{side}_{stat}"
            vals = []
            for _, row in master.iterrows():
                gd_str = row["game_date"].strftime("%Y-%m-%d")
                team = row[team_col]
                snap = fg_cache.get(gd_str, {})
                team_data = snap.get(team, {})
                raw = team_data.get(stat)
                if raw is None:
                    vals.append(np.nan)
                elif stat in ("k_pct", "bb_pct"):
                    vals.append(_parse_fg_pct(raw))
                else:
                    try:
                        vals.append(float(raw))
                    except (TypeError, ValueError):
                        vals.append(np.nan)
            master[col_name] = vals

    # --- Implied probabilities ---
    def _ml_to_raw(s):
        s = pd.to_numeric(s, errors="coerce")
        prob = pd.Series(index=s.index, dtype=float)
        fav = s <= -100
        dog = s >= 100
        prob[fav] = (-s[fav]) / (-s[fav] + 100)
        prob[dog] = 100 / (s[dog] + 100)
        return prob

    if "close_home_ml" in master.columns:
        h_raw = _ml_to_raw(master["close_home_ml"])
        a_raw = _ml_to_raw(master["close_away_ml"])
        total = h_raw + a_raw
        master["home_implied_prob"] = h_raw / total
        master["away_implied_prob"] = a_raw / total

    return _ensure_columns(master)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fetch 2026 MLB data for model predictions.")
    parser.add_argument("--today-only", action="store_true",
                        help="Only scrape odds for today's games")
    args = parser.parse_args()

    t0 = time.time()
    today = pd.to_datetime(datetime.today().date())
    tomorrow = today + timedelta(days=1)

    # --- Load existing CSV ---
    existing = _load_existing()
    if not existing.empty:
        print(f"Loaded existing {OUTPUT_CSV}: {len(existing)} rows")
    else:
        print(f"No existing {OUTPUT_CSV} found, starting fresh.")

    # ── STEP 1: Schedule ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1/6: Fetching fresh schedule from MLB Stats API")
    print("=" * 60)
    full_schedule = fetch_schedule()
    if full_schedule.empty:
        print("No games found. Exiting.")
        return

    # ── Only past games + today/tomorrow window (no full-season skeleton) ─
    past_games = full_schedule[full_schedule["game_date"] < today].copy()
    window_games = full_schedule[
        (full_schedule["game_date"] >= today)
        & (full_schedule["game_date"] <= tomorrow)
    ].copy()

    print(f"  Past games:        {len(past_games)}")
    print(f"  Today + tomorrow:  {len(window_games)} games (always refreshed)")

    # ── Determine which past games are already complete in CSV ─────────────
    complete_pks, _ = _split_existing(existing, set(full_schedule["game_pk"]), today)
    past_to_process = past_games[~past_games["game_pk"].isin(complete_pks)].copy()

    # Today/tomorrow are always re-processed for schedule accuracy
    to_process = pd.concat([past_to_process, window_games], ignore_index=True)
    to_process = to_process.drop_duplicates(subset=["game_pk"], keep="last")

    if to_process.empty:
        print("\n  No games to process.")
        if not existing.empty:
            result = existing[existing["game_pk"].isin(complete_pks)].copy()
            result = _ensure_columns(result)
            result = result.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
            result.to_csv(OUTPUT_CSV, index=False)
            try:
                DB.upsert_games(result)
            except Exception as e:
                print(f"  WARNING: DB upsert failed ({e}).")
            print(f"  Saved {len(result)} rows to {OUTPUT_CSV} (stale rows removed)")
        return

    print(f"  Games to process: {len(to_process)} "
          f"({to_process['is_completed'].sum()} completed, "
          f"{(~to_process['is_completed']).sum()} upcoming)")

    # ── STEP 2: Odds ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2/6: Fetching odds")
    print("=" * 60)
    odds_df = fetch_odds(to_process, today_only=args.today_only)

    # ── STEP 3: Pitcher data ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3/6: Fetching pitcher data")
    print("=" * 60)
    starters_df = fetch_starters(to_process)
    pitcher_df = fetch_pitcher_features(starters_df, to_process)
    print(f"  Pitcher features: {pitcher_df.shape}")

    # ── STEP 4: Weather (force re-fetch today/tomorrow forecasts) ────────
    print("\n" + "=" * 60)
    print("STEP 4/6: Fetching weather")
    print("=" * 60)
    weather_df = fetch_weather(
        to_process, refresh_dates={today.date(), tomorrow.date()})

    # ── STEP 5: FanGraphs stats ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5/6: Fetching FanGraphs team stats")
    print("=" * 60)
    fg_df = fetch_fangraphs_stats()

    # ── STEP 6: Assembly ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6/6: Assembling CSV")
    print("=" * 60)
    new_rows = assemble(to_process, odds_df, pitcher_df, weather_df, fg_df)

    # Keep existing complete rows that aren't being re-processed
    new_pks = set(to_process["game_pk"])
    if not existing.empty and complete_pks:
        keep_pks = complete_pks - new_pks
        complete_rows = existing[existing["game_pk"].isin(keep_pks)].copy()
        complete_rows = _ensure_columns(complete_rows)
    else:
        complete_rows = pd.DataFrame()

    parts = [df for df in [complete_rows, new_rows] if not df.empty]
    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates(subset=["game_pk"], keep="first")
    result = result.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    result.to_csv(OUTPUT_CSV, index=False)
    try:
        DB.upsert_games(result)
    except Exception as e:
        print(f"  WARNING: DB upsert failed ({e}). CSV still written.")

    elapsed = time.time() - t0
    n_complete = result["home_win"].notna().sum()
    n_upcoming = result["home_win"].isna().sum()
    n_with_odds = result["close_home_ml"].notna().sum()
    n_with_fg = result["home_avg"].notna().sum()
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.0f}s")
    print(f"  Output:      {OUTPUT_CSV}")
    print(f"  Total rows:  {len(result)}")
    print(f"  Completed:   {n_complete} (with scores)")
    print(f"  Upcoming:    {n_upcoming} (today/tomorrow)")
    print(f"  With odds:   {n_with_odds}")
    print(f"  With FG:     {n_with_fg}")
    gd = pd.to_datetime(result["game_date"])
    print(f"  Date range:  {gd.min().date()} -> {gd.max().date()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
