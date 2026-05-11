#!/usr/bin/env python3
"""Real sandbox data source pipelines.

Current sources:
- MLB StatsAPI boxscore/feed: lineups, catchers, umpires, starter workload,
  bullpen workload/freshness, roof/venue metadata.
- Open-Meteo archive: hourly game-time weather.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab import planned_features  # noqa: E402


CACHE_DIR = LAB_DIR / "cache"
OUTPUT_DIR = LAB_DIR / "output"
MASTER_PATH = OUTPUT_DIR / "master_sandbox_mlb.csv"
MLB_RAW_PATH = CACHE_DIR / "mlb_statsapi_game_rows.parquet"
MLB_FEATURES_PATH = CACHE_DIR / "mlb_statsapi_features.parquet"
WEATHER_HOURLY_PATH = CACHE_DIR / "openmeteo_hourly_features.parquet"
PITCHER_HANDS_PATH = CACHE_DIR / "mlb_pitcher_hands.parquet"
MLB_GAME_TIMES_PATH = CACHE_DIR / "mlb_game_times.parquet"

MLB_BOX_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
MLB_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

TEAM_TIMEZONES = {
    "ARI": "America/Phoenix",
    "ATL": "America/New_York",
    "BAL": "America/New_York",
    "BOS": "America/New_York",
    "CHC": "America/Chicago",
    "CHW": "America/Chicago",
    "CIN": "America/New_York",
    "CLE": "America/New_York",
    "COL": "America/Denver",
    "DET": "America/New_York",
    "HOU": "America/Chicago",
    "KCR": "America/Chicago",
    "LAA": "America/Los_Angeles",
    "LAD": "America/Los_Angeles",
    "MIA": "America/New_York",
    "MIL": "America/Chicago",
    "MIN": "America/Chicago",
    "NYM": "America/New_York",
    "NYY": "America/New_York",
    "ATH": "America/Los_Angeles",
    "OAK": "America/Los_Angeles",
    "PHI": "America/New_York",
    "PIT": "America/New_York",
    "SDP": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "SFG": "America/Los_Angeles",
    "STL": "America/Chicago",
    "TBR": "America/New_York",
    "TEX": "America/Chicago",
    "TOR": "America/Toronto",
    "WSN": "America/New_York",
}


def _num(value) -> float:
    if value in (None, "", ".---", "-.--"):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _ip_to_float(value) -> float:
    if value in (None, "", ".---"):
        return np.nan
    text = str(value)
    if "." not in text:
        return _num(text)
    whole, frac = text.split(".", 1)
    try:
        return float(whole) + (float(frac[:1]) / 3.0)
    except ValueError:
        return np.nan


def _request_json(url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _player_map(feed: dict | None) -> dict[int, dict]:
    players = ((feed or {}).get("gameData", {}).get("players") or {})
    out: dict[int, dict] = {}
    for key, player in players.items():
        try:
            pid = int(str(key).replace("ID", ""))
        except ValueError:
            continue
        out[pid] = player
    return out


def _official_id(box: dict | None, feed: dict | None) -> int | None:
    officials = ((box or {}).get("officials") or []) or (
        ((feed or {}).get("liveData", {}).get("boxscore", {}) or {}).get("officials") or []
    )
    for item in officials:
        if item.get("officialType") == "Home Plate":
            oid = (item.get("official") or {}).get("id")
            return int(oid) if oid is not None else None
    return None


def _venue_fields(feed: dict | None) -> dict:
    venue = ((feed or {}).get("gameData", {}) or {}).get("venue", {}) or {}
    field = venue.get("fieldInfo", {}) or {}
    loc = venue.get("location", {}) or {}
    roof_type = str(field.get("roofType") or "").lower()
    return {
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "venue_azimuth_angle": _num(loc.get("azimuthAngle")),
        "venue_elevation_ft": _num(loc.get("elevation")),
        "roof_type": field.get("roofType"),
        "roof_possible_flag": 1.0 if "retractable" in roof_type or "dome" in roof_type else 0.0,
        "roof_closed_flag": 1.0 if "dome" in roof_type else np.nan,
    }


def _lineup_fields(side: str, team: dict, players: dict[int, dict]) -> dict:
    out: dict[str, float | int | None] = {
        f"{side}_lineup_lhb_count": np.nan,
        f"{side}_lineup_rhb_count": np.nan,
        f"{side}_catcher_id": np.nan,
    }
    lhb = 0
    rhb = 0
    starters = 0
    catcher_id = None
    for key, player in (team.get("players") or {}).items():
        order = player.get("battingOrder")
        if not order:
            continue
        try:
            if int(order) > 900:
                continue
        except ValueError:
            continue
        pid = int(str(key).replace("ID", ""))
        starters += 1
        meta = players.get(pid, {})
        bat = ((meta.get("batSide") or {}).get("code") or "").upper()
        if bat == "L":
            lhb += 1
        elif bat == "R":
            rhb += 1
        elif bat == "S":
            lhb += 0.5
            rhb += 0.5
        pos = ((player.get("position") or {}).get("abbreviation") or "").upper()
        if pos == "C" and catcher_id is None:
            catcher_id = pid
    if starters:
        out[f"{side}_lineup_lhb_count"] = float(lhb)
        out[f"{side}_lineup_rhb_count"] = float(rhb)
    out[f"{side}_catcher_id"] = catcher_id if catcher_id is not None else np.nan
    return out


def _pitcher_rows(game_pk: int, game_date: pd.Timestamp, side: str, team_abbr: str, team: dict) -> list[dict]:
    pitchers = [int(pid) for pid in team.get("pitchers") or []]
    rows: list[dict] = []
    for order, pid in enumerate(pitchers):
        p = (team.get("players") or {}).get(f"ID{pid}", {})
        stats = ((p.get("stats") or {}).get("pitching") or {})
        rows.append(
            {
                "game_pk": game_pk,
                "game_date": game_date,
                "team": team_abbr,
                "side": side,
                "pitcher_id": pid,
                "is_starter": order == 0 or int(stats.get("gamesStarted", 0) or 0) > 0,
                "pitch_count": _num(stats.get("numberOfPitches", stats.get("pitchesThrown"))),
                "bf": _num(stats.get("battersFaced")),
                "ip": _ip_to_float(stats.get("inningsPitched")),
                "outs": _num(stats.get("outs")),
                "er": _num(stats.get("earnedRuns")),
                "runs": _num(stats.get("runs")),
                "hits": _num(stats.get("hits")),
                "bb": _num(stats.get("baseOnBalls")),
                "so": _num(stats.get("strikeOuts")),
                "hr": _num(stats.get("homeRuns")),
                "holds": _num(stats.get("holds")),
                "saves": _num(stats.get("saves")),
                "games_finished": _num(stats.get("gamesFinished")),
            }
        )
    return rows


def fetch_one_mlb_game(row: dict) -> dict:
    game_pk = int(row["game_pk"])
    game_date = pd.Timestamp(row["game_date"])
    feed = _request_json(MLB_FEED_URL.format(game_pk=game_pk))
    box = (((feed or {}).get("liveData") or {}).get("boxscore") or None)
    if not box:
        box = _request_json(MLB_BOX_URL.format(game_pk=game_pk))
    if not box:
        return {"game_pk": game_pk, "ok": False}

    players = _player_map(feed)
    box_teams = box.get("teams", {}) or {}
    home_team = box_teams.get("home", {}) or {}
    away_team = box_teams.get("away", {}) or {}
    out: dict = {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "ok": True,
        "umpire_id": _official_id(box, feed),
        "pitcher_appearances": [],
    }
    out.update(_venue_fields(feed))
    out.update(_lineup_fields("home", home_team, players))
    out.update(_lineup_fields("away", away_team, players))
    out["pitcher_appearances"] = (
        _pitcher_rows(game_pk, game_date, "home", str(row.get("home_team")), home_team)
        + _pitcher_rows(game_pk, game_date, "away", str(row.get("away_team")), away_team)
    )
    return out


def fetch_mlb_statsapi(master: pd.DataFrame, limit: int | None = None, workers: int = 16) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    needed = master[["game_pk", "game_date", "home_team", "away_team"]].dropna(subset=["game_pk"]).copy()
    needed["game_pk"] = needed["game_pk"].astype(int)

    cached = pd.read_parquet(MLB_RAW_PATH) if MLB_RAW_PATH.exists() else pd.DataFrame()
    done = set(cached["game_pk"].astype(int)) if not cached.empty else set()
    todo = needed[~needed["game_pk"].isin(done)].copy()
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print(f"MLB StatsAPI cache hit ({len(cached):,} games)", flush=True)
    else:
        print(f"MLB StatsAPI fetching {len(todo):,} games ({len(done):,} cached)", flush=True)
        fetched_total = 0
        chunk_size = max(workers * 50, 500)
        for start in range(0, len(todo), chunk_size):
            chunk = todo.iloc[start:start + chunk_size]
            rows: list[dict] = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(fetch_one_mlb_game, r.to_dict()): int(r["game_pk"]) for _, r in chunk.iterrows()}
                for fut in as_completed(futures):
                    rows.append(fut.result())
            fetched_total += len(rows)
            cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            cached = cached.drop_duplicates(subset=["game_pk"], keep="last")
            cached.to_parquet(MLB_RAW_PATH, index=False)
            print(f"  fetched {fetched_total:,}/{len(todo):,}; cache={len(cached):,}", flush=True)

    features = build_mlb_statsapi_features(cached)
    features.to_parquet(MLB_FEATURES_PATH, index=False)
    print(f"MLB StatsAPI features wrote {MLB_FEATURES_PATH} rows={len(features):,}")


def _appearance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, game in raw.iterrows():
        appearances = game.get("pitcher_appearances")
        if appearances is None or (isinstance(appearances, float) and math.isnan(appearances)):
            appearances = []
        for item in list(appearances):
            rows.append(item)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["season"] = out["game_date"].dt.year
    for col in ["pitcher_id", "pitch_count", "bf", "ip", "er", "bb", "so", "hr", "holds", "saves", "games_finished"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["game_date", "game_pk", "team"]).reset_index(drop=True)


def _fetch_pitcher_hands(appearances: pd.DataFrame) -> pd.DataFrame:
    ids = sorted({int(pid) for pid in appearances["pitcher_id"].dropna().astype(int)})
    cached = pd.read_parquet(PITCHER_HANDS_PATH) if PITCHER_HANDS_PATH.exists() else pd.DataFrame(columns=["pitcher_id", "pitch_hand"])
    done = set(cached["pitcher_id"].dropna().astype(int)) if not cached.empty else set()
    missing = [pid for pid in ids if pid not in done]
    rows: list[dict] = []
    for start in range(0, len(missing), 100):
        chunk = missing[start:start + 100]
        if not chunk:
            continue
        try:
            r = requests.get(MLB_PEOPLE_URL, params={"personIds": ",".join(map(str, chunk))}, timeout=30)
            r.raise_for_status()
            for person in r.json().get("people", []):
                pid = person.get("id")
                hand = ((person.get("pitchHand") or {}).get("code") or "").upper()
                if pid is not None:
                    rows.append({"pitcher_id": int(pid), "pitch_hand": hand if hand in {"L", "R"} else None})
        except Exception:
            rows.extend({"pitcher_id": pid, "pitch_hand": None} for pid in chunk)
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached = cached.drop_duplicates("pitcher_id", keep="last")
        cached.to_parquet(PITCHER_HANDS_PATH, index=False)
    return cached


def _starter_workload(master: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    starts = appearances[appearances["is_starter"]].copy()
    rows: list[dict] = []
    history: dict[int, list[dict]] = {}
    for _, row in starts.sort_values(["game_date", "game_pk"]).iterrows():
        pid = int(row["pitcher_id"])
        prev = history.get(pid, [])
        prev3 = prev[-3:]
        prior_date = prev[-1]["game_date"] if prev else pd.NaT
        rows.append(
            {
                "game_pk": int(row["game_pk"]),
                "side": row["side"],
                "sp_days_rest": (pd.Timestamp(row["game_date"]) - pd.Timestamp(prior_date)).days if pd.notna(prior_date) else np.nan,
                "sp_pitch_count_l3": np.nanmean([p["pitch_count"] for p in prev3]) if prev3 else np.nan,
                "sp_bf_l3": np.nanmean([p["bf"] for p in prev3]) if prev3 else np.nan,
                "sp_ip_l3": np.nanmean([p["ip"] for p in prev3]) if prev3 else np.nan,
            }
        )
        history.setdefault(pid, []).append(row.to_dict())
    wide = pd.DataFrame({"game_pk": master["game_pk"].astype(int)})
    if not rows:
        return wide
    workload = pd.DataFrame(rows)
    for side in ["home", "away"]:
        part = workload[workload["side"] == side].drop(columns=["side"]).rename(
            columns={
                "sp_days_rest": f"{side}_sp_days_rest",
                "sp_pitch_count_l3": f"{side}_sp_pitch_count_l3",
                "sp_bf_l3": f"{side}_sp_bf_l3",
                "sp_ip_l3": f"{side}_sp_ip_l3",
            }
        )
        wide = wide.merge(part, on="game_pk", how="left")
    return wide


def _bullpen_features(master: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    relievers = appearances[~appearances["is_starter"]].copy()
    out = master[["game_pk", "game_date", "home_team", "away_team"]].copy()
    out["game_pk"] = out["game_pk"].astype(int)
    if relievers.empty:
        return out[["game_pk"]]

    hands = _fetch_pitcher_hands(appearances)
    if not hands.empty:
        relievers = relievers.merge(hands, on="pitcher_id", how="left")
    elif "pitch_hand" not in relievers.columns:
        relievers["pitch_hand"] = None

    relievers["game_date"] = pd.to_datetime(relievers["game_date"])
    relievers["game_day"] = relievers["game_date"].dt.normalize()
    relievers["is_high_leverage_rp"] = ((relievers["saves"].fillna(0) > 0) | (relievers["holds"].fillna(0) > 0)).astype(float)
    relievers["high_lev_pitch_count"] = relievers["pitch_count"].fillna(0) * relievers["is_high_leverage_rp"]
    relievers["closer_used_flag"] = (relievers["saves"].fillna(0) > 0).astype(float)
    daily = (
        relievers.groupby(["team", "game_day"], as_index=False)
        .agg(
            bp_pitch_count=("pitch_count", "sum"),
            bp_bf=("bf", "sum"),
            bp_ip=("ip", "sum"),
            bp_high_lev_pitches=("high_lev_pitch_count", "sum"),
            bp_er=("er", "sum"),
            bp_bb=("bb", "sum"),
            bp_so=("so", "sum"),
            bp_hr=("hr", "sum"),
            closer_used=("closer_used_flag", "max"),
            top3_used=("pitcher_id", "nunique"),
        )
        .sort_values(["team", "game_day"])
    )

    frames: list[pd.DataFrame] = []
    for team, grp in daily.groupby("team", sort=False):
        grp = grp.sort_values("game_day").set_index("game_day")
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_idx)
        grp["team"] = team
        num_cols = [c for c in grp.columns if c != "team"]
        grp[num_cols] = grp[num_cols].fillna(0.0)
        shifted = grp[num_cols].shift(1).fillna(0.0)
        feat = pd.DataFrame({"team": team, "game_day": full_idx})
        for days in [1, 2, 3]:
            feat[f"bp_pitches_{days}d"] = shifted["bp_pitch_count"].rolling(days, min_periods=1).sum().values
        feat["bp_bf_2d"] = shifted["bp_bf"].rolling(2, min_periods=1).sum().values
        feat["bp_ip_2d"] = shifted["bp_ip"].rolling(2, min_periods=1).sum().values
        feat["bp_high_leverage_pitches_2d"] = shifted["bp_high_lev_pitches"].rolling(2, min_periods=1).sum().values
        feat["closer_used_yesterday"] = shifted["closer_used"].rolling(1, min_periods=1).sum().clip(0, 1).values
        feat["top3_rp_used_2d"] = shifted["top3_used"].rolling(2, min_periods=1).sum().values

        er30 = shifted["bp_er"].rolling(30, min_periods=1).sum()
        ip30 = shifted["bp_ip"].rolling(30, min_periods=1).sum().replace(0, np.nan)
        bb30 = shifted["bp_bb"].rolling(30, min_periods=1).sum()
        so30 = shifted["bp_so"].rolling(30, min_periods=1).sum()
        hr30 = shifted["bp_hr"].rolling(30, min_periods=1).sum()
        bf30 = shifted["bp_bf"].rolling(30, min_periods=1).sum().replace(0, np.nan)
        feat["bp_era_30d"] = (er30 / ip30 * 9.0).values
        feat["bp_fip_30d"] = (((13 * hr30) + (3 * bb30) - (2 * so30)) / ip30).values
        feat["bp_k_minus_bb_30d"] = ((so30 - bb30) / bf30).values
        feat["bp_freshness_score"] = -(feat["bp_pitches_1d"] + 0.5 * feat["bp_pitches_2d"])
        feat["bp_fatigue_expected"] = -feat["bp_freshness_score"]
        frames.append(feat)

    by_team_day = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out["game_day"] = pd.to_datetime(out["game_date"]).dt.normalize()
    for side in ["home", "away"]:
        part = by_team_day.rename(columns={"team": f"{side}_team"})
        rename = {
            c: f"{side}_{c}"
            for c in part.columns
            if c not in {f"{side}_team", "game_day"}
        }
        part = part.rename(columns=rename)
        out = out.merge(part, on=[f"{side}_team", "game_day"], how="left")

    game_sides: list[dict] = []
    for row in out[["game_pk", "game_day", "home_team", "away_team"]].itertuples(index=False):
        game_sides.append({"game_pk": int(row.game_pk), "game_day": row.game_day, "side": "home", "team": row.home_team})
        game_sides.append({"game_pk": int(row.game_pk), "game_day": row.game_day, "side": "away", "team": row.away_team})
    game_sides_df = pd.DataFrame(game_sides)
    role_rows: list[dict] = []
    pitcher_day = (
        relievers.groupby(["team", "pitcher_id", "game_day"], as_index=False)
        .agg(
            saves=("saves", "sum"),
            holds=("holds", "sum"),
            games_finished=("games_finished", "sum"),
            pitch_count=("pitch_count", "sum"),
            bf=("bf", "sum"),
            bb=("bb", "sum"),
            so=("so", "sum"),
            pitch_hand=("pitch_hand", "last"),
        )
        .sort_values(["team", "game_day", "pitcher_id"])
    )
    for team, games in game_sides_df.groupby("team", sort=False):
        team_relief = pitcher_day[pitcher_day["team"].eq(team)].copy()
        if team_relief.empty:
            continue
        by_day: dict[pd.Timestamp, dict] = {}
        unique_days = sorted(pd.Timestamp(d) for d in games["game_day"].dropna().unique())
        events = team_relief.sort_values(["game_day", "pitcher_id"]).reset_index(drop=True)
        event_ords = events["game_day"].to_numpy(dtype="datetime64[D]").astype("int64")
        records = events.to_dict("records")
        stats30: dict[int, dict[str, float | str | None]] = {}
        pitches2: dict[int, float] = {}
        pitches1: dict[int, float] = {}
        add30 = drop30 = add2 = drop2 = add1 = drop1 = 0

        def apply_event(store: dict[int, dict[str, float | str | None]], rec: dict, sign: float) -> None:
            pid = int(rec["pitcher_id"])
            item = store.setdefault(
                pid,
                {"saves": 0.0, "holds": 0.0, "games_finished": 0.0, "pitch_count": 0.0, "bf": 0.0, "bb": 0.0, "so": 0.0, "pitch_hand": rec.get("pitch_hand")},
            )
            for metric in ["saves", "holds", "games_finished", "pitch_count", "bf", "bb", "so"]:
                item[metric] = float(item.get(metric, 0.0) or 0.0) + sign * float(rec.get(metric, 0.0) or 0.0)
            if rec.get("pitch_hand"):
                item["pitch_hand"] = rec.get("pitch_hand")
            if sign < 0 and all(abs(float(item.get(metric, 0.0) or 0.0)) < 1e-9 for metric in ["saves", "holds", "games_finished", "pitch_count", "bf", "bb", "so"]):
                store.pop(pid, None)

        def apply_pitch(store: dict[int, float], rec: dict, sign: float) -> None:
            pid = int(rec["pitcher_id"])
            store[pid] = store.get(pid, 0.0) + sign * float(rec.get("pitch_count", 0.0) or 0.0)

        for gd in unique_days:
            day_ord = np.datetime64(gd.date()).astype("datetime64[D]").astype("int64")
            while add30 < len(records) and event_ords[add30] < day_ord:
                apply_event(stats30, records[add30], 1.0)
                add30 += 1
            while drop30 < len(records) and event_ords[drop30] < day_ord - 30:
                apply_event(stats30, records[drop30], -1.0)
                drop30 += 1
            while add2 < len(records) and event_ords[add2] < day_ord:
                apply_pitch(pitches2, records[add2], 1.0)
                add2 += 1
            while drop2 < len(records) and event_ords[drop2] < day_ord - 2:
                apply_pitch(pitches2, records[drop2], -1.0)
                drop2 += 1
            while add1 < len(records) and event_ords[add1] < day_ord:
                apply_pitch(pitches1, records[add1], 1.0)
                add1 += 1
            while drop1 < len(records) and event_ords[drop1] < day_ord - 1:
                apply_pitch(pitches1, records[drop1], -1.0)
                drop1 += 1
            data: dict[str, float | int | None] = {}
            role_by_pid: dict[int, dict[str, float | str | None]] = {}
            for pid, item in stats30.items():
                if float(item.get("bf", 0.0) or 0.0) <= 0 and float(item.get("saves", 0.0) or 0.0) <= 0 and float(item.get("holds", 0.0) or 0.0) <= 0:
                    continue
                closer_score = float(item.get("saves", 0.0) or 0.0) * 3.0 + float(item.get("games_finished", 0.0) or 0.0) + float(item.get("holds", 0.0) or 0.0) * 0.25
                setup_score = float(item.get("holds", 0.0) or 0.0) * 3.0 + float(item.get("games_finished", 0.0) or 0.0) * 0.25
                role_by_pid[pid] = {
                    **item,
                    "pitches_2d": pitches2.get(pid, 0.0),
                    "pitches_yesterday": pitches1.get(pid, 0.0),
                    "closer_score": closer_score,
                    "setup_score": setup_score,
                }
            if not role_by_pid:
                by_day[gd] = data
                continue
            closer_id, closer = max(role_by_pid.items(), key=lambda kv: float(kv[1].get("closer_score", 0.0) or 0.0))
            if float(closer.get("closer_score", 0.0) or 0.0) <= 0:
                closer_id = None
            setup_candidates = [(pid, item) for pid, item in role_by_pid.items() if closer_id is None or pid != closer_id]
            setup_id = None
            if setup_candidates:
                setup_id, setup = max(setup_candidates, key=lambda kv: float(kv[1].get("setup_score", 0.0) or 0.0))
                if float(setup.get("setup_score", 0.0) or 0.0) <= 0:
                    setup_id = None
            left_candidates = [(pid, item) for pid, item in role_by_pid.items() if item.get("pitch_hand") == "L"]
            right_candidates = [(pid, item) for pid, item in role_by_pid.items() if item.get("pitch_hand") == "R"]
            top_lhrp_id = max(left_candidates, key=lambda kv: float(kv[1].get("setup_score", 0.0) or 0.0))[0] if left_candidates else None
            top_rhrp_id = max(right_candidates, key=lambda kv: float(kv[1].get("setup_score", 0.0) or 0.0))[0] if right_candidates else None
            data.update(
                {
                    "closer_id": closer_id,
                    "setup_id": setup_id,
                    "top_lhrp_id": top_lhrp_id,
                    "top_rhrp_id": top_rhrp_id,
                    "bp_lefty_available": float(any(float(item.get("pitches_2d", 0.0) or 0.0) <= 35.0 for _, item in left_candidates)) if left_candidates else np.nan,
                    "bp_righty_available": float(any(float(item.get("pitches_2d", 0.0) or 0.0) <= 35.0 for _, item in right_candidates)) if right_candidates else np.nan,
                }
            )
            for label, pid in [("closer", closer_id), ("setup", setup_id)]:
                if pid is None or pid not in role_by_pid:
                    continue
                r = role_by_pid[pid]
                bf = float(r.get("bf", 0.0) or 0.0)
                data[f"{label}_k_minus_bb_30d"] = ((float(r.get("so", 0.0) or 0.0) - float(r.get("bb", 0.0) or 0.0)) / bf) if bf else np.nan
                data[f"{label}_pitches_yesterday"] = float(r.get("pitches_yesterday", 0.0) or 0.0)
                data[f"{label}_pitches_2d"] = float(r.get("pitches_2d", 0.0) or 0.0)
                data[f"{label}_available_flag"] = float(float(r.get("pitches_2d", 0.0) or 0.0) <= 35.0)
            for label, pid in [("top_lhrp", top_lhrp_id), ("top_rhrp", top_rhrp_id)]:
                if pid is not None and pid in role_by_pid:
                    data[f"{label}_pitches_yesterday"] = float(role_by_pid[pid].get("pitches_yesterday", 0.0) or 0.0)
            by_day[gd] = data
        for game in games.sort_values(["game_day", "game_pk"]).itertuples(index=False):
            data = {"game_pk": int(game.game_pk), "side": game.side}
            data.update(by_day.get(pd.Timestamp(game.game_day), {}))
            role_rows.append(data)

    if role_rows:
        roles = pd.DataFrame(role_rows)
        wide = out[["game_pk"]].copy()
        for side in ["home", "away"]:
            part = roles[roles["side"].eq(side)].drop(columns=["side"]).rename(
                columns={c: f"{side}_{c}" for c in roles.columns if c not in {"game_pk", "side"}}
            )
            wide = wide.merge(part, on="game_pk", how="left")
        out = out.merge(wide, on="game_pk", how="left")

    return out.drop(columns=["game_date", "home_team", "away_team", "game_day"])

    # Legacy per-game role inference kept below for reference; unreachable.
    for side in ["home", "away"]:
        team_col = f"{side}_team"
        for idx, game in out.iterrows():
            gd = pd.Timestamp(game["game_date"])
            team = game[team_col]
            prior = relievers[(relievers["team"] == team) & (relievers["game_date"] < gd)].copy()
            for days in [1, 2, 3]:
                since = gd - pd.Timedelta(days=days)
                window = prior[prior["game_date"] >= since]
                out.at[idx, f"{side}_bp_pitches_{days}d"] = float(window["pitch_count"].sum()) if not window.empty else 0.0
            window2 = prior[prior["game_date"] >= gd - pd.Timedelta(days=2)]
            window30 = prior[prior["game_date"] >= gd - pd.Timedelta(days=30)]
            out.at[idx, f"{side}_bp_bf_2d"] = float(window2["bf"].sum()) if not window2.empty else 0.0
            out.at[idx, f"{side}_bp_ip_2d"] = float(window2["ip"].sum()) if not window2.empty else 0.0
            out.at[idx, f"{side}_bp_high_leverage_pitches_2d"] = float(
                window2.loc[(window2["saves"].fillna(0) > 0) | (window2["holds"].fillna(0) > 0), "pitch_count"].sum()
            )
            yesterday = prior[prior["game_date"] >= gd - pd.Timedelta(days=1)]
            role = window30.groupby("pitcher_id")[["saves", "holds", "games_finished", "pitch_count", "bf", "ip", "er", "bb", "so", "hr"]].sum()
            closer_id = None
            setup_id = None
            if not role.empty:
                role["closer_score"] = role["saves"].fillna(0) * 3 + role["games_finished"].fillna(0)
                role["setup_score"] = role["holds"].fillna(0) * 3 + role["games_finished"].fillna(0) * 0.25
                closer_id = int(role["closer_score"].idxmax()) if role["closer_score"].max() > 0 else None
                setup_id = int(role["setup_score"].idxmax()) if role["setup_score"].max() > 0 else None
                ip = role["ip"].sum()
                bf = role["bf"].sum()
                out.at[idx, f"{side}_bp_era_30d"] = (role["er"].sum() / ip * 9.0) if ip else np.nan
                out.at[idx, f"{side}_bp_fip_30d"] = ((13 * role["hr"].sum()) + (3 * role["bb"].sum()) - (2 * role["so"].sum())) / ip if ip else np.nan
                out.at[idx, f"{side}_bp_k_minus_bb_30d"] = ((role["so"].sum() - role["bb"].sum()) / bf) if bf else np.nan
            out.at[idx, f"{side}_closer_used_yesterday"] = float(
                closer_id is not None and closer_id in set(yesterday["pitcher_id"].astype(int))
            )
            out.at[idx, f"{side}_top3_rp_used_2d"] = 0.0
            if not role.empty:
                top3 = set(role.sort_values("closer_score", ascending=False).head(3).index.astype(int))
                out.at[idx, f"{side}_top3_rp_used_2d"] = float(len(top3 & set(window2["pitcher_id"].dropna().astype(int))))
            for label, pid in [("closer", closer_id), ("setup", setup_id)]:
                if pid is None:
                    continue
                p2 = window2[window2["pitcher_id"] == pid]
                py = yesterday[yesterday["pitcher_id"] == pid]
                out.at[idx, f"{side}_{label}_pitches_yesterday"] = float(py["pitch_count"].sum()) if not py.empty else 0.0
                out.at[idx, f"{side}_{label}_pitches_2d"] = float(p2["pitch_count"].sum()) if not p2.empty else 0.0
                out.at[idx, f"{side}_{label}_available_flag"] = float(out.at[idx, f"{side}_{label}_pitches_2d"] <= 35.0)
                if pid in role.index:
                    r = role.loc[pid]
                    bf = r["bf"]
                    out.at[idx, f"{side}_{label}_k_minus_bb_30d"] = ((r["so"] - r["bb"]) / bf) if bf else np.nan
            out.at[idx, f"{side}_bp_freshness_score"] = -(
                out.at[idx, f"{side}_bp_pitches_1d"] + 0.5 * out.at[idx, f"{side}_bp_pitches_2d"]
            )
            out.at[idx, f"{side}_bp_fatigue_expected"] = -out.at[idx, f"{side}_bp_freshness_score"]
    return out.drop(columns=["game_date", "home_team", "away_team"])


def build_mlb_statsapi_features(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw.get("ok", False) == True].copy()  # noqa: E712
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["game_pk"] = raw["game_pk"].astype(int)
    base_cols = [
        "game_pk",
        "umpire_id",
        "venue_id",
        "venue_name",
        "venue_azimuth_angle",
        "venue_elevation_ft",
        "roof_type",
        "roof_closed_flag",
        "roof_possible_flag",
        "home_lineup_lhb_count",
        "home_lineup_rhb_count",
        "away_lineup_lhb_count",
        "away_lineup_rhb_count",
        "home_catcher_id",
        "away_catcher_id",
    ]
    base = raw[[c for c in base_cols if c in raw.columns]].copy()
    appearances = _appearance_frame(raw)
    if appearances.empty:
        return base
    workload = _starter_workload(raw, appearances)
    bullpen = _bullpen_features(raw[["game_pk", "game_date", "home_team", "away_team"]], appearances)
    out = base.merge(workload, on="game_pk", how="left").merge(bullpen, on="game_pk", how="left")
    if {"home_lineup_lhb_count", "away_lineup_lhb_count"} <= set(out.columns):
        out["lineup_handedness_balance_DIFF"] = out["home_lineup_lhb_count"] - out["away_lineup_lhb_count"]
    if {"home_bp_pitches_2d", "away_bp_pitches_2d"} <= set(out.columns):
        out["bp_pitches_2d_DIFF"] = out["away_bp_pitches_2d"] - out["home_bp_pitches_2d"]
    if {"home_bp_freshness_score", "away_bp_freshness_score"} <= set(out.columns):
        out["bp_freshness_DIFF"] = out["home_bp_freshness_score"] - out["away_bp_freshness_score"]
    if {"home_bp_era_30d", "away_bp_era_30d"} <= set(out.columns):
        out["bp_quality_DIFF"] = out["away_bp_era_30d"] - out["home_bp_era_30d"]
    if {"home_closer_available_flag", "away_closer_available_flag"} <= set(out.columns):
        out["closer_freshness_DIFF"] = out["home_closer_available_flag"] - out["away_closer_available_flag"]
    if {"home_setup_available_flag", "away_setup_available_flag"} <= set(out.columns):
        out["setup_freshness_DIFF"] = out["home_setup_available_flag"] - out["away_setup_available_flag"]
    return out.drop_duplicates(subset=["game_pk"], keep="last")


def _nearest_hourly(hourly_df: pd.DataFrame, local_time: pd.Timestamp) -> dict:
    if hourly_df.empty:
        return {}
    times = hourly_df["time"]
    if len(times) == 0:
        return {}
    target = pd.Timestamp(local_time).tz_localize(None)
    deltas = (times - target).abs()
    idx = int(deltas.argmin())
    return hourly_df.iloc[idx].drop(labels=["time"]).to_dict()


def fetch_openmeteo_hourly(master: pd.DataFrame, limit: int | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["game_pk", "game_date", "game_time_utc", "home_team"]
    needed = master[cols].dropna(subset=["game_pk", "game_date", "home_team"]).copy()
    needed["game_pk"] = needed["game_pk"].astype(int)
    cached = pd.read_parquet(WEATHER_HOURLY_PATH) if WEATHER_HOURLY_PATH.exists() else pd.DataFrame()
    if not cached.empty and "temp_c_game_time" in cached.columns:
        valid_cached = cached[cached["temp_c_game_time"].notna()].copy()
    else:
        valid_cached = cached
    done = set(valid_cached["game_pk"].astype(int)) if not valid_cached.empty else set()
    todo = needed[~needed["game_pk"].isin(done)].copy()
    if limit:
        todo = todo.head(limit)
    print(f"Open-Meteo hourly fetching {len(todo):,} games ({len(done):,} cached)")
    rows: list[dict] = []
    todo["game_date"] = pd.to_datetime(todo["game_date"])
    todo["season"] = todo["game_date"].dt.year
    groups = list(todo.groupby(["home_team", "season"]))
    for gi, ((team, season), group) in enumerate(groups, start=1):
        team = str(team)
        coords = planned_features.TEAM_PARKS.get(team)
        if not coords:
            continue
        lat, lon = coords[:2]
        start_date = group["game_date"].min().date().isoformat()
        end_date = group["game_date"].max().date().isoformat()
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        try:
            data = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30).json()
            hourly = pd.DataFrame(data.get("hourly", {}))
            if not hourly.empty:
                hourly["time"] = pd.to_datetime(hourly["time"], errors="coerce")
            for row in group.itertuples(index=False):
                utc = pd.to_datetime(row.game_time_utc, utc=True, errors="coerce")
                if pd.isna(utc):
                    local = pd.Timestamp(row.game_date).replace(hour=19)
                else:
                    tz = ZoneInfo(TEAM_TIMEZONES.get(team, "America/New_York"))
                    local = utc.tz_convert(tz)
                h = _nearest_hourly(hourly, local)
                rows.append(
                    {
                        "game_pk": int(row.game_pk),
                        "temp_c_game_time": h.get("temperature_2m"),
                        "relative_humidity_game_time": h.get("relative_humidity_2m"),
                        "dew_point_c_game_time": h.get("dew_point_2m"),
                        "surface_pressure_hpa_game_time": h.get("surface_pressure"),
                        "wind_speed_kmh_game_time": h.get("wind_speed_10m"),
                        "wind_dir_deg_game_time": h.get("wind_direction_10m"),
                    }
                )
        except Exception:
            for row in group.itertuples(index=False):
                rows.append({"game_pk": int(row.game_pk)})
        if gi % 50 == 0:
            print(f"  weather groups {gi:,}/{len(groups):,} rows={len(rows):,}")
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached = cached.drop_duplicates(subset=["game_pk"], keep="last")
        cached.to_parquet(WEATHER_HOURLY_PATH, index=False)
    print(f"Open-Meteo hourly wrote {WEATHER_HOURLY_PATH} rows={len(cached):,}")


def load_master(path: Path = MASTER_PATH) -> pd.DataFrame:
    if path == MASTER_PATH:
        import db

        hist = db.get_games_df()
        curr_season = int(pd.Timestamp.utcnow().year)
        curr = db.get_games_df(season=curr_season)
        frames = [df for df in (hist, curr) if df is not None and not df.empty]
        if not frames:
            raise RuntimeError("DB returned no games for v2 source refresh")
        all_cols = list(dict.fromkeys(c for df in frames for c in df.columns))
        out = pd.concat([df.reindex(columns=all_cols) for df in frames], ignore_index=True)
        out = out.drop_duplicates(subset=["game_pk"], keep="last")
        print(f"Loaded {len(out):,} games from DB for v2 source refresh")
        return out
    return pd.read_csv(path, low_memory=False)


def fetch_mlb_game_times(master: pd.DataFrame) -> None:
    """Fetch UTC game start times from MLB schedule endpoint, one call per season."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seasons = sorted(pd.to_datetime(master["game_date"]).dt.year.dropna().astype(int).unique())
    rows: list[dict] = []
    for season in seasons:
        params = {
            "sportId": 1,
            "startDate": f"{season}-01-01",
            "endDate": f"{season}-12-31",
        }
        try:
            data = requests.get(MLB_SCHEDULE_URL, params=params, timeout=60).json()
        except Exception as exc:
            print(f"  schedule {season} failed: {exc}")
            continue
        count = 0
        for date_block in data.get("dates", []) or []:
            for game in date_block.get("games", []) or []:
                pk = game.get("gamePk")
                gd = game.get("gameDate")
                if pk is None or gd is None:
                    continue
                rows.append({"game_pk": int(pk), "game_time_utc": str(gd)})
                count += 1
        print(f"  schedule {season} games={count:,}")
    if not rows:
        print("no game times fetched")
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["game_pk"], keep="last")
    df.to_parquet(MLB_GAME_TIMES_PATH, index=False)
    print(f"wrote {MLB_GAME_TIMES_PATH} rows={len(df):,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["fetch-mlb", "fetch-weather", "fetch-times", "fetch-all"]:
        p = sub.add_parser(name)
        p.add_argument("--master", type=Path, default=MASTER_PATH)
        p.add_argument("--limit", type=int)
        p.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    master = load_master(args.master)
    if args.cmd in {"fetch-mlb", "fetch-all"}:
        fetch_mlb_statsapi(master, limit=args.limit, workers=args.workers)
    if args.cmd in {"fetch-times", "fetch-all"}:
        fetch_mlb_game_times(master)
    if args.cmd in {"fetch-weather", "fetch-all"}:
        fetch_openmeteo_hourly(master, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
