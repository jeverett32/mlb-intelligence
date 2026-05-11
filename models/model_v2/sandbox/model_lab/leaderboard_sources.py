#!/usr/bin/env python3
"""Savant leaderboard feature pipeline for sandbox-only modeling."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab.savant_sources import SAVANT_DIR  # noqa: E402


CACHE_DIR = LAB_DIR / "cache"
OUTPUT_DIR = LAB_DIR / "output"
MASTER_PATH = OUTPUT_DIR / "master_sandbox_mlb.csv"
LEADERBOARD_DIR = CACHE_DIR / "savant_leaderboards"
LEADERBOARD_FEATURES_PATH = CACHE_DIR / "savant_leaderboard_features.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_text(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=45)
            response.raise_for_status()
            return response.text.lstrip("\ufeff")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""


def _read_csv_url(url: str) -> pd.DataFrame:
    text = _fetch_text(url)
    if not text.strip() or not text.lstrip().startswith('"'):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), low_memory=False)


def _write_year_csv(kind: str, season: int, df: pd.DataFrame) -> None:
    path = LEADERBOARD_DIR / kind
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / f"{season}.parquet", index=False)


def _read_kind(kind: str) -> pd.DataFrame:
    path = LEADERBOARD_DIR / kind
    files = sorted(path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def fetch_leaderboards(start_season: int = 2014, end_season: int = 2025, force: bool = False) -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    for season in range(start_season, end_season + 1):
        targets = {
            "sprint_speed": (
                LEADERBOARD_DIR / "sprint_speed" / f"{season}.parquet",
                f"https://baseballsavant.mlb.com/leaderboard/sprint_speed?csv=true&min=1&min_season={season}&max_season={season}",
            ),
            "active_spin": (
                LEADERBOARD_DIR / "active_spin" / f"{season}.parquet",
                f"https://baseballsavant.mlb.com/leaderboard/active-spin?csv=true&year={season}&min=1",
            ),
            "catcher_blocking": (
                LEADERBOARD_DIR / "catcher_blocking" / f"{season}.parquet",
                "https://baseballsavant.mlb.com/leaderboard/catcher-blocking"
                f"?game_type=Regular&n=1&season_start={season}&season_end={season}&split=no&team=&type=Catchers&with_team_only=0&csv=true",
            ),
            "catcher_throwing": (
                LEADERBOARD_DIR / "catcher_throwing" / f"{season}.parquet",
                "https://baseballsavant.mlb.com/leaderboard/catcher-throwing"
                f"?game_type=Regular&n=1&season_start={season}&season_end={season}&split=no&team=&type=Cat&with_team_only=0&target_base=All&csv=true",
            ),
            "batter_custom": (
                LEADERBOARD_DIR / "batter_custom" / f"{season}.parquet",
                "https://baseballsavant.mlb.com/leaderboard/custom"
                f"?csv=true&chart=false&chartType=beeswarm&min=1&r=no&selections=pa%2Cwoba%2Con_base_percent"
                f"&sort=woba&sortDir=desc&type=batter&x=pa&y=pa&year={season}",
            ),
        }
        for kind, (path, url) in targets.items():
            if path.exists() and not force:
                continue
            df = _read_csv_url(url)
            if df.empty:
                df = pd.DataFrame({"season": [season]})
            df["season"] = season
            _write_year_csv(kind, season, df)
            print(f"{kind} {season}: rows={len(df):,}", flush=True)

        park_path = LEADERBOARD_DIR / "park_factors" / f"{season}.parquet"
        if park_path.exists() and not force:
            continue
        park = fetch_park_factor_season(season)
        _write_year_csv("park_factors", season, park)
        print(f"park_factors {season}: rows={len(park):,}", flush=True)


def fetch_park_factor_season(season: int) -> pd.DataFrame:
    rows: list[dict] = []
    for bat_side in ["L", "R"]:
        url = (
            "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
            f"?type=year&year={season}&batSide={bat_side}&stat=index_HR&condition=All&rolling=&parks="
        )
        text = _fetch_text(url)
        match = re.search(r"var data = (.*?);\s*var queryString", text, re.S)
        if not match:
            continue
        raw = match.group(1).strip()
        data = json.loads(raw) if raw.startswith("[") else []
        for item in data:
            rows.append(
                {
                    "season": season,
                    "venue_id": item.get("venue_id"),
                    "bat_side": bat_side,
                    "park_run_factor": pd.to_numeric(item.get("index_runs"), errors="coerce") / 100.0,
                    "park_hr_factor": pd.to_numeric(item.get("index_hr"), errors="coerce") / 100.0,
                }
            )
    return pd.DataFrame(rows)


def _prep_player_season(df: pd.DataFrame, player_col: str = "player_id") -> pd.DataFrame:
    if df.empty or player_col not in df.columns or "season" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[player_col] = pd.to_numeric(out[player_col], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    return out.dropna(subset=[player_col, "season"])


def _lineup_long() -> pd.DataFrame:
    lineups = pd.read_parquet(SAVANT_DIR / "game_lineups")
    rows: list[dict] = []
    for row in lineups.itertuples(index=False):
        game_pk = int(row.game_pk)
        for side in ["home", "away"]:
            ids = getattr(row, f"{side}_lineup_ids", [])
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            if isinstance(ids, tuple):
                ids = list(ids)
            if not isinstance(ids, list):
                ids = []
            for player_id in ids[:9]:
                if pd.notna(player_id):
                    rows.append({"game_pk": game_pk, "side": side, "player_id": int(player_id)})
    return pd.DataFrame(rows)


def build_leaderboard_features(master_path: Path = MASTER_PATH) -> None:
    master = pd.read_csv(master_path, low_memory=False).copy()
    master["game_pk"] = pd.to_numeric(master["game_pk"], errors="coerce")
    master["season"] = pd.to_numeric(master["season"], errors="coerce")
    master["prior_source_season"] = master["season"] - 1
    out = master[["game_pk", "season", "prior_source_season"]].copy()

    out = out.merge(_starter_active_spin(master), on="game_pk", how="left")
    out = out.merge(_catcher_leaderboard_features(master), on="game_pk", how="left")
    out = out.merge(_lineup_sprint_speed(master), on="game_pk", how="left")
    out = out.merge(_lineup_batting_custom(master), on="game_pk", how="left")
    out = out.merge(_park_factor_features(master), on="game_pk", how="left")
    out = out.drop(columns=["season", "prior_source_season"], errors="ignore")
    out.to_parquet(LEADERBOARD_FEATURES_PATH, index=False)
    print(f"leaderboard features wrote {LEADERBOARD_FEATURES_PATH} rows={len(out):,} cols={len(out.columns):,}", flush=True)


def _starter_active_spin(master: pd.DataFrame) -> pd.DataFrame:
    active = _prep_player_season(_read_kind("active_spin"), "entity_id")
    if active.empty:
        return master[["game_pk"]].copy()
    spin_cols = [c for c in active.columns if c.startswith("active_spin_")]
    active["sp_active_spin"] = active[spin_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    active = active[["entity_id", "season", "sp_active_spin"]].dropna(subset=["sp_active_spin"])
    out = master[["game_pk", "prior_source_season", "home_starter_id", "away_starter_id"]].copy()
    for side in ["home", "away"]:
        part = active.rename(columns={"entity_id": f"{side}_starter_id", "season": "prior_source_season", "sp_active_spin": f"{side}_sp_active_spin"})
        out = out.merge(part, on=[f"{side}_starter_id", "prior_source_season"], how="left")
    return out[["game_pk", "home_sp_active_spin", "away_sp_active_spin"]]


def _catcher_leaderboard_features(master: pd.DataFrame) -> pd.DataFrame:
    blocking = _prep_player_season(_read_kind("catcher_blocking"))
    throwing = _prep_player_season(_read_kind("catcher_throwing"))
    out = master[["game_pk", "prior_source_season", "home_catcher_id", "away_catcher_id"]].copy()
    if not blocking.empty:
        blocking = blocking[["player_id", "season", "catcher_blocking_runs", "blocks_above_average"]].copy()
    if not throwing.empty:
        throwing = throwing[["player_id", "season", "pop_time", "arm_strength", "rate_cs"]].copy()
    catcher = blocking.merge(throwing, on=["player_id", "season"], how="outer") if not blocking.empty or not throwing.empty else pd.DataFrame()
    if catcher.empty:
        return out[["game_pk"]]
    for col in ["catcher_blocking_runs", "blocks_above_average", "pop_time", "arm_strength", "rate_cs"]:
        if col in catcher.columns:
            catcher[col] = pd.to_numeric(catcher[col], errors="coerce")
    for side in ["home", "away"]:
        part = catcher.rename(
            columns={
                "player_id": f"{side}_catcher_id",
                "season": "prior_source_season",
                "catcher_blocking_runs": f"{side}_catcher_blocking_runs",
                "blocks_above_average": f"{side}_catcher_blocks_above_avg",
                "pop_time": f"{side}_catcher_pop_time",
                "arm_strength": f"{side}_catcher_arm_strength",
                "rate_cs": f"{side}_catcher_caught_stealing_rate",
            }
        )
        out = out.merge(part, on=[f"{side}_catcher_id", "prior_source_season"], how="left")
    return out.drop(columns=["prior_source_season", "home_catcher_id", "away_catcher_id"], errors="ignore")


def _lineup_sprint_speed(master: pd.DataFrame) -> pd.DataFrame:
    sprint = _prep_player_season(_read_kind("sprint_speed"))
    if sprint.empty:
        return master[["game_pk"]].copy()
    sprint = sprint[["player_id", "season", "sprint_speed"]].copy()
    sprint["sprint_speed"] = pd.to_numeric(sprint["sprint_speed"], errors="coerce")
    lineup = _lineup_long()
    if lineup.empty:
        return master[["game_pk"]].copy()
    work = lineup.merge(master[["game_pk", "prior_source_season"]], on="game_pk", how="inner")
    work = work.merge(sprint.rename(columns={"season": "prior_source_season"}), on=["player_id", "prior_source_season"], how="left")
    means = work.groupby(["game_pk", "side"], as_index=False)["sprint_speed"].mean()
    out = master[["game_pk"]].copy()
    for side in ["home", "away"]:
        part = means[means["side"].eq(side)][["game_pk", "sprint_speed"]].rename(columns={"sprint_speed": f"{side}_lineup_sprint_speed"})
        out = out.merge(part, on="game_pk", how="left")
    return out


def _lineup_batting_custom(master: pd.DataFrame) -> pd.DataFrame:
    bat = _prep_player_season(_read_kind("batter_custom"))
    if bat.empty:
        return master[["game_pk"]].copy()
    bat = bat.rename(columns={"on_base_percent": "obp"})
    for col in ["pa", "woba", "obp"]:
        if col in bat.columns:
            bat[col] = pd.to_numeric(bat[col], errors="coerce")
    if not {"player_id", "season", "woba", "obp"}.issubset(bat.columns):
        return master[["game_pk"]].copy()
    league = (
        bat.dropna(subset=["woba"])
        .assign(woba_x_pa=lambda d: d["woba"] * d["pa"].fillna(1.0))
        .groupby("season", as_index=False)
        .agg(woba_x_pa=("woba_x_pa", "sum"), pa=("pa", "sum"))
    )
    league["league_woba"] = league["woba_x_pa"] / league["pa"].replace(0, np.nan)
    bat = bat.merge(league[["season", "league_woba"]], on="season", how="left")
    # FanGraphs player wRC+ is not available from Savant; use a documented
    # same-scale wOBA+ proxy so the sandbox column carries real offensive signal.
    bat["wrc_plus_proxy"] = 100.0 * bat["woba"] / bat["league_woba"].replace(0, np.nan)
    bat = bat[["player_id", "season", "woba", "obp", "wrc_plus_proxy"]].copy()
    lineup = _lineup_long()
    if lineup.empty:
        return master[["game_pk"]].copy()
    work = lineup.merge(master[["game_pk", "prior_source_season"]], on="game_pk", how="inner")
    work = work.merge(bat.rename(columns={"season": "prior_source_season"}), on=["player_id", "prior_source_season"], how="left")
    means = work.groupby(["game_pk", "side"], as_index=False)[["woba", "obp", "wrc_plus_proxy"]].mean()
    out = master[["game_pk"]].copy()
    for side in ["home", "away"]:
        part = means[means["side"].eq(side)][["game_pk", "woba", "obp", "wrc_plus_proxy"]].rename(
            columns={
                "woba": f"{side}_lineup_avg_woba",
                "obp": f"{side}_lineup_avg_obp",
                "wrc_plus_proxy": f"{side}_lineup_avg_wrc_plus",
            }
        )
        out = out.merge(part, on="game_pk", how="left")
    return out


def _park_factor_features(master: pd.DataFrame) -> pd.DataFrame:
    parks = _read_kind("park_factors")
    if parks.empty or "venue_id" not in master.columns:
        return master[["game_pk"]].copy()
    parks["venue_id"] = pd.to_numeric(parks["venue_id"], errors="coerce")
    parks["season"] = pd.to_numeric(parks["season"], errors="coerce")
    wide = parks.pivot_table(index=["venue_id", "season"], columns="bat_side", values=["park_run_factor", "park_hr_factor"], aggfunc="first")
    wide.columns = [f"{metric}_{side.lower()}" for metric, side in wide.columns]
    wide = wide.reset_index().rename(
        columns={
            "season": "prior_source_season",
            "park_hr_factor_l": "park_hr_factor_lhb",
            "park_hr_factor_r": "park_hr_factor_rhb",
            "park_run_factor_l": "park_run_factor_lhb",
            "park_run_factor_r": "park_run_factor_rhb",
        }
    )
    work = master[["game_pk", "venue_id", "prior_source_season"]].copy()
    work["venue_id"] = pd.to_numeric(work["venue_id"], errors="coerce")
    return work.merge(wide, on=["venue_id", "prior_source_season"], how="left").drop(columns=["venue_id", "prior_source_season"], errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--start-season", type=int, default=2014)
    fetch.add_argument("--end-season", type=int, default=2025)
    fetch.add_argument("--force", action="store_true")
    build = sub.add_parser("build")
    build.add_argument("--master", type=Path, default=MASTER_PATH)
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch_leaderboards(args.start_season, args.end_season, args.force)
    elif args.cmd == "build":
        build_leaderboard_features(args.master)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
