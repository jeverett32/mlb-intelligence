#!/usr/bin/env python3
"""Baseball Savant / Statcast sandbox pipeline.

Fetches Statcast pitch-level data in small date chunks, immediately reduces it
to game-level aggregates, then builds pregame features from prior-only history.
Statcast coverage is 2015+.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab.real_sources import MLB_RAW_PATH, _appearance_frame  # noqa: E402


CACHE_DIR = LAB_DIR / "cache"
SAVANT_DIR = CACHE_DIR / "savant"
SAVANT_FEATURES_PATH = CACHE_DIR / "savant_features.parquet"
MASTER_PATH = LAB_DIR / "output" / "master_sandbox_mlb.csv"
SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH"]
FASTBALL_TYPES = {"FF", "SI", "FC"}
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}
WALK_EVENTS = {"walk", "intent_walk"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
BARREL_CODES = {6}
SHADOW_ZONES = {1, 2, 3, 4, 5, 6, 7, 8, 9}


def _date_chunks(start: str, end: str, days: int, game_dates_only: bool = True) -> list[tuple[date, date]]:
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    stop = datetime.strptime(end, "%Y-%m-%d").date()
    if game_dates_only and MASTER_PATH.exists():
        master = pd.read_csv(MASTER_PATH, usecols=["game_date"], low_memory=False)
        dates = pd.to_datetime(master["game_date"], errors="coerce").dt.date
        dates = sorted({d for d in dates if pd.notna(d) and cur <= d <= stop})
        if days == 1:
            return [(d, d) for d in dates]
        chunks: list[tuple[date, date]] = []
        idx = 0
        while idx < len(dates):
            lo = dates[idx]
            hi_limit = lo + timedelta(days=days - 1)
            hi = lo
            while idx + 1 < len(dates) and dates[idx + 1] <= hi_limit:
                idx += 1
                hi = dates[idx]
            chunks.append((lo, hi))
            idx += 1
        return chunks
    chunks: list[tuple[date, date]] = []
    while cur <= stop:
        hi = min(cur + timedelta(days=days - 1), stop)
        chunks.append((cur, hi))
        cur = hi + timedelta(days=1)
    return chunks


def _chunk_name(start: date, end: date, kind: str) -> Path:
    return SAVANT_DIR / kind / f"{start.isoformat()}_{end.isoformat()}.parquet"


def _to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def fetch_statcast_chunk(start: date, end: date, retries: int = 3) -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
    }
    for attempt in range(retries):
        try:
            r = requests.get(SAVANT_URL, params=params, timeout=90)
            r.raise_for_status()
            text = r.text.lstrip("\ufeff")
            if not text.strip():
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            keep = [
                "game_pk",
                "game_date",
                "pitch_type",
                "release_speed",
                "release_spin_rate",
                "release_extension",
                "pitcher",
                "batter",
                "events",
                "description",
                "zone",
                "type",
                "stand",
                "p_throws",
                "inning_topbot",
                "at_bat_number",
                "pitch_number",
                "bb_type",
                "launch_speed",
                "launch_angle",
                "launch_speed_angle",
                "estimated_woba_using_speedangle",
                "estimated_ba_using_speedangle",
                "estimated_slg_using_speedangle",
                "woba_value",
                "woba_denom",
                "n_thruorder_pitcher",
                "fielder_2",
            ]
            return df[[c for c in keep if c in df.columns]].copy()
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  Savant chunk failed {start}..{end}: {exc}", flush=True)
                return pd.DataFrame()
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def aggregate_chunk(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame()
        return {
            "pitcher_game": empty,
            "pitcher_pitch_type_game": empty,
            "batter_game": empty,
            "batter_pitch_type_game": empty,
            "catcher_game": empty,
            "game_lineups": empty,
        }
    df = df.copy()
    _to_num(
        df,
        [
            "game_pk",
            "pitcher",
            "batter",
            "zone",
            "release_speed",
            "release_spin_rate",
            "release_extension",
            "launch_speed",
            "launch_angle",
            "launch_speed_angle",
            "estimated_woba_using_speedangle",
            "estimated_ba_using_speedangle",
            "estimated_slg_using_speedangle",
            "woba_value",
            "woba_denom",
            "n_thruorder_pitcher",
            "fielder_2",
            "at_bat_number",
        ],
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(float)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(float)
    df["is_called_strike"] = df["description"].isin(CALLED_STRIKE_DESCRIPTIONS).astype(float)
    df["is_csw"] = ((df["is_whiff"] == 1) | (df["is_called_strike"] == 1)).astype(float)
    df["is_out_zone"] = ~df["zone"].isin(range(1, 10))
    df["is_chase"] = (df["is_out_zone"] & df["description"].isin(SWING_DESCRIPTIONS)).astype(float)
    df["is_bip"] = df["type"].eq("X").astype(float)
    df["is_gb"] = df["bb_type"].eq("ground_ball").astype(float)
    df["is_hard_hit"] = (df["launch_speed"] >= 95).astype(float)
    df["is_barrel"] = df["launch_speed_angle"].isin(BARREL_CODES).astype(float)
    df["is_sweet_spot"] = df["launch_angle"].between(8, 32).astype(float)
    df["is_pa"] = df["events"].notna().astype(float)
    df["is_k"] = df["events"].isin(STRIKEOUT_EVENTS).astype(float)
    df["is_bb"] = df["events"].isin(WALK_EVENTS).astype(float)
    df["bat_team_side"] = np.where(df["inning_topbot"].eq("Top"), "away", "home")

    pitcher_game = (
        df.groupby(["pitcher", "game_pk", "game_date"], as_index=False)
        .agg(
            pitches=("pitch_type", "size"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            called_strikes=("is_called_strike", "sum"),
            csw=("is_csw", "sum"),
            out_zone=("is_out_zone", "sum"),
            chases=("is_chase", "sum"),
            bip=("is_bip", "sum"),
            gb=("is_gb", "sum"),
            hard_hit=("is_hard_hit", "sum"),
            barrels=("is_barrel", "sum"),
            avg_ev_allowed=("launch_speed", "mean"),
            xwoba_sum=("estimated_woba_using_speedangle", "sum"),
            xwoba_count=("estimated_woba_using_speedangle", "count"),
            pa=("is_pa", "sum"),
        )
        .rename(columns={"pitcher": "pitcher_id"})
    )

    for order in [1, 2, 3]:
        part = (
            df[df["n_thruorder_pitcher"].eq(order)]
            .groupby(["pitcher", "game_pk"], as_index=False)
            .agg(xwoba_sum=("estimated_woba_using_speedangle", "sum"), xwoba_count=("estimated_woba_using_speedangle", "count"))
            .rename(columns={"pitcher": "pitcher_id", "xwoba_sum": f"tto{order}_xwoba_sum", "xwoba_count": f"tto{order}_xwoba_count"})
        )
        pitcher_game = pitcher_game.merge(part, on=["pitcher_id", "game_pk"], how="left")

    for stand in ["L", "R"]:
        part = (
            df[df["stand"].eq(stand)]
            .groupby(["pitcher", "game_pk"], as_index=False)
            .agg(xwoba_sum=("estimated_woba_using_speedangle", "sum"), xwoba_count=("estimated_woba_using_speedangle", "count"))
            .rename(columns={"pitcher": "pitcher_id", "xwoba_sum": f"vs_{stand.lower()}hb_xwoba_sum", "xwoba_count": f"vs_{stand.lower()}hb_xwoba_count"})
        )
        pitcher_game = pitcher_game.merge(part, on=["pitcher_id", "game_pk"], how="left")

    pitcher_pitch_type_game = (
        df[df["pitch_type"].isin(PITCH_TYPES)]
        .groupby(["pitcher", "game_pk", "game_date", "pitch_type"], as_index=False)
        .agg(
            pitches=("pitch_type", "size"),
            release_speed=("release_speed", "mean"),
            release_spin_rate=("release_spin_rate", "mean"),
            release_extension=("release_extension", "mean"),
        )
        .rename(columns={"pitcher": "pitcher_id"})
    )

    pa = df[df["is_pa"].eq(1)].copy()
    batter_game = (
        pa.groupby(["batter", "game_pk", "game_date", "bat_team_side"], as_index=False)
        .agg(
            pa=("is_pa", "sum"),
            k=("is_k", "sum"),
            bb=("is_bb", "sum"),
            xwoba_sum=("estimated_woba_using_speedangle", "sum"),
            xwoba_count=("estimated_woba_using_speedangle", "count"),
            xba_sum=("estimated_ba_using_speedangle", "sum"),
            xba_count=("estimated_ba_using_speedangle", "count"),
            xslg_sum=("estimated_slg_using_speedangle", "sum"),
            xslg_count=("estimated_slg_using_speedangle", "count"),
            hard_hit=("is_hard_hit", "sum"),
            barrels=("is_barrel", "sum"),
            bip=("is_bip", "sum"),
            avg_ev=("launch_speed", "mean"),
            avg_la=("launch_angle", "mean"),
            sweet_spot=("is_sweet_spot", "sum"),
        )
        .rename(columns={"batter": "batter_id"})
    )

    batter_pitch_type_game = (
        pa[pa["pitch_type"].isin(PITCH_TYPES)]
        .groupby(["batter", "game_pk", "game_date", "pitch_type"], as_index=False)
        .agg(xwoba_sum=("estimated_woba_using_speedangle", "sum"), xwoba_count=("estimated_woba_using_speedangle", "count"))
        .rename(columns={"batter": "batter_id"})
    )

    catcher_rows = df[df["fielder_2"].notna()].copy()
    catcher_rows["is_shadow"] = catcher_rows["zone"].isin(SHADOW_ZONES).astype(float)
    catcher_game = (
        catcher_rows.groupby(["fielder_2", "game_pk", "game_date"], as_index=False)
        .agg(
            called_strikes=("is_called_strike", "sum"),
            balls=("type", lambda s: (s == "B").sum()),
            shadow_called_strikes=("is_called_strike", lambda s: s[catcher_rows.loc[s.index, "is_shadow"].eq(1).values].sum()),
            shadow_pitches=("is_shadow", "sum"),
        )
        .rename(columns={"fielder_2": "catcher_id"})
    )

    first_pas = pa.sort_values(["game_pk", "bat_team_side", "at_bat_number"]).drop_duplicates(["game_pk", "bat_team_side", "batter"])
    lineups = []
    for (game_pk, side), grp in first_pas.groupby(["game_pk", "bat_team_side"]):
        ids = grp.sort_values("at_bat_number")["batter"].dropna().astype(int).head(9).tolist()
        lineups.append({"game_pk": int(game_pk), f"{side}_lineup_ids": ids})
    game_lineups = pd.DataFrame(lineups)
    if not game_lineups.empty:
        game_lineups = game_lineups.groupby("game_pk", as_index=False).first()

    return {
        "pitcher_game": pitcher_game,
        "pitcher_pitch_type_game": pitcher_pitch_type_game,
        "batter_game": batter_game,
        "batter_pitch_type_game": batter_pitch_type_game,
        "catcher_game": catcher_game,
        "game_lineups": game_lineups,
    }


def _fetch_aggregate_write(lo: date, hi: date) -> tuple[date, date, int]:
    df = fetch_statcast_chunk(lo, hi)
    aggs = aggregate_chunk(df)
    for kind, agg in aggs.items():
        agg.to_parquet(_chunk_name(lo, hi, kind), index=False)
    return lo, hi, len(df)


def fetch_savant(start: str = "2015-03-01", end: str = "2025-11-30", days: int = 1, limit_chunks: int | None = None, workers: int = 4) -> None:
    SAVANT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ["pitcher_game", "pitcher_pitch_type_game", "batter_game", "batter_pitch_type_game", "catcher_game", "game_lineups"]:
        (SAVANT_DIR / kind).mkdir(parents=True, exist_ok=True)

    chunks = _date_chunks(start, end, days)
    if limit_chunks:
        chunks = chunks[:limit_chunks]
    todo = [(lo, hi) for lo, hi in chunks if not _chunk_name(lo, hi, "pitcher_game").exists()]
    print(f"Savant fetching {len(todo):,} chunks ({len(chunks) - len(todo):,} cached)", flush=True)
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_fetch_aggregate_write, lo, hi): (lo, hi) for lo, hi in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            lo, hi, pitch_count = fut.result()
            print(f"  savant {i:,}/{len(todo):,} {lo}..{hi} pitches={pitch_count:,}", flush=True)


def _read_kind(kind: str) -> pd.DataFrame:
    path = SAVANT_DIR / kind
    files = sorted(path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(path)


def _prior_player_game(df: pd.DataFrame, player_col: str, stat_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values([player_col, "game_date", "game_pk"]).copy()
    for col in stat_cols:
        out[f"prior_{col}"] = out.groupby(player_col)[col].cumsum() - out[col]
    return out


def _mean_from(out: pd.DataFrame, num: str, den: str) -> pd.Series:
    return _rate(out[f"prior_{num}"], out[f"prior_{den}"])


def _build_pitcher_prior(pitcher_game: pd.DataFrame, pitch_type_game: pd.DataFrame) -> pd.DataFrame:
    if pitcher_game.empty:
        return pd.DataFrame()
    pg = pitcher_game.copy()
    pg["avg_ev_allowed_sum"] = pd.to_numeric(pg.get("avg_ev_allowed"), errors="coerce").fillna(0.0)
    pg["avg_ev_allowed_obs"] = pd.to_numeric(pg.get("avg_ev_allowed"), errors="coerce").notna().astype(float)
    pg = pg.fillna(0)
    stat_cols = [
        "pitches",
        "swings",
        "whiffs",
        "called_strikes",
        "csw",
        "out_zone",
        "chases",
        "bip",
        "gb",
        "hard_hit",
        "barrels",
        "xwoba_sum",
        "xwoba_count",
        "pa",
        "tto1_xwoba_sum",
        "tto1_xwoba_count",
        "tto2_xwoba_sum",
        "tto2_xwoba_count",
        "tto3_xwoba_sum",
        "tto3_xwoba_count",
        "vs_lhb_xwoba_sum",
        "vs_lhb_xwoba_count",
        "vs_rhb_xwoba_sum",
        "vs_rhb_xwoba_count",
        "avg_ev_allowed_sum",
        "avg_ev_allowed_obs",
    ]
    for col in stat_cols:
        if col not in pg.columns:
            pg[col] = 0.0
    prior = _prior_player_game(pg, "pitcher_id", stat_cols)
    prior["sp_whiff_rate"] = _mean_from(prior, "whiffs", "swings")
    prior["sp_chase_rate"] = _mean_from(prior, "chases", "out_zone")
    prior["sp_cs_rate"] = _mean_from(prior, "called_strikes", "pitches")
    prior["sp_csw_rate"] = _mean_from(prior, "csw", "pitches")
    prior["sp_groundball_rate"] = _mean_from(prior, "gb", "bip")
    prior["sp_barrel_allowed_rate"] = _mean_from(prior, "barrels", "bip")
    prior["sp_hard_hit_allowed_rate"] = _mean_from(prior, "hard_hit", "bip")
    prior["sp_avg_ev_allowed"] = _mean_from(prior, "avg_ev_allowed_sum", "avg_ev_allowed_obs")
    prior["sp_xwoba_allowed"] = _mean_from(prior, "xwoba_sum", "xwoba_count")
    prior["sp_xera"] = 2.8 + ((prior["sp_xwoba_allowed"] - 0.300) * 20.0)
    prior["sp_first_time_woba"] = _mean_from(prior, "tto1_xwoba_sum", "tto1_xwoba_count")
    prior["sp_second_time_woba"] = _mean_from(prior, "tto2_xwoba_sum", "tto2_xwoba_count")
    prior["sp_third_time_woba"] = _mean_from(prior, "tto3_xwoba_sum", "tto3_xwoba_count")
    prior["sp_tto3_penalty"] = prior["sp_third_time_woba"] - prior["sp_first_time_woba"]
    prior["sp_platoon_split_vs_lhb"] = _mean_from(prior, "vs_lhb_xwoba_sum", "vs_lhb_xwoba_count")
    prior["sp_platoon_split_vs_rhb"] = _mean_from(prior, "vs_rhb_xwoba_sum", "vs_rhb_xwoba_count")
    prior["sp_times_through_order_avg"] = np.nan

    keep = [
        "pitcher_id",
        "game_pk",
        "sp_whiff_rate",
        "sp_chase_rate",
        "sp_cs_rate",
        "sp_csw_rate",
        "sp_groundball_rate",
        "sp_barrel_allowed_rate",
        "sp_hard_hit_allowed_rate",
        "sp_avg_ev_allowed",
        "sp_xwoba_allowed",
        "sp_xera",
        "sp_first_time_woba",
        "sp_second_time_woba",
        "sp_third_time_woba",
        "sp_tto3_penalty",
        "sp_platoon_split_vs_lhb",
        "sp_platoon_split_vs_rhb",
        "sp_times_through_order_avg",
    ]
    out = prior[keep].copy()

    if not pitch_type_game.empty:
        pt = pitch_type_game.copy()
        for col in ["pitches", "release_speed", "release_spin_rate", "release_extension"]:
            pt[col] = pd.to_numeric(pt[col], errors="coerce")
        pt["velo_x_pitches"] = pt["release_speed"] * pt["pitches"]
        pt["spin_x_pitches"] = pt["release_spin_rate"] * pt["pitches"]
        pt["ext_x_pitches"] = pt["release_extension"] * pt["pitches"]
        stat = (
            pt.groupby(["pitcher_id", "game_pk", "game_date", "pitch_type"], as_index=False)
            .agg(pitches=("pitches", "sum"), velo_x_pitches=("velo_x_pitches", "sum"), spin_x_pitches=("spin_x_pitches", "sum"), ext_x_pitches=("ext_x_pitches", "sum"))
        )
        stat = stat.sort_values(["pitcher_id", "pitch_type", "game_date", "game_pk"])
        for col in ["pitches", "velo_x_pitches", "spin_x_pitches", "ext_x_pitches"]:
            stat[f"prior_{col}"] = stat.groupby(["pitcher_id", "pitch_type"])[col].cumsum() - stat[col]
        total = stat.groupby(["pitcher_id", "game_pk"], as_index=False)["prior_pitches"].sum().rename(columns={"prior_pitches": "prior_total_pitches"})
        stat = stat.merge(total, on=["pitcher_id", "game_pk"], how="left")
        stat["mix"] = stat["prior_pitches"] / stat["prior_total_pitches"].replace(0, np.nan)
        stat["velo"] = stat["prior_velo_x_pitches"] / stat["prior_pitches"].replace(0, np.nan)
        stat["spin"] = stat["prior_spin_x_pitches"] / stat["prior_pitches"].replace(0, np.nan)
        stat["extension"] = stat["prior_ext_x_pitches"] / stat["prior_pitches"].replace(0, np.nan)
        piv = stat.pivot_table(index=["pitcher_id", "game_pk"], columns="pitch_type", values="mix", aggfunc="first")
        piv = piv.rename(columns={pt: f"sp_pitch_mix_{pt.lower()}" for pt in piv.columns}).reset_index()
        fb = stat[stat["pitch_type"].isin(FASTBALL_TYPES)].groupby(["pitcher_id", "game_pk"], as_index=False).agg(
            fastball_velo=("velo", "mean"),
            fastball_spin=("spin", "mean"),
            extension=("extension", "mean"),
        )
        out = out.merge(piv, on=["pitcher_id", "game_pk"], how="left").merge(fb, on=["pitcher_id", "game_pk"], how="left")
        out = out.rename(columns={"fastball_velo": "sp_fastball_velo", "fastball_spin": "sp_fastball_spin", "extension": "sp_extension"})
    return out


def _build_batter_prior(batter_game: pd.DataFrame, batter_pitch_type_game: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if batter_game.empty:
        return pd.DataFrame(), pd.DataFrame()
    bg = batter_game.copy()
    for metric in ["avg_ev", "avg_la"]:
        bg[f"{metric}_sum"] = pd.to_numeric(bg.get(metric), errors="coerce").fillna(0.0)
        bg[f"{metric}_obs"] = pd.to_numeric(bg.get(metric), errors="coerce").notna().astype(float)
    bg = bg.fillna(0)
    stat_cols = [
        "pa",
        "k",
        "bb",
        "xwoba_sum",
        "xwoba_count",
        "xba_sum",
        "xba_count",
        "xslg_sum",
        "xslg_count",
        "hard_hit",
        "barrels",
        "bip",
        "sweet_spot",
        "avg_ev_sum",
        "avg_ev_obs",
        "avg_la_sum",
        "avg_la_obs",
    ]
    prior = _prior_player_game(bg, "batter_id", stat_cols)
    prior["avg_xwoba"] = _mean_from(prior, "xwoba_sum", "xwoba_count")
    prior["avg_xba"] = _mean_from(prior, "xba_sum", "xba_count")
    prior["avg_slg"] = _mean_from(prior, "xslg_sum", "xslg_count")
    prior["avg_iso"] = prior["avg_slg"] - prior["avg_xba"]
    prior["avg_k_rate"] = _mean_from(prior, "k", "pa")
    prior["avg_bb_rate"] = _mean_from(prior, "bb", "pa")
    prior["hard_hit_rate"] = _mean_from(prior, "hard_hit", "bip")
    prior["barrel_rate"] = _mean_from(prior, "barrels", "bip")
    prior["sweet_spot_rate"] = _mean_from(prior, "sweet_spot", "bip")
    prior["avg_ev"] = _mean_from(prior, "avg_ev_sum", "avg_ev_obs")
    prior["avg_la"] = _mean_from(prior, "avg_la_sum", "avg_la_obs")

    # Rolling recent values.
    recent_rows = []
    for bid, grp in bg.sort_values(["batter_id", "game_date", "game_pk"]).groupby("batter_id"):
        grp = grp.copy()
        for window in [14, 30]:
            shifted = grp.set_index("game_date")[["xwoba_sum", "xwoba_count", "hard_hit", "barrels", "bip", "k", "pa"]].shift(1)
            roll = shifted.rolling(f"{window}D", min_periods=1).sum()
            tmp = pd.DataFrame({
                "batter_id": bid,
                "game_pk": grp["game_pk"].values,
                f"recent_xwoba_{window}d": roll["xwoba_sum"].values / roll["xwoba_count"].replace(0, np.nan).values,
                f"recent_hard_hit_{window}d": roll["hard_hit"].values / roll["bip"].replace(0, np.nan).values,
                f"recent_barrel_{window}d": roll["barrels"].values / roll["bip"].replace(0, np.nan).values,
                f"recent_k_rate_{window}d": roll["k"].values / roll["pa"].replace(0, np.nan).values,
            })
            recent_rows.append(tmp)
    recent = pd.concat(recent_rows, ignore_index=True) if recent_rows else pd.DataFrame()
    if not recent.empty:
        recent = recent.groupby(["batter_id", "game_pk"], as_index=False).first()
        prior = prior.merge(recent, on=["batter_id", "game_pk"], how="left")

    keep = [
        "batter_id",
        "game_pk",
        "avg_xwoba",
        "avg_xba",
        "avg_slg",
        "avg_iso",
        "avg_k_rate",
        "avg_bb_rate",
        "hard_hit_rate",
        "barrel_rate",
        "avg_ev",
        "avg_la",
        "sweet_spot_rate",
        "recent_xwoba_14d",
        "recent_hard_hit_14d",
        "recent_barrel_14d",
        "recent_k_rate_14d",
        "recent_xwoba_30d",
        "recent_hard_hit_30d",
        "recent_barrel_30d",
        "recent_k_rate_30d",
    ]
    batter_prior = prior[[c for c in keep if c in prior.columns]].copy()

    if batter_pitch_type_game.empty:
        return batter_prior, pd.DataFrame()
    bt = batter_pitch_type_game.fillna(0).copy()
    bt = bt.sort_values(["batter_id", "pitch_type", "game_date", "game_pk"])
    for col in ["xwoba_sum", "xwoba_count"]:
        bt[f"prior_{col}"] = bt.groupby(["batter_id", "pitch_type"])[col].cumsum() - bt[col]
    bt["pitch_type_xwoba"] = bt["prior_xwoba_sum"] / bt["prior_xwoba_count"].replace(0, np.nan)
    return batter_prior, bt[["batter_id", "game_pk", "pitch_type", "pitch_type_xwoba"]]


def _avg_lineup(ids, game_pk: int, table: pd.DataFrame, id_col: str = "batter_id") -> pd.Series:
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if not isinstance(ids, list) or not ids:
        return pd.Series(dtype=float)
    subset = table[(table["game_pk"].eq(game_pk)) & (table[id_col].isin(ids))]
    return subset.mean(numeric_only=True)


def _lineup_index_mean(ids, game_pk: int, indexed: pd.DataFrame) -> pd.Series:
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if not isinstance(ids, list) or not ids or indexed.empty:
        return pd.Series(dtype=float)
    idx = pd.MultiIndex.from_product([[int(game_pk)], [int(i) for i in ids if pd.notna(i)]], names=["game_pk", "batter_id"])
    subset = indexed.reindex(idx)
    return subset.mean(numeric_only=True)


def _pitch_type_lineup_means(ids, game_pk: int, batter_pt_indexed: pd.DataFrame) -> dict[str, float]:
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if not isinstance(ids, list) or not ids or batter_pt_indexed.empty:
        return {}
    idx = pd.MultiIndex.from_product(
        [[int(game_pk)], [int(i) for i in ids if pd.notna(i)], PITCH_TYPES],
        names=["game_pk", "batter_id", "pitch_type"],
    )
    subset = batter_pt_indexed.reindex(idx).reset_index()
    if subset.empty:
        return {}
    return subset.groupby("pitch_type")["pitch_type_xwoba"].mean().to_dict()


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def _build_lineup_features(master: pd.DataFrame, game_lineups: pd.DataFrame, batter_prior: pd.DataFrame, batter_pt: pd.DataFrame, pitcher_prior: pd.DataFrame) -> pd.DataFrame:
    base = master[["game_pk", "home_starter_id", "away_starter_id"]].copy()
    out = base[["game_pk"]].copy()
    if game_lineups.empty or batter_prior.empty:
        return out
    lineups = game_lineups.copy()
    for col in ["home_lineup_ids", "away_lineup_ids"]:
        if col not in lineups.columns:
            lineups[col] = [[] for _ in range(len(lineups))]
    lineup_rows: list[dict] = []
    for row in lineups.itertuples(index=False):
        game_pk = int(row.game_pk)
        for side in ["home", "away"]:
            ids = getattr(row, f"{side}_lineup_ids")
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            if isinstance(ids, tuple):
                ids = list(ids)
            if not isinstance(ids, list):
                ids = []
            for slot, pid in enumerate(ids[:9], start=1):
                if pd.notna(pid):
                    lineup_rows.append({"game_pk": game_pk, "side": side, "slot": slot, "batter_id": int(pid)})
    if not lineup_rows:
        return out
    lineup_long = pd.DataFrame(lineup_rows)
    bp = batter_prior.copy()
    bp["game_pk"] = pd.to_numeric(bp["game_pk"], errors="coerce").astype("Int64")
    bp["batter_id"] = pd.to_numeric(bp["batter_id"], errors="coerce").astype("Int64")
    merged = lineup_long.merge(bp, on=["game_pk", "batter_id"], how="left")
    metric_cols = [
        "avg_xwoba",
        "avg_xba",
        "avg_slg",
        "avg_iso",
        "avg_k_rate",
        "avg_bb_rate",
        "hard_hit_rate",
        "barrel_rate",
        "avg_ev",
        "avg_la",
        "sweet_spot_rate",
        "recent_xwoba_14d",
        "recent_hard_hit_14d",
        "recent_barrel_14d",
        "recent_k_rate_14d",
        "recent_xwoba_30d",
        "recent_hard_hit_30d",
        "recent_barrel_30d",
        "recent_k_rate_30d",
    ]
    means = merged.groupby(["game_pk", "side"], as_index=False)[[c for c in metric_cols if c in merged.columns]].mean()
    wide = pd.DataFrame({"game_pk": base["game_pk"].astype(int)})
    for side in ["home", "away"]:
        part = means[means["side"].eq(side)].drop(columns=["side"]).rename(columns={c: f"{side}_lineup_{c}" for c in metric_cols if c in means.columns})
        wide = wide.merge(part, on="game_pk", how="left")
        prefix = f"{side}_lineup"
        wide[f"{side}_contact_quality_recent"] = wide[[f"{prefix}_recent_xwoba_14d", f"{prefix}_recent_hard_hit_14d", f"{prefix}_recent_barrel_14d"]].mean(axis=1)
        wide[f"{prefix}_depth_score"] = wide.get(f"{prefix}_avg_xwoba")

        top4 = merged[(merged["side"].eq(side)) & (merged["slot"] <= 4)].groupby("game_pk")["avg_xwoba"].mean().rename(f"{prefix}_top4_xwoba")
        bottom5 = merged[(merged["side"].eq(side)) & (merged["slot"] >= 5)].groupby("game_pk")["avg_xwoba"].mean().rename(f"{prefix}_bottom5_xwoba")
        wide = wide.merge(top4, on="game_pk", how="left").merge(bottom5, on="game_pk", how="left")

    pitch_type_lookup: dict[tuple[int, str, str], float] = {}
    if not batter_pt.empty:
        bt = batter_pt.copy()
        bt["game_pk"] = pd.to_numeric(bt["game_pk"], errors="coerce").astype("Int64")
        bt["batter_id"] = pd.to_numeric(bt["batter_id"], errors="coerce").astype("Int64")
        pt_means = (
            lineup_long.merge(bt, on=["game_pk", "batter_id"], how="left")
            .groupby(["game_pk", "side", "pitch_type"])["pitch_type_xwoba"]
            .mean()
            .dropna()
        )
        pitch_type_lookup = {(int(g), str(s), str(pt)): float(v) for (g, s, pt), v in pt_means.items()}

    pitcher_lookup = pitcher_prior.set_index(["pitcher_id", "game_pk"]) if not pitcher_prior.empty else pd.DataFrame()
    wide_lookup = wide.set_index("game_pk")
    rows = []
    for row in base.itertuples(index=False):
        game_pk = int(row.game_pk)
        data = {"game_pk": game_pk}
        game_vals = wide_lookup.loc[game_pk] if game_pk in wide_lookup.index else pd.Series(dtype=float)
        for bat_side, pitch_side, spid in [("home", "away", row.away_starter_id), ("away", "home", row.home_starter_id)]:
            lineup_avg = game_vals.get(f"{bat_side}_lineup_avg_xwoba", np.nan)
            sp_xwoba = sp_whiff = sp_gb = sp_barrel = np.nan
            sp_row = pd.Series(dtype=float)
            if pd.notna(spid) and not pitcher_prior.empty:
                key = (float(spid), game_pk)
                if key in pitcher_lookup.index:
                    sp_row = pitcher_lookup.loc[key]
                    sp_xwoba = sp_row.get("sp_xwoba_allowed", np.nan)
                    sp_whiff = sp_row.get("sp_whiff_rate", np.nan)
                    sp_gb = sp_row.get("sp_groundball_rate", np.nan)
                    sp_barrel = sp_row.get("sp_barrel_allowed_rate", np.nan)
            data[f"{bat_side}_lineup_vs_{pitch_side}_sp_xwoba"] = _safe_mean([lineup_avg, sp_xwoba])
            data[f"{bat_side}_lineup_whiff_risk_vs_{pitch_side}_sp"] = _safe_mean([game_vals.get(f"{bat_side}_lineup_avg_k_rate", np.nan), sp_whiff])
            data[f"{bat_side}_lineup_gb_fit_vs_{pitch_side}_sp"] = sp_gb
            data[f"{bat_side}_lineup_power_vs_{pitch_side}_sp"] = _safe_mean([game_vals.get(f"{bat_side}_lineup_barrel_rate", np.nan), sp_barrel])
            fit = 0.0
            total_w = 0.0
            for pt in PITCH_TYPES:
                val = pitch_type_lookup.get((game_pk, bat_side, pt), np.nan)
                w = sp_row.get(f"sp_pitch_mix_{pt.lower()}", np.nan)
                if pd.notna(val) and pd.notna(w):
                    fit += val * w
                    total_w += w
                if pt in ["FF", "SL", "CH"]:
                    data[f"{bat_side}_lineup_vs_{pitch_side}_sp_pitch_type_{pt.lower()}"] = val
            data[f"{bat_side}_lineup_vs_{pitch_side}_sp_pitch_mix_fit"] = fit / total_w if total_w else np.nan
        rows.append(data)
    return wide.merge(pd.DataFrame(rows), on="game_pk", how="left")


def _build_catcher_features(catcher_game: pd.DataFrame) -> pd.DataFrame:
    if catcher_game.empty:
        return pd.DataFrame()
    cg = catcher_game.fillna(0).copy()
    stat_cols = ["called_strikes", "balls", "shadow_called_strikes", "shadow_pitches"]
    prior = _prior_player_game(cg, "catcher_id", stat_cols)
    prior["catcher_strike_rate"] = prior["prior_called_strikes"] / (prior["prior_called_strikes"] + prior["prior_balls"]).replace(0, np.nan)
    prior["catcher_shadow_strike_rate"] = prior["prior_shadow_called_strikes"] / prior["prior_shadow_pitches"].replace(0, np.nan)
    prior["catcher_framing_runs"] = (prior["catcher_shadow_strike_rate"] - 0.47) * prior["prior_shadow_pitches"] * 0.125
    prior["catcher_framing_runs_1000"] = prior["catcher_framing_runs"] / prior["prior_shadow_pitches"].replace(0, np.nan) * 1000
    return prior[["catcher_id", "game_pk", "catcher_strike_rate", "catcher_shadow_strike_rate", "catcher_framing_runs", "catcher_framing_runs_1000"]]


def _build_pitcher_recent_quality(pitcher_game: pd.DataFrame) -> pd.DataFrame:
    if pitcher_game.empty:
        return pd.DataFrame()
    pg = pitcher_game.fillna(0).copy()
    for col in ["pitcher_id", "game_pk", "swings", "whiffs", "xwoba_sum", "xwoba_count"]:
        if col not in pg.columns:
            pg[col] = 0.0
        pg[col] = pd.to_numeric(pg[col], errors="coerce")
    pg["game_date"] = pd.to_datetime(pg["game_date"], errors="coerce")
    rows: list[pd.DataFrame] = []
    for pid, grp in pg.sort_values(["pitcher_id", "game_date", "game_pk"]).groupby("pitcher_id", sort=False):
        grp = grp.copy()
        shifted = grp.set_index("game_date")[["swings", "whiffs", "xwoba_sum", "xwoba_count"]].shift(1)
        roll = shifted.rolling("30D", min_periods=1).sum()
        rows.append(
            pd.DataFrame(
                {
                    "pitcher_id": int(pid),
                    "game_pk": grp["game_pk"].astype(int).values,
                    "rp_xwoba_allowed_30d": roll["xwoba_sum"].values / roll["xwoba_count"].replace(0, np.nan).values,
                    "rp_whiff_rate_30d": roll["whiffs"].values / roll["swings"].replace(0, np.nan).values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_role_quality_features(master: pd.DataFrame, pitcher_recent: pd.DataFrame) -> pd.DataFrame:
    out = master[["game_pk"]].copy()
    if pitcher_recent.empty:
        return out
    recent = pitcher_recent.copy()
    recent["pitcher_id"] = pd.to_numeric(recent["pitcher_id"], errors="coerce")
    recent["game_pk"] = pd.to_numeric(recent["game_pk"], errors="coerce")
    work = master.copy()
    work["game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce")

    role_specs = [
        ("closer", ["xwoba_allowed_30d", "whiff_rate_30d"]),
        ("setup", ["xwoba_allowed_30d", "whiff_rate_30d"]),
        ("top_lhrp", ["xwoba_allowed_30d"]),
        ("top_rhrp", ["xwoba_allowed_30d"]),
    ]
    for side in ["home", "away"]:
        for role, metrics in role_specs:
            id_col = f"{side}_{role}_id"
            if id_col not in work.columns:
                continue
            part = recent.rename(columns={"pitcher_id": id_col})
            rename: dict[str, str] = {}
            if "xwoba_allowed_30d" in metrics:
                rename["rp_xwoba_allowed_30d"] = f"{side}_{role}_xwoba_allowed_30d"
            if "whiff_rate_30d" in metrics:
                rename["rp_whiff_rate_30d"] = f"{side}_{role}_whiff_rate_30d"
            cols = [id_col, "game_pk"] + list(rename.keys())
            tmp = work[["game_pk", id_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp = tmp.merge(part[cols].rename(columns=rename), on=[id_col, "game_pk"], how="left")
            out = out.merge(tmp.drop(columns=[id_col]), on="game_pk", how="left")

    diff_pairs = {
        "closer_quality_DIFF": ("home_closer_xwoba_allowed_30d", "away_closer_xwoba_allowed_30d"),
        "setup_quality_DIFF": ("home_setup_xwoba_allowed_30d", "away_setup_xwoba_allowed_30d"),
        "top_lhrp_quality_DIFF": ("home_top_lhrp_xwoba_allowed_30d", "away_top_lhrp_xwoba_allowed_30d"),
        "top_rhrp_quality_DIFF": ("home_top_rhrp_xwoba_allowed_30d", "away_top_rhrp_xwoba_allowed_30d"),
    }
    for dest, (home_col, away_col) in diff_pairs.items():
        if home_col in out.columns and away_col in out.columns:
            out[dest] = out[away_col] - out[home_col]
    return out


def _build_bullpen_team_savant_features(master: pd.DataFrame, pitcher_game: pd.DataFrame) -> pd.DataFrame:
    out = master[["game_pk"]].copy()
    if pitcher_game.empty or not MLB_RAW_PATH.exists():
        return out
    raw = pd.read_parquet(MLB_RAW_PATH)
    appearances = _appearance_frame(raw)
    if appearances.empty:
        return out
    relievers = appearances[~appearances["is_starter"]][["game_pk", "pitcher_id", "team"]].copy()
    relievers["game_pk"] = pd.to_numeric(relievers["game_pk"], errors="coerce")
    relievers["pitcher_id"] = pd.to_numeric(relievers["pitcher_id"], errors="coerce")
    pg = pitcher_game.copy()
    pg["game_pk"] = pd.to_numeric(pg["game_pk"], errors="coerce")
    pg["pitcher_id"] = pd.to_numeric(pg["pitcher_id"], errors="coerce")
    joined = pg.merge(relievers, on=["game_pk", "pitcher_id"], how="inner")
    if joined.empty:
        return out
    joined["game_day"] = pd.to_datetime(joined["game_date"], errors="coerce").dt.normalize()
    daily = (
        joined.groupby(["team", "game_day"], as_index=False)
        .agg(
            swings=("swings", "sum"),
            whiffs=("whiffs", "sum"),
            xwoba_sum=("xwoba_sum", "sum"),
            xwoba_count=("xwoba_count", "sum"),
        )
        .sort_values(["team", "game_day"])
    )
    frames: list[pd.DataFrame] = []
    for team, grp in daily.groupby("team", sort=False):
        grp = grp.sort_values("game_day").set_index("game_day")
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_idx)
        grp[["swings", "whiffs", "xwoba_sum", "xwoba_count"]] = grp[["swings", "whiffs", "xwoba_sum", "xwoba_count"]].fillna(0.0)
        shifted = grp[["swings", "whiffs", "xwoba_sum", "xwoba_count"]].shift(1).fillna(0.0)
        roll = shifted.rolling(30, min_periods=1).sum()
        frames.append(
            pd.DataFrame(
                {
                    "team": team,
                    "game_day": full_idx,
                    "bp_whiff_rate_30d": roll["whiffs"].values / roll["swings"].replace(0, np.nan).values,
                    "bp_xwoba_allowed_30d": roll["xwoba_sum"].values / roll["xwoba_count"].replace(0, np.nan).values,
                }
            )
        )
    by_team_day = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if by_team_day.empty:
        return out
    work = master[["game_pk", "game_date", "home_team", "away_team"]].copy()
    work["game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce")
    work["game_day"] = pd.to_datetime(work["game_date"], errors="coerce").dt.normalize()
    for side in ["home", "away"]:
        part = by_team_day.rename(columns={"team": f"{side}_team"})
        part = part.rename(columns={c: f"{side}_{c}" for c in ["bp_whiff_rate_30d", "bp_xwoba_allowed_30d"]})
        work = work.merge(part, on=[f"{side}_team", "game_day"], how="left")
    return work[
        [
            "game_pk",
            "home_bp_whiff_rate_30d",
            "home_bp_xwoba_allowed_30d",
            "away_bp_whiff_rate_30d",
            "away_bp_xwoba_allowed_30d",
        ]
    ]


def _build_umpire_features(master: pd.DataFrame, catcher_game: pd.DataFrame) -> pd.DataFrame:
    out = master[["game_pk"]].copy()
    if catcher_game.empty or "umpire_id" not in master.columns:
        return out
    game_zone = (
        catcher_game.groupby("game_pk", as_index=False)
        .agg(
            called_strikes=("called_strikes", "sum"),
            balls=("balls", "sum"),
            shadow_called_strikes=("shadow_called_strikes", "sum"),
            shadow_pitches=("shadow_pitches", "sum"),
        )
    )
    work = master[["game_pk", "game_date", "umpire_id"]].merge(game_zone, on="game_pk", how="inner")
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    for col in ["umpire_id", "called_strikes", "balls", "shadow_called_strikes", "shadow_pitches"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["umpire_id"]).sort_values(["umpire_id", "game_date", "game_pk"])
    stat_cols = ["called_strikes", "balls", "shadow_called_strikes", "shadow_pitches"]
    prior = _prior_player_game(work.fillna(0), "umpire_id", stat_cols)
    total_calls = prior["prior_called_strikes"] + prior["prior_balls"]
    prior["umpire_called_strike_rate"] = prior["prior_called_strikes"] / total_calls.replace(0, np.nan)
    prior["umpire_zone_size"] = prior["umpire_called_strike_rate"]
    prior["umpire_zone_tightness"] = prior["prior_shadow_called_strikes"] / prior["prior_shadow_pitches"].replace(0, np.nan)
    prior["umpire_consistency"] = 1.0 - (prior["umpire_zone_tightness"] - 0.47).abs()
    return prior[["game_pk", "umpire_zone_size", "umpire_zone_tightness", "umpire_called_strike_rate", "umpire_consistency"]]


def _build_pair_features(master: pd.DataFrame, catcher_prior: pd.DataFrame) -> pd.DataFrame:
    out = master[["game_pk"]].copy()
    if catcher_prior.empty:
        return out
    rows: list[dict] = []
    cp = catcher_prior.copy()
    cp["game_pk"] = pd.to_numeric(cp["game_pk"], errors="coerce")
    for side in ["home", "away"]:
        needed = [f"{side}_starter_id", f"{side}_catcher_id", "game_pk", "game_date"]
        if not set(needed) <= set(master.columns):
            continue
        part = master[needed].rename(columns={f"{side}_starter_id": "starter_id", f"{side}_catcher_id": "catcher_id"}).copy()
        part["starter_id"] = pd.to_numeric(part["starter_id"], errors="coerce")
        part["catcher_id"] = pd.to_numeric(part["catcher_id"], errors="coerce")
        part["game_pk"] = pd.to_numeric(part["game_pk"], errors="coerce")
        part["game_date"] = pd.to_datetime(part["game_date"], errors="coerce")
        part = part.merge(cp, on=["game_pk", "catcher_id"], how="left").sort_values(["starter_id", "catcher_id", "game_date", "game_pk"])
        for col in ["catcher_framing_runs", "catcher_strike_rate"]:
            part[f"pair_prior_{col}_sum"] = part.groupby(["starter_id", "catcher_id"])[col].cumsum() - part[col].fillna(0)
            part[f"pair_prior_{col}_count"] = part.groupby(["starter_id", "catcher_id"])[col].transform(lambda s: s.notna().cumsum()) - part[col].notna().astype(int)
        tmp = pd.DataFrame({"game_pk": part["game_pk"].astype(int)})
        tmp[f"{side}_sp_catcher_pair_framing"] = part["pair_prior_catcher_framing_runs_sum"] / part["pair_prior_catcher_framing_runs_count"].replace(0, np.nan)
        tmp[f"{side}_sp_catcher_pair_cs_rate"] = part["pair_prior_catcher_strike_rate_sum"] / part["pair_prior_catcher_strike_rate_count"].replace(0, np.nan)
        rows.append(tmp)
    for frame in rows:
        out = out.merge(frame, on="game_pk", how="left")
    return out


def build_savant_features(master_path: Path = MASTER_PATH) -> None:
    master = pd.read_csv(master_path, low_memory=False)
    master["game_pk"] = pd.to_numeric(master["game_pk"], errors="coerce").astype("Int64")
    print("Savant build: reading aggregate parquet", flush=True)
    pitcher_game = _read_kind("pitcher_game")
    print(f"  pitcher_game rows={len(pitcher_game):,}", flush=True)
    pitch_type_game = _read_kind("pitcher_pitch_type_game")
    print(f"  pitcher_pitch_type_game rows={len(pitch_type_game):,}", flush=True)
    batter_game = _read_kind("batter_game")
    print(f"  batter_game rows={len(batter_game):,}", flush=True)
    batter_pt = _read_kind("batter_pitch_type_game")
    print(f"  batter_pitch_type_game rows={len(batter_pt):,}", flush=True)
    catcher_game = _read_kind("catcher_game")
    print(f"  catcher_game rows={len(catcher_game):,}", flush=True)
    lineups = _read_kind("game_lineups")
    print(f"  game_lineups rows={len(lineups):,}", flush=True)

    print("Savant build: pitcher priors", flush=True)
    pitcher_prior = _build_pitcher_prior(pitcher_game, pitch_type_game)
    pitcher_recent = _build_pitcher_recent_quality(pitcher_game)
    print("Savant build: batter priors", flush=True)
    batter_prior, batter_pt_prior = _build_batter_prior(batter_game, batter_pt)
    print("Savant build: lineup features", flush=True)
    lineup_features = _build_lineup_features(master, lineups, batter_prior, batter_pt_prior, pitcher_prior)
    print("Savant build: catcher/role features", flush=True)
    catcher_prior = _build_catcher_features(catcher_game)
    role_quality = _build_role_quality_features(master, pitcher_recent)
    bullpen_team = _build_bullpen_team_savant_features(master, pitcher_game)
    umpire_features = _build_umpire_features(master, catcher_game)
    pair_features = _build_pair_features(master, catcher_prior)

    out = master[["game_pk", "home_starter_id", "away_starter_id", "home_catcher_id", "away_catcher_id"]].copy()
    for side, id_col in [("home", "home_starter_id"), ("away", "away_starter_id")]:
        if not pitcher_prior.empty:
            pp = pitcher_prior.rename(columns={"pitcher_id": id_col})
            rename = {c: f"{side}_{c}" for c in pp.columns if c.startswith("sp_")}
            pp = pp.rename(columns=rename)
            out = out.merge(pp.drop_duplicates([id_col, "game_pk"]), on=[id_col, "game_pk"], how="left")
    out = out.merge(lineup_features, on="game_pk", how="left")
    out = out.merge(role_quality, on="game_pk", how="left")
    out = out.merge(bullpen_team, on="game_pk", how="left")
    out = out.merge(umpire_features, on="game_pk", how="left")
    out = out.merge(pair_features, on="game_pk", how="left")
    for side, id_col in [("home", "home_catcher_id"), ("away", "away_catcher_id")]:
        if not catcher_prior.empty and id_col in out.columns:
            cp = catcher_prior.rename(columns={"catcher_id": id_col})
            rename = {c: f"{side}_{c}" for c in cp.columns if c.startswith("catcher_")}
            cp = cp.rename(columns=rename)
            out = out.merge(cp.drop_duplicates([id_col, "game_pk"]), on=[id_col, "game_pk"], how="left")

    out = out.drop(columns=["home_starter_id", "away_starter_id", "home_catcher_id", "away_catcher_id"], errors="ignore")
    out.to_parquet(SAVANT_FEATURES_PATH, index=False)
    print(f"Savant features wrote {SAVANT_FEATURES_PATH} rows={len(out):,} cols={len(out.columns):,}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--start", default="2015-03-01")
    fetch.add_argument("--end", default="2026-11-30")
    fetch.add_argument("--days", type=int, default=1)
    fetch.add_argument("--limit-chunks", type=int)
    fetch.add_argument("--workers", type=int, default=4)
    build = sub.add_parser("build")
    build.add_argument("--master", type=Path, default=MASTER_PATH)
    all_cmd = sub.add_parser("fetch-build")
    all_cmd.add_argument("--start", default="2015-03-01")
    all_cmd.add_argument("--end", default="2026-11-30")
    all_cmd.add_argument("--days", type=int, default=1)
    all_cmd.add_argument("--limit-chunks", type=int)
    all_cmd.add_argument("--master", type=Path, default=MASTER_PATH)
    all_cmd.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.cmd in {"fetch", "fetch-build"}:
        fetch_savant(args.start, args.end, args.days, args.limit_chunks, args.workers)
    if args.cmd in {"build", "fetch-build"}:
        build_savant_features(args.master)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
