"""Cumulative-to-date starter pitcher stats for sandbox master.

Replaces the season-final values that fetch_data writes into
``home_sp_era`` / ``away_sp_era`` (and bb9/k9/whip) with cumulative-to-date
values computed from per-game pitching logs. Each (pitcher, season) is
sorted by game_date, components are cumsum'd, then shift(1) drops the
current game so only prior-game information feeds the feature.

Source: ``data/cache/starter_fip_game_logs.parquet`` (populated by
``scripts/backfill_starter_fip.py``). Schema: pitcher_id, season,
game_date, ip, er, hr, h, bb, hbp, so.

Sandbox-only. Production ``home_sp_*`` columns remain season-final until
a parallel backfill writes ``h_sp_era_lag1`` etc. into ``games.extra``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOGS_PATH = REPO_ROOT / "data" / "cache" / "starter_fip_game_logs.parquet"

MIN_IP_FOR_RATE = 10.0
STAT_COLS = ("era", "bb9", "k9", "whip")


def _coerce(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_cumulative_lookup(logs_path: Path | None = None) -> pd.DataFrame:
    """Per-pitcher per-game cumulative-to-date rate stats.

    Output columns: pitcher_id, season, game_date, sp_era_todate,
    sp_bb9_todate, sp_k9_todate, sp_whip_todate, sp_ip_todate.
    Values reflect prior games in the same season (shift(1) applied
    after cumsum). Below MIN_IP_FOR_RATE prior IP, rate stats are NaN.
    """
    path = logs_path or DEFAULT_LOGS_PATH
    cols = [
        "pitcher_id",
        "season",
        "game_date",
        "sp_era_todate",
        "sp_bb9_todate",
        "sp_k9_todate",
        "sp_whip_todate",
        "sp_ip_todate",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df["pitcher_id"] = _coerce(df["pitcher_id"]).astype("Int64")
    df["season"] = _coerce(df["season"]).astype("Int64")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    for c in ("ip", "er", "hr", "h", "bb", "hbp", "so"):
        df[c] = _coerce(df.get(c, pd.Series(dtype=float)))
    df = df.dropna(subset=["pitcher_id", "season", "game_date"])

    df = df.sort_values(["pitcher_id", "season", "game_date"]).reset_index(drop=True)
    grp = df.groupby(["pitcher_id", "season"], sort=False)

    cum_ip = grp["ip"].cumsum().shift(1)
    cum_er = grp["er"].cumsum().shift(1)
    cum_h = grp["h"].cumsum().shift(1)
    cum_bb = grp["bb"].cumsum().shift(1)
    cum_so = grp["so"].cumsum().shift(1)

    # Reset shift(1) carry across (pitcher, season) boundaries.
    first_row_mask = grp.cumcount() == 0
    for s in (cum_ip, cum_er, cum_h, cum_bb, cum_so):
        s[first_row_mask] = np.nan

    safe_ip = cum_ip.where(cum_ip >= MIN_IP_FOR_RATE)

    return pd.DataFrame(
        {
            "pitcher_id": df["pitcher_id"].astype("Int64"),
            "season": df["season"].astype("Int64"),
            "game_date": df["game_date"],
            "sp_ip_todate": cum_ip,
            "sp_era_todate": 9.0 * cum_er / safe_ip,
            "sp_bb9_todate": 9.0 * cum_bb / safe_ip,
            "sp_k9_todate": 9.0 * cum_so / safe_ip,
            "sp_whip_todate": (cum_h + cum_bb) / safe_ip,
        }
    )


def apply_cumulative_to_master(
    master: pd.DataFrame,
    logs_path: Path | None = None,
) -> pd.DataFrame:
    """Overwrite home_sp_*/away_sp_* with cumulative-to-date values.

    Falls back to existing (season-final) values when the pitcher has
    < MIN_IP_FOR_RATE prior IP. Those early-season rows are still
    leaky; downstream model code may want to filter or weight them
    separately.
    """
    if not {"game_date", "home_starter_id", "away_starter_id"} <= set(master.columns):
        return master

    lookup = build_cumulative_lookup(logs_path)
    if lookup.empty:
        return master

    out = master.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year

    join_date = out["game_date"].dt.date
    out_season = pd.to_numeric(out["season"], errors="coerce").astype("Int64")

    # Index lookup once for fast per-side joins.
    lookup = lookup.set_index(["pitcher_id", "season", "game_date"])

    for side in ("home", "away"):
        pid_col = f"{side}_starter_id"
        pid = pd.to_numeric(out[pid_col], errors="coerce").astype("Int64")
        keys = pd.MultiIndex.from_arrays([pid, out_season, join_date])
        joined = lookup.reindex(keys).reset_index(drop=True)

        for stat in STAT_COLS:
            target = f"{side}_sp_{stat}"
            todate = pd.to_numeric(joined[f"sp_{stat}_todate"], errors="coerce")
            existing = pd.to_numeric(out.get(target, pd.Series(np.nan, index=out.index)), errors="coerce")
            out[target] = todate.combine_first(existing).values

        out[f"{side}_sp_ip_todate"] = pd.to_numeric(joined["sp_ip_todate"], errors="coerce").values

    return _recompute_diffs(out)


def _recompute_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute pitcher DIFF cols from updated home_sp_*/away_sp_* values.

    Mirrors model/train.py conventions:
      - sp_era_DIFF, sp_bb9_DIFF, sp_whip_DIFF: away - home
      - sp_k9_DIFF: home - away
      - rolling_*_DIFF chain falls back to sp_* when rolling missing.
    """

    def col(name: str) -> pd.Series:
        return pd.to_numeric(df.get(name, pd.Series(np.nan, index=df.index)), errors="coerce")

    h_era = col("home_sp_era")
    a_era = col("away_sp_era")
    h_bb9 = col("home_sp_bb9")
    a_bb9 = col("away_sp_bb9")
    h_k9 = col("home_sp_k9")
    a_k9 = col("away_sp_k9")
    h_whip = col("home_sp_whip")
    a_whip = col("away_sp_whip")

    df["sp_era_DIFF"] = a_era - h_era
    df["sp_bb9_DIFF"] = a_bb9 - h_bb9
    df["sp_k9_DIFF"] = h_k9 - a_k9
    df["sp_whip_DIFF"] = a_whip - h_whip

    h_re = col("home_rolling_era").combine_first(h_era)
    a_re = col("away_rolling_era").combine_first(a_era)
    df["rolling_era_DIFF"] = a_re - h_re

    h_rw = col("home_rolling_whip").combine_first(h_whip)
    a_rw = col("away_rolling_whip").combine_first(a_whip)
    df["rolling_whip_DIFF"] = a_rw - h_rw

    h_rk = col("home_rolling_k9").combine_first(h_k9)
    a_rk = col("away_rolling_k9").combine_first(a_k9)
    df["rolling_k9_DIFF"] = h_rk - a_rk

    return df
