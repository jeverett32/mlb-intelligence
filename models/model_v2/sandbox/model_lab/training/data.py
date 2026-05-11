"""Sandbox training data loader, leakage audit, feature selection."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LAB_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = LAB_DIR / "output" / "master_sandbox_mlb.csv"

LEAKY_KEYWORDS = (
    "home_score",
    "away_score",
    "winner",
    "result",
    "final",
    "actual",
    "_postgame",
    "outcome",
    "live_",
    "current_",
)

NEVER_FEATURE = {
    "game_id",
    "game_pk",
    "home_win",
    "home_team",
    "away_team",
    "game_date",
    "season",
    "home_score",
    "away_score",
    "venue_name",
    "venue_id",
    "home_starter_name",
    "away_starter_name",
    "umpire_name",
    "home_starter_id",
    "away_starter_id",
    "umpire_id",
    "catcher_id_home",
    "catcher_id_away",
    "game_time_utc",
    "season_start_date",
}

EARLY_FEATURES = (
    "sp_era_DIFF",
    "sp_whip_DIFF",
    "sp_k9_DIFF",
    "sp_bb9_DIFF",
    "sp_fip_DIFF",
    "rolling_era_DIFF",
    "rolling_whip_DIFF",
    "rolling_k9_DIFF",
    "market_implied_prob",
    "park_factor",
    "home_rest_days",
    "away_rest_days",
    "month",
    "is_night_game",
)


def load_master(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_MASTER, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date", "home_win"]).copy()
    df["home_win"] = df["home_win"].astype(int)
    df["season"] = df["game_date"].dt.year
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def leakage_audit(cols: list[str]) -> list[str]:
    flagged: list[str] = []
    for c in cols:
        if c in NEVER_FEATURE:
            continue
        for kw in LEAKY_KEYWORDS:
            if re.search(kw, c, re.IGNORECASE):
                flagged.append(c)
                break
    return sorted(set(flagged))


def select_numeric_features(
    df: pd.DataFrame,
    *,
    min_coverage: float = 0.5,
    extra_drop: tuple[str, ...] = (),
) -> list[str]:
    """Numeric cols with coverage >= threshold. Drops leakage and id cols."""
    numeric = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    drop = NEVER_FEATURE | set(extra_drop) | set(leakage_audit(numeric))
    feats: list[str] = []
    for c in numeric:
        if c in drop:
            continue
        if df[c].notna().mean() < min_coverage:
            continue
        if df[c].nunique(dropna=True) < 2:
            continue
        feats.append(c)
    return feats


def early_features(df: pd.DataFrame) -> list[str]:
    return [c for c in EARLY_FEATURES if c in df.columns]


def split_early(df: pd.DataFrame, cutoff: int) -> tuple[pd.Series, pd.Series]:
    if cutoff is None or cutoff <= 0:
        return pd.Series(False, index=df.index), pd.Series(True, index=df.index)
    home = pd.to_numeric(df.get("home_games_played"), errors="coerce")
    away = pd.to_numeric(df.get("away_games_played"), errors="coerce")
    early = (home < cutoff) | (away < cutoff)
    early = early.fillna(False)
    return early, ~early
