"""Sandbox feature helpers.

Keep experimental feature code here until a feature group earns production
promotion. Functions in this module must be deterministic and read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_CUTOFF = "2026-01-01"

LIVE_STATE_COLS = (
    "balls",
    "strikes",
    "inning",
    "outs",
    "detailed_state",
    "game_state",
    "game_status",
    "inning_state",
    "inning_ordinal",
    "is_completed",
    "is_top_inning",
    "live_updated_at",
    "live_status_text",
)

LEGACY_DROP_COLS = (
    "over_under",
    "home_starter_era",
    "home_starter_whip",
    "home_starter_k9",
    "home_starter_bb9",
    "home_starter_fip",
    "home_starter_hand",
    "away_starter_era",
    "away_starter_whip",
    "away_starter_k9",
    "away_starter_bb9",
    "away_starter_fip",
    "away_starter_hand",
    "wind_speed_kph",
    "precip_mm",
    "home_k9",
    "home_bb9",
    "away_k9",
    "away_bb9",
    # Keep probable IDs in sandbox master so we can simulate "listed pitcher"
    # constraints and measure scratch/opener frequency.
    "home_bp_xfip_30d",
    "away_bp_xfip_30d",
)


@dataclass(frozen=True)
class SandboxContract:
    cutoff: str = DEFAULT_CUTOFF
    target_col: str = "home_win"


def apply_sandbox_contract(df: pd.DataFrame, contract: SandboxContract | None = None) -> pd.DataFrame:
    """Return rows allowed for sandbox training/testing."""
    contract = contract or SandboxContract()
    out = df.copy()
    if "game_date" not in out.columns:
        raise ValueError("sandbox master requires game_date")

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out[out["game_date"].notna()].copy()
    out = out[out["game_date"] < pd.Timestamp(contract.cutoff)].copy()

    drop_cols = [c for c in LIVE_STATE_COLS if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year

    bad = out[out["game_date"] >= pd.Timestamp(contract.cutoff)]
    if not bad.empty:
        raise ValueError(f"sandbox master contains rows on/after {contract.cutoff}")

    return out.sort_values(["game_date", "game_id"], na_position="last").reset_index(drop=True)


def feature_coverage(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Summarize feature coverage by season."""
    if "season" not in df.columns:
        raise ValueError("feature coverage requires season")
    rows: list[dict] = []
    for season, season_df in df.groupby("season"):
        row: dict[str, float | int] = {"season": int(season), "rows": int(len(season_df))}
        for col in feature_cols:
            if col in season_df.columns:
                row[f"{col}__notna"] = float(season_df[col].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
