#!/usr/bin/env python3
"""Build sandbox master CSV through completed 2025 season only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "model_v1"))

from models.model_v2.sandbox.model_lab import features, pitcher_cumulative, planned_features  # noqa: E402


OUTPUT_PATH = LAB_DIR / "output" / "master_sandbox_mlb.csv"
MANIFEST_PATH = LAB_DIR / "output" / "master_sandbox_manifest.json"
FEATURE_STATUS_PATH = LAB_DIR / "output" / "planned_feature_status.json"
CACHE_DIR = LAB_DIR / "cache"
REAL_SOURCE_FEATURE_FILES = [
    CACHE_DIR / "mlb_game_times.parquet",
    CACHE_DIR / "mlb_statsapi_features.parquet",
    CACHE_DIR / "openmeteo_hourly_features.parquet",
    CACHE_DIR / "savant_features.parquet",
    CACHE_DIR / "savant_leaderboard_features.parquet",
]


def _load_production_engineered_frame(*, allow_csv_fallback: bool = True) -> pd.DataFrame:
    """Read current engineered frame without writing production artifacts."""
    import models.model_v1.train as T

    return T.load_and_engineer_features(allow_csv_fallback=allow_csv_fallback)


def build_master(cutoff: str = features.DEFAULT_CUTOFF, *, allow_csv_fallback: bool = True) -> pd.DataFrame:
    raw = _load_production_engineered_frame(allow_csv_fallback=allow_csv_fallback)
    filtered = features.apply_sandbox_contract(raw, features.SandboxContract(cutoff=cutoff))
    planned = planned_features.add_planned_features(filtered)
    for path in REAL_SOURCE_FEATURE_FILES:
        if not path.exists():
            continue
        source = pd.read_parquet(path)
        if "game_pk" not in source.columns:
            continue
        source["game_pk"] = pd.to_numeric(source["game_pk"], errors="coerce")
        planned["game_pk"] = pd.to_numeric(planned["game_pk"], errors="coerce")
        overlap = [c for c in source.columns if c in planned.columns and c != "game_pk"]
        planned = planned.drop(columns=overlap).merge(source, on="game_pk", how="left")
    planned = planned_features.derive_after_real_sources(planned)
    drop_cols = [c for c in features.LEGACY_DROP_COLS if c in planned.columns]
    if drop_cols:
        planned = planned.drop(columns=drop_cols)
    planned = apply_manual_venue_patches(planned)
    planned = pitcher_cumulative.apply_cumulative_to_master(planned)
    return planned


def apply_manual_venue_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Patch known venue/park-factor data errors in sandbox master.

    This is sandbox-only and exists to keep backtests honest when upstream
    sources mis-attribute a venue for a season.
    """
    out = df.copy()
    if {"game_date", "home_team"} <= set(out.columns):
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        # Athletics 2025: Sutter Health Park (Sacramento).
        mask = (out["game_date"] >= pd.Timestamp("2025-01-01")) & (out["home_team"] == "ATH")
        if mask.any():
            if "venue_id" in out.columns:
                out.loc[mask, "venue_id"] = 2529
            if "venue_name" in out.columns:
                out.loc[mask, "venue_name"] = "Sutter Health Park"
            # Override park_factor to reflect a hitter-friendly environment.
            # Keep this as a single scalar to stay compatible with existing models.
            if "park_factor" in out.columns:
                out.loc[mask, "park_factor"] = 1.17
    return out


def write_manifest(df: pd.DataFrame, output_path: Path, cutoff: str) -> None:
    seasons = sorted(int(s) for s in df["season"].dropna().unique()) if "season" in df.columns else []
    manifest = {
        "output": str(output_path.relative_to(REPO_ROOT)),
        "cutoff": cutoff,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "min_game_date": str(df["game_date"].min().date()) if len(df) else None,
        "max_game_date": str(df["game_date"].max().date()) if len(df) else None,
        "seasons": seasons,
        "contains_2026_or_later": bool((pd.to_datetime(df["game_date"]) >= pd.Timestamp(cutoff)).any()),
        "planned_feature_count": len(planned_features.PLANNED_FEATURE_COLUMNS),
        "selectable_planned_feature_count": len(planned_features.selectable_sandbox_features(df)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    FEATURE_STATUS_PATH.write_text(
        json.dumps(planned_features.planned_feature_status(df), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="sandbox CSV output path")
    parser.add_argument("--cutoff", default=features.DEFAULT_CUTOFF, help="exclusive game_date cutoff")
    args = parser.parse_args()

    df = build_master(cutoff=args.cutoff)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    write_manifest(df, args.output, args.cutoff)

    print(f"wrote {args.output}")
    print(f"rows={len(df):,} columns={len(df.columns):,}")
    print(f"max_game_date={df['game_date'].max().date() if len(df) else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
