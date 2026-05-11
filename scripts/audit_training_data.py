#!/usr/bin/env python3
"""Read-only DB audit for model training inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db  # noqa: E402
import models.model_v1.train as T  # noqa: E402


def _records(df: pd.DataFrame, cols: list[str], limit: int) -> list[dict]:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return []
    return df[existing].head(limit).to_dict("records")


def _american_overround(df: pd.DataFrame) -> pd.Series:
    home = df["close_home_ml"].map(T.american_to_raw_implied)
    away = df["close_away_ml"].map(T.american_to_raw_implied)
    return home + away


def build_report(example_limit: int = 10) -> dict:
    raw = db.get_games_df()
    engineered = T.load_and_engineer_features()

    active = [c for c in T.FEATURE_COLUMNS if c in engineered.columns]
    missing_features = [c for c in T.FEATURE_COLUMNS if c not in engineered.columns]
    required = ["home_win", "market_implied_prob", "game_date", "home_games_played", "away_games_played"]
    train = engineered.dropna(subset=[c for c in required if c in engineered.columns])

    settled = raw[raw["home_win"].notna()].copy()
    no_close = settled[settled[["close_home_ml", "close_away_ml"]].isna().any(axis=1)]

    overround = _american_overround(settled)
    bad_overround = settled[overround.notna() & (overround < 0.99)].copy()
    bad_overround["overround"] = overround.loc[bad_overround.index]

    wrc_issues = {}
    for side in ["home", "away"]:
        col = f"{side}_wrc_plus"
        values = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else pd.Series(dtype=float)
        bad_scale = raw[values.notna() & ((values < 10) | (values > 300))].copy()
        wrc_issues[col] = {
            "count": int(len(bad_scale)),
            "min_date": str(pd.to_datetime(bad_scale["game_date"]).min().date()) if len(bad_scale) else None,
            "max_date": str(pd.to_datetime(bad_scale["game_date"]).max().date()) if len(bad_scale) else None,
            "examples": _records(
                bad_scale,
                ["game_pk", "game_date", "away_team", "home_team", col],
                example_limit,
            ),
        }

    null_rates = train[active].isna().mean().sort_values(ascending=False) if active else pd.Series(dtype=float)
    all_null_active = [c for c in active if train[c].isna().all()]

    return {
        "raw_rows": int(len(raw)),
        "settled_rows": int(len(settled)),
        "training_ready_rows": int(len(train)),
        "active_feature_count": int(len(active)),
        "missing_configured_features": missing_features,
        "all_null_active_features": all_null_active,
        "top_null_rates": {k: round(float(v), 6) for k, v in null_rates.head(20).items()},
        "missing_close_odds": {
            "count": int(len(no_close)),
            "examples": _records(
                no_close,
                ["game_pk", "game_date", "away_team", "home_team", "close_home_ml", "close_away_ml", "odds_source"],
                example_limit,
            ),
        },
        "bad_overround_lt_099": {
            "count": int(len(bad_overround)),
            "examples": _records(
                bad_overround,
                ["game_pk", "game_date", "away_team", "home_team", "close_home_ml", "close_away_ml", "odds_source", "overround"],
                example_limit,
            ),
        },
        "wrc_plus_scale_issues": wrc_issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--fail", action="store_true", help="exit non-zero if critical issues exist")
    parser.add_argument("--examples", type=int, default=10, help="example row limit")
    args = parser.parse_args()

    report = build_report(example_limit=args.examples)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"raw_rows={report['raw_rows']:,}")
        print(f"settled_rows={report['settled_rows']:,}")
        print(f"training_ready_rows={report['training_ready_rows']:,}")
        print(f"active_feature_count={report['active_feature_count']}")
        print(f"missing_configured_features={report['missing_configured_features']}")
        print(f"all_null_active_features={report['all_null_active_features']}")
        print(f"top_null_rates={report['top_null_rates']}")
        print(f"missing_close_odds={report['missing_close_odds']['count']}")
        print(f"bad_overround_lt_099={report['bad_overround_lt_099']['count']}")
        for col, issue in report["wrc_plus_scale_issues"].items():
            print(f"{col}_scale_issues={issue['count']} range={issue['min_date']}..{issue['max_date']}")

    critical = bool(report["all_null_active_features"]) or any(
        issue["count"] for issue in report["wrc_plus_scale_issues"].values()
    )
    return 1 if args.fail and critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
