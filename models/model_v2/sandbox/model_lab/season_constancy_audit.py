#!/usr/bin/env python3
"""Detect season-final / lookahead leakage via temporal-constancy heuristic.

For each numeric feature, group rows by an "entity" (pitcher_id for sp_*
columns, home_team for team-level columns) and season, then count unique
values per group. Features whose values stay constant within (entity,
season) for groups with many games are likely season-final stats —
i.e. lookahead leakage like the original ``home_sp_bb9`` problem.

Output: sandbox/model_lab/output/season_constancy_report.json plus a
list of suspect feature names. Exit 2 if any "hard" hits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab import features as sf  # noqa: E402
from models.model_v2.sandbox.model_lab.training import data as td  # noqa: E402

DEFAULT_INPUT = LAB_DIR / "output" / "master_sandbox_mlb.csv"
DEFAULT_OUT = LAB_DIR / "output" / "season_constancy_report.json"

MIN_GAMES_PER_GROUP = 10
MIN_GROUPS = 5
HARD_RATIO = 0.80  # if ≥80% of qualifying groups are constant → hard hit
SOFT_RATIO = 0.50


def _entity_for(col: str) -> str | None:
    """Pick grouping entity for a feature.

    sp_*/pitcher columns → pitcher_id of that side.
    home_*/away_* (non-sp) → that team.
    DIFF columns and team-paired columns: skip — they vary by opponent.
    """
    lower = col.lower()
    if "_diff" in lower or lower.endswith("_diff"):
        return None
    if col.startswith("home_sp_") or col.startswith("home_starter_") or col.startswith("h_sp_"):
        return "home_starter_id"
    if col.startswith("away_sp_") or col.startswith("away_starter_") or col.startswith("a_sp_"):
        return "away_starter_id"
    if col.startswith("home_") or col.startswith("h_"):
        return "home_team"
    if col.startswith("away_") or col.startswith("a_"):
        return "away_team"
    return None


def audit_constancy(df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    if "season" not in df.columns:
        df = df.copy()
        df["season"] = pd.to_datetime(df["game_date"], errors="coerce").dt.year

    results: list[dict] = []
    for col in feature_cols:
        entity = _entity_for(col)
        if entity is None or entity not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 200:
            continue
        sub = pd.DataFrame(
            {
                "ent": df[entity],
                "season": df["season"],
                "v": s,
            }
        ).dropna()
        if sub.empty:
            continue
        grp = sub.groupby(["ent", "season"], sort=False)
        sizes = grp.size()
        unique_counts = grp["v"].nunique(dropna=True)
        # Suppress groups where the entity simply doesn't have the trait
        # (constant zero / constant NaN). A constant-zero pitch_mix means
        # the pitcher never throws that pitch — that's signal, not leakage.
        nz_unique = grp["v"].apply(lambda s: s[s != 0].nunique())
        big_mask = sizes >= MIN_GAMES_PER_GROUP
        if big_mask.sum() < MIN_GROUPS:
            continue
        big_unique = unique_counts[big_mask]
        big_nz = nz_unique[big_mask]
        # Constant means: 1 unique value AND that value is non-zero (real
        # season-aggregate signature). A pitcher with cum_BB9 that never
        # changes within a season is a season-final stat.
        constant_mask = (big_unique <= 1) & (big_nz >= 1)
        constant_ratio = float(constant_mask.mean())
        median_unique = float(big_unique.median())
        results.append(
            {
                "feature": col,
                "entity": entity,
                "qualifying_groups": int(big_mask.sum()),
                "constant_ratio": constant_ratio,
                "median_unique_values": median_unique,
            }
        )
    return results


def classify(report: list[dict]) -> tuple[list[str], list[str]]:
    hard = [r["feature"] for r in report if r["constant_ratio"] >= HARD_RATIO]
    soft = [
        r["feature"]
        for r in report
        if SOFT_RATIO <= r["constant_ratio"] < HARD_RATIO
    ]
    return hard, soft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cutoff", default=sf.DEFAULT_CUTOFF)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df = sf.apply_sandbox_contract(df, sf.SandboxContract(cutoff=args.cutoff))
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    feats = td.select_numeric_features(df, min_coverage=args.min_coverage)
    report = audit_constancy(df, feats)
    report.sort(key=lambda r: r["constant_ratio"], reverse=True)
    hard, soft = classify(report)

    out = {
        "input": str(args.input),
        "rows": int(len(df)),
        "feature_count": len(feats),
        "audited_count": len(report),
        "thresholds": {
            "min_games_per_group": MIN_GAMES_PER_GROUP,
            "min_qualifying_groups": MIN_GROUPS,
            "hard_constant_ratio": HARD_RATIO,
            "soft_constant_ratio": SOFT_RATIO,
        },
        "hard_hits": hard,
        "soft_hits": soft,
        "details": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"hard={len(hard)} soft={len(soft)} audited={len(report)}")
    if hard:
        print("hard hits:")
        for h in hard[:20]:
            print(f"  {h}")
    return 2 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
