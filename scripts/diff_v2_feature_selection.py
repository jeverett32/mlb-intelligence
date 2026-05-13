#!/usr/bin/env python3
"""Compare V2 top-K feature selection: sandbox CSV+engineered vs production games_v2.

Prints columns present in CSV selection but not DB, and vice versa, plus pool sizes.
Requires DB env for the DB leg (--skip-db to only run CSV path).

Usage (repo root):
  uv run python scripts/diff_v2_feature_selection.py
  uv run python scripts/diff_v2_feature_selection.py --skip-db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2 import config as C  # noqa: E402
from models.model_v2.sandbox.model_lab import features as sf  # noqa: E402


def _feature_sets_for_df(df, *, cache_subdir: str, use_cache: bool) -> dict:
    from models.model_v2.sandbox.model_lab.feature_engineer import load_or_build_feature_sets

    cache_dir = REPO_ROOT / "models" / "model_v2" / "sandbox" / "model_lab" / "output" / "cache" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return load_or_build_feature_sets(
        df,
        min_coverage=C.MIN_COVERAGE,
        corr_threshold=C.CORR_THRESHOLD,
        top_k=C.K_FEATURES,
        cache_dir=cache_dir,
        use_cache=use_cache,
        selected_by=C.SELECTED_BY,
    )


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-csv",
        type=Path,
        default=REPO_ROOT / "models" / "model_v2" / "sandbox" / "model_lab" / "output" / "master_sandbox_mlb.csv",
    )
    parser.add_argument("--skip-db", action="store_true", help="Do not load games_v2 (CSV-only).")
    parser.add_argument("--no-cache", action="store_true", help="Recompute feature sets (ignore pickle/json cache).")
    args = parser.parse_args()

    use_cache = not args.no_cache

    if not args.master_csv.exists():
        print(f"Missing master CSV: {args.master_csv}", file=sys.stderr)
        print("Build with: uv run python models/model_v2/sandbox/model_lab/build_master.py", file=sys.stderr)
        return 2

    from models.model_v2.sandbox.model_lab.feature_engineer import load_or_build_engineered_frame

    print("[csv] load_or_build_engineered_frame …")
    df_csv = load_or_build_engineered_frame(
        input_path=args.master_csv,
        cutoff=sf.DEFAULT_CUTOFF,
        cache_dir=REPO_ROOT / "models" / "model_v2" / "sandbox" / "model_lab" / "output" / "cache" / "csv_engineered",
        use_cache=use_cache,
    )
    fs_csv = _feature_sets_for_df(df_csv, cache_subdir="diff_csv_fs", use_cache=use_cache)
    sel_csv = set(fs_csv["selected"])
    print(
        f"[csv] rows={len(df_csv):,} cols={df_csv.shape[1]:,} "
        f"cand={len(fs_csv['cand'])} pruned={len(fs_csv['pruned'])} selected={len(sel_csv)}"
    )

    if args.skip_db:
        print("[db] skipped")
        return 0

    if not os.environ.get("DB_HOST") and not os.environ.get("DATABASE_URL"):
        print("[db] No DB_HOST / DATABASE_URL; use --skip-db or configure .env", file=sys.stderr)
        return 3

    from models.model_v2.feature_loader import load_or_build_engineered_frame_from_db

    print("[db] load_or_build_engineered_frame_from_db …")
    df_db = load_or_build_engineered_frame_from_db(
        cutoff=None,
        cache_dir=str(REPO_ROOT / "models" / "model_v2" / "sandbox" / "model_lab" / "output" / "cache" / "diff_db_frame"),
        use_cache=use_cache,
    )
    if df_db.empty:
        print("[db] empty frame", file=sys.stderr)
        return 4

    fs_db = _feature_sets_for_df(df_db, cache_subdir="diff_db_fs", use_cache=use_cache)
    sel_db = set(fs_db["selected"])
    print(
        f"[db]  rows={len(df_db):,} cols={df_db.shape[1]:,} "
        f"cand={len(fs_db['cand'])} pruned={len(fs_db['pruned'])} selected={len(sel_db)}"
    )

    only_csv = sorted(sel_csv - sel_db)
    only_db = sorted(sel_db - sel_csv)
    both = sorted(sel_csv & sel_db)

    print()
    print(f"Intersection (both): {len(both)}")
    print(f"In CSV selected only (missing from production selection): {len(only_csv)}")
    for c in only_csv:
        print(f"  + {c}")
    print()
    print(f"In DB selected only (not chosen on CSV path): {len(only_db)}")
    for c in only_db:
        print(f"  - {c}")

    eng_only = {
        "market_logit",
        "market_open_logit",
        "market_move_logit",
        "park_x_wrc_DIFF",
        "park_x_woba_DIFF",
        "sp_x_bp_quality",
        "sp_x_bp_freshness",
        "bp_quality_x_freshness",
        "matchup_x_park",
        "matchup_x_sp",
        "weather_x_park",
        "weather_x_lineup_power",
        "travel_fatigue_idx",
        "tz_x_getaway",
        "pyth_x_recent_form",
        "luck_regress",
        "rolling_staff_composite",
        "high_lev_bp_DIFF",
        "sp_csw_DIFF",
        "framing_x_csw",
        "recent_form_DIFF_15",
        "recent_run_diff_DIFF_15",
    }
    eng_missing = [c for c in only_csv if c in eng_only]
    if eng_missing:
        print()
        print(
            "Note: some CSV-only names are in-memory composites from "
            "feature_engineer.engineer_combined_features (not stored in games_v2)."
        )
        print(f"  ({len(eng_missing)} of CSV-only are known composites: {', '.join(eng_missing[:10])}…)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
