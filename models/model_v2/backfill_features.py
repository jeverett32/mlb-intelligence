
import os
import sys
import argparse
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

# Absolute imports
_V2_DIR = Path(__file__).resolve().parent
_LAB_DIR = _V2_DIR / "sandbox" / "model_lab"
_REPO_ROOT = _V2_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import db as DB

def run_cmd(cmd):
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def backfill_features(start_year=2022, end_year=2026):
    py = sys.executable
    
    print(f"Starting feature backfill from {start_year} to {end_year}...")
    
    # 1. Fetch missing data for 2026 (and any others requested)
    for year in range(start_year, end_year + 1):
        print(f"\n--- Fetching sources for {year} ---")
        run_cmd([
            py, str(_LAB_DIR / "leaderboard_sources.py"), "fetch",
            "--start-season", str(year), "--end-season", str(year)
        ])
        
        start_date = f"{year}-03-01"
        end_date = f"{year}-11-30"
        run_cmd([
            py, str(_LAB_DIR / "savant_sources.py"), "fetch",
            "--start", start_date, "--end", end_date
        ])

    print("\n--- Refreshing weather and MLB raw data ---")
    run_cmd([py, str(_LAB_DIR / "real_sources.py"), "fetch-mlb"])
    run_cmd([py, str(_LAB_DIR / "real_sources.py"), "fetch-weather"])

    # Ordering matters:
    #   build_savant_features and build_leaderboard_features both read the master
    #   CSV on disk and need accurate venue_id / catcher_id / starter_id / lineup
    #   fields (from mlb_statsapi_features.parquet) for the current season. So we
    #   build the master CSV once first with the latest statsapi parquet, then
    #   rebuild savant + leaderboard parquets against it, then rebuild master a
    #   final time pulling in the freshly built savant/leaderboard features.
    from models.model_v2.sandbox.model_lab import build_master as BM

    print("\n--- Pre-build master CSV (statsapi-driven IDs) ---")
    pre_master = BM.build_master(cutoff="2027-01-01", allow_csv_fallback=False)
    BM.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pre_master.to_csv(BM.OUTPUT_PATH, index=False)
    print(f"  wrote {BM.OUTPUT_PATH} rows={len(pre_master):,} cols={len(pre_master.columns):,}")

    print("\n--- Building source feature parquets ---")
    run_cmd([py, str(_LAB_DIR / "savant_sources.py"), "build"])
    run_cmd([py, str(_LAB_DIR / "leaderboard_sources.py"), "build"])

    print("\n--- Rebuilding master feature frame ---")
    df = BM.build_master(cutoff="2027-01-01", allow_csv_fallback=False)
    df.to_csv(BM.OUTPUT_PATH, index=False)
    BM.write_manifest(df, BM.OUTPUT_PATH, "2027-01-01")
    
    # 3. Filter for requested years
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    target_df = df[(df['season'] >= start_year) & (df['season'] <= end_year)].copy()
    
    if target_df.empty:
        print("No games found to backfill.")
        return
        
    print(f"Upserting {len(target_df)} rows into games_v2...")
    
    METADATA_COLS = [
        "game_pk", "game_date", "season", "home_team", "away_team", 
        "home_win", "close_home_ml", "close_away_ml", "market_implied_prob"
    ]
    all_cols = target_df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in METADATA_COLS]
    
    rows = []
    for _, r in target_df.iterrows():
        meta = {}
        for col in METADATA_COLS:
            val = r.get(col)
            if col == 'game_date' and isinstance(val, pd.Timestamp):
                meta[col] = val.strftime("%Y-%m-%d")
            elif pd.isna(val):
                meta[col] = None
            else:
                meta[col] = val
                
        features = {}
        for col in feature_cols:
            val = r.get(col)
            if not pd.isna(val):
                features[col] = val
        
        row = meta.copy()
        row['features'] = features
        rows.append(row)
        
    DB.bulk_upsert_games_v2(rows)
    print("Backfill complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill V2 features for multiple seasons.")
    parser.add_argument("--start", type=int, default=2022, help="Start year.")
    parser.add_argument("--end", type=int, default=2026, help="End year.")
    args = parser.parse_args()
    
    backfill_features(start_year=args.start, end_year=args.end)
