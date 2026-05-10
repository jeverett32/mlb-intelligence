import os
import sys
import argparse
import subprocess
import pandas as pd
from datetime import datetime, timedelta

# Absolute imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import db as DB

def run_cmd(cmd):
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def ingest_features(season, start_date=None, end_date=None):
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        # Default to a few days ago to capture recent games if season is current
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
    print(f"Refreshing features for {season} ({start_date} to {end_date})...")
    
    py = sys.executable

    # 1. Refresh sandbox parquet caches
    run_cmd([
        py, "sandbox/model_lab/leaderboard_sources.py", "fetch",
        "--start-season", str(season), "--end-season", str(season)
    ])

    run_cmd([
        py, "sandbox/model_lab/savant_sources.py", "fetch",
        "--start", start_date, "--end", end_date
    ])

    run_cmd([
        py, "sandbox/model_lab/real_sources.py", "fetch-mlb"
    ])
    run_cmd([
        py, "sandbox/model_lab/real_sources.py", "fetch-weather"
    ])
    
    # 2. Rebuild the master frame (in memory)
    # We use build_master from sandbox.model_lab.build_master
    sys.path.insert(0, os.path.join(ROOT, "sandbox/model_lab"))
    from sandbox.model_lab.build_master import build_master
    
    # Use tomorrow as cutoff to include today's games if they are in prod db
    cutoff = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    df = build_master(cutoff=cutoff)
    
    # Filter for requested window
    df['game_date'] = pd.to_datetime(df['game_date'])
    mask = (df['game_date'] >= pd.Timestamp(start_date)) & (df['game_date'] <= pd.Timestamp(end_date))
    target_df = df.loc[mask].copy()
    
    if target_df.empty:
        print("No games found in the requested window.")
        return
        
    print(f"Upserting {len(target_df)} rows into games_v2...")
    
    # Re-identify feature columns
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
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and ingest V2 features for a window.")
    parser.add_argument("--season", type=int, default=2026, help="Season to fetch.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()
    
    ingest_features(season=args.season, start_date=args.start, end_date=args.end)
