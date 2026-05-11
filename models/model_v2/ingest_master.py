import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm

# Absolute imports to reach db and config
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

import db as DB

METADATA_COLS = [
    "game_pk", "game_date", "season", "home_team", "away_team", 
    "home_win", "close_home_ml", "close_away_ml", "market_implied_prob"
]

def ingest_master(limit=None):
    csv_path = os.path.join(_V2_DIR, "sandbox", "model_lab", "output", "master_sandbox_mlb.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    # Use low_memory=False to avoid DtypeWarning
    df = pd.read_csv(csv_path, low_memory=False)
    
    if limit:
        df = df.head(limit)
        
    print(f"Total rows to ingest: {len(df)}")
    
    # Identify feature columns
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in METADATA_COLS]
    
    # Prepare rows for bulk upsert
    rows = []
    batch_size = 5000
    
    # Initialize DB tables
    DB.init_games_v2()
    
    for i, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing rows")):
        # Extract metadata
        meta = {}
        for col in METADATA_COLS:
            val = r.get(col)
            if pd.isna(val):
                meta[col] = None
            else:
                meta[col] = val
                
        # Extract features
        features = {}
        for col in feature_cols:
            val = r.get(col)
            if not pd.isna(val):
                features[col] = val
        
        row = meta.copy()
        row['features'] = features
        rows.append(row)
        
        if len(rows) >= batch_size:
            DB.bulk_upsert_games_v2(rows)
            rows = []
            
    if rows:
        DB.bulk_upsert_games_v2(rows)
        
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest master sandbox CSV into games_v2 table.")
    parser.add_argument("--limit", type=int, help="Limit number of rows to ingest.")
    args = parser.parse_args()
    
    ingest_master(limit=args.limit)
