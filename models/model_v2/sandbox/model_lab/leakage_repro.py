import pandas as pd
import numpy as np
from pathlib import Path

MASTER_PATH = Path(__file__).resolve().parent / "output" / "master_sandbox_mlb.csv"

def audit_leakage():
    if not MASTER_PATH.exists():
        print(f"File not found: {MASTER_PATH}")
        return

    df = pd.read_csv(MASTER_PATH, low_memory=False)
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    selected = [
        "sp_era_DIFF", "rolling_k9_DIFF", "home_sp_bb9", "war_DIFF", 
        "away_bp_era_30d", "home_lineup_avg_k_rate"
    ]
    
    print(f"Total rows: {len(df)}")
    
    for f in selected:
        if f not in df.columns:
            print(f"Feature {f} not found.")
            continue
            
        print(f"\n--- Auditing {f} ---")
        # Check a few teams/pitchers to see if the value changes
        if "pitcher" in f or "sp_" in f:
            group_col = "home_starter_id"
        elif "lineup" in f or "bp_" in f or "war" in f:
            group_col = "home_team"
        else:
            continue
            
        entities = df[group_col].dropna().unique()[:3]
        for ent in entities:
            e_df = df[df[group_col] == ent].sort_values("game_date")
            for season, s_df in e_df.groupby("season"):
                if len(s_df) > 10:
                    vals = s_df[f].unique()
                    if len(vals) == 1:
                        print(f"  FAILED: {group_col} {ent}, Season {season}: Constant {f} = {vals[0]}")
                    else:
                        print(f"  PASSED: {group_col} {ent}, Season {season}: {len(vals)} unique values")

if __name__ == "__main__":
    audit_leakage()
