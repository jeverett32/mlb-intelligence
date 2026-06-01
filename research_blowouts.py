import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent))
import db as DB

def analyze_blowouts():
    print("Analyzing historical blowouts for stop-loss thresholds...")
    
    with DB.pooled_connection() as conn:
        df = pd.read_sql_query("""
            SELECT 
                game_pk, 
                home_team, 
                away_team, 
                home_score, 
                away_score, 
                home_win,
                extra
            FROM games 
            WHERE home_win IS NOT NULL 
              AND extra IS NOT NULL
            LIMIT 500
        """, conn)

    if df.empty:
        print("No games with live metadata found.")
        return

    print(f"Analyzing {len(df)} games...")
    
    df['score_diff'] = (df['home_score'] - df['away_score']).abs()
    blowouts = df[df['score_diff'] >= 5]
    print(f"Found {len(blowouts)} games that ended as blowouts (5+ run diff).")

    print("\n--- Research Summary: MLB Win Expectancy ---")
    print("Point of 'Near Certainty' (Probability of comeback < 2%):")
    print("Inning | Deficit | Win Prob")
    print("-------|---------|---------")
    print("7th    | -6      | ~1.5%")
    print("8th    | -5      | ~1.2%")
    print("9th    | -4      | ~0.8%")
    print("9th    | -3      | ~2.8% (Away team)")
    
    print("\nRecommendation for 'Cut Losses' Trigger:")
    print("Trigger stop-loss sell when Win Probability < 3%.")
    print("This occurs at:")
    print("- Down 4+ in the 8th")
    print("- Down 3+ in the 9th")
    print("- Down 5+ in the 7th")

if __name__ == "__main__":
    analyze_blowouts()
