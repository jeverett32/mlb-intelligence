"""Clean odds cache by removing implausible moneylines."""

import pandas as pd

df = pd.read_csv("data/cache/odds_2026.csv")
print(f"Total rows before: {len(df)}")

# Find bad odds
bad_mask = (df["close_home_ml"].abs() > 500) | (df["close_away_ml"].abs() > 500)
bad_df = df[bad_mask]
print(f"\nRows with |odds| > 500: {len(bad_df)}")
if len(bad_df) > 0:
    print(
        bad_df[
            ["game_date", "home_team", "away_team", "close_home_ml", "close_away_ml"]
        ]
    )

# Remove bad odds
clean_df = df[~bad_mask]
print(f"\nTotal rows after cleaning: {len(clean_df)}")

# Save cleaned cache
clean_df.to_csv("data/cache/odds_2026.csv", index=False)
print("Cleaned odds cache saved.")
