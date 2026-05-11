import argparse
import sys
import os
from datetime import datetime

# Absolute imports to reach sandbox and main repo
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from models.model_v2 import predict as P

def main():
    parser = argparse.ArgumentParser(description="V2 Model Standalone CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--game_pk", type=str, help="Predict for a specific game PK")
    group.add_argument("--game_date", type=str, help="Predict for all games on a date (ISO format)")
    
    args = parser.parse_args()
    
    target_date = args.game_date
    pks = []
    
    # We use the DB to resolve PK/Date mapping
    import db as DB
    try:
        if args.game_pk:
            with DB.pooled_connection() as conn:
                import pandas as pd
                df = pd.read_sql_query(
                    "SELECT game_date FROM games WHERE game_pk = %s", 
                    conn, 
                    params=(int(args.game_pk),)
                )
                if not df.empty:
                    target_date = str(df.iloc[0]["game_date"])[:10]
                else:
                    target_date = datetime.now().strftime("%Y-%m-%d")
            pks = [args.game_pk]
        else:
            with DB.pooled_connection() as conn:
                import pandas as pd
                df = pd.read_sql_query(
                    "SELECT game_pk FROM games WHERE game_date::text LIKE %s", 
                    conn, 
                    params=(f"{args.game_date}%",)
                )
                pks = [str(pk) for pk in df["game_pk"].tolist()]
    except Exception as e:
        print(f"Database error resolving PK/Date: {e}")
        if args.game_pk and not target_date:
            target_date = datetime.now().strftime("%Y-%m-%d")
            pks = [args.game_pk]
        elif not pks:
            return

    if not pks:
        print(f"No games found for input.")
        return

    print(f"Preparing shared V2 model for {target_date}...")
    try:
        shared = P.prepare_shared(pks, target_date)
    except Exception as e:
        print(f"Error preparing model: {e}")
        return
    
    print(f"\nResults (v2-lgbm-k306):")
    print(f"{'Game PK':<12} | {'Prob':<8} | {'Mkt':<8} | {'Edge':<8} | {'Side':<6} | {'Stake':<8}")
    print("-" * 75)
    
    for pk in pks:
        try:
            res = P.predict_one(pk, shared, dry_run=True)
            print(f"{res['game_pk']:<12} | {res['prob']:<8.4f} | {res['market_implied_prob']:<8.4f} | "
                  f"{res['edge']:<8.4f} | {res['bet_side']:<6} | {res['bet_frac']:<8.4f}")
        except Exception as e:
            print(f"{pk:<12} | Error: {e}")

if __name__ == "__main__":
    main()
