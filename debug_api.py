import db
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "dashboard"))

def test_get_performance_v2():
    print("\nTesting get_performance(version='v2')...")
    from dashboard import app as APP
    # Mocking request isn't easy here, let's just call the internal builder
    perf = APP._build_public_performance(email=None, mode='paper', model='v2')
    print(f"Performance: {perf.model_dump()}")

def test_get_all_bets_v2():
    print("Testing get_all_bets(version='v2')...")
    df = db.get_all_bets(version='v2')
    print(f"Result count: {len(df)}")
    if not df.empty:
        print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    try:
        test_get_all_bets_v2()
        test_get_performance_v2()
        print("\nTests passed.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
