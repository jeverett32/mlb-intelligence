"""
Fetch current Kalshi account balance and append a row to data/balance.csv.
Run from project root: uv run fetch/fetch_balance.py
"""

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# Allow importing kalshi_client from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials

BALANCE_CSV = Path("data/balance.csv")


FALLBACK_BALANCE_CENTS = 10000  # $100 placeholder when API is unreachable


def fetch_balance() -> int:
    """Fetch balance from Kalshi and append to balance.csv. Returns balance in cents."""
    try:
        key_id, private_key = load_credentials()
        base_url = get_base_url()

        path    = api_path("portfolio/balance")
        headers = auth_headers(key_id, private_key, "GET", path)

        resp = requests.get(base_url + "/portfolio/balance", headers=headers, timeout=10)
        resp.raise_for_status()

        data          = resp.json()
        balance_cents = int(data["balance"])
    except Exception as e:
        print(f"  WARNING: Kalshi balance fetch failed: {e}")
        if BALANCE_CSV.exists() and BALANCE_CSV.stat().st_size > 0:
            import pandas as pd
            df = pd.read_csv(BALANCE_CSV)
            if not df.empty:
                balance_cents = int(df.iloc[-1]["balance_cents"])
                print(f"  Using last known balance: {balance_cents} cents")
                return balance_cents
        balance_cents = FALLBACK_BALANCE_CENTS
        print(f"  Using fallback balance: ${balance_cents / 100:.2f}")

    balance_dollars = balance_cents / 100.0
    timestamp       = datetime.now(timezone.utc).isoformat()

    print(f"  Kalshi balance: ${balance_dollars:.2f} ({balance_cents} cents)")

    write_header = not BALANCE_CSV.exists() or BALANCE_CSV.stat().st_size == 0
    with open(BALANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "balance_cents", "balance_dollars"])
        writer.writerow([timestamp, balance_cents, f"{balance_dollars:.2f}"])

    return balance_cents


if __name__ == "__main__":
    fetch_balance()
