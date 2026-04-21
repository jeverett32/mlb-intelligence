"""
Fetch current Kalshi account balance and insert a row into the balance table.
Run from project root: uv run fetch/fetch_balance.py
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials
import db as DB

def fetch_balance() -> int:
    """Fetch balance from Kalshi, persist to DB, and return balance in cents."""
    try:
        key_id, private_key = load_credentials()
        base_url = get_base_url()

        path    = api_path("portfolio/balance")
        headers = auth_headers(key_id, private_key, "GET", path)

        resp = requests.get(base_url + "/portfolio/balance", headers=headers, timeout=10)
        resp.raise_for_status()

        balance_cents = int(resp.json()["balance"])
    except Exception as e:
        print(f"  WARNING: Kalshi balance fetch failed: {e}")
        last = DB.get_last_balance_cents()
        if last is not None:
            print(f"  Using last known balance: {last} cents")
            return last
        sys.exit("ERROR: Kalshi balance unavailable and no prior balance in DB. Cannot size bets safely.")

    print(f"  Kalshi balance: ${balance_cents / 100:.2f} ({balance_cents} cents)")
    DB.insert_balance(balance_cents)
    return balance_cents


if __name__ == "__main__":
    fetch_balance()
