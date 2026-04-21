"""
Fetch current Kalshi account balance and insert a row into the balance table.
Run from project root: uv run fetch/fetch_balance.py
"""

import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials
import db as DB


def fetch_balance_for_account(
    *,
    key_id: str,
    key_path: str,
    kalshi_env: str,
    email: str | None = None,
) -> int:
    try:
        key_id, private_key = load_credentials(key_id=key_id, key_path=key_path)
        base_url = get_base_url(kalshi_env)

        path    = api_path("portfolio/balance")
        headers = auth_headers(key_id, private_key, "GET", path)

        resp = requests.get(base_url + "/portfolio/balance", headers=headers, timeout=10)
        resp.raise_for_status()

        balance_cents = int(resp.json()["balance"])
    except Exception as e:
        print(f"  WARNING: Kalshi balance fetch failed: {e}")
        last = DB.get_last_user_balance_cents(email) if email else DB.get_last_balance_cents()
        if last is not None:
            print(f"  Using last known balance: {last} cents")
            return last
        raise RuntimeError(
            "Kalshi balance unavailable and no prior balance in DB. Cannot size bets safely."
        )

    print(f"  Kalshi balance: ${balance_cents / 100:.2f} ({balance_cents} cents)")
    if email:
        DB.insert_user_balance(email, balance_cents)
    else:
        DB.insert_balance(balance_cents)
    return balance_cents


def fetch_balance() -> int:
    """Fetch balance from the legacy env-configured Kalshi account."""
    return fetch_balance_for_account(
        key_id=os.environ.get("KALSHI_KEY_ID", ""),
        key_path=os.environ.get("KALSHI_KEY_PATH", "kalshi-key.pem"),
        kalshi_env=os.environ.get("KALSHI_ENV", "prod"),
        email=None,
    )


if __name__ == "__main__":
    fetch_balance()
