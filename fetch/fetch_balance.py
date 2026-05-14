"""
Fetch current Kalshi account balance and insert a row into the user_balance table.
"""

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
    email: str,
    private_key_pem: str | None = None,
) -> int:
    try:
        key_id, private_key = load_credentials(
            key_id=key_id,
            key_path=key_path,
            private_key_pem=private_key_pem,
        )
        base_url = get_base_url(kalshi_env)

        path    = api_path("portfolio/balance")
        headers = auth_headers(key_id, private_key, "GET", path)

        resp = requests.get(base_url + "/portfolio/balance", headers=headers, timeout=10)
        # 4xx (auth, bad key, account issue) — fail closed; do not size with stale balance.
        if 400 <= resp.status_code < 500:
            raise RuntimeError(
                f"Kalshi balance fetch returned {resp.status_code} for {email or 'legacy account'}: "
                f"{resp.text[:200]}. Refusing to fall back to stale balance."
            )
        resp.raise_for_status()

        balance_cents = int(resp.json()["balance"])
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  WARNING: Kalshi balance fetch failed (transient): {e}")
        last = DB.get_last_user_balance_cents(email)
        if last is not None:
            print(f"  Using last known balance: {last} cents")
            return last
        raise RuntimeError(
            "Kalshi balance unavailable and no prior balance in DB. Cannot size bets safely."
        )

    print(f"  Kalshi balance: ${balance_cents / 100:.2f} ({balance_cents} cents)")
    DB.insert_user_balance(email, balance_cents)
    return balance_cents
