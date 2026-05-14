"""
Migrate an existing Kalshi PEM file into encrypted Postgres storage.

Run on the app LXC, where the legacy PEM file exists:

    uv run python scripts/migrate_kalshi_pem_to_db.py --email user@example.com

The script does not print the PEM. It verifies the key by fetching the Kalshi
balance before updating the account row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db as DB
from fetch.fetch_balance import fetch_balance_for_account


def _db_secret_ref_for_email(email: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in email.lower())
    return f"db://kalshi_accounts/{safe}/private_key_pem"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move a Kalshi PEM file into encrypted DB storage.")
    parser.add_argument("--email", required=True, help="Account email to migrate.")
    parser.add_argument(
        "--pem-path",
        default=None,
        help="Override PEM path. Defaults to the account's current key_path.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    email = args.email.strip().lower()

    DB.init_auth_tables()
    account = DB.get_kalshi_account(email)
    if not account:
        raise SystemExit(f"No Kalshi account found for {email}.")

    pem_path = Path(args.pem_path or account.get("key_path") or "").expanduser()
    if not pem_path.exists() or not pem_path.is_file():
        raise SystemExit(f"PEM file not found: {pem_path}")

    pem_text = pem_path.read_text(encoding="utf-8").strip()
    if "BEGIN" not in pem_text or "PRIVATE KEY" not in pem_text:
        raise SystemExit(f"File does not look like a PEM private key: {pem_path}")

    key_ref = _db_secret_ref_for_email(email)
    balance_cents = fetch_balance_for_account(
        key_id=account["key_id"],
        key_path=key_ref,
        private_key_pem=pem_text,
        kalshi_env=account["kalshi_env"],
        email=email,
    )
    DB.upsert_kalshi_account(
        email,
        label=account.get("label") or "Primary account",
        key_id=account["key_id"],
        key_path=key_ref,
        private_key_pem=pem_text,
        kalshi_env=account["kalshi_env"],
        is_active=bool(account.get("is_active", True)),
        last_verified=True,
        last_error="",
    )
    print(f"Migrated {email} to encrypted DB PEM storage. Balance: ${balance_cents / 100:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
