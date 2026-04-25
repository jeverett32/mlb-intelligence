"""
Cancel today's bad bets on Kalshi and remove them from the DB.

Usage:
    uv run python scripts/cancel_bad_bets.py [--dry-run]

Finds all user_orders placed today (result IS NULL, not dry_run),
attempts to cancel each on Kalshi, then deletes the rows from user_orders.
"""

import argparse
import sys
from datetime import date, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials
import db as DB


def get_todays_live_orders():
    conn = DB.get_connection()
    try:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, email, game_pk, game_date, home_team, away_team,
                       kalshi_order_id, bet_dollars, n_contracts, bet_side
                FROM user_orders
                WHERE game_date = %s
                  AND dry_run = FALSE
                  AND result IS NULL
                ORDER BY id
            """, (date.today(),))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def cancel_kalshi_order(order_id: str, key_id: str, private_key, base_url: str) -> bool:
    path = api_path(f"portfolio/orders/{order_id}/cancel")
    headers = auth_headers(key_id, private_key, "DELETE", path)
    resp = requests.delete(base_url + f"/portfolio/orders/{order_id}/cancel",
                           headers=headers, timeout=10)
    if resp.status_code in (200, 204):
        return True
    # 400 = already filled/cancelled — treat as ok to delete from DB
    if resp.status_code == 400:
        print(f"  Kalshi 400 for {order_id}: {resp.text} (may already be settled/cancelled)")
        return True
    print(f"  Kalshi error {resp.status_code} for {order_id}: {resp.text}")
    return False


def delete_user_order(order_id_pk: int):
    conn = DB.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_orders WHERE id = %s", (order_id_pk,))
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without making changes")
    args = parser.parse_args()

    orders = get_todays_live_orders()
    if not orders:
        print("No live orders found for today. Nothing to do.")
        return

    print(f"Found {len(orders)} order(s) to cancel:\n")
    for o in orders:
        print(f"  id={o['id']}  game={o['away_team']} @ {o['home_team']}  "
              f"date={o['game_date']}  side={o['bet_side']}  "
              f"${o['bet_dollars']}  kalshi_id={o['kalshi_order_id']}")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        return

    confirm = input("\nCancel these on Kalshi and delete from DB? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    key_id, private_key = load_credentials()
    base_url = get_base_url()

    for o in orders:
        kalshi_id = o["kalshi_order_id"]
        print(f"\nOrder id={o['id']}  kalshi_order_id={kalshi_id}")

        kalshi_ok = True
        if kalshi_id:
            print(f"  Cancelling on Kalshi...")
            kalshi_ok = cancel_kalshi_order(kalshi_id, key_id, private_key, base_url)
        else:
            print("  No kalshi_order_id — skipping Kalshi cancel.")

        if kalshi_ok:
            delete_user_order(o["id"])
            print(f"  Deleted from DB.")
        else:
            print(f"  Kalshi cancel failed — NOT deleting from DB. Check manually.")

    print("\nDone.")


if __name__ == "__main__":
    main()
