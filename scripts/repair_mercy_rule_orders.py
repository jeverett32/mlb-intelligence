#!/usr/bin/env python3
"""
Repair historical mercy-rule (sold_stop_loss) user_orders rows.

Fixes:
  - Clears game `result` (mercy exits are not game settlements)
  - Zeros `n_contracts` after a full exit
  - Recomputes `profit_loss_cents` from Kalshi buy/sell fills when available
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import RealDictCursor

import db as DB
from bet.place_bet import _order_fill_cost, _order_fill_count, _order_fill_revenue
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials


def _retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _fetch_orders_for_ticker(
    session: requests.Session,
    *,
    base_url: str,
    key_id: str,
    private_key,
    ticker: str,
) -> list[dict]:
    path = api_path("portfolio/orders")
    headers = auth_headers(key_id, private_key, "GET", path)
    resp = session.get(
        base_url + "/portfolio/orders",
        headers=headers,
        params={"ticker": ticker, "limit": 100},
        timeout=20,
    )
    resp.raise_for_status()
    orders = resp.json().get("orders") or []
    return orders if isinstance(orders, list) else [orders]


def _pnl_from_kalshi_orders(orders: list[dict]) -> tuple[int | None, str]:
    buy_total = 0.0
    sell_total = 0.0
    buy_fills = 0.0
    sell_fills = 0.0
    for order in orders:
        action = str(order.get("action") or "").lower()
        fills = float(_order_fill_count(order) or 0.0)
        if fills <= 0:
            continue
        if action == "buy":
            dollars = round(float(_order_fill_cost(order) or 0.0), 2)
            if dollars <= 0:
                continue
            buy_total += dollars
            buy_fills += fills
        elif action == "sell":
            dollars = round(float(_order_fill_revenue(order) or 0.0), 2)
            if dollars <= 0:
                continue
            sell_total += dollars
            sell_fills += fills

    if sell_fills <= 0 or buy_total <= 0:
        return None, "missing sell or buy fill on Kalshi"

    pnl_dollars = round(sell_total - buy_total, 2)
    return int(round(pnl_dollars * 100)), "kalshi_fills"


def _rows_needing_repair() -> list[dict]:
    conn = DB.get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT email, game_pk, game_date, kalshi_ticker, bet_cents,
                       profit_loss_cents, n_contracts, result, status, updated_at
                FROM user_orders
                WHERE status = %s
                  AND dry_run = FALSE
                ORDER BY game_date, game_pk, email
                """,
                (DB.SOLD_STOP_LOSS_STATUS,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _apply_repair(
    email: str,
    game_pk: int,
    *,
    profit_loss_cents: int | None,
    dry_run: bool,
) -> bool:
    if dry_run:
        return True
    conn = DB.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_orders
                SET result = NULL,
                    n_contracts = 0,
                    profit_loss_cents = COALESCE(%s, profit_loss_cents),
                    updated_at = NOW()
                WHERE email = %s
                  AND game_pk = %s
                  AND status = %s
                """,
                (
                    profit_loss_cents,
                    email,
                    int(game_pk),
                    DB.SOLD_STOP_LOSS_STATUS,
                ),
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        conn.close()


def complete_stale_pending_payout_jobs(*, dry_run: bool = True) -> int:
    if dry_run:
        conn = DB.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM kalshi_balance_refresh_jobs j
                    JOIN games g ON g.game_pk = j.game_pk
                    WHERE j.status <> 'completed'
                      AND g.home_win IS NOT NULL
                      AND g.game_date < CURRENT_DATE
                    """
                )
                return int(cur.fetchone()[0])
        finally:
            conn.close()
    count = DB.complete_stale_kalshi_balance_refresh_jobs()
    if count:
        print(f"Completed {count} stale balance refresh job(s)")
    return count


def repair_mercy_rule_orders(*, dry_run: bool = True) -> dict:
    rows = _rows_needing_repair()
    session = _retry_session()
    account_cache: dict[str, dict | None] = {}
    cred_cache: dict[str, tuple[str, object, str]] = {}
    stats = {
        "checked": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "changes": [],
    }

    for row in rows:
        stats["checked"] += 1
        email = row["email"]
        game_pk = int(row["game_pk"])
        ticker = str(row.get("kalshi_ticker") or "").strip()
        old_pnl = row.get("profit_loss_cents")
        old_result = row.get("result")
        old_contracts = row.get("n_contracts")

        new_pnl = old_pnl
        source = "unchanged"
        err = ""

        if ticker:
            try:
                if email not in account_cache:
                    account_cache[email] = DB.get_kalshi_account(email)
                account = account_cache[email]
                if not account or not account.get("is_active"):
                    raise RuntimeError("Kalshi account inactive or missing")
                if email not in cred_cache:
                    key_id, private_key = load_credentials(
                        key_id=account["key_id"],
                        key_path=account["key_path"],
                        private_key_pem=account.get("private_key_pem") or None,
                    )
                    cred_cache[email] = (
                        key_id,
                        private_key,
                        get_base_url(account.get("kalshi_env")),
                    )
                key_id, private_key, base_url = cred_cache[email]
                orders = _fetch_orders_for_ticker(
                    session,
                    base_url=base_url,
                    key_id=key_id,
                    private_key=private_key,
                    ticker=ticker,
                )
                computed, source = _pnl_from_kalshi_orders(orders)
                if computed is not None:
                    new_pnl = computed
            except Exception as exc:
                err = str(exc)
                stats["errors"] += 1

        needs_update = (
            old_result is not None
            or int(old_contracts or 0) != 0
            or (new_pnl is not None and new_pnl != old_pnl)
        )
        if not needs_update:
            stats["skipped"] += 1
            continue

        change = {
            "email": email,
            "game_pk": game_pk,
            "ticker": ticker,
            "old_profit_loss_cents": old_pnl,
            "new_profit_loss_cents": new_pnl,
            "old_result": old_result,
            "old_n_contracts": old_contracts,
            "source": source,
            "error": err or None,
        }
        stats["changes"].append(change)

        if _apply_repair(email, game_pk, profit_loss_cents=new_pnl, dry_run=dry_run):
            stats["updated"] += 1
            action = "WOULD UPDATE" if dry_run else "UPDATED"
            print(
                f"{action}: {email} game_pk={game_pk} "
                f"pnl {old_pnl} -> {new_pnl} result={old_result!r}->NULL "
                f"contracts={old_contracts}->0 ({source})"
            )
        else:
            stats["errors"] += 1
            print(f"FAILED: {email} game_pk={game_pk}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair mercy-rule user_orders rows.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist repairs (default is dry-run).",
    )
    parser.add_argument(
        "--skip-pending-payouts",
        action="store_true",
        help="Do not complete stale balance refresh jobs.",
    )
    args = parser.parse_args()
    dry_run = not args.apply
    if dry_run:
        print("Dry-run mode — pass --apply to write changes.")
    if not args.skip_pending_payouts:
        stale = complete_stale_pending_payout_jobs(dry_run=dry_run)
        if dry_run:
            print(f"WOULD complete {stale} stale balance refresh job(s)")
    stats = repair_mercy_rule_orders(dry_run=dry_run)
    print(
        "\nSummary: "
        f"checked={stats['checked']} updated={stats['updated']} "
        f"skipped={stats['skipped']} errors={stats['errors']}"
    )


if __name__ == "__main__":
    main()
