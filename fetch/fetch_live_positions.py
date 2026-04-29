"""
Refresh live Kalshi order mark-to-market data.

Run once:
    uv run python -m fetch.fetch_live_positions --once

Loop:
    uv run python -m fetch.fetch_live_positions --interval 300
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent.parent))
import db as DB
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials


REFRESH_STALE_SECONDS = 300
MLB_SERIES = "KXMLBGAME"


@dataclass
class RefreshStats:
    checked: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0


def _retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values):
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _signed_get(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key,
    endpoint: str,
    params: dict | None = None,
) -> dict:
    path = api_path(endpoint)
    resp = session.get(
        base_url + "/" + endpoint.lstrip("/"),
        headers=auth_headers(key_id, private_key, "GET", path),
        params=params or {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_order_ticker(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key,
    order_id: str,
) -> str | None:
    if not order_id:
        return None
    body = _signed_get(session, base_url, key_id, private_key, f"portfolio/orders/{order_id}")
    order = body.get("order") or body
    ticker = order.get("ticker")
    return str(ticker) if ticker else None


def _fetch_position(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key,
    ticker: str,
) -> dict:
    body = _signed_get(
        session,
        base_url,
        key_id,
        private_key,
        "portfolio/positions",
        params={"ticker": ticker, "count_filter": "position,total_traded", "limit": 1000},
    )
    for pos in body.get("market_positions", []):
        if str(pos.get("ticker", "")).upper() == ticker.upper():
            return pos
    return {}


def _fetch_market(session: requests.Session, base_url: str, ticker: str) -> dict:
    resp = session.get(
        base_url + "/markets",
        params={"tickers": ticker, "limit": 1},
        timeout=15,
    )
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    return markets[0] if markets else {}


def _market_mark_price(market: dict) -> float | None:
    # Conservative liquidation estimate for a YES position: highest YES bid.
    return _first_float(
        market.get("yes_bid_dollars"),
        market.get("yes_bid"),
        market.get("yes_bid_cents"),
        market.get("last_price_dollars"),
        market.get("last_price"),
        market.get("last_price_cents"),
    )


def refresh_order(row: dict, session: requests.Session | None = None) -> bool:
    session = session or _retry_session()
    email = str(row["email"])
    game_pk = row["game_pk"]
    key_id, private_key = load_credentials(
        key_id=row.get("key_id"),
        key_path=row.get("key_path"),
    )
    base_url = get_base_url(row.get("kalshi_env"))

    ticker = (row.get("kalshi_ticker") or "").strip()
    if not ticker:
        ticker = _fetch_order_ticker(
            session,
            base_url,
            key_id,
            private_key,
            str(row.get("kalshi_order_id") or ""),
        ) or ""
        if ticker:
            DB.update_user_order_kalshi_ticker(email, game_pk, ticker)
    if not ticker:
        DB.update_user_order_live_snapshot(
            email,
            game_pk,
            last_check_error="missing Kalshi ticker",
        )
        return False

    position = _fetch_position(session, base_url, key_id, private_key, ticker)
    market = _fetch_market(session, base_url, ticker)

    position_count = _first_float(
        position.get("position_fp"),
        position.get("position"),
        row.get("n_contracts"),
    )
    mark_price = _market_mark_price(market)
    if mark_price is not None and mark_price > 1:
        mark_price = mark_price / 100.0
    current_value = None
    if position_count is not None and mark_price is not None:
        current_value = round(float(position_count) * float(mark_price), 2)
    bet_dollars = _to_float(row.get("bet_dollars"))
    unrealized_pnl = None
    if current_value is not None and bet_dollars is not None:
        unrealized_pnl = round(current_value - bet_dollars, 2)

    DB.update_user_order_live_snapshot(
        email,
        game_pk,
        kalshi_ticker=ticker,
        current_price=mark_price,
        current_value=current_value,
        unrealized_pnl=unrealized_pnl,
        position_count=position_count,
        market_status=market.get("status"),
        last_check_error=None,
    )
    return True


def refresh_due_orders(
    stale_seconds: int = REFRESH_STALE_SECONDS,
    limit: int = 200,
) -> RefreshStats:
    rows = DB.get_open_live_user_orders_for_refresh(stale_seconds=stale_seconds, limit=limit)
    stats = RefreshStats(checked=len(rows))
    if not rows:
        return stats
    session = _retry_session()
    for row in rows:
        try:
            if refresh_order(row, session=session):
                stats.updated += 1
            else:
                stats.skipped += 1
        except Exception as exc:
            stats.errors += 1
            try:
                DB.update_user_order_live_snapshot(
                    str(row["email"]),
                    row["game_pk"],
                    kalshi_ticker=row.get("kalshi_ticker"),
                    last_check_error=str(exc)[:500],
                )
            except Exception:
                pass
            print(f"Live position refresh failed for {row.get('email')} game_pk={row.get('game_pk')}: {exc}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live Kalshi position marks.")
    parser.add_argument("--once", action="store_true", help="Run one refresh pass and exit")
    parser.add_argument("--interval", type=int, default=REFRESH_STALE_SECONDS, help="Loop interval in seconds")
    parser.add_argument("--limit", type=int, default=200, help="Max orders per pass")
    args = parser.parse_args()

    while True:
        stats = refresh_due_orders(stale_seconds=args.interval, limit=args.limit)
        print(
            "Live position refresh: "
            f"checked={stats.checked} updated={stats.updated} "
            f"skipped={stats.skipped} errors={stats.errors}"
        )
        if args.once:
            return
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
