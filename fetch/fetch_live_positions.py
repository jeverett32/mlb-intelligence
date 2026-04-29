"""
Refresh Kalshi market marks for open live and paper orders.

One Kalshi market request updates every user/order sharing that ticker.

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
from bet.place_bet import find_kalshi_market
from kalshi_client import get_base_url


REFRESH_STALE_SECONDS = 300


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


def _market_mark_price(market: dict) -> float | None:
    # Conservative liquidation estimate for a YES position: highest YES bid.
    mark = _first_float(
        market.get("yes_bid_dollars"),
        market.get("yes_bid"),
        market.get("yes_bid_cents"),
        market.get("last_price_dollars"),
        market.get("last_price"),
        market.get("last_price_cents"),
    )
    if mark is not None and mark > 1:
        mark = mark / 100.0
    return mark


def _fetch_market(session: requests.Session, base_url: str, ticker: str) -> dict:
    resp = session.get(
        base_url + "/markets",
        params={"tickers": ticker, "limit": 1},
        timeout=15,
    )
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    return markets[0] if markets else {}


def _resolve_ticker(row: dict, base_url: str) -> str:
    ticker = (row.get("kalshi_ticker") or "").strip()
    if ticker:
        return ticker
    bet_side = str(row.get("bet_side") or "").lower()
    if bet_side not in {"home", "away"}:
        return ""
    ticker, _market = find_kalshi_market(
        str(row.get("home_team") or ""),
        str(row.get("away_team") or ""),
        str(row.get("game_date") or "")[:10],
        bet_side,
        game_time_utc=str(row.get("game_time_utc") or "") or None,
        base_url=base_url,
    )
    if ticker:
        DB.update_order_kalshi_ticker(
            str(row.get("mode") or "live"),
            str(row["email"]),
            row["game_pk"],
            ticker,
        )
    return ticker or ""


def _refresh_ticker(session: requests.Session, ticker: str, base_url: str) -> int:
    try:
        market = _fetch_market(session, base_url, ticker)
        mark_price = _market_mark_price(market)
        market_status = market.get("status")
        DB.upsert_kalshi_market_snapshot(
            ticker,
            current_price=mark_price,
            market_status=market_status,
            last_error=None,
        )
        return DB.apply_market_snapshot_to_open_orders(
            ticker,
            current_price=mark_price,
            market_status=market_status,
            last_error=None,
        )
    except Exception as exc:
        msg = str(exc)[:500]
        DB.upsert_kalshi_market_snapshot(ticker, last_error=msg)
        DB.apply_market_snapshot_to_open_orders(ticker, last_error=msg)
        raise


def refresh_due_orders(
    stale_seconds: int = REFRESH_STALE_SECONDS,
    limit: int = 200,
) -> RefreshStats:
    rows = DB.get_open_orders_for_market_refresh(stale_seconds=stale_seconds, limit=limit)
    stats = RefreshStats(checked=len(rows))
    if not rows:
        return stats

    session = _retry_session()
    tickers_by_env: dict[str, set[str]] = {}
    for row in rows:
        kalshi_env = str(row.get("kalshi_env") or "").strip().lower() or None
        base_url = get_base_url(kalshi_env)
        try:
            ticker = _resolve_ticker(row, base_url)
        except Exception as exc:
            stats.errors += 1
            print(f"Ticker resolve failed for {row.get('mode')} game_pk={row.get('game_pk')}: {exc}")
            continue
        if not ticker:
            stats.skipped += 1
            continue
        tickers_by_env.setdefault(base_url, set()).add(ticker)

    for base_url, tickers in tickers_by_env.items():
        for ticker in sorted(tickers):
            try:
                stats.updated += _refresh_ticker(session, ticker, base_url)
            except Exception as exc:
                stats.errors += 1
                print(f"Market refresh failed for {ticker}: {exc}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Kalshi market marks.")
    parser.add_argument("--once", action="store_true", help="Run one refresh pass and exit")
    parser.add_argument("--interval", type=int, default=REFRESH_STALE_SECONDS, help="Loop interval in seconds")
    parser.add_argument("--limit", type=int, default=200, help="Max order rows per pass")
    args = parser.parse_args()

    while True:
        stats = refresh_due_orders(stale_seconds=args.interval, limit=args.limit)
        print(
            "Market mark refresh: "
            f"checked={stats.checked} updated={stats.updated} "
            f"skipped={stats.skipped} errors={stats.errors}"
        )
        if args.once:
            return
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
