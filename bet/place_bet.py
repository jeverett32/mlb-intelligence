"""
Place the model's bet on Kalshi for the target game.
Reads bet_frac and bet_side from the bets table, sizes against current balance,
finds the Kalshi market, and places an immediate-or-cancel limit order.

Run from project root: uv run bet/place_bet.py --game_pk 12345
"""

import argparse
import math
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials
import db as DB

MLB_SERIES = "KXMLBGAME"
ORDER_SLIPPAGE_CENTS = int(os.environ.get("KALSHI_ORDER_SLIPPAGE_CENTS", "3"))
EXECUTION_MIN_EDGE = float(os.environ.get("KALSHI_EXECUTION_MIN_EDGE", "0.0"))
KELLY_FRACTION = 0.25
MAX_BET_FRAC = 0.25

# Kalshi uses different abbreviations than the MLB Stats API for some teams.
MLB_TO_KALSHI = {
    "KCR": "KC",
    "CHW": "CWS",
    "WSN": "WSH",
    "SDP": "SD",
    "SFG": "SF",
    "ARI": "AZ",
}


def _to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _kelly_stake(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (b * prob - (1.0 - prob)) / b
    stake = kelly * KELLY_FRACTION
    return float(max(0.0, min(stake, MAX_BET_FRAC)))


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------


def find_kalshi_market(home_team: str, away_team: str, game_date: str):
    """
    Dynamically find the open Kalshi market for this game.
    Searches KXMLBGAME series and matches by team abbreviations + date.
    Returns (ticker, market_dict) or (None, None).
    """
    base_url = get_base_url()
    resp = requests.get(
        base_url + "/markets",
        params={"series_ticker": MLB_SERIES, "status": "open", "limit": 200},
        timeout=10,
    )
    resp.raise_for_status()
    markets = resp.json().get("markets", [])

    home = MLB_TO_KALSHI.get(home_team.upper(), home_team.upper())
    away = MLB_TO_KALSHI.get(away_team.upper(), away_team.upper())
    # "2026-04-01" → "260401" (YY + MMDD, how Kalshi formats dates in tickers)
    date_compact = game_date.replace("-", "")[2:8]

    # Priority 1: ticker contains both teams AND date
    for m in markets:
        ticker = m.get("ticker", "").upper()
        if home in ticker and away in ticker and date_compact in ticker:
            return m["ticker"], m

    # Priority 2: ticker contains both teams (date format may differ)
    for m in markets:
        ticker = m.get("ticker", "").upper()
        if home in ticker and away in ticker:
            return m["ticker"], m

    return None, None


# ---------------------------------------------------------------------------
# Main bet placement
# ---------------------------------------------------------------------------


def place_bet(game_pk: str):
    # --- Load bet row from DB ---
    row = DB.get_bet(game_pk)
    if row is None:
        sys.exit(f"ERROR: game_pk={game_pk} not found in bets table.")

    original_bet_frac = float(row.get("bet_frac") or 0)
    bet_side = str(row.get("bet_side") or "none").strip().lower()

    if original_bet_frac <= 0 or bet_side == "none":
        print("No bet indicated — skipping.")
        return

    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    game_date = str(row["game_date"])[:10]
    market_prob = row.get("market_implied_prob")
    predicted_prob = row.get("predicted_prob")

    if market_prob is None or predicted_prob is None:
        sys.exit(
            "ERROR: predicted_prob or market_implied_prob missing in bets table. Cannot size the bet."
        )
    market_prob = float(market_prob)
    predicted_prob = float(predicted_prob)

    # --- Load Kalshi credentials ---
    key_id, private_key = load_credentials()
    base_url = get_base_url()

    # --- Find market ---
    print(f"\n  Searching Kalshi market: {away_team} @ {home_team} on {game_date}...")
    ticker, _ = find_kalshi_market(home_team, away_team, game_date)

    if ticker is None:
        sys.exit(
            f"ERROR: No open Kalshi market found for {away_team} @ {home_team} on {game_date}.\n"
            "       The market may not be listed yet, or the game has been postponed."
        )
    print(f"  Market found:    {ticker}")

    # Confirm market is still open (Kalshi uses "open" or "active")
    resp = requests.get(base_url + f"/markets/{ticker}", timeout=10)
    resp.raise_for_status()
    market = resp.json().get("market", {})
    mkt_status = market.get("status", "unknown")
    if mkt_status not in ("open", "active"):
        sys.exit(f"ERROR: Market {ticker} is '{mkt_status}' — cannot place order.")

    if bet_side == "home":
        side = "yes"
        price_field = "yes_price"
        live_price = _to_float(market.get("yes_ask_dollars"))
        model_prob = predicted_prob
    else:
        side = "no"
        price_field = "no_price"
        live_price = _to_float(market.get("no_ask_dollars"))
        model_prob = 1.0 - predicted_prob

    if live_price is None or not (0 < live_price < 1):
        sys.exit(f"ERROR: Market {ticker} has no live {price_field} available.")

    live_edge = model_prob - live_price
    if live_edge <= EXECUTION_MIN_EDGE:
        print(f"  Sportsbook market: {market_prob:.4f}")
        print(f"  Live Kalshi price: {live_price:.4f} ({price_field})")
        print(f"  Live edge:         {live_edge:.4f}")
        print("  Live Kalshi price removed the edge — skipping.")
        return

    live_bet_frac = _kelly_stake(model_prob, 1.0 / live_price)
    bet_frac = min(original_bet_frac, live_bet_frac) if original_bet_frac > 0 else live_bet_frac

    balance_cents = DB.get_last_balance_cents()
    if balance_cents is None:
        sys.exit("ERROR: No balance in DB. Run fetch/fetch_balance.py first.")
    balance_dollars = balance_cents / 100.0
    bet_dollars = round(balance_dollars * bet_frac, 2)

    print(f"  Bankroll:         ${balance_dollars:.2f}")
    print(f"  Sportsbook market:{market_prob:>9.4f}")
    print(f"  Live Kalshi price:{live_price:>9.4f}")
    print(f"  Live edge:       {live_edge:>10.4f}")
    print(f"  Kelly fraction:  {bet_frac * 100:.2f}%")
    print(f"  Bet amount:      ${bet_dollars:.2f} on {bet_side.upper()}")

    if bet_dollars < 0.01:
        print("  Bet rounds to $0.00 — skipping.")
        return

    live_price_cents = max(1, min(99, math.ceil(live_price * 100)))
    limit_price_cents = max(1, min(99, live_price_cents + ORDER_SLIPPAGE_CENTS))
    cost_per_contract = limit_price_cents / 100.0
    n_contracts = int(bet_dollars / cost_per_contract)
    if n_contracts < 1:
        print(
            f"  Kelly bet ${bet_dollars:.2f} is below 1-contract cost "
            f"(${cost_per_contract:.2f}) — skipping to respect sizing."
        )
        return

    print(f"  Side:            {side} ({price_field}={limit_price_cents}c IOC)")
    max_cost = round(n_contracts * cost_per_contract, 2)
    print(
        f"  Contracts:       {n_contracts} × ${cost_per_contract:.2f} max = ${max_cost:.2f}"
    )

    # --- Place order ---
    dry_run = not DB.is_live_betting()
    if dry_run:
        print(f"\n  [DRY RUN] Would place market order:")
        print(f"    Ticker:        {ticker}")
        print(f"    Side:          {side}")
        print(f"    Limit:         {price_field}={limit_price_cents}c IOC")
        print(f"    Contracts:     {n_contracts}")
        print(f"    Max cost:      ${max_cost:.2f}")
        print(f"  [DRY RUN] No real order was placed.")
        return

    order_path = api_path("portfolio/orders")
    headers = auth_headers(key_id, private_key, "POST", order_path)

    order_body = {
        "ticker": ticker,
        "side": side,
        "action": "buy",
        "time_in_force": "immediate_or_cancel",
        "count": n_contracts,
        "client_order_id": str(uuid.uuid4()),
        price_field: limit_price_cents,
    }

    resp = requests.post(
        base_url + "/portfolio/orders",
        headers=headers,
        json=order_body,
        timeout=10,
    )

    if resp.status_code == 201:
        order = resp.json()["order"]
        order_id = order["order_id"]
        fill_count = _to_float(order.get("fill_count_fp")) or 0.0
        actual_cost = (_to_float(order.get("taker_fill_cost_dollars")) or 0.0) + (
            _to_float(order.get("maker_fill_cost_dollars")) or 0.0
        )
        print(f"\n  Order placed:    {order_id}")
        print(f"  Status:          {order['status']}")
        print(f"  Filled:          {fill_count:.2f}")
        print(f"  Fill cost:       ${actual_cost:.2f}")
    else:
        sys.exit(f"ERROR placing order ({resp.status_code}): {resp.text}")

    if fill_count <= 0 or actual_cost <= 0:
        print("  Order was accepted but not filled. Not recording it as a placed bet.")
        return

    DB.update_bet_order(
        game_pk,
        order_id,
        round(actual_cost, 2),
        max(1, int(round(fill_count))),
    )
    print("  Filled order recorded in bets table.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place Kalshi bet for a game.")
    parser.add_argument(
        "--game_pk", required=True, type=str, help="MLB Stats API game_pk"
    )
    args = parser.parse_args()
    place_bet(args.game_pk)
