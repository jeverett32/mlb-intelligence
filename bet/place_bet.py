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
from fetch.fetch_balance import fetch_balance_for_account

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


class PlaceBetError(RuntimeError):
    pass


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


def find_kalshi_market(home_team: str, away_team: str, game_date: str, *, base_url: str | None = None):
    """
    Dynamically find the open Kalshi market for this game.
    Searches KXMLBGAME series and matches by team abbreviations + date.
    Returns (ticker, market_dict) or (None, None).
    """
    base_url = base_url or get_base_url()
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


def _execute_bet_row(
    row: dict,
    *,
    key_id: str,
    key_path: str,
    kalshi_env: str,
    balance_cents: int,
    dry_run: bool,
) -> dict:
    original_bet_frac = float(row.get("bet_frac") or 0)
    bet_side = str(row.get("bet_side") or "none").strip().lower()
    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    game_date = str(row["game_date"])[:10]
    market_prob = row.get("market_implied_prob")
    predicted_prob = row.get("predicted_prob")

    if market_prob is None or predicted_prob is None:
        raise PlaceBetError(
            "predicted_prob or market_implied_prob missing in bets table."
        )
    market_prob = float(market_prob)
    predicted_prob = float(predicted_prob)

    key_id, private_key = load_credentials(key_id=key_id, key_path=key_path)
    base_url = get_base_url(kalshi_env)

    # --- Find market ---
    print(f"\n  Searching Kalshi market: {away_team} @ {home_team} on {game_date}...")
    ticker, _ = find_kalshi_market(home_team, away_team, game_date, base_url=base_url)

    if ticker is None:
        raise PlaceBetError(
            f"No open Kalshi market found for {away_team} @ {home_team} on {game_date}."
        )
    print(f"  Market found:    {ticker}")

    resp = requests.get(base_url + f"/markets/{ticker}", timeout=10)
    resp.raise_for_status()
    market = resp.json().get("market", {})
    mkt_status = market.get("status", "unknown")
    if mkt_status not in ("open", "active"):
        raise PlaceBetError(f"Market {ticker} is '{mkt_status}' — cannot place order.")

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
        raise PlaceBetError(f"Market {ticker} has no live {price_field} available.")

    live_edge = model_prob - live_price
    if live_edge <= EXECUTION_MIN_EDGE:
        print(f"  Sportsbook market: {market_prob:.4f}")
        print(f"  Live Kalshi price: {live_price:.4f} ({price_field})")
        print(f"  Live edge:         {live_edge:.4f}")
        print("  Live Kalshi price removed the edge — skipping.")
        return {
            "game_pk": str(row.get("game_pk")),
            "status": "skipped_no_live_edge",
            "live_edge": live_edge,
            "live_price": live_price,
            "bet_dollars": 0.0,
        }

    live_bet_frac = _kelly_stake(model_prob, 1.0 / live_price)
    bet_frac = min(original_bet_frac, live_bet_frac) if original_bet_frac > 0 else live_bet_frac

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
        return {"game_pk": str(game_pk), "status": "skipped_too_small"}

    live_price_cents = max(1, min(99, math.ceil(live_price * 100)))
    limit_price_cents = max(1, min(99, live_price_cents + ORDER_SLIPPAGE_CENTS))
    cost_per_contract = limit_price_cents / 100.0
    n_contracts = int(bet_dollars / cost_per_contract)
    if n_contracts < 1:
        print(
            f"  Kelly bet ${bet_dollars:.2f} is below 1-contract cost "
            f"(${cost_per_contract:.2f}) — skipping to respect sizing."
        )
        return {"game_pk": str(game_pk), "status": "skipped_below_contract"}

    print(f"  Side:            {side} ({price_field}={limit_price_cents}c IOC)")
    max_cost = round(n_contracts * cost_per_contract, 2)
    print(
        f"  Contracts:       {n_contracts} × ${cost_per_contract:.2f} max = ${max_cost:.2f}"
    )

    # --- Place order ---
    if dry_run:
        print(f"\n  [DRY RUN] Would place market order:")
        print(f"    Ticker:        {ticker}")
        print(f"    Side:          {side}")
        print(f"    Limit:         {price_field}={limit_price_cents}c IOC")
        print(f"    Contracts:     {n_contracts}")
        print(f"    Max cost:      ${max_cost:.2f}")
        print(f"  [DRY RUN] No real order was placed.")
        return {
            "game_pk": str(row.get("game_pk")),
            "status": "dry_run",
            "ticker": ticker,
            "bet_dollars": max_cost,
            "contracts": n_contracts,
            "live_price": live_price,
            "live_edge": live_edge,
        }

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
        raise PlaceBetError(f"order rejected ({resp.status_code}): {resp.text}")

    if fill_count <= 0 or actual_cost <= 0:
        print("  Order was accepted but not filled. Not recording it as a placed bet.")
        return {"game_pk": str(game_pk), "status": "unfilled", "order_id": order_id}

    DB.update_bet_order(
        game_pk,
        order_id,
        round(actual_cost, 2),
        max(1, int(round(fill_count))),
    )
    print("  Filled order recorded in bets table.")
    return {
        "game_pk":   str(game_pk),
        "status":    "filled",
        "order_id":  order_id,
        "fill_cost": round(actual_cost, 2),
        "contracts": max(1, int(round(fill_count))),
        "live_price": live_price,
        "live_edge": live_edge,
    }


def place_bet(game_pk: str) -> dict:
    """Legacy single-account placement path using env-configured Kalshi credentials."""
    row = DB.get_bet(game_pk)
    if row is None:
        raise PlaceBetError(f"game_pk={game_pk} not found in bets table.")
    original_bet_frac = float(row.get("bet_frac") or 0)
    bet_side = str(row.get("bet_side") or "none").strip().lower()
    if original_bet_frac <= 0 or bet_side == "none":
        print("No bet indicated — skipping.")
        return {"game_pk": str(game_pk), "status": "skipped_no_bet"}
    balance_cents = DB.get_last_balance_cents()
    if balance_cents is None:
        raise PlaceBetError("No balance in DB. Run fetch/fetch_balance.py first.")
    result = _execute_bet_row(
        row,
        key_id=os.environ.get("KALSHI_KEY_ID", ""),
        key_path=os.environ.get("KALSHI_KEY_PATH", "kalshi-key.pem"),
        kalshi_env=os.environ.get("KALSHI_ENV", "prod"),
        balance_cents=balance_cents,
        dry_run=not DB.is_live_betting(),
    )
    if result.get("status") == "filled":
        DB.update_bet_order(
            game_pk,
            result["order_id"],
            result["fill_cost"],
            result["contracts"],
        )
        print("  Filled order recorded in bets table.")
    return result


def place_user_bet(email: str, game_pk: str) -> dict:
    email = email.strip().lower()
    user = DB.get_user(email)
    if not user or user["approval_status"] != DB.USER_STATUS_APPROVED:
        return {"game_pk": str(game_pk), "email": email, "status": "skipped_unapproved"}
    account = DB.get_kalshi_account(email)
    if not account or not account.get("is_active"):
        return {"game_pk": str(game_pk), "email": email, "status": "skipped_no_account"}
    row = DB.get_bet(game_pk)
    if row is None:
        raise PlaceBetError(f"game_pk={game_pk} not found in bets table.")

    original_bet_frac = float(row.get("bet_frac") or 0)
    bet_side = str(row.get("bet_side") or "none").strip().lower()
    if original_bet_frac <= 0 or bet_side == "none":
        return {"game_pk": str(game_pk), "email": email, "status": "skipped_no_bet"}

    balance_cents = fetch_balance_for_account(
        key_id=account["key_id"],
        key_path=account["key_path"],
        kalshi_env=account["kalshi_env"],
        email=email,
    )
    dry_run = not (DB.is_global_live_betting() and DB.is_user_live_betting(email))
    result = _execute_bet_row(
        row,
        key_id=account["key_id"],
        key_path=account["key_path"],
        kalshi_env=account["kalshi_env"],
        balance_cents=balance_cents,
        dry_run=dry_run,
    )
    DB.upsert_user_order(
        email,
        game_pk,
        game_date=row.get("game_date", ""),
        home_team=row.get("home_team", ""),
        away_team=row.get("away_team", ""),
        predicted_prob=row.get("predicted_prob"),
        market_implied_prob=row.get("market_implied_prob"),
        edge=row.get("edge"),
        bet_side=row.get("bet_side") or "none",
        bet_frac=row.get("bet_frac") or 0.0,
        bet_dollars=result.get("fill_cost") or result.get("bet_dollars"),
        n_contracts=result.get("contracts"),
        kalshi_order_id=result.get("order_id"),
        live_price=result.get("live_price"),
        live_edge=result.get("live_edge"),
        status=result.get("status", "pending"),
        dry_run=dry_run,
    )
    return result


def execute_for_all_users(game_pk: str) -> list[dict]:
    results = []
    for user in DB.list_approved_users_with_accounts():
        try:
            results.append(place_user_bet(user["email"], game_pk))
        except Exception as exc:
            results.append(
                {"game_pk": str(game_pk), "email": user["email"], "status": "error", "error": str(exc)}
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place Kalshi bet for a game.")
    parser.add_argument(
        "--game_pk", required=True, type=str, help="MLB Stats API game_pk"
    )
    args = parser.parse_args()
    try:
        place_bet(args.game_pk)
    except PlaceBetError as e:
        sys.exit(f"ERROR: {e}")
