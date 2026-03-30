"""
Place the model's bet on Kalshi for the target game.
Reads bet_frac and bet_side from data/games.csv, sizes against data/balance.csv,
finds the Kalshi market, and places a limit order.

Run from project root: uv run bet/place_bet.py --game_pk 12345
"""

import argparse
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials

DRY_RUN = True  # Set to False to place real bets on Kalshi

GAMES_CSV   = Path("data/games.csv")
BALANCE_CSV = Path("data/balance.csv")
MLB_SERIES  = "KXMLBGAME"


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

    home = home_team.upper()
    away = away_team.upper()
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
    # --- Load game row ---
    if not GAMES_CSV.exists():
        sys.exit(f"ERROR: {GAMES_CSV} not found.")

    games = pd.read_csv(GAMES_CSV, dtype=str)
    mask  = games["game_pk"].astype(str) == str(game_pk)
    if not mask.any():
        sys.exit(f"ERROR: game_pk={game_pk} not found in {GAMES_CSV}")

    row      = games[mask].iloc[0]
    bet_frac = float(row.get("bet_frac") or 0)
    bet_side = str(row.get("bet_side") or "none").strip().lower()

    if bet_frac <= 0 or bet_side == "none":
        print("No bet indicated — skipping.")
        return

    home_team   = str(row["home_team"])
    away_team   = str(row["away_team"])
    game_date   = str(row["game_date"])[:10]
    mkt_prob_str = str(row.get("market_implied_prob") or "")

    if not mkt_prob_str or mkt_prob_str.lower() in ("nan", ""):
        sys.exit("ERROR: market_implied_prob missing in games.csv. Cannot size the bet.")

    market_prob = float(mkt_prob_str)

    # --- Size the bet ---
    if not BALANCE_CSV.exists() or BALANCE_CSV.stat().st_size == 0:
        sys.exit(f"ERROR: {BALANCE_CSV} empty. Run fetch_kalshi_balance.py first.")

    balance_df     = pd.read_csv(BALANCE_CSV)
    balance_cents  = int(balance_df.iloc[-1]["balance_cents"])
    balance_dollars = balance_cents / 100.0
    bet_dollars     = round(balance_dollars * bet_frac, 2)

    print(f"  Bankroll:        ${balance_dollars:.2f}")
    print(f"  Kelly fraction:  {bet_frac*100:.2f}%")
    print(f"  Bet amount:      ${bet_dollars:.2f} on {bet_side.upper()}")

    if bet_dollars < 0.01:
        print("  Bet rounds to $0.00 — skipping.")
        return

    # Kalshi prices are integers 1–99 (cents per contract)
    yes_price_cents = max(1, min(99, round(market_prob * 100)))

    if bet_side == "home":
        side              = "yes"
        cost_per_contract = yes_price_cents / 100.0        # cost of one YES contract
    else:  # away
        side              = "no"
        cost_per_contract = (100 - yes_price_cents) / 100.0  # cost of one NO contract

    n_contracts = max(1, int(bet_dollars / cost_per_contract))
    actual_cost = round(n_contracts * cost_per_contract, 2)

    print(f"  Side:            {side} (yes_price={yes_price_cents}¢)")
    print(f"  Contracts:       {n_contracts} × ${cost_per_contract:.2f} = ${actual_cost:.2f}")

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

    # Confirm market is still open
    resp = requests.get(base_url + f"/markets/{ticker}", timeout=10)
    resp.raise_for_status()
    mkt_status = resp.json().get("market", {}).get("status", "unknown")
    if mkt_status != "open":
        sys.exit(f"ERROR: Market {ticker} is '{mkt_status}' — cannot place order.")

    # --- Place order ---
    if DRY_RUN:
        print(f"\n  [DRY RUN] Would place order:")
        print(f"    Ticker:        {ticker}")
        print(f"    Side:          {side}")
        print(f"    Contracts:     {n_contracts}")
        print(f"    Yes price:     {yes_price_cents}¢")
        print(f"    Est. cost:     ${actual_cost:.2f}")
        print(f"  [DRY RUN] No real order was placed.")
        return

    order_path = api_path("portfolio/orders")
    headers    = auth_headers(key_id, private_key, "POST", order_path)

    order_body = {
        "ticker":          ticker,
        "side":            side,
        "action":          "buy",
        "type":            "limit",
        "count":           n_contracts,
        "yes_price":       yes_price_cents,
        "time_in_force":   "good_till_canceled",
        "client_order_id": str(uuid.uuid4()),
    }

    resp = requests.post(
        base_url + "/portfolio/orders",
        headers=headers,
        json=order_body,
        timeout=10,
    )

    if resp.status_code == 201:
        order    = resp.json()["order"]
        order_id = order["order_id"]
        print(f"\n  Order placed:    {order_id}")
        print(f"  Status:          {order['status']}")
    else:
        sys.exit(f"ERROR placing order ({resp.status_code}): {resp.text}")

    # --- Write order details back to games.csv ---
    games.loc[mask, "kalshi_order_id"] = order_id
    games.loc[mask, "bet_dollars"]     = f"{actual_cost:.2f}"
    games.loc[mask, "n_contracts"]     = str(n_contracts)
    games.to_csv(GAMES_CSV, index=False)
    print(f"  games.csv updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place Kalshi bet for a game.")
    parser.add_argument("--game_pk", required=True, type=str, help="MLB Stats API game_pk")
    args = parser.parse_args()
    place_bet(args.game_pk)
