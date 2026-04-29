"""
Place the model's bet on Kalshi for the target game.
Reads bet_frac and bet_side from the bets table, sizes against current balance,
finds the Kalshi market, and places an immediate-or-cancel limit order.

Run from project root: uv run bet/place_bet.py --game_pk 12345
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials
import db as DB
from fetch.fetch_balance import fetch_balance_for_account

MLB_SERIES = "KXMLBGAME"
ORDER_SLIPPAGE_CENTS = int(os.environ.get("KALSHI_ORDER_SLIPPAGE_CENTS", "3"))
EXECUTION_MIN_EDGE = float(os.environ.get("KALSHI_EXECUTION_MIN_EDGE", "0.0"))
KALSHI_ORDER_ENDPOINT = os.environ.get("KALSHI_ORDER_ENDPOINT", "legacy").strip().lower()
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
    "TBR": "TB",
    "OAK": "ATH",
}


def _retry_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


class PlaceBetError(RuntimeError):
    pass


def _to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _first_float(*values):
    for value in values:
        try:
            parsed = _to_float(value)
        except (TypeError, ValueError):
            continue
        if parsed is not None:
            return parsed
    return None


def _price_to_dollars(value):
    parsed = _first_float(value)
    if parsed is None:
        return None
    if parsed > 1:
        return parsed / 100.0
    return parsed


def _market_yes_price(market: dict) -> float | None:
    return _price_to_dollars(_first_float(
        market.get("yes_ask_dollars"),
        market.get("yes_ask"),
        market.get("yes_ask_cents"),
        market.get("yes_bid_dollars"),
        market.get("yes_bid"),
        market.get("yes_bid_cents"),
    ))


def _order_fill_count(order: dict) -> float:
    return _first_float(
        order.get("fill_count"),
        order.get("fill_count_fp"),
        order.get("filled_count"),
        order.get("filled_count_fp"),
    ) or 0.0


def _order_fill_cost(order: dict) -> float:
    taker_dollars = _first_float(order.get("taker_fill_cost_dollars"))
    maker_dollars = _first_float(order.get("maker_fill_cost_dollars"))
    if taker_dollars is not None or maker_dollars is not None:
        return (taker_dollars or 0.0) + (maker_dollars or 0.0)
    filled_dollars = _first_float(order.get("filled_cost_dollars"))
    if filled_dollars is not None:
        return filled_dollars
    taker_cents = _first_float(order.get("taker_fill_cost"))
    maker_cents = _first_float(order.get("maker_fill_cost"))
    if taker_cents is not None or maker_cents is not None:
        return ((taker_cents or 0.0) + (maker_cents or 0.0)) / 100.0
    filled_cents = _first_float(order.get("filled_cost"))
    return (filled_cents or 0.0) / 100.0 if filled_cents is not None else 0.0


def _events_order_fill_cost(order: dict) -> float:
    fill_count = _order_fill_count(order)
    avg_price = _price_to_dollars(_first_float(
        order.get("average_fill_price"),
        order.get("average_fill_price_dollars"),
        order.get("price"),
    ))
    if fill_count > 0 and avg_price is not None:
        return fill_count * avg_price
    return _order_fill_cost(order)


def _is_safe_events_fallback(resp: requests.Response) -> bool:
    if resp.status_code not in {400, 404, 405, 422}:
        return False
    text = (resp.text or "").lower()
    hints = ("portfolio/events/orders", "side", "price", "count", "schema", "invalid")
    return any(hint in text for hint in hints)


def _post_legacy_order(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key,
    ticker: str,
    n_contracts: int,
    limit_price_cents: int,
) -> tuple[dict, str]:
    order_path = api_path("portfolio/orders")
    headers = auth_headers(key_id, private_key, "POST", order_path)
    order_body = {
        "ticker": ticker,
        "side": "yes",
        "action": "buy",
        "time_in_force": "immediate_or_cancel",
        "count": n_contracts,
        "client_order_id": str(uuid.uuid4()),
        "yes_price": limit_price_cents,
    }
    resp = session.post(
        base_url + "/portfolio/orders",
        headers=headers,
        json=order_body,
        timeout=15,
    )
    if resp.status_code != 201:
        raise PlaceBetError(f"order rejected ({resp.status_code}): {resp.text}")
    return resp.json()["order"], "legacy"


def _post_events_order(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key,
    ticker: str,
    n_contracts: int,
    limit_price_cents: int,
) -> tuple[dict, str]:
    order_path = api_path("portfolio/events/orders")
    headers = auth_headers(key_id, private_key, "POST", order_path)
    order_body = {
        "ticker": ticker,
        "side": "bid",
        "count": f"{float(n_contracts):.2f}",
        "price": f"{limit_price_cents / 100.0:.4f}",
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": str(uuid.uuid4()),
    }
    resp = session.post(
        base_url + "/portfolio/events/orders",
        headers=headers,
        json=order_body,
        timeout=15,
    )
    if resp.status_code == 201:
        return resp.json(), "events"
    if KALSHI_ORDER_ENDPOINT in {"events", "auto"} and _is_safe_events_fallback(resp):
        print(
            "  Events order endpoint rejected request before execution; "
            "falling back to legacy /portfolio/orders."
        )
        return _post_legacy_order(
            session, base_url, key_id, private_key, ticker, n_contracts, limit_price_cents
        )
    raise PlaceBetError(f"order rejected ({resp.status_code}): {resp.text}")


def _post_kalshi_order(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key,
    ticker: str,
    n_contracts: int,
    limit_price_cents: int,
) -> tuple[dict, str]:
    if KALSHI_ORDER_ENDPOINT == "legacy":
        return _post_legacy_order(
            session, base_url, key_id, private_key, ticker, n_contracts, limit_price_cents
        )
    return _post_events_order(
        session, base_url, key_id, private_key, ticker, n_contracts, limit_price_cents
    )


def _place_bet_error_status(exc: Exception) -> str:
    msg = str(exc).lower()
    if "no open kalshi market" in msg:
        return "skipped_no_market"
    if "no live" in msg and "price" in msg:
        return "skipped_no_live_price"
    return "error"


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


_MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def _kalshi_date_prefix(game_date: str) -> str:
    """'2026-04-27' → '26APR27' (Kalshi ticker date format)."""
    parts = game_date[:10].split("-")
    yy = parts[0][2:]
    mon = _MONTH_ABBR[int(parts[1])]
    dd = parts[2]
    return f"{yy}{mon}{dd}"


def find_kalshi_market(
    home_team: str,
    away_team: str,
    game_date: str,
    bet_side: str,
    *,
    game_time_utc: str | None = None,
    base_url: str | None = None,
):
    """
    Find the correct Kalshi market for this game and bet side.

    Kalshi MLB tickers: KXMLBGAME-{YY}{MON}{DD}{HHMM_ET}{AWAY}{HOME}-{TEAM}
    Two markets per game (one per team). We match the event by teams + date,
    then pick the market for the team we're betting on.

    For doubleheaders, game_time_utc disambiguates (matched to ET HHMM in ticker).
    Returns (ticker, market_dict) or (None, None).
    """
    base_url = base_url or get_base_url()
    session = _retry_session()
    markets = []
    for status in ("open",):
        cursor = None
        while True:
            params = {"series_ticker": MLB_SERIES, "status": status, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = session.get(base_url + "/markets", params=params, timeout=15)
            resp.raise_for_status()
            body = resp.json()
            markets.extend(body.get("markets", []))
            cursor = body.get("cursor")
            if not cursor:
                break

    home = MLB_TO_KALSHI.get(home_team.upper(), home_team.upper())
    away = MLB_TO_KALSHI.get(away_team.upper(), away_team.upper())
    date_prefix = _kalshi_date_prefix(game_date)
    bet_team = home if bet_side == "home" else away

    # Parse game time for doubleheader disambiguation (convert UTC → ET HHMM)
    game_hhmm_et = None
    if game_time_utc:
        try:
            from datetime import datetime, timezone
            from zoneinfo import ZoneInfo
            utc_dt = datetime.strptime(game_time_utc[:16], "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )
            et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
            game_hhmm_et = et_dt.strftime("%H%M")
        except Exception:
            pass

    # Group by event (same game, different team markets)
    events: dict[str, list[dict]] = {}
    for m in markets:
        event = m.get("event_ticker", "").upper()
        if home in event and away in event and date_prefix in event:
            events.setdefault(event, []).append(m)

    if not events:
        return None, None

    # If multiple events match (doubleheader), use game time to disambiguate.
    # Without a unique event, fail closed rather than risk betting the wrong game.
    if len(events) > 1 and game_hhmm_et:
        for event_key, event_markets in events.items():
            if game_hhmm_et in event_key:
                events = {event_key: event_markets}
                break
    if len(events) > 1:
        return None, None

    # Take first (or only) matching event
    event_markets = next(iter(events.values()))

    # Pick the market for the team we're betting on (buy yes = team wins)
    for m in event_markets:
        ticker = m.get("ticker", "").upper()
        if ticker.endswith(f"-{bet_team}"):
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
    session = _retry_session()

    # --- Find market ---
    game_time_utc = str(row.get("game_time_utc") or "")
    print(f"\n  Searching Kalshi market: {away_team} @ {home_team} on {game_date}...")
    ticker, _ = find_kalshi_market(
        home_team, away_team, game_date, bet_side,
        game_time_utc=game_time_utc or None,
        base_url=base_url,
    )

    if ticker is None:
        raise PlaceBetError(
            f"No open Kalshi market found for {away_team} @ {home_team} on {game_date}."
        )
    print(f"  Market found:    {ticker}")

    resp = session.get(base_url + f"/markets/{ticker}", timeout=15)
    resp.raise_for_status()
    market = resp.json().get("market", {})
    mkt_status = market.get("status", "unknown")
    if mkt_status not in ("open", "active"):
        raise PlaceBetError(f"Market {ticker} is '{mkt_status}' — cannot place order.")

    # Each Kalshi MLB market is team-specific: yes = that team wins.
    # find_kalshi_market already picked the market for our bet_side team.
    side = "yes"
    price_field = "yes_price"
    live_price = _market_yes_price(market)
    if bet_side == "home":
        model_prob = predicted_prob
    elif bet_side == "away":
        model_prob = 1.0 - predicted_prob
    else:
        raise PlaceBetError(f"Invalid bet_side {bet_side!r}; expected 'home' or 'away'.")

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
        return {"game_pk": str(row.get("game_pk")), "status": "skipped_too_small"}

    live_price_cents = max(1, min(99, int(round(live_price * 100))))
    limit_price_cents = max(1, min(99, live_price_cents + ORDER_SLIPPAGE_CENTS))
    cost_per_contract = limit_price_cents / 100.0
    n_contracts = max(1, int(bet_dollars / cost_per_contract))

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

    order, order_api = _post_kalshi_order(
        session,
        base_url,
        key_id,
        private_key,
        ticker,
        n_contracts,
        limit_price_cents,
    )
    order_id = order["order_id"]
    fill_count = _order_fill_count(order)
    actual_cost = (
        _events_order_fill_cost(order)
        if order_api == "events"
        else _order_fill_cost(order)
    )
    print(f"\n  Order placed:    {order_id}")
    print(f"  API:             {order_api}")
    print(f"  Status:          {order.get('status', 'submitted')}")
    print(f"  Filled:          {fill_count:.2f}")
    print(f"  Fill cost:       ${actual_cost:.2f}")

    if fill_count <= 0 or actual_cost <= 0:
        print("  Order was accepted but not filled. Not recording it as a placed bet.")
        return {"game_pk": str(row.get("game_pk")), "status": "unfilled", "order_id": order_id, "ticker": ticker}

    return {
        "game_pk":   str(row.get("game_pk")),
        "status":    "filled",
        "order_id":  order_id,
        "ticker":    ticker,
        "fill_cost": round(actual_cost, 2),
        "contracts": max(1, int(round(fill_count))),
        "live_price": live_price,
        "live_edge": live_edge,
    }


def _upsert_paper_result(email: str, game_pk: str, row: dict, result: dict, paper_bankroll: float) -> None:
    # Paper mode is universal. Per-user dry-run rows are legacy noise now;
    # dashboard paper state comes from DB.PAPER_UNIVERSAL_EMAIL only.
    _ = email
    paper_after = None
    if result.get("status") in {"dry_run", "skipped_no_live_edge"}:
        paper_after = paper_bankroll
    DB.upsert_paper_order(
        DB.PAPER_UNIVERSAL_EMAIL,
        game_pk,
        game_date=row.get("game_date", ""),
        home_team=row.get("home_team", ""),
        away_team=row.get("away_team", ""),
        predicted_prob=row.get("predicted_prob"),
        market_implied_prob=row.get("market_implied_prob"),
        edge=row.get("edge"),
        bet_side=row.get("bet_side") or "none",
        bet_frac=row.get("bet_frac") or 0.0,
        bet_dollars=result.get("bet_dollars"),
        n_contracts=result.get("contracts"),
        kalshi_ticker=result.get("ticker"),
        live_price=result.get("live_price"),
        live_edge=result.get("live_edge"),
        status=result.get("status", "pending"),
        paper_bankroll_before=paper_bankroll,
        paper_bankroll_after=paper_after,
    )


def _execution_mode_for_email(email: str) -> str:
    try:
        mode = DB.get_user_setting(email, "execution_mode", "server_managed")
    except Exception:
        return "server_managed"
    return mode if mode in {"server_managed", "self_custody", "paper_only"} else "server_managed"


def place_user_bet(email: str, game_pk: str) -> dict:
    email = email.strip().lower()
    user = DB.get_user(email)
    if not user or user["approval_status"] != DB.USER_STATUS_APPROVED:
        return {"game_pk": str(game_pk), "email": email, "status": "skipped_unapproved"}
    execution_mode = _execution_mode_for_email(email)
    if execution_mode == "self_custody":
        return {"game_pk": str(game_pk), "email": email, "status": "skipped_self_custody", "mode": "self_custody"}
    account = DB.get_kalshi_account(email)
    if not account or not account.get("is_active"):
        return {"game_pk": str(game_pk), "email": email, "status": "skipped_no_account"}
    row = DB.get_bet(game_pk)
    if row is None:
        raise PlaceBetError(f"game_pk={game_pk} not found in bets table.")

    existing_order = DB.get_user_order(email, game_pk)
    if existing_order is not None:
        existing_status = existing_order.get("status", "")
        if existing_status in ("filled", "dry_run"):
            return {"game_pk": str(game_pk), "email": email, "status": "skipped_already_bet"}

    original_bet_frac = float(row.get("bet_frac") or 0)
    bet_side = str(row.get("bet_side") or "none").strip().lower()
    if original_bet_frac <= 0 or bet_side == "none":
        return {"game_pk": str(game_pk), "email": email, "status": "skipped_no_bet"}

    paper_bankroll = DB.get_paper_bankroll_dollars(email)
    try:
        paper_result = _execute_bet_row(
            row,
            key_id=account["key_id"],
            key_path=account["key_path"],
            kalshi_env=account["kalshi_env"],
            balance_cents=int(round(paper_bankroll * 100)),
            dry_run=True,
        )
    except PlaceBetError as exc:
        paper_result = {
            "game_pk": str(game_pk),
            "status": _place_bet_error_status(exc),
            "error": str(exc),
        }
    _upsert_paper_result(email, game_pk, row, paper_result, paper_bankroll)

    if execution_mode == "paper_only":
        return {**paper_result, "email": email, "mode": "paper"}

    live_enabled = DB.is_global_live_betting() and DB.is_user_live_betting(email)
    if not live_enabled or paper_result.get("status", "").startswith("skipped_no_"):
        return {**paper_result, "email": email, "mode": "paper"}

    balance_cents = fetch_balance_for_account(
        key_id=account["key_id"],
        key_path=account["key_path"],
        kalshi_env=account["kalshi_env"],
        email=email,
    )
    try:
        result = _execute_bet_row(
            row,
            key_id=account["key_id"],
            key_path=account["key_path"],
            kalshi_env=account["kalshi_env"],
            balance_cents=balance_cents,
            dry_run=False,
        )
    except PlaceBetError as exc:
        result = {
            "game_pk": str(game_pk),
            "status": _place_bet_error_status(exc),
            "error": str(exc),
        }
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
        kalshi_ticker=result.get("ticker"),
        live_price=result.get("live_price"),
        live_edge=result.get("live_edge"),
        status=result.get("status", "pending"),
        dry_run=False,
        last_check_error=result.get("error"),
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
    parser.add_argument(
        "--email", required=True, type=str, help="User email for bet placement"
    )
    args = parser.parse_args()
    try:
        place_user_bet(args.email, args.game_pk)
    except PlaceBetError as e:
        sys.exit(f"ERROR: {e}")
