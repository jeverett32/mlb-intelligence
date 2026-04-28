"""
Standalone MLB Intelligence self-custody follower.

Edit the CONFIG values below, install dependencies, then run:
  python self_custody_follower.py

Dependencies:
  pip install requests cryptography
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# ---------------------------------------------------------------------------
# CONFIG - edit these values.
# ---------------------------------------------------------------------------

MLBI_API_BASE_URL = "https://your-mlb-intelligence-domain"
MLBI_API_TOKEN = "paste-your-api-token"

KALSHI_KEY_ID = "paste-your-kalshi-key-id"
KALSHI_KEY_PATH = "kalshi-key.pem"
KALSHI_ENV = "prod"  # "prod" or "demo"

DRY_RUN = True
POLL_SECONDS = 60
SIGNAL_LIMIT = 25
STATE_FILE = "signal_follower_state.json"
ORDER_SLIPPAGE_CENTS = 3
EXECUTION_MIN_EDGE = 0.0


KALSHI_BASES = {
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
}
API_PATH_PREFIX = "/trade-api/v2"
MLB_SERIES = "KXMLBGAME"
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
MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


class FollowerError(RuntimeError):
    pass


def kalshi_base_url() -> str:
    return KALSHI_BASES.get(KALSHI_ENV.lower(), KALSHI_BASES["prod"])


def api_path(endpoint: str) -> str:
    return f"{API_PATH_PREFIX}/{endpoint.lstrip('/')}"


def load_private_key():
    key_path = Path(KALSHI_KEY_PATH).expanduser()
    if not key_path.exists():
        raise FollowerError(f"Kalshi private key not found: {key_path}")
    with key_path.open("rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend(),
        )


def kalshi_headers(method: str, path: str) -> dict:
    private_key = load_private_key()
    ts = str(int(time.time() * 1000))
    message = f"{ts}{method.upper()}{path.split('?')[0]}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "Content-Type": "application/json",
    }


def mlbi_headers() -> dict:
    return {
        "Authorization": f"Bearer {MLBI_API_TOKEN}",
        "Content-Type": "application/json",
    }


def load_state() -> dict:
    path = Path(STATE_FILE).expanduser()
    if not path.exists():
        return {"signals": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"signals": {}}


def save_state(state: dict) -> None:
    path = Path(STATE_FILE).expanduser()
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def mlbi_get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(
        MLBI_API_BASE_URL.rstrip("/") + path,
        params=params or {},
        headers=mlbi_headers(),
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def mlbi_post(path: str, payload: dict) -> dict:
    resp = requests.post(
        MLBI_API_BASE_URL.rstrip("/") + path,
        headers=mlbi_headers(),
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_balance_cents() -> int:
    path = api_path("portfolio/balance")
    resp = requests.get(
        kalshi_base_url() + "/portfolio/balance",
        headers=kalshi_headers("GET", path),
        timeout=15,
    )
    resp.raise_for_status()
    return int(resp.json()["balance"])


def kalshi_date_prefix(game_date: str) -> str:
    year, month, day = game_date[:10].split("-")
    return f"{year[2:]}{MONTH_ABBR[int(month)]}{day}"


def event_hhmm_et(game_time_utc: str | None) -> str | None:
    if not game_time_utc:
        return None
    try:
        from zoneinfo import ZoneInfo

        dt = datetime.fromisoformat(str(game_time_utc).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%H%M")
    except Exception:
        return None


def market_yes_price(market: dict) -> float | None:
    for key in (
        "yes_ask_dollars",
        "yes_ask",
        "yes_ask_cents",
        "yes_bid_dollars",
        "yes_bid",
        "yes_bid_cents",
    ):
        value = market.get(key)
        if value is None or value == "":
            continue
        price = float(value)
        return price / 100.0 if price > 1 else price
    return None


def find_kalshi_market(signal: dict) -> tuple[str, dict]:
    home = MLB_TO_KALSHI.get(signal["home_team"].upper(), signal["home_team"].upper())
    away = MLB_TO_KALSHI.get(signal["away_team"].upper(), signal["away_team"].upper())
    date_prefix = kalshi_date_prefix(signal["game_date"])
    bet_team = home if signal["bet_side"] == "home" else away
    game_hhmm = event_hhmm_et(signal.get("game_time_utc"))

    markets = []
    cursor = None
    while True:
        params = {"series_ticker": MLB_SERIES, "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(kalshi_base_url() + "/markets", params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        markets.extend(body.get("markets", []))
        cursor = body.get("cursor")
        if not cursor:
            break

    events: dict[str, list[dict]] = {}
    for market in markets:
        event = market.get("event_ticker", "").upper()
        if home in event and away in event and date_prefix in event:
            events.setdefault(event, []).append(market)

    if len(events) > 1 and game_hhmm:
        matches = {k: v for k, v in events.items() if game_hhmm in k}
        if matches:
            events = matches
    if len(events) != 1:
        raise FollowerError(f"Could not identify unique Kalshi market for {away} @ {home}.")

    for market in next(iter(events.values())):
        ticker = market.get("ticker", "").upper()
        if ticker.endswith(f"-{bet_team}"):
            return market["ticker"], market
    raise FollowerError(f"Could not find team market for {bet_team}.")


def is_expired(signal: dict) -> bool:
    expires = signal.get("expires_at_utc")
    if not expires:
        return False
    try:
        expires_dt = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
    except Exception:
        return False
    return expires_dt <= datetime.now(timezone.utc)


def already_handled(signal_id: str, state: dict) -> bool:
    status = ((state.get("signals") or {}).get(signal_id) or {}).get("status", "")
    return status in {"filled", "dry_run", "unfilled", "skipped_too_small"}


def record_signal(state: dict, signal_id: str, result: dict) -> None:
    state.setdefault("signals", {})[signal_id] = {
        "status": result.get("status", "unknown"),
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "order_id": result.get("order_id"),
    }


def execute_signal(signal: dict) -> tuple[dict, int]:
    balance_cents = fetch_balance_cents()
    ticker, _ = find_kalshi_market(signal)

    resp = requests.get(kalshi_base_url() + f"/markets/{ticker}", timeout=15)
    resp.raise_for_status()
    market = resp.json().get("market", {})
    if market.get("status", "unknown") not in {"open", "active"}:
        raise FollowerError(f"Market {ticker} is not open.")

    live_price = market_yes_price(market)
    model_prob = signal.get("model_side_prob")
    if live_price is None or not (0 < live_price < 1):
        raise FollowerError(f"Market {ticker} has no live yes price.")
    if model_prob is None:
        raise FollowerError("Signal missing model_side_prob.")

    live_edge = float(model_prob) - live_price
    if live_edge <= EXECUTION_MIN_EDGE:
        return {
            "game_pk": signal["game_pk"],
            "status": "skipped_no_live_edge",
            "live_edge": live_edge,
            "live_price": live_price,
            "bet_dollars": 0.0,
        }, balance_cents

    stake_frac = float(signal.get("recommended_stake_frac") or 0.0)
    bet_dollars = round((balance_cents / 100.0) * stake_frac, 2)
    if bet_dollars < 0.01:
        return {"game_pk": signal["game_pk"], "status": "skipped_too_small"}, balance_cents

    live_price_cents = max(1, min(99, int(round(live_price * 100))))
    limit_price_cents = max(1, min(99, live_price_cents + ORDER_SLIPPAGE_CENTS))
    n_contracts = max(1, int(bet_dollars / (limit_price_cents / 100.0)))
    max_cost = round(n_contracts * (limit_price_cents / 100.0), 2)

    if DRY_RUN:
        return {
            "game_pk": signal["game_pk"],
            "status": "dry_run",
            "ticker": ticker,
            "bet_dollars": max_cost,
            "contracts": n_contracts,
            "live_price": live_price,
            "live_edge": live_edge,
        }, balance_cents

    order_path = api_path("portfolio/orders")
    order_body = {
        "ticker": ticker,
        "side": "yes",
        "action": "buy",
        "time_in_force": "immediate_or_cancel",
        "count": n_contracts,
        "client_order_id": str(uuid.uuid4()),
        "yes_price": limit_price_cents,
    }
    resp = requests.post(
        kalshi_base_url() + "/portfolio/orders",
        headers=kalshi_headers("POST", order_path),
        json=order_body,
        timeout=15,
    )
    if resp.status_code != 201:
        raise FollowerError(f"order rejected ({resp.status_code}): {resp.text}")

    order = resp.json()["order"]
    fill_count = float(order.get("fill_count") or order.get("filled_count") or 0)
    actual_cost = float(order.get("taker_fill_cost_dollars") or order.get("filled_cost_dollars") or 0)
    if fill_count <= 0 or actual_cost <= 0:
        return {"game_pk": signal["game_pk"], "status": "unfilled", "order_id": order["order_id"]}, balance_cents

    return {
        "game_pk": signal["game_pk"],
        "status": "filled",
        "order_id": order["order_id"],
        "fill_cost": round(actual_cost, 2),
        "contracts": max(1, int(round(fill_count))),
        "live_price": live_price,
        "live_edge": live_edge,
    }, balance_cents


def sync_balance(balance_cents: int) -> None:
    mlbi_post("/api/client/balance", {"balance_cents": int(balance_cents), "source": "self_custody"})


def sync_order(signal: dict, result: dict) -> None:
    mlbi_post(
        "/api/client/orders",
        {
            "signal_id": signal.get("signal_id", ""),
            "game_pk": signal["game_pk"],
            "game_date": signal.get("game_date", ""),
            "home_team": signal.get("home_team", ""),
            "away_team": signal.get("away_team", ""),
            "predicted_prob": signal.get("predicted_prob"),
            "market_implied_prob": signal.get("market_implied_prob"),
            "edge": signal.get("edge"),
            "bet_side": signal.get("bet_side", "none"),
            "bet_frac": signal.get("recommended_stake_frac") or 0.0,
            "bet_dollars": result.get("fill_cost") or result.get("bet_dollars"),
            "n_contracts": result.get("contracts"),
            "kalshi_order_id": result.get("order_id"),
            "live_price": result.get("live_price"),
            "live_edge": result.get("live_edge"),
            "status": result.get("status", "pending"),
            "dry_run": bool(DRY_RUN),
        },
    )


def run_once(state: dict) -> None:
    mlbi_post("/api/client/heartbeat", {"version": "standalone-1", "kalshi_env": KALSHI_ENV})
    signals = mlbi_get("/api/signals/upcoming", {"limit": SIGNAL_LIMIT}).get("signals", [])
    print(f"Fetched {len(signals)} signal(s).")
    for signal in signals:
        signal_id = signal["signal_id"]
        if already_handled(signal_id, state):
            print(f"Skip {signal_id}: already handled.")
            continue
        if is_expired(signal):
            print(f"Skip {signal_id}: expired.")
            continue

        try:
            result, balance_cents = execute_signal(signal)
            sync_balance(balance_cents)
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}

        print(f"{signal_id}: {result.get('status')}")
        if result.get("error"):
            print(f"  {result['error']}")
        try:
            sync_order(signal, result)
        except Exception as exc:
            print(f"  WARNING: telemetry sync failed: {exc}")
        record_signal(state, signal_id, result)


def main() -> None:
    state = load_state()
    while True:
        run_once(state)
        save_state(state)
        time.sleep(max(5, int(POLL_SECONDS)))


if __name__ == "__main__":
    main()
