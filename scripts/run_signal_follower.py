"""
Pull read-only betting signals from MLB Intelligence and place orders directly
from your own machine with your own Kalshi credentials.

Example:
  MLBI_API_BASE_URL=https://mlb.example.com \
  MLBI_API_TOKEN=... \
  KALSHI_KEY_ID=... \
  KALSHI_KEY_PATH=kalshi-key.pem \
  uv run python scripts/run_signal_follower.py --once
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bet.place_bet import _execute_bet_row, PlaceBetError
from kalshi_client import api_path, auth_headers, get_base_url, load_credentials

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow MLB Intelligence signals with self-custody Kalshi execution.")
    parser.add_argument("--base-url", default=os.environ.get("MLBI_API_BASE_URL", "").strip(), help="API base URL, e.g. https://mlb.example.com")
    parser.add_argument("--api-token", default=os.environ.get("MLBI_API_TOKEN", "").strip(), help="Read-only signal API token")
    parser.add_argument("--kalshi-env", default=os.environ.get("KALSHI_ENV", "prod").strip(), choices=["prod", "demo"], help="Kalshi environment")
    parser.add_argument("--state-file", default=os.environ.get("MLBI_STATE_FILE", str(ROOT / "tmp" / "signal_follower_state.json")), help="Local state file for placed signals")
    parser.add_argument("--poll-seconds", type=int, default=int(os.environ.get("MLBI_POLL_SECONDS", "60")), help="Poll interval for loop mode")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("MLBI_SIGNAL_LIMIT", "25")), help="Max signals to fetch each poll")
    parser.add_argument("--dry-run", action="store_true", help="Do not send real orders to Kalshi")
    parser.add_argument("--no-sync", action="store_true", help="Do not push balance/order telemetry back to MLB Intelligence")
    parser.add_argument("--once", action="store_true", help="Run one poll and exit")
    return parser.parse_args()


def fetch_balance_cents(*, key_id: str, key_path: str, kalshi_env: str) -> int:
    key_id, private_key = load_credentials(key_id=key_id, key_path=key_path)
    base_url = get_base_url(kalshi_env)
    path = api_path("portfolio/balance")
    headers = auth_headers(key_id, private_key, "GET", path)
    resp = requests.get(base_url + "/portfolio/balance", headers=headers, timeout=15)
    resp.raise_for_status()
    return int(resp.json()["balance"])


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"signals": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"signals": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def fetch_signals(base_url: str, api_token: str, limit: int) -> list[dict]:
    resp = requests.get(
        base_url.rstrip("/") + "/api/signals/upcoming",
        params={"limit": limit},
        headers={"Authorization": f"Bearer {api_token}"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("signals", [])


def post_client_json(base_url: str, api_token: str, path: str, payload: dict) -> dict:
    resp = requests.post(
        base_url.rstrip("/") + path,
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def sync_balance(args: argparse.Namespace, balance_cents: int) -> None:
    if args.no_sync:
        return
    post_client_json(
        args.base_url,
        args.api_token,
        "/api/client/balance",
        {"balance_cents": int(balance_cents), "source": "self_custody"},
    )


def sync_heartbeat(args: argparse.Namespace) -> None:
    if args.no_sync:
        return
    post_client_json(
        args.base_url,
        args.api_token,
        "/api/client/heartbeat",
        {"version": "1", "kalshi_env": args.kalshi_env},
    )


def sync_order(args: argparse.Namespace, signal: dict, result: dict) -> None:
    if args.no_sync:
        return
    payload = {
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
        "dry_run": bool(args.dry_run),
    }
    post_client_json(args.base_url, args.api_token, "/api/client/orders", payload)


def is_expired(signal: dict) -> bool:
    expires = signal.get("expires_at_utc")
    if not expires:
        return False
    try:
        expires_dt = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
    except Exception:
        return False
    return expires_dt <= datetime.now(timezone.utc)


def should_skip(signal_id: str, state: dict) -> bool:
    status = ((state.get("signals") or {}).get(signal_id) or {}).get("status", "")
    return status in {"filled", "dry_run", "unfilled", "skipped_too_small"}


def record_signal_state(state: dict, signal_id: str, result: dict) -> None:
    state.setdefault("signals", {})[signal_id] = {
        "status": result.get("status", "unknown"),
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "order_id": result.get("order_id"),
    }


def execute_signal(
    signal: dict,
    *,
    key_id: str,
    key_path: str,
    kalshi_env: str,
    dry_run: bool,
) -> tuple[dict, int]:
    balance_cents = fetch_balance_cents(key_id=key_id, key_path=key_path, kalshi_env=kalshi_env)
    row = {
        "game_pk": signal["game_pk"],
        "game_date": signal["game_date"],
        "game_time_utc": signal.get("game_time_utc"),
        "home_team": signal["home_team"],
        "away_team": signal["away_team"],
        "predicted_prob": signal["predicted_prob"],
        "market_implied_prob": signal["market_implied_prob"],
        "edge": signal.get("edge"),
        "bet_side": signal["bet_side"],
        "bet_frac": signal["recommended_stake_frac"],
    }
    result = _execute_bet_row(
        row,
        key_id=key_id,
        key_path=key_path,
        kalshi_env=kalshi_env,
        balance_cents=balance_cents,
        dry_run=dry_run,
    )
    return result, balance_cents


def run_once(args: argparse.Namespace, state: dict) -> None:
    key_id = os.environ.get("KALSHI_KEY_ID", "").strip()
    key_path = os.environ.get("KALSHI_KEY_PATH", "kalshi-key.pem").strip()
    try:
        sync_heartbeat(args)
    except Exception as exc:
        print(f"WARNING: heartbeat sync failed: {exc}")
    signals = fetch_signals(args.base_url, args.api_token, args.limit)
    print(f"Fetched {len(signals)} signal(s).")
    for signal in signals:
        signal_id = signal["signal_id"]
        if should_skip(signal_id, state):
            print(f"Skip {signal_id}: already handled.")
            continue
        if is_expired(signal):
            print(f"Skip {signal_id}: expired.")
            continue
        try:
            result, balance_cents = execute_signal(
                signal,
                key_id=key_id,
                key_path=key_path,
                kalshi_env=args.kalshi_env,
                dry_run=args.dry_run,
            )
            try:
                sync_balance(args, balance_cents)
            except Exception as exc:
                print(f"  WARNING: balance sync failed: {exc}")
        except PlaceBetError as exc:
            result = {"status": "error", "error": str(exc)}
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        print(f"{signal_id}: {result.get('status')}")
        if result.get("error"):
            print(f"  {result['error']}")
        try:
            sync_order(args, signal, result)
        except Exception as exc:
            print(f"  WARNING: telemetry sync failed: {exc}")
        record_signal_state(state, signal_id, result)


def main() -> int:
    args = parse_args()
    if not args.base_url:
        raise SystemExit("Missing --base-url or MLBI_API_BASE_URL")
    if not args.api_token:
        raise SystemExit("Missing --api-token or MLBI_API_TOKEN")
    state_path = Path(args.state_file).expanduser()
    state = load_state(state_path)
    while True:
        run_once(args, state)
        save_state(state_path, state)
        if args.once:
            return 0
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
