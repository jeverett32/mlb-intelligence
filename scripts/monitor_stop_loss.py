#!/usr/bin/env python3
"""
MLB Pipeline stop-loss ("Mercy Rule").

Automatically exits open Kalshi positions for ALL active users when the 
tracked side is in a deep late-inning deficit.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import db as DB
from bet.place_bet import exit_kalshi_position


def evaluate_stop_loss(order: dict) -> bool:
    """Return True when the order satisfies the stop-loss trigger."""
    try:
        inning = int(order.get("inning") or 0)
        home_score = int(order.get("home_score") or 0)
        away_score = int(order.get("away_score") or 0)
        over_under = float(order.get("over_under") or 0.0)
    except (TypeError, ValueError):
        return False

    if inning < 7:
        return False

    bet_side = str(order.get("bet_side") or "").strip().lower()
    if bet_side == "home":
        deficit = away_score - home_score
    elif bet_side == "away":
        deficit = home_score - away_score
    else:
        return False

    if deficit <= 0:
        return False

    # Higher-total games can swing more, so allow one extra run before exiting.
    leash = 1 if over_under >= 10.0 else 0
    if inning == 7:
        return deficit >= (5 + leash)
    if inning == 8:
        return deficit >= (4 + leash)
    return inning >= 9 and deficit >= (3 + leash)


def _try_exit_order(order: dict, dry_run: bool) -> tuple[bool, bool]:
    """
    Attempt to exit one order.
    Returns (triggered, executed_successfully).
    """
    game_desc = f"{order['away_team']} @ {order['home_team']}"
    score_desc = f"{order['away_score']}-{order['home_score']}"
    inning_desc = f"{order.get('inning_state') or ''} {order.get('inning') or ''}".strip()
    user_label = order.get("email", "unknown")
    
    print(
        f"[{user_label}] Checking {game_desc}: {score_desc}, {inning_desc} "
        f"(bet={order['bet_side']}, contracts={order['n_contracts']})"
    )

    if not evaluate_stop_loss(order):
        return False, False

    print(f"  [{user_label}] !!! Stop-loss triggered for {game_desc} !!!")
    if dry_run:
        print(
            f"  [{user_label}] [DRY RUN] Would sell "
            f"{order['n_contracts']} contracts for {order['kalshi_ticker']}"
        )
        return True, True

    try:
        exit_result = exit_kalshi_position(
            order["kalshi_ticker"],
            int(order["n_contracts"]),
            order["kalshi_env"],
            order["email"],
        )
        if not exit_result or int(exit_result.get("fill_count") or 0) <= 0:
            print(
                f"  [{user_label}] Exit attempt returned no fills "
                f"for {order['kalshi_ticker']} (no bids/liquidity)"
            )
            return True, False

        fill_count = int(exit_result["fill_count"])
        requested = int(order["n_contracts"])
        if fill_count < requested:
            # Partial fills remain 'filled' to reflect residual risk.
            print(
                f"  [{user_label}] Partial stop-loss fill ({fill_count}/{requested}); "
                "leaving order status unchanged."
            )
            return True, False

        revenue = float(exit_result["revenue"])
        cost = float(order.get("bet_cents") or 0) / 100.0
        pnl = revenue - cost
        
        # Update DB with terminal status 'sold_stop_loss'
        DB.update_user_order_status(
            order["email"],
            order["game_pk"],
            status="sold_stop_loss",
            profit_loss=pnl,
        )
        print(f"  [{user_label}] SUCCESS: Sold {fill_count} contracts for ${revenue:.2f} (PnL ${pnl:.2f})")
        return True, True
    except Exception as exc:
        print(f"  [{user_label}] ERROR exiting position for {order['kalshi_ticker']}: {exc}")
        return True, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and execute MLB stop-loss for all accounts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate triggers without placing exit orders.",
    )
    args = parser.parse_args()

    # DB.get_open_orders_with_live_data() already filters for active users with linked accounts.
    print("Fetching live positions across all active users...")
    orders = DB.get_open_orders_with_live_data()
    if not orders:
        print("No live positions needing stop-loss monitoring.")
        return

    checked = len(orders)
    triggered = 0
    executed = 0
    for order in orders:
        did_trigger, did_execute = _try_exit_order(order, dry_run=args.dry_run)
        triggered += int(did_trigger)
        executed += int(did_execute)

    print(
        "\n--- Stop-Loss Summary ---"
        f"\nTotal Checked: {checked}"
        f"\nTriggered:     {triggered}"
        f"\nExecuted:      {executed}"
    )


if __name__ == "__main__":
    main()
