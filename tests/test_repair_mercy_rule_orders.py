from bet.place_bet import _order_fill_revenue
from scripts.repair_mercy_rule_orders import _pnl_from_kalshi_orders


def test_order_fill_revenue_uses_yes_price_not_complement_on_sell():
    order = {
        "action": "sell",
        "side": "yes",
        "fill_count_fp": "1.00",
        "yes_price_dollars": "0.0200",
        "taker_fill_cost_dollars": "0.980000",
    }
    assert _order_fill_revenue(order) == 0.02


def test_pnl_from_kalshi_orders_uses_sell_minus_buy_totals():
    orders = [
        {
            "action": "buy",
            "side": "yes",
            "fill_count": 1,
            "taker_fill_cost_dollars": "0.43",
        },
        {
            "action": "sell",
            "side": "yes",
            "fill_count": 1,
            "yes_price_dollars": "0.0200",
            "taker_fill_cost_dollars": "0.980000",
        },
    ]
    pnl_cents, source = _pnl_from_kalshi_orders(orders)
    assert source == "kalshi_fills"
    assert pnl_cents == -41


def test_pnl_from_kalshi_orders_supports_multi_contract_rows():
    orders = [
        {
            "action": "buy",
            "side": "yes",
            "fill_count": 2,
            "taker_fill_cost_dollars": "1.32",
        },
        {
            "action": "sell",
            "side": "yes",
            "fill_count": 2,
            "yes_price_dollars": "0.0400",
            "taker_fill_cost_dollars": "1.880000",
        },
    ]
    pnl_cents, source = _pnl_from_kalshi_orders(orders)
    assert source == "kalshi_fills"
    assert pnl_cents == -124


def test_pnl_from_kalshi_orders_requires_sell_fill():
    orders = [
        {
            "action": "buy",
            "fill_count": 1,
            "taker_fill_cost_dollars": "0.43",
        }
    ]
    pnl_cents, source = _pnl_from_kalshi_orders(orders)
    assert pnl_cents is None
    assert source == "missing sell or buy fill on Kalshi"
