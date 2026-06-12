from scripts.repair_mercy_rule_orders import _pnl_from_kalshi_orders


def test_pnl_from_kalshi_orders_uses_sell_minus_buy_totals():
    orders = [
        {
            "action": "buy",
            "fill_count": 1,
            "taker_fill_cost_dollars": "0.43",
        },
        {
            "action": "sell",
            "fill_count": 1,
            "taker_fill_cost_dollars": "0.98",
        },
    ]
    pnl_cents, source = _pnl_from_kalshi_orders(orders)
    assert source == "kalshi_fills"
    assert pnl_cents == 55


def test_pnl_from_kalshi_orders_supports_multi_contract_rows():
    orders = [
        {
            "action": "buy",
            "fill_count": 3,
            "taker_fill_cost_dollars": "1.56",
        },
        {
            "action": "sell",
            "fill_count": 3,
            "taker_fill_cost_dollars": "2.88",
        },
    ]
    pnl_cents, source = _pnl_from_kalshi_orders(orders)
    assert source == "kalshi_fills"
    assert pnl_cents == 132


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
