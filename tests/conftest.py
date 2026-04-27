import pandas as pd
import pytest

import db
from fastapi.testclient import TestClient


def _empty_bets() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["result", "bet_dollars", "bet_side", "market_implied_prob"]
    )


def _empty_model_picks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["predicted_prob", "home_win", "market_implied_prob"]
    )


db.init_auth_tables = lambda: None
db.purge_expired_sessions = lambda: 0
db.get_all_bets = _empty_bets
db.get_all_user_orders = _empty_bets
db.get_user_orders = lambda email: _empty_bets()
db.get_all_paper_orders = _empty_bets
db.get_paper_orders = lambda email: _empty_bets()
db.get_paper_bankroll_dollars = lambda email: 10000.0
db.get_model_picks = _empty_model_picks
db.get_setting = lambda key, default="": default
db.PAPER_STARTING_BANKROLL_DOLLARS = 10000.0

import dashboard.app as dashboard_app


@pytest.fixture
def app_module():
    return dashboard_app


@pytest.fixture
def client():
    with TestClient(dashboard_app.app) as test_client:
        test_client.cookies.clear()
        yield test_client
        test_client.cookies.clear()
