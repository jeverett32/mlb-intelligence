import os

import pytest
import pandas as pd
import numpy as np
import db as DB


def _requires_db_env() -> None:
    missing = [n for n in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD") if not os.environ.get(n)]
    if missing:
        pytest.skip("DB env missing: " + ", ".join(missing))


def test_games_v2_roundtrip(monkeypatch):
    _requires_db_env()
    """Verify bulk_upsert_games_v2 and load_games_v2_frame work as expected."""
    # Ensure tables exist
    DB.init_games_v2()
    
    test_pk = 999999999
    test_rows = [
        {
            'game_pk': test_pk,
            'game_date': '2026-05-10',
            'season': 2026,
            'home_team': 'TEST_H',
            'away_team': 'TEST_A',
            'home_win': True,
            'close_home_ml': -150.0,
            'close_away_ml': 130.0,
            'market_implied_prob': 0.6,
            'features': {
                'feat_1': 1.23,
                'feat_2': 'abc',
                'feat_3': None
            }
        }
    ]
    
    # 1. Bulk Upsert
    DB.bulk_upsert_games_v2(test_rows)
    
    # 2. Load
    df = DB.load_games_v2_frame()
    
    # 3. Verify
    row = df[df['game_pk'] == test_pk]
    assert not row.empty
    assert row.iloc[0]['home_team'] == 'TEST_H'
    assert row.iloc[0]['feat_1'] == 1.23
    assert row.iloc[0]['feat_2'] == 'abc'
    
    # Cleanup (manual delete to not affect other tests if DB is shared)
    with DB.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM games_v2 WHERE game_pk = %s", (test_pk,))
        conn.commit()

