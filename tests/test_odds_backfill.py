from datetime import date

import pandas as pd

import db
from fetch.fetch_data import SBR_NAME_TO_ABB, sbr_normalize
from scripts.backfill_sbr_odds import build_backfill_updates
from scripts.reset_2025_live_lines import build_updates as build_live_line_updates


def test_sbr_maps_athletics_duplicate_name():
    assert SBR_NAME_TO_ABB[sbr_normalize("Athletics Athletics")] == "ATH"


def test_upsert_odds_assignments_preserve_existing_snapshot_overwrites():
    assignments = db._build_upsert_assignments(
        [
            "game_pk",
            "open_home_ml",
            "open_away_ml",
            "close_home_ml",
            "close_away_ml",
            "home_implied_prob",
            "home_score",
            "away_score",
            "home_win",
            "odds_source",
            "extra",
        ]
    )

    assert any(
        "open_home_ml = CASE WHEN EXCLUDED.close_home_ml IS NOT NULL" in assignment
        and "EXCLUDED.odds_source IS NOT NULL" in assignment
        and "ELSE COALESCE(games.open_home_ml, EXCLUDED.open_home_ml)" in assignment
        for assignment in assignments
    )
    assert any(
        "close_home_ml = CASE WHEN EXCLUDED.close_home_ml IS NOT NULL" in assignment
        and "EXCLUDED.odds_source IS NOT NULL" in assignment
        and "ELSE COALESCE(games.close_home_ml, EXCLUDED.close_home_ml)" in assignment
        for assignment in assignments
    )
    assert any(
        "home_implied_prob = CASE WHEN" in assignment
        and "COALESCE(games.home_implied_prob, EXCLUDED.home_implied_prob)" in assignment
        for assignment in assignments
    )
    assert "extra = COALESCE(games.extra, '{}'::jsonb) || COALESCE(EXCLUDED.extra, '{}'::jsonb)" in assignments
    odds_source_assignment = next(a for a in assignments if a.startswith("odds_source = "))
    assert "WHEN EXCLUDED.odds_source IS NULL THEN games.odds_source" in odds_source_assignment
    assert "games.odds_source NOT LIKE 'odds-api:%'" in odds_source_assignment
    assert "EXCLUDED.odds_source LIKE 'odds-api:%' THEN games.odds_source" in odds_source_assignment
    assert any(
        "home_score = CASE WHEN COALESCE(EXCLUDED.extra->>'game_status', '') IN ('postponed', 'cancelled') "
        "THEN EXCLUDED.home_score ELSE COALESCE(EXCLUDED.home_score, games.home_score) END" == assignment
        for assignment in assignments
    )
    assert any(
        "away_score = CASE WHEN COALESCE(EXCLUDED.extra->>'game_status', '') IN ('postponed', 'cancelled') "
        "THEN EXCLUDED.away_score ELSE COALESCE(EXCLUDED.away_score, games.away_score) END" == assignment
        for assignment in assignments
    )
    assert any(
        "home_win = CASE WHEN COALESCE(EXCLUDED.extra->>'game_status', '') IN ('postponed', 'cancelled') "
        "THEN EXCLUDED.home_win ELSE COALESCE(EXCLUDED.home_win, games.home_win) END" == assignment
        for assignment in assignments
    )


def test_backfill_updates_only_valid_real_sbr_lines():
    candidates = [
        {
            "game_pk": 1,
            "game_date": date(2026, 4, 1),
            "home_team": "NYY",
            "away_team": "BOS",
        },
        {
            "game_pk": 2,
            "game_date": date(2026, 4, 1),
            "home_team": "LAD",
            "away_team": "SFG",
        },
        {
            "game_pk": 3,
            "game_date": date(2026, 4, 1),
            "home_team": "SEA",
            "away_team": "TEX",
        },
    ]
    sbr_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-01",
                "home_team": "NYY",
                "away_team": "BOS",
                "open_home_ml": -120,
                "open_away_ml": 110,
                "close_home_ml": -135,
                "close_away_ml": 125,
                "odds_source": "pinnacle",
            },
            {
                "game_date": "2026-04-01",
                "home_team": "LAD",
                "away_team": "SFG",
                "open_home_ml": -150,
                "open_away_ml": 140,
                "close_home_ml": -150,
                "close_away_ml": 140,
                "odds_source": "pinnacle",
            },
            {
                "game_date": "2026-04-01",
                "home_team": "SEA",
                "away_team": "TEX",
                "open_home_ml": -110,
                "open_away_ml": 100,
                "close_home_ml": -510,
                "close_away_ml": 420,
                "odds_source": "pinnacle",
            },
        ]
    )

    updates = build_backfill_updates(candidates, sbr_df)

    assert [update["game_pk"] for update in updates] == [1]
    assert updates[0]["close_home_ml"] == -135
    assert updates[0]["home_implied_prob"] is not None


def test_backfill_skips_ambiguous_doubleheader_keys():
    candidates = [
        {
            "game_pk": 1,
            "game_date": date(2026, 4, 1),
            "home_team": "NYY",
            "away_team": "BOS",
        },
        {
            "game_pk": 2,
            "game_date": date(2026, 4, 1),
            "home_team": "NYY",
            "away_team": "BOS",
        },
    ]
    sbr_df = pd.DataFrame(
        [
            {
                "game_date": "2026-04-01",
                "home_team": "NYY",
                "away_team": "BOS",
                "open_home_ml": -120,
                "open_away_ml": 110,
                "close_home_ml": -135,
                "close_away_ml": 125,
                "odds_source": "pinnacle",
            }
        ]
    )

    assert build_backfill_updates(candidates, sbr_df) == []


def test_2025_live_line_updates_reset_close_to_open():
    updates = build_live_line_updates(
        [
            {
                "game_pk": 776300,
                "open_home_ml": -120,
                "open_away_ml": 110,
                "close_home_ml": -740,
                "close_away_ml": 600,
            }
        ]
    )

    assert updates == [
        {
            "game_pk": 776300,
            "close_home_ml": -120,
            "close_away_ml": 110,
            "home_implied_prob": updates[0]["home_implied_prob"],
            "away_implied_prob": updates[0]["away_implied_prob"],
        }
    ]
    assert round(updates[0]["home_implied_prob"] + updates[0]["away_implied_prob"], 12) == 1
