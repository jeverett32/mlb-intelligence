# MLB Automated Betting Pipeline — Orchestration Guide

This document is the master instruction set for the automated agent. Follow steps 1–5 in order for every game run.

---

## Running the pipeline

Start the orchestrator once and it handles everything automatically:
```bash
uv run run_pipeline.py
```

To run immediately for a specific game (testing / manual override):
```bash
uv run run_pipeline.py --game_pk 12345
uv run run_pipeline.py --now
```

## Overview

The orchestrator runs automatically **5 minutes before each scheduled MLB game**. Each run is scoped to one specific game. The run order is driven by `data/mlb_2026.csv`.

**Key files (all in `data/`):**
| File | Purpose |
|---|---|
| `data/mlb_2026.csv` | Current season game data + features (written by fetch step) |
| `data/master_mlb.csv` | Full historical data used to train the model — **you supply this** |
| `data/games.csv` | One row per game the system has run on; tracks prediction, bet, result |
| `data/balance.csv` | Kalshi account balance history (one row per run) |
| `data/results.tsv` | Walk-forward validation results log (written by train.py) |

---

## Step 1 — Initialize the Game Row

Before anything else, add a row to `data/games.csv` for the upcoming game.

`data/games.csv` already exists with headers. If for any reason it is missing, recreate it with these columns:
```
game_pk,game_date,home_team,away_team,predicted_prob,edge,bet_side,bet_frac,market_implied_prob,kalshi_order_id,bet_dollars,n_contracts,result
```

Find the next game from `data/mlb_2026.csv` whose `game_date` (UTC) is within the next 10 minutes. Add a row for it with `game_pk`, `game_date`, `home_team`, `away_team` filled in; leave all other columns blank.

---

## Step 2 — Fetch Data

**Instruction file:** `1_fetch_data/FETCH_MLB_DATA.md`

Run the data pipeline to refresh `data/mlb_2026.csv`. This fetches:
- Schedule and scores from the MLB Stats API
- Moneyline and total odds from Sportsbook Review
- Pitcher stats and weather
- FanGraphs team batting/pitching stats

Also fetch the current Kalshi account balance and append to `data/balance.csv`.

**Instruction file:** `1_fetch_data/FETCH_KALSHI_DATA.md`

---

## Step 3 — Run Model

**Instruction file:** `2_run_model/RUN_MODEL.md`

Run `predict.py` for the target game. This trains the model on all historical data, predicts the probability of the home team winning, calculates edge vs. the market, and writes the Kelly bet fraction to `data/games.csv`.

---

## Step 4 — Place Bet

**Instruction file:** `3_place_bet/PLACE_BET.md`

Read `data/games.csv` for the current game. If `bet_frac > 0`, find the game on Kalshi and place the bet.

---

## Step 5 — Schedule Next Run

Determine the next upcoming game from `data/mlb_2026.csv` (first row with a future `game_date` that has not yet been initialized in `data/games.csv`). Schedule the pipeline to run again 5 minutes before that game's UTC start time.

---

## Notes

- All times in `data/mlb_2026.csv` are **UTC**.
- A game is considered "complete" when `home_score`, `away_score`, and `home_win` are populated.
- After a completed game, update the `result` column in `data/games.csv` (`win` / `loss` / `push`) based on the bet side and final score.
- Never skip the fetch step — odds close to game time are the most accurate and the model depends on closing line data.
