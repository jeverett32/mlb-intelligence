# Run Model

Trains the model on all historical data and predicts the edge + Kelly bet size for the target game. Writes results to `data/games.csv`.

## Script

`2_run_model/predict.py`

This script replaces the walk-forward validation in `train.py` for live use. `train.py` remains the tool for model development and hyperparameter tuning. `predict.py` is the production entry point.

## Command

Run from the project root. Identify the game by its `game_pk`:
```bash
uv run 2_run_model/predict.py --game_pk 12345
```

Or by date and teams if `game_pk` is unknown:
```bash
uv run 2_run_model/predict.py --game_date 2026-04-01 --home_team NYY --away_team BOS
```

The `game_pk` is preferred — it uniquely identifies the game and is available in `data/mlb_2026.csv` and `data/games.csv` after the fetch step.

## What it does

1. Loads `data/master_mlb.csv` (historical training data) + `data/mlb_2026.csv` (current season)
2. Combines them and runs the full feature engineering pipeline (same 99+ features as `train.py`)
3. Trains the configured model (`MODEL = "lr"` by default in `train.py`) on all historical rows that have a known outcome (`home_win` is not null)
4. Predicts the probability of the **home team winning** for the target game
5. Calculates edge: `model_prob - market_implied_prob` (closing line, devigged)
6. If edge exceeds `CONFIDENCE_THRESHOLD` (default: 0.14), sizes a fractional Kelly bet
7. Writes to `data/games.csv`: `predicted_prob`, `edge`, `bet_side`, `bet_frac`, `market_implied_prob`

## Output in data/games.csv

| Column | Description |
|---|---|
| `predicted_prob` | Model's probability of home team winning (0–1) |
| `market_implied_prob` | Closing market probability (devigged) |
| `edge` | `abs(model_prob - market_prob)` — only meaningful when >= threshold |
| `bet_side` | `"home"`, `"away"`, or `"none"` |
| `bet_frac` | Fraction of bankroll to bet (0 = no bet) |

## Early-season behavior

When either team has played fewer than 25 games (`EARLY_CUTOFF`), the model routes to a simpler early-season specialist (logistic regression on pitcher + market features only). Rolling team stats are unreliable this early in the season. Kelly stake is also halved (`WARMUP_KELLY_MULT = 0.5`).

## Model configuration

All model parameters live in `train.py`:
- `MODEL` — which model type (`"lr"`, `"lgb"`, `"xgb"`, `"mlp"`, `"ensemble_avg"`, `"ensemble_stack"`)
- `CONFIDENCE_THRESHOLD` — minimum edge to place a bet (default `0.14`)
- `KELLY_FRACTION` — fractional Kelly multiplier (default `0.25`)
- `PROB_CAP` — clips model probability before edge calc to `(0.34, 0.66)` to prevent overconfidence
- `MAX_BET_FRAC` — hard cap on any single bet as fraction of bankroll (default `0.25`)

## Prerequisites

- `data/games.csv` must have a row for the target game (created in Step 1)
- `data/mlb_2026.csv` must exist and contain the target game with odds populated (run Step 2 first)
- `data/master_mlb.csv` must exist in the project root

## No-bet conditions

The script always runs. If it outputs `NO BET`, it means:
- Edge is below threshold, OR
- Market odds are missing (odds scraped as NaN)

In both cases `bet_frac = 0` and `bet_side = "none"` in `data/games.csv`. Proceed to Step 4 which will read these values and skip the bet.
