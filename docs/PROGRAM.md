# MLB Automated Betting Pipeline — Orchestration Guide

This document describes how the pipeline runs today.

## Secrets / hygiene

- Do not put secrets, hostnames, or IP addresses in docs.
- Infrastructure secrets live in `.env` (DB creds, encryption key, etc.).
- Per-user Kalshi account configuration is stored in Postgres and can be encrypted at rest.

## Primary entry point

Run the orchestrator:

```bash
uv run run_pipeline.py
```

One-off modes:

```bash
uv run run_pipeline.py --game_pk <game_pk>   # run immediately for a specific game
uv run run_pipeline.py --now                 # run immediately for the next upcoming game
```

## Overview

The orchestrator continuously:

1. Identifies upcoming games that need predictions.
2. Runs the data refresh (MLB schedule/odds/stats).
3. Fetches balances (Kalshi) for configured user accounts.
4. Runs model inference.
5. Places bets for approved users (when a bet is indicated).
6. Periodically refreshes live positions and scores.

Implementation notes (from `run_pipeline.py`):

- The pipeline triggers a fixed lead time before first pitch (`LEAD_MINUTES`).
- Games sharing the same scheduled start minute are treated as a batch:
  - data refresh runs once per batch
  - per-game prediction + bet placement can run in parallel

## Data storage

The pipeline is **DB-first**.

- `fetch/fetch_data.py` upserts the season’s game rows into the `games` table.
- `model/predict.py` writes predictions/sizing back into the DB.
- `bet/place_bet.py` records orders into `user_orders`.
- `fetch/fetch_balance.py` records balances into `user_balance`.

A local CSV (`data/mlb_<season>.csv`) exists as a fallback cache if the DB is temporarily unavailable.

## Related docs

- MLB data ingest: `docs/fetch_mlb_data.md`
- Model inference: `docs/run_model.md`
- Kalshi integration: `docs/kalshi.md`
- Homelab SSH helper: `docs/homelab_access.md`
