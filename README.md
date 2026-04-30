<p align="center">
  <img src="dashboard/static/favicon.svg" width="72" height="72" alt="MLB Pipeline logo">
</p>

<h1 align="center">MLB Pipeline</h1>

<p align="center">MLB betting intelligence system with ML-driven predictions, live execution on Kalshi, and transparent performance analytics.</p>

<p align="center">
  <a href="https://github.com/jeverett32/mlb-pipeline/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/jeverett32/mlb-pipeline/test.yml?branch=main&label=tests" alt="Tests">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/license-BUSL--1.1-4b5563" alt="BUSL-1.1">
  </a>
  <a href="https://github.com/jeverett32/mlb-pipeline">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  </a>
</p>

<p align="center">
  <img src="docs/images/dashboard-overview.png" alt="MLB Pipeline dashboard" width="100%">
</p>

## Problem

Sports betting (especially MLB) has a few persistent problems:

- **Too much data, too little time** — schedule context, pitcher changes, weather, team form, and live odds all move quickly.
- **Execution is the hard part** — even good models fail if betting decisions aren’t consistent, timed correctly, and recorded.
- **Post-hoc analysis is usually missing** — without clean history, you can’t answer “did this strategy work?” with confidence.

## Action

MLB Pipeline turns the day-to-day work into a repeatable system:

1. **Ingest** schedules, odds, weather, pitcher stats, and team stats
2. **Engineer features** (80+ game-level features)
3. **Predict** win probabilities (LR default; optional LightGBM/XGBoost/MLP/ensemble)
4. **Compare to the market** to identify edge
5. **Execute** on Kalshi (live or dry-run)
6. **Settle + audit** results and ROI in a single database

## Solution

MLB Pipeline is an end-to-end stack for MLB trading:

- **One orchestrator** that runs the full workflow (`run_pipeline.py`)
- **A dashboard** (FastAPI) for transparent performance analytics + operator controls
- **A Postgres-backed history** for bets, balances, orders, and model artifacts
- **A public modeling snapshot** (`data/master_mlb.csv`) that can be updated periodically (not continuously)

## What’s inside

- **Dashboard** — FastAPI app serving public analytics + private operator controls
- **Pipeline** — runs ahead of scheduled first pitch; predicts and places bets in parallel
- **Model** — calibrated classifiers + walk-forward validation; Kelly stake sizing
- **DB** — PostgreSQL source of truth for games, bets, and run history

Key components:

```text
run_pipeline.py       # Main orchestrator
fetch/                # Data ingestion — odds, weather, stats
model/train.py        # Model training + walk-forward evaluation
model/predict.py      # Inference — win probabilities + sizing
bet/place_bet.py      # Kalshi execution
settle_games.py       # Post-game settlement
dashboard/app.py      # Dashboard API
homelab.py            # SSH helper for app/db LXCs
```

## Data model philosophy

- **Postgres is the source of truth.**
- Files under `data/` are primarily **caches/backups** and may be stale.
- `data/master_mlb.csv` is intended as a **public snapshot** of modeling data and can be refreshed on a controlled cadence.

## Docs

- [Pipeline (end-to-end)](./docs/pipeline.md)
- [Homelab SSH helper](./docs/homelab_access.md)

## Contributing / local development

See [CONTRIBUTING.md](./CONTRIBUTING.md) for local setup, testing, and dev commands.

## Deployment

Push to GitHub → GitHub Actions runner auto-deploys to homelab LXC.

SSH debugging via [`homelab.py`](./homelab.py):

```bash
python3 homelab.py app "systemctl status mlb-dashboard --no-pager -l | tail -40"
python3 homelab.py app "journalctl -u mlb-dashboard -n 100 --no-pager"
```

## Data sources

- [MLB Stats API](https://statsapi.mlb.com)
- [The Odds API](https://the-odds-api.com)
- [Kalshi](https://kalshi.com)
- [Open-Meteo](https://open-meteo.com)
- [FanGraphs](https://www.fangraphs.com)

## License

BUSL-1.1 (Business Source License 1.1)

- See [LICENSE](./LICENSE)
- See [LICENSE-FAQ.md](./LICENSE-FAQ.md)
