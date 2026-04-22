<p align="center">
  <img src="dashboard/static/favicon.svg" width="72" height="72" alt="MLB Pipeline logo">
</p>

<h1 align="center">MLB Pipeline</h1>

<p align="center">Automated MLB moneyline modeling, EV detection, Kalshi execution, and dashboard monitoring.</p>

<p align="center">
  <a href="https://github.com/jeverett32/mlb-pipeline/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/jeverett32/mlb-pipeline/test.yml?branch=main&label=tests" alt="Tests">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/license-AGPL--3.0-4b5563" alt="AGPL-3.0">
  </a>
  <a href="https://github.com/jeverett32/mlb-pipeline">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  </a>
</p>

MLB Pipeline is a Python-based betting workflow that pulls MLB game and market data, trains a gradient-boosted model, identifies positive expected value opportunities, and can place or dry-run Kalshi orders. It also exposes a FastAPI dashboard for monitoring bankroll, open positions, model performance, and execution state.

> [!WARNING]
> This project can place real-money trades when live betting is enabled. Review the configuration, risk logic, and execution safeguards before using it against a funded Kalshi account.

## At a glance

- **Automated data pipeline:** Fetches schedules, odds, pitcher data, weather, and team statistics.
- **Model-driven betting:** Scores games with gradient-boosted models and compares predictions against market prices.
- **Execution controls:** Supports dry runs, live execution toggles, stake sizing, and balance sync.
- **Operational dashboard:** FastAPI UI for bankroll snapshots, open bets, action queues, and performance review.
- **Self-hosted stack:** Runs on your own infrastructure with PostgreSQL and standard Python tooling.

## Dashboard

Drop your dashboard screenshot at `docs/images/dashboard-overview.png` to render it here:

```md
<p align="center">
  <img src="docs/images/dashboard-overview.png" alt="MLB Pipeline dashboard overview" width="100%">
</p>
```

The current UI includes operational status cards, live betting state, bankroll history, open positions, and model-versus-market tracking.

## Quick Start

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- PostgreSQL
- A [Kalshi](https://kalshi.com) account with API credentials
- An [Odds API](https://the-odds-api.com) key

### Install

```bash
git clone https://github.com/jeverett32/mlb-pipeline.git
cd mlb-pipeline
uv sync
cp .env.example .env
```

Update `.env` with your Kalshi credentials, Odds API key, and PostgreSQL connection details.

### Load historical data

```bash
uv run migrate_to_postgres.py
```

This imports the historical dataset in `data/master_mlb.csv` so the model has baseline training data available in PostgreSQL.

### Run the main surfaces

```bash
# dashboard API + UI
uv run uvicorn dashboard.app:app --host 0.0.0.0 --port 8080

# pipeline loop
uv run python run_pipeline.py
```

Open `http://localhost:8080` to access the dashboard.

## How It Works

```text
fetch/fetch_data.py    -> collect schedules, odds, weather, pitcher, and team data
model/predict.py       -> score games and compute edge versus market price
bet/place_bet.py       -> place or dry-run Kalshi orders
fetch/fetch_balance.py -> sync account balance
run_pipeline.py        -> orchestrate the loop around upcoming games
```

The pipeline stores games, bets, balances, settings, and dashboard state in PostgreSQL. In production, it is typically run as a long-lived service that wakes ahead of scheduled games, refreshes the latest inputs, and evaluates whether any wager meets the execution thresholds.

## Architecture

The repo is organized around four main concerns:

- `fetch/`: data ingestion from MLB Stats API, odds providers, weather, and team stat sources
- `model/`: training and prediction logic for win probabilities and edge calculations
- `bet/`: Kalshi order construction and execution
- `dashboard/`: FastAPI routes, auth, templates, and static assets for monitoring

The current production deployment uses a separate PostgreSQL host and app server managed on a homelab. That operational setup is documented outside the main README so the primary onboarding path stays focused on the software itself.

## Configuration

Core environment variables:

| Variable | Purpose |
|---|---|
| `KALSHI_KEY_ID` | Kalshi API key ID |
| `KALSHI_KEY_PATH` | Path to the Kalshi PEM private key |
| `KALSHI_ENV` | `prod` for live trading or `demo` for paper trading |
| `ODDS_API_KEY` | API key for The Odds API fallback feed |
| `DB_HOST` | PostgreSQL host |
| `DB_PORT` | PostgreSQL port |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `MLB_ALERT_WEBHOOK` | Optional Discord webhook for failures and alerts |
| `MLB_COOKIE_SECURE` | Set to `1` when serving the dashboard over HTTPS |

See [.env.example](/home/everjohn/projects/mlb-pipeline/.env.example) for the full set of supported variables.

## Development

Common commands:

```bash
# run tests
uv run pytest -q tests/

# syntax check
uv run python -m compileall -q bet dashboard fetch model db.py notify.py run_pipeline.py settle_games.py

# dashboard
uv run uvicorn dashboard.app:app --reload --host 0.0.0.0 --port 8080

# pipeline
uv run python run_pipeline.py
```

CI currently runs the test workflow in [.github/workflows/test.yml](/home/everjohn/projects/mlb-pipeline/.github/workflows/test.yml), and the repo also contains a homelab deployment workflow in [.github/workflows](/home/everjohn/projects/mlb-pipeline/.github/workflows).

## Project Structure

```text
mlb-pipeline/
├── bet/                Kalshi order placement
├── dashboard/          FastAPI app, templates, and static assets
├── data/               Historical MLB dataset and cache artifacts
├── docs/               Operational and module-specific documentation
├── fetch/              External data ingestion
├── model/              Training and prediction logic
├── db.py               PostgreSQL access helpers
├── kalshi_client.py    Kalshi API authentication and requests
├── migrate_to_postgres.py
├── run_pipeline.py
└── tests/
```

## Data Sources

- [MLB Stats API](https://statsapi.mlb.com)
- [Sportsbook Review](https://www.sportsbookreview.com)
- [The Odds API](https://the-odds-api.com)
- [Open-Meteo](https://open-meteo.com)
- [FanGraphs](https://www.fangraphs.com)
- [Kalshi](https://kalshi.com)

## Docs

- [Program overview](/home/everjohn/projects/mlb-pipeline/docs/PROGRAM.md)
- [Model workflow](/home/everjohn/projects/mlb-pipeline/docs/run_model.md)
- [MLB data fetch notes](/home/everjohn/projects/mlb-pipeline/docs/fetch_mlb_data.md)
- [Kalshi data fetch notes](/home/everjohn/projects/mlb-pipeline/docs/fetch_kalshi_data.md)
- [Bet placement notes](/home/everjohn/projects/mlb-pipeline/docs/place_bet.md)
- [Homelab access](/home/everjohn/projects/mlb-pipeline/docs/homelab_access.md)
- [Auto-deploy workflow](/home/everjohn/projects/mlb-pipeline/docs/auto_deploy.md)

## License

AGPL-3.0
