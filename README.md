<p align="center">
  <img src="dashboard/static/favicon.svg" width="72" height="72" alt="MLB Pipeline logo">
</p>

<h1 align="center">MLB Pipeline</h1>

<p align="center">A public-facing MLB betting intelligence site with transparent analytics, private operator controls, and live model tracking.</p>

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

<p align="center">
  <img src="docs/images/dashboard-overview.png" alt="MLB Pipeline private dashboard overview" width="100%">
</p>

MLB Pipeline is a deployed web product with two distinct surfaces: a public site that explains the model and publishes performance, and a private dashboard used to operate the system behind it. Visitors can review the public record first; approved users can then access live controls, bankroll monitoring, and execution workflows.

> [!WARNING]
> This software can be connected to real-money betting infrastructure. Live execution should stay disabled unless the operator has reviewed the model behavior, bankroll rules, and account configuration.

## Product Surfaces

### Public landing page

The root site is the front door. It explains what the system is, how the workflow works, and where to go next.

- Presents the product in plain language instead of dropping users into raw internal tooling
- Shows live headline metrics pulled from the same public endpoints as the analytics page
- Directs visitors to either explore the public record or request private access

### Public analytics

The `/public` surface is the proof layer.

- Publishes model accuracy, market comparison, calibration, and betting performance
- Separates evidence from marketing by focusing on live numbers and historical results
- Lets users inspect whether the model has actually outperformed market baselines

### Private dashboard

The authenticated dashboard is the operator workspace shown in the screenshot above.

- Monitor bankroll, open bets, next actions, and current system state
- Review model-vs-market deltas, exposure, and live betting status
- Manage private settings and operational controls without exposing them on the public site

## User Journey

For a normal user, the product flow is straightforward:

1. Land on the homepage and understand what the system does.
2. Open the public analytics page to inspect performance, calibration, and results.
3. Decide whether the public record is credible enough to keep following.
4. Request access if private dashboard features are relevant.
5. Use the authenticated dashboard only after approval.

That is the framing the README should emphasize, because that is how someone actually experiences the deployed website.

## What The Site Shows

- **Public trust layer:** landing page, methodology framing, live summary metrics, public analytics
- **Proof layer:** model accuracy, market accuracy, calibration, ROI, and public operating history
- **Operator layer:** bankroll snapshots, actionable signals, open positions, timezone-aware dashboard views, and live execution state

## How The System Works

Behind the website, the application runs a model-driven MLB betting workflow:

```text
Data collection -> feature engineering -> win probability model -> market comparison -> bet sizing -> dashboard + public reporting
```

More concretely:

- `fetch/` gathers schedules, odds, weather, pitcher data, and team statistics
- `model/` produces predictions and compares them to market-implied pricing
- `bet/` handles execution and dry-run behavior
- `dashboard/` serves the public site, analytics pages, auth flows, and private operator UI

## Why This Repo Exists

This repository is not just an internal pipeline dump. It is the code behind a product that tries to make a betting model legible:

- the public site explains the system,
- the analytics page shows the evidence,
- the private dashboard runs the operation.

That split matters. Most projects in this space either hide the process entirely or publish picks without giving users a way to inspect the track record. This repo is built to expose the record clearly while keeping execution controls private.

## Development

This repository is still a Python application with a FastAPI frontend and PostgreSQL-backed state. If you are working on the codebase itself, the main local commands are:

```bash
uv sync
uv run pytest -q tests/
uv run uvicorn dashboard.app:app --reload --host <REDACTED_IP> --port <REDACTED_PORT>
uv run python run_pipeline.py
```

For environment variables and deployment details, start with [.env.example](/home/everjohn/projects/mlb-pipeline/.env.example) and the documents in [docs](/home/everjohn/projects/mlb-pipeline/docs).

## Project Structure

```text
mlb-pipeline/
├── bet/                Kalshi order placement and execution logic
├── dashboard/          Public site, auth flows, private dashboard, static assets
├── data/               Historical MLB data and cache artifacts
├── docs/               Operational and module-specific documentation
├── fetch/              External data ingestion
├── model/              Training and prediction logic
├── db.py               PostgreSQL access helpers
├── kalshi_client.py    Kalshi API authentication and requests
├── run_pipeline.py     Main orchestration loop
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
