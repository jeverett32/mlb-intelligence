# MLB Betting Pipeline

This file is for the coding agent.

Primary operating rules live in **`.clauderules`** (local-only / gitignored).
If this is a fresh clone and the agent files don’t exist yet, initialize them from templates:

```bash
python3 scripts/init_agent_files.py
```

High-level guide: **AGENTS.md**.

## What this repo does

Automated MLB betting pipeline:
- fetch data → build features → train/predict win probabilities
- compare to market odds to find edge
- place bets on Kalshi (or dry-run)
- settle games and track ROI in Postgres

Managed with `uv`.

## Fast commands (run from repo root)

```bash
uv sync

# Full pipeline
uv run run_pipeline.py

# Dashboard (dev)
uv run dashboard/app.py --reload --host <bind_addr> --port <port>
# (Equivalent: uv run uvicorn dashboard.app:app --reload ...)

# Tests
uv run pytest -q tests/
```

For additional conventions and guardrails, see `.clauderules`.

## Token efficiency

Shell commands auto-route through `rtk` via hook. Prefer `rg`/`find` over opening lots of files.

## Data handling (avoid huge reads)

- **Prefer DB reads via `db.py`** — Postgres is the source of truth.
- Treat `data/*.csv` / `data/*.parquet` as caches/backups and **possibly stale**.
- Avoid opening large data files with file-read tools.

Example (targeted DB query):
```python
import pandas as pd
import db

with db.pooled_connection() as conn:
    df = pd.read_sql_query(
        "SELECT game_id, game_date, home_team, away_team FROM games ORDER BY game_date DESC LIMIT 50",
        conn,
    )

print(df.head())
```

## Secrets

- Never read/print/log `.env`.
- Never read/output any `*.pem` (especially `kalshi-key.pem`).
- Use `.env.example` for variable names.

## Key entry points

| Path | Purpose |
|------|---------|
| `run_pipeline.py` | Orchestrates end-to-end pipeline |
| `db.py` | All Postgres access |
| `fetch/` | Schedules, odds, weather, stats, balances |
| `model/` | Training + inference |
| `bet/place_bet.py` | Kalshi execution |
| `dashboard/app.py` | FastAPI dashboard |
| `settle_games.py` | Post-game settlement |
| `homelab.py` | SSH helper for app/db LXCs |
