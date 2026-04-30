# Agent Guide — MLB Betting Pipeline

Canonical human-facing overview: **README.md**. Deeper system notes: **docs/pipeline.md**.

## Agent config files (local-only)

This repo uses local, gitignored agent/editor config files:

- `/.clauderules` — primary agent operating rules
- `/.claudeignore` — files to exclude from agent context
- `/.cursor/rules/mlb-pipeline.mdc` — Cursor project rules
- `/.cursorignore` — files to exclude from Cursor context

These are intentionally **not committed**.

If you clone this repo and they don’t exist yet, initialize them from templates:

```bash
python3 scripts/init_agent_files.py
```

## Token efficiency

- Shell commands auto-route through `rtk` via hook — transparent, no overhead.
- Prefer `rg`/`find` to locate code rather than reading many files.
- For broad exploration, use an Explore subagent to protect main context.

## Project layout

```text
run_pipeline.py      # Orchestrator entry point
config.py            # Runtime constants / season config
db.py                # All DB access (Postgres via psycopg2)

bet/place_bet.py     # Kalshi bet placement
fetch/               # Data fetchers (stats, odds, weather, balance)
model/               # Train + predict (LR/LightGBM/XGBoost/MLP/ensemble)

dashboard/app.py     # FastAPI dashboard backend
settle_games.py      # Post-game settlement
scripts/             # One-off ops scripts
tests/               # pytest suite
```

## Setup

```bash
uv sync
```

Copy `.env.example` → `.env` and fill values (never commit `.env`).

## Running things (from repo root)

```bash
uv run run_pipeline.py

# Dashboard (dev)
uv run uvicorn dashboard.app:app --reload --host <bind_addr> --port <port>

# Tests
uv run pytest -q tests/
```

## Data rules

- **Source of truth is Postgres**. Prefer DB reads via `db.py`.
- Treat files under `data/` (CSV/Parquet/TSV) as caches/backups and **possibly stale**.
- Avoid opening large `*.csv` / `*.parquet` with file-read tools.

Example (targeted DB query):
```python
import pandas as pd
import db

with db.pooled_connection() as conn:
    df = pd.read_sql_query(
        """
        SELECT game_id, game_date, home_team, away_team
        FROM games
        ORDER BY game_date DESC
        LIMIT 50
        """,
        conn,
    )

print(df.head())
```

## Secrets

- `.env` holds all keys — never read/print/log it.
- `kalshi-key.pem` is the Kalshi private key — never read or output it.
- Use `.env.example` for variable names.

## Deploy

Push to GitHub → runner deploys automatically. No manual SSH needed for deploys.

## SSH debugging

`homelab.py` connects **directly to each LXC** via SSH — no Proxmox/pct involved.

```bash
# App LXC — reads HOMELAB_HOST / HOMELAB_USER / HOMELAB_PASSWORD from .env
python3 homelab.py app "systemctl status mlb-dashboard --no-pager -l | tail -40"
python3 homelab.py app "journalctl -u mlb-dashboard -n 100 --no-pager"
python3 homelab.py app "curl -s http://localhost:<port>/health"

# DB LXC — reads HOMELAB_DB_SSH_HOST (or DB_HOST) + HOMELAB_DB_SSH_USER/PASSWORD
# Falls back to HOMELAB_USER / HOMELAB_PASSWORD if no DB-specific SSH creds set
python3 homelab.py db "pg_lsclusters"

# For psql: DB creds live in .env (DB_USER, DB_PASSWORD, DB_NAME).
# Load them locally and inject into the remote command string, e.g.:
#   from dotenv import load_dotenv; load_dotenv(); import os
#   pw = os.getenv("DB_PASSWORD"); user = os.getenv("DB_USER"); db = os.getenv("DB_NAME")
#   run: python3 homelab.py db f"PGPASSWORD={pw} psql -U {user} -d {db} -h localhost -c '\\dt'"
```

Set these in `.env` for DB SSH if credentials differ from app:
- `HOMELAB_DB_SSH_HOST` (falls back to `DB_HOST`)
- `HOMELAB_DB_SSH_USER`, `HOMELAB_DB_SSH_PASSWORD`, `HOMELAB_DB_SSH_PORT`

## Repo hygiene

Local agent/editor state is intentionally untracked and gitignored:
- `/.claude/`, `/.cursor/`, `/.claudeignore`, `/.cursorignore`, `/.clauderules`

## Conventions

- Python 3.10+, type hints, short functions
- All datetimes UTC unless suffixed `_et`
- `argparse` for CLI, pandas for I/O, `pathlib.Path` for paths
- Keep DB access inside `db.py`
