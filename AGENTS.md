# Agent Guide — MLB Betting Pipeline

## Token efficiency

Shell commands auto-route through `rtk` via hook — transparent, no overhead.
For broad codebase exploration, use an Explore subagent to protect main context.

## Project layout

```
run_pipeline.py      # Orchestrator entry point
db.py                # All DB access (postgres via psycopg2)
bet/place_bet.py     # Kalshi bet placement
fetch/               # Data fetchers (stats, odds, weather, balance)
model/               # LightGBM/XGBoost training + prediction
dashboard/app.py     # FastAPI dashboard (deployed on app LXC, port <REDACTED_PORT>)
settle_games.py      # Post-game settlement
scripts/             # One-off ops scripts
tests/               # pytest suite
```

## Running things

```bash
uv run run_pipeline.py          # Full pipeline
uv run dashboard/app.py         # Dev server
uv run -m pytest tests/         # Tests
```

## Data rules

Never read CSV/Parquet directly. Use pandas with targeted queries only.

## Secrets

`.env` holds all keys. `kalshi-key.pem` is the Kalshi private key — never read or output it.
Check `.env.example` for variable names.

## Deploy

Push to GitHub → runner deploys automatically. No manual SSH needed for deploys.

## Git workflow

Commit frequently after tested, coherent checkpoints so deployable fixes do not sit uncommitted.

## SSH debugging

`homelab.py` connects **directly to each LXC** via SSH — no Proxmox/pct involved.

```bash
# App LXC — reads HOMELAB_HOST / HOMELAB_USER / HOMELAB_PASSWORD from .env
python3 homelab.py app "systemctl status mlb-dashboard --no-pager -l | tail -40"
python3 homelab.py app "journalctl -u mlb-dashboard -n 100 --no-pager"
python3 homelab.py app "curl -s http://localhost:<REDACTED_PORT>/health"

# DB LXC — reads HOMELAB_DB_SSH_HOST (or DB_HOST) + HOMELAB_DB_SSH_USER/PASSWORD
# Falls back to HOMELAB_USER / HOMELAB_PASSWORD if no DB-specific SSH creds set
python3 homelab.py db "pg_lsclusters"

# For psql: DB creds live in .env (DB_USER, DB_PASSWORD, DB_NAME).
# Load them locally and inject into the remote command string, e.g.:
#   from dotenv import load_dotenv; load_dotenv(); import os
#   pw = os.getenv("DB_PASSWORD"); user = os.getenv("DB_USER"); db = os.getenv("DB_NAME")
#   run: python3 homelab.py db f"PGPASSWORD={pw} psql -U {user} -d {db} -h <REDACTED_IP> -c '\\dt'"
```

Set these in `.env` for DB SSH if credentials differ from app:
- `HOMELAB_DB_SSH_HOST` (falls back to `DB_HOST`)
- `HOMELAB_DB_SSH_USER`, `HOMELAB_DB_SSH_PASSWORD`, `HOMELAB_DB_SSH_PORT`

## Conventions

- Python 3.10+, type hints, short functions
- All datetimes UTC unless suffixed `_et`
- argparse for CLI, pandas for I/O, pathlib for paths
