# MLB Pipeline (End-to-End)

This is the single, end-to-end operational doc for the MLB betting pipeline.

## Security / hygiene

- Do not put secrets, hostnames, or IP addresses in docs.
- Store infrastructure secrets in `.env` (loaded via `python-dotenv`).
- Store per-user Kalshi account configuration in Postgres. Enable encryption-at-rest for credential fields.

## Components

- Orchestrator: `run_pipeline.py`
- MLB ingest: `fetch/fetch_data.py`
- Model inference: `model/predict.py`
- Bet execution: `bet/place_bet.py`
- Dashboard: `dashboard/app.py`
- DB access layer: `db.py`

## Setup

### 1) Install dependencies

```bash
uv sync
```

### 2) Configure environment

Copy `.env.example` → `.env`, then fill in values.

At minimum you typically need:

- Postgres: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- Optional but recommended: `ENCRYPTION_KEY` (Fernet key) to encrypt sensitive DB fields

Generate an encryption key:

```bash
python3 - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

## Running the pipeline

### Normal operation (loop)

```bash
uv run run_pipeline.py
```

### One-off operation

```bash
uv run run_pipeline.py --game_pk <game_pk>
uv run run_pipeline.py --now
```

## What the orchestrator does

High-level loop (see `run_pipeline.py`):

1. Identify upcoming games needing prediction (DB-first; CSV fallback).
2. Refresh MLB data.
3. Refresh Kalshi balances (per configured account).
4. Run model inference.
5. Place bets for approved users.
6. Periodically refresh live positions and scores.

Batching:
- Games with the same scheduled start minute may be processed as a batch.
- Data refresh runs once per batch; prediction/betting can run in parallel per game.

## Step details

### Step A — Fetch MLB data

Script: `fetch/fetch_data.py`

```bash
uv run fetch/fetch_data.py
```

Optional:

```bash
uv run fetch/fetch_data.py --today-only
```

Behavior:
- Upserts into Postgres `games`.
- Uses `data/cache/` for caches.
- Maintains a local CSV (`data/mlb_<season>.csv`) as a fallback cache if DB is unavailable.

### Step B — Run model inference

Script: `model/predict.py`

Predict by `game_pk`:

```bash
uv run model/predict.py --game_pk <game_pk>
```

Or by date + teams:

```bash
uv run model/predict.py --game_date <YYYY-MM-DD> --home_team <HOME> --away_team <AWAY>
```

Inputs:
- `data/master_mlb.csv` (historical training data)
- Current season games from Postgres (`games`)

Outputs:
- Writes prediction + sizing back to Postgres (exact tables/columns handled by `db.py`).

### Step C — Kalshi: balances + orders

Kalshi integration details (auth, balance, orders, credential storage):

- `docs/kalshi.md`

The pipeline is DB-first:
- Balances recorded in `user_balance`
- Orders recorded in `user_orders`

To place a bet manually for one user:

```bash
uv run bet/place_bet.py --game_pk <game_pk> --email <user_email>
```

## Dashboard

Dev server:

```bash
uv run uvicorn dashboard.app:app --reload --host <bind_addr> --port <REDACTED_PORT>
```

## Auto-deploy

Auto-deploy is configured via:

- `.github/workflows/deploy.yml`
- `scripts/deploy.sh`

Operational notes:
- Keep secrets on the deployed machine in `.env` (not in the repo).
- Use a dedicated, minimally-privileged runner user.
- Scope sudo permissions tightly to only what deployment needs.

See: `docs/auto_deploy.md`

## Homelab SSH helper

For live debugging via SSH:

- `docs/homelab_access.md`
