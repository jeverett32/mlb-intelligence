# MLB Pipeline (End-to-End)

This is the single, end-to-end operational doc for the MLB betting pipeline.

## Security / hygiene

- Do not put secrets, hostnames, or IP addresses in docs.
- Store infrastructure secrets in `.env` (loaded via `python-dotenv`).
- Store per-user Kalshi account configuration in Postgres. Enable encryption-at-rest for credential fields.

## Components

- Orchestrator: `run_pipeline.py`
- MLB ingest: `fetch/fetch_data.py`
- Model inference: `models/model_v1/predict.py`
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

Script: `models/model_v1/predict.py`

Predict by `game_pk`:

```bash
uv run models/model_v1/predict.py --game_pk <game_pk>
```

Or by date + teams:

```bash
uv run models/model_v1/predict.py --game_date <YYYY-MM-DD> --home_team <HOME> --away_team <AWAY>
```

Inputs:
- Postgres `games` table (source of truth)
- `data/master_mlb.csv` only as a **public snapshot / bootstrap fallback** (used if DB is unavailable or to seed a new DB)

Outputs:
- Writes prediction + sizing back to Postgres (exact tables/columns handled by `db.py`).

### Step C — Kalshi: balances + orders

Kalshi integration is implemented in code via `kalshi_client.py` (request signing) and is orchestrated through DB-backed workflows.

Key points:

- Per-user Kalshi credentials are stored in Postgres in `kalshi_accounts`.
  - `key_id` (API key id)
  - `key_path` (path on the deployed machine to the PEM private key)
  - `kalshi_env` (e.g. prod vs demo)
- If `ENCRYPTION_KEY` is set in `.env`, sensitive credential fields can be encrypted at rest in the DB (see `db.py`).
- Balances are recorded in `user_balance`.
- Orders are recorded in `user_orders`.

To place a bet manually for one user:

```bash
uv run bet/place_bet.py --game_pk <game_pk> --email <user_email>
```

## V2 parallel pipeline (LightGBM, `games_v2`)

V2 runs alongside V1: `run_pipeline_v2.py`, inference in `models/model_v2/predict.py`
(single deterministic LGBM + `SimpleImputer` per training fingerprint — no bootstrap).

**Nightly walk-forward metrics** (dashboard admin / `model_artifacts_v2`):

```bash
uv run python -m models.model_v2.eval
```

- One LGBM fit per fold with recency-weighted training (Phase 2.5 / `experiments_advanced.py` style — **not** `sandbox/model_lab/bayes_lgbm_experiment.py`, which is Phase 2.9 research only).
- Persists JSON metrics onto the **latest** `model_artifacts_v2` row (see `db.py`).

**Homelab:** `deploy/mlb-pipeline-v2-eval.service` + `deploy/mlb-pipeline-v2-eval.timer` (systemd oneshot + daily timer).

**Data:** engineered features live in Postgres `games_v2` (`models/model_v2/ingest_features.py`, `db.bulk_upsert_games_v2`).

## Dashboard

Dev server:

```bash
uv run dashboard/app.py --reload --host <bind_addr> --port <port>
# (Equivalent: uv run uvicorn dashboard.app:app --reload ...)
```

## Auto-deploy

Auto-deploy is configured via:

- `.github/workflows/deploy.yml`
- `scripts/deploy.sh`

Operational notes:
- Keep secrets on the deployed machine in `.env` (not in the repo).
- Use a dedicated, minimally-privileged runner user.
- Scope sudo permissions tightly to only what deployment needs.

(See `.github/workflows/deploy.yml` for the exact runner expectations and steps.)

## LXC SSH helper

For live debugging via SSH:

- `docs/homelab_access.md`
