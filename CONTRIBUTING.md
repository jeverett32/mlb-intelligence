# Contributing

Thanks for your interest in contributing.

This repo is a production-oriented MLB betting/trading pipeline. The dashboard and pipeline assume a working Postgres instance and a populated `games` table.

## Development setup

### 1) Install dependencies

This project uses [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

### 2) Environment variables

Copy the template and fill in values:

```bash
cp .env.example .env
```

Notes:
- **Never commit** `.env`.
- **Never share** any `*.pem` (especially `kalshi-key.pem`).

### 3) Local agent/editor files (optional)

This repo uses local-only (gitignored) agent/editor files such as `.clauderules` / `.claudeignore` / `.cursor/*`.

On a fresh clone, you can initialize safe defaults from templates:

```bash
python3 scripts/init_agent_files.py
```

## Running locally

### Tests

```bash
uv run pytest -q tests/
```

### Dashboard (dev)

```bash
uv run dashboard/app.py --reload --host 127.0.0.1 --port 8000
```

### Pipeline

```bash
uv run run_pipeline.py
```

### V2 nightly eval (walk-forward metrics)

```bash
uv run python -m models.model_v2.eval
```

Writes deterministic LGBM walk-forward metrics to the latest `model_artifacts_v2` row (see `docs/pipeline.md`).

## Data / DB rules

- **Postgres is the source of truth.** Prefer DB reads via `db.py`.
- Treat files under `data/` as caches/backups and **possibly stale**.
- `data/master_mlb.csv` is intended as a periodically refreshed public snapshot of modeling data.

## Pull request guidelines

- Keep changes focused and easy to review.
- Add/adjust tests where practical.
- Avoid large refactors mixed with functional changes.
- Don’t introduce new required secrets/config without updating `.env.example`.
