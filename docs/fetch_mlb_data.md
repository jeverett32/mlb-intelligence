# Fetch MLB Data

Fetches MLB schedule, odds, pitcher stats, weather, and team stats, then upserts results into Postgres.

## Script

- `fetch/fetch_data.py`

## Command

```bash
uv run fetch/fetch_data.py
```

Optional (faster near game time):

```bash
uv run fetch/fetch_data.py --today-only
```

## What it does

- Pulls the active-season schedule from the MLB Stats API.
- Scrapes odds and caches them under `data/cache/`.
- Fetches starter + team stats + weather.
- **Upserts** rows into the Postgres `games` table.
- Maintains a local CSV (`data/mlb_<season>.csv`) as a fallback cache if the DB is temporarily unavailable.

## Season selection

The season comes from:

- `MLB_SEASON` env var (optional), else
- current year.

The output CSV path is derived from the active season (see `config.py`).

## Verification

Prefer verifying via the DB (so you’re checking what the pipeline will actually use):

```bash
python3 - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv()
import db as DB
from config import ACTIVE_SEASON

df = DB.get_games_df(season=ACTIVE_SEASON)
print(df[['game_pk','game_date','away_team','home_team']].tail(10).to_string(index=False))
PY
```

## No sensitive data in docs

Do not paste connection strings, hostnames, or IP addresses into docs. Configure DB access via `.env`.
