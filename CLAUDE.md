# MLB Betting Pipeline

Automated MLB betting pipeline: fetch data → train model → predict edge → place bets on Kalshi.
Managed with `uv`. All commands: `uv run <script>`. See `.clauderules` for full conventions.

## Token efficiency

All shell commands run through `rtk` automatically via hook. No manual wrapping needed.
For broad exploration, spawn an Explore subagent rather than reading many files raw.

## Data files — never read raw

CSV/Parquet in `data/` can be huge. Query with pandas:
```python
import pandas as pd
df = pd.read_csv("data/games.csv")
print(df.tail(5))
```

## Secrets

- Keys in `.env` (python-dotenv). Never log or print.
- `kalshi-key.pem` — never read or output contents.
- Use `.env.example` for variable names.

## Deploy

Push to GitHub → GitHub Actions runner deploys to app LXC automatically. No manual SSH needed for deploys.

## Git workflow

Commit frequently after tested, coherent checkpoints so deployable fixes do not sit uncommitted.

## Key entry points

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Orchestrator |
| `db.py` | All DB access |
| `bet/place_bet.py` | Kalshi bet placement |
| `dashboard/app.py` | FastAPI dashboard |
| `settle_games.py` | Settlement logic |
| `homelab.py` | Direct SSH into app/db LXCs for live debugging (`python3 homelab.py app "cmd"` / `db "cmd"`) |
