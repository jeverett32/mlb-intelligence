# Scheduled Jobs — Production LXCs

All times UTC.

## App LXC (10.1.23.161)

### Cron (`crontab -l`)

| Time | Script | Purpose |
|------|--------|---------|
| 08:00 daily | `settle_games.py` | Settle previous day's games |
| 08:30 daily | `scripts/backfill_sbr_odds.py --recent-days 14 --apply` | Replace fallback odds (odds-api/fanduel) with SBR odds for last 14 days |
| 12:00 daily | `fetch/fetch_data.py --prefetch-today-odds` | Prefetch today's odds |

### systemd timers

| Time | Unit | Purpose |
|------|------|---------|
| 03:30 daily | `mlb-backup.timer` → `mlb-backup.service` | Run `scripts/backup_db.sh` (Postgres backup) |
| 08:00 daily | `mlb-retrain.timer` → `mlb-retrain.service` | Run `models/model_v1/train.py` (daily model retrain) |

### Persistent services

| Unit | Purpose |
|------|---------|
| `mlb-dashboard.service` | FastAPI dashboard on :8080 |
| `mlb-pipeline.service` | `run_pipeline.py` (full pipeline) |
| `actions.runner.*.mlb-app-runner.service` | GitHub Actions self-hosted runner |

## DB LXC

### systemd timers

| Time | Unit | Purpose |
|------|------|---------|
| 03:00 daily (±5 min) | `mlb-backup.timer` → `mlb-backup.service` | `backup_to_gdrive.py` — dump DB and upload to Google Drive |
