# Fetch MLB Data

Refreshes `data/mlb_2026.csv` with the latest schedule, odds, pitcher stats, weather, and FanGraphs team stats.

## Command

Run from the project root:
```bash
uv run 1_fetch_data/fetch_2026_data.py
```

Use `--today-only` to skip historical backfill and only scrape today's odds (faster when running close to game time):
```bash
uv run 1_fetch_data/fetch_2026_data.py --today-only
```

## What it does

The script runs a 6-step pipeline and writes results to `data/mlb_2026.csv`:

1. **Schedule** — Pulls all 2026 regular season games from the MLB Stats API. Updates final scores for recently completed games.
2. **Odds** — Scrapes moneyline and totals from Sportsbook Review (SBR). Prefers Pinnacle > DraftKings > FanDuel > BetMGM. Caches to `1_fetch_data/cache_2026/odds_2026.csv`.
3. **Pitcher stats** — Fetches ERA, WHIP, K/9, BB/9 and handedness for probable starters. Uses MLB Stats API boxscores for completed games. Caches to `cache_2026/`.
4. **Weather** — Open-Meteo API (free, no key required). Historical data for past games, 7-day forecast for upcoming.
5. **FanGraphs** — Season-to-date batting (wRC+, wOBA, AVG, OBP, SLG) and pitching (FIP, ERA, K/9) per team. Cached as daily snapshots.
6. **Assembly** — Merges all sources; fills gaps with NaN; writes `data/mlb_2026.csv`.

## Incremental behavior

The script skips rows that are already complete (have scores + stats). It always re-fetches today's and tomorrow's games so odds and probable pitchers stay current.

A row is considered complete when these fields are populated:
`home_score`, `away_score`, `home_win`, `home_starter_id`, `temp_c`, `home_avg`

## Timing note

All times in `data/mlb_2026.csv` are **UTC**. The system runs 5 minutes before each game's UTC start time. Example: ATL plays ATH at 14:00 UTC — the fetch runs at 13:55 UTC, refreshing today's data and backfilling any unscored games from the previous day.

## Output columns relevant to the model

| Column | Description |
|---|---|
| `game_pk` | MLB Stats API game ID (used to identify the game in predict.py) |
| `game_date` | UTC date |
| `home_team` / `away_team` | 2–3 letter MLB team abbreviation |
| `close_home_ml` / `close_away_ml` | Closing moneyline (American format) |
| `open_home_ml` / `open_away_ml` | Opening moneyline |
| `close_total` / `open_total` | Over/under line |
| `home_starter_id` / `away_starter_id` | MLB pitcher ID |
| `home_sp_era`, `home_sp_whip`, `home_sp_k9`, `home_sp_bb9` | Starting pitcher stats |
| `home_avg`, `home_woba`, `home_wrc_plus` | Team batting stats (FanGraphs) |
| `home_fip`, `home_era` | Team pitching stats (FanGraphs) |
| `temp_c`, `wind_speed_kmh`, `wind_dir_deg` | Game-time weather |
| `is_night_game` | 1 if night game |

## Verification

After running, confirm the target game row has odds:
```python
import pandas as pd
df = pd.read_csv("data/mlb_2026.csv")
game = df[df["game_pk"] == YOUR_GAME_PK]
print(game[["game_date", "home_team", "away_team", "close_home_ml", "close_away_ml"]])
```

If `close_home_ml` is NaN, odds have not been posted yet or the game is too far out. The model will still run but cannot calculate edge or place a bet.
