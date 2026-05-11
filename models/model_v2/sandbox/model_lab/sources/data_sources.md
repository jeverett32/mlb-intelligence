# Sandbox Data Sources

Document source, pull method, coverage, and leakage rule for each sandbox feature group.

## Baseline Master

| Field | Value |
|---|---|
| Source | Existing production feature engineering via `model.train.load_and_engineer_features()` |
| Output | `sandbox/model_lab/output/master_sandbox_mlb.csv` |
| Cutoff | `game_date < 2026-01-01` |
| Mutation | Read-only; no production writes |

## Planned Schema

| Field | Value |
|---|---|
| Source | `docs/model_feature_catalog.md` |
| Expansion | `sandbox/model_lab/planned_features.py` |
| Status output | `sandbox/model_lab/output/planned_feature_status.json` |
| Coverage report | `sandbox/model_lab/output/feature_coverage_report.json` (per-feature non-null counts and rates inside each source's coverage window) |
| Rule | All planned features exist as columns; unavailable source data remains null |

## Source Coverage Windows

Source-coverage start dates determine the earliest game season where a real
value can be filled. Every leaderboard-derived feature is joined on
`prior_source_season = season - 1` to avoid current-season leakage, so the
first usable game season is one year after the source coverage start.

| Source | Raw coverage start | First usable game season | Notes |
|---|---|---|---|
| MLB StatsAPI boxscore/feed | 2010 | 2010 | Lineups, catchers, umpires, starter/bullpen workload, roof/venue |
| Open-Meteo hourly archive | 2010 | 2010 | Hourly reanalysis weather, all parks |
| Schedule (internal) | 2010 | 2010 | Travel miles, tz shift, getaway flag |
| Baseball Savant pitch-level (Statcast) | 2015 | 2015 | Pitcher/batter/catcher priors, lineup priors |
| Baseball Savant sprint speed leaderboard | 2015 | 2016 | Joined prior season |
| Baseball Savant catcher pop time / throwing leaderboard | 2016 | 2017 | Joined prior season |
| Baseball Savant catcher blocking leaderboard | 2018 | 2019 | Joined prior season |
| Baseball Savant active spin leaderboard | 2020 | 2021 | Joined prior season; sparse before |
| Baseball Savant park factors leaderboard | 2015 | 2016 | LHB/RHB park HR/run factors joined prior season |
| Baseball Savant batter custom leaderboard (wOBA/OBP) | 2015 | 2016 | wRC+ proxy = `100 * wOBA / league_wOBA` (FanGraphs API blocked here) |

## Source Pipelines

### MLB StatsAPI

| Field | Value |
|---|---|
| Fetch | `uv run python sandbox/model_lab/real_sources.py fetch-mlb` |
| Raw cache | `sandbox/model_lab/cache/mlb_statsapi_game_rows.parquet` |
| Built cache | `sandbox/model_lab/cache/mlb_statsapi_features.parquet` |
| Endpoints | `statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live`, `statsapi.mlb.com/api/v1/game/{game_pk}/boxscore`, `statsapi.mlb.com/api/v1/people` |
| Filled features | umpire id, venue id/azimuth/elevation, roof, lineup handedness counts, catcher id, starter days_rest/pitch_count_l3/bf_l3/ip_l3, bullpen 30d ERA/FIP/K-BB, bullpen freshness windows (1/2/3d, high-leverage 2d, lefty/righty available, freshness/fatigue scores), closer/setup/top-LHRP/top-RHRP role IDs and freshness flags |
| Leakage | Prior pitcher appearances only; bullpen rolling windows shifted one day before game date |
| Joins | `game_pk` |
| Notes | `pitch_hand` cached separately at `mlb_pitcher_hands.parquet`; bullpen FIP uses no league constant for parity with displayed FIP shape |

### MLB schedule game times

| Field | Value |
|---|---|
| Fetch | `uv run python sandbox/model_lab/real_sources.py fetch-times` |
| Built cache | `sandbox/model_lab/cache/mlb_game_times.parquet` |
| Endpoint | `statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=YYYY-01-01&endDate=YYYY-12-31` |
| Filled features | `game_time_utc` (ISO UTC) for every game_pk |
| Joins | `game_pk` (merged in `build_master.py`) |
| Leakage | Schedule metadata; no leakage. |
| Notes | One call per season (16 calls total). Required for accurate Open-Meteo nearest-hour join. |

### Open-Meteo hourly archive

| Field | Value |
|---|---|
| Fetch | `uv run python sandbox/model_lab/real_sources.py fetch-weather` |
| Raw cache | `sandbox/model_lab/cache/openmeteo_hourly_features.parquet` |
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Filled features | temp_c, relative humidity, dew point, surface pressure, wind speed/direction at game start (nearest hour). Air density / density altitude / wind components / weather HR boost / weather run env factor derived in `_derive_weather_features`. |
| Coverage | 100% after `fetch-times` populates `game_time_utc`. Fetch retries any rows where `temp_c_game_time IS NULL`. |
| Joins | `game_pk` |
| Leakage | Pure environmental; no leakage. |

### Baseball Savant pitch-level (Statcast)

| Field | Value |
|---|---|
| Fetch | `uv run python sandbox/model_lab/savant_sources.py fetch --start 2015-03-01 --end 2025-11-30 --days 1` |
| Aggregates | `sandbox/model_lab/cache/savant/{pitcher_game,pitcher_pitch_type_game,batter_game,batter_pitch_type_game,catcher_game,game_lineups}/<date>.parquet` |
| Built cache | `sandbox/model_lab/cache/savant_features.parquet` |
| Endpoint | `https://baseballsavant.mlb.com/statcast_search/csv` (player_type=pitcher) |
| Filled features | starter pitch shape (velo/spin/extension), pitch mix, TTO/wOBA splits, lineup priors and recent form (14/30d), matchup vectors, catcher framing, umpire zone metrics, bullpen 30d xwOBA/whiff, starter-catcher pair priors |
| Leakage | All player priors are cumulative excluding current game; rolling priors use 1-day shift before game date. |
| Joins | `game_pk` plus pitcher_id / batter_id / catcher_id |
| Notes | Aggregate retains only counts/sums needed; raw pitch CSV is not cached. |

### Baseball Savant leaderboards

| Field | Value |
|---|---|
| Fetch | `uv run python sandbox/model_lab/leaderboard_sources.py fetch --start-season 2014 --end-season 2025` |
| Built cache | `sandbox/model_lab/cache/savant_leaderboard_features.parquet` |
| Cache layout | `sandbox/model_lab/cache/savant_leaderboards/<kind>/<season>.parquet` for `sprint_speed`, `active_spin`, `catcher_blocking`, `catcher_throwing`, `batter_custom`, `park_factors` |
| Endpoints | sprint_speed, active-spin, catcher-blocking, catcher-throwing, leaderboard/custom (batter wOBA/OBP), statcast-park-factors |
| Filled features | starter active spin, catcher blocking runs / blocks above avg, catcher pop time / arm strength / caught stealing rate, lineup sprint speed, lineup avg wOBA / OBP / wRC+ proxy, park handedness HR/run factors |
| Leakage | All joined on `prior_source_season = season - 1`. |
| Joins | `(player_id, prior_source_season)` for player metrics, `(venue_id, prior_source_season, bat_side)` for parks. |
| Notes on wRC+ | Official FanGraphs API responds 403 from this environment, and Savant `wrc_plus` selection returns blank. Filled column is a wOBA+ proxy: `100 * player_wOBA / league_wOBA(season)` weighted by PA. Same scale as wRC+, no park or league-runs adjustment. |

## Park / Travel / Roof

| Feature group | Source | Notes |
|---|---|---|
| Travel / body clock | Schedule order plus static park coordinates/time zones | `travel_miles` and `travel_tz_shift` are away travel minus home travel, so positive means away traveled more. |
| Roof possible / closed | MLB StatsAPI feed venue metadata | `roof_closed_flag=1` only for dome roof type; retractable open/closed state not exposed in feed. |
| Park handedness HR/Run factors | Baseball Savant park factors leaderboard | Joined prior season. |

## Known Remaining Gaps

| Feature | Status | Reason | Mitigation |
|---|---|---|---|
| `home_bp_xfip_30d`, `away_bp_xfip_30d` | schema_only | xFIP requires reliever fly-ball counts plus league HR/FB. Current Statcast aggregate cache (`pitcher_game`) holds only `gb`/`hard_hit`/`barrels`, no `fb` count. FanGraphs reliever-splits API returns 403 from this environment. Savant custom-leaderboard `xfip` field returns blank. | Either (a) augment `aggregate_chunk` to write `fb` from `bb_type=='fly_ball'` and re-fetch all Statcast daily chunks (≈1881 days) plus join MLB StatsAPI reliever HR totals, or (b) use a fronted FanGraphs scrape with the API allowed. Until then, leave schema_only. |
| `roof_closed_flag` | sparse | Feed exposes static roof type only; closed/open dome state not separable for retractable roofs. | Capture from a per-game roof-status source if available. |

## Source Template

Copy this section for each new data source.

```text
Feature group:
Source system:
Fetch script / command:
Raw fields:
Join keys:
Historical coverage:
Known missingness:
Pregame availability:
Leakage risk:
Formula summary:
Validation checks:
```
