# Training Data Pipelines

This document details how the MLB betting pipeline obtains and processes its training data, distinguishing between the baseline (v1) production pipeline and the advanced (v2) LightGBM sandbox pipeline.

## Overview

*   **v1 (Production Baseline):** Relies on `fetch/fetch_data.py` pulling daily data into a Postgres `games` table, which is then engineered by `model/train.py` (`engineer_features`). It uses a small set of hand-picked features (k=24-64) combined with Logistic Regression. The fallback cache is `data/master_mlb.csv`.
*   **v2 (LightGBM Sandbox):** A larger-scale feature lab (`sandbox/model_lab/`) pulling from diverse sources (MLB StatsAPI, Open-Meteo, Baseball Savant). It combines these into `sandbox/model_lab/output/master_sandbox_mlb.csv`. Feature selection (`feature_engineer.py`) uses correlation pruning (r ≥ 0.85) and L1 stability, ranking the top 306 features via LightGBM gain.

## Source Map (v2 Sandbox)

| Source | Fetch Command | Raw Cache Path | First Usable Season | Leakage Rule |
| :--- | :--- | :--- | :--- | :--- |
| MLB StatsAPI | `uv run python sandbox/model_lab/real_sources.py fetch-mlb` | `cache/mlb_statsapi_game_rows.parquet` | 2010 | Prior appearances only; bullpens 1-day shifted |
| Schedule / Times | `uv run python sandbox/model_lab/real_sources.py fetch-times` | `cache/mlb_game_times.parquet` | 2010 | None (metadata) |
| Open-Meteo Weather | `uv run python sandbox/model_lab/real_sources.py fetch-weather` | `cache/openmeteo_hourly_features.parquet` | 2010 | None (environmental) |
| Savant Pitch-Level | `uv run python sandbox/model_lab/savant_sources.py fetch ...` | `cache/savant/<group>/<date>.parquet` | 2015 | 1-day shift before game date |
| Savant Sprint Speed | `uv run python sandbox/model_lab/leaderboard_sources.py fetch ...` | `cache/savant_leaderboards/sprint_speed/<yr>.parquet`| 2016 | Joined on `season - 1` |
| Savant Park Factors | `uv run python sandbox/model_lab/leaderboard_sources.py fetch ...` | `cache/savant_leaderboards/park_factors/<yr>.parquet`| 2016 | Joined on `season - 1` |
| Savant Catchers | `uv run python sandbox/model_lab/leaderboard_sources.py fetch ...` | `cache/savant_leaderboards/catcher_*/<yr>.parquet` | 2017/2019 | Joined on `season - 1` |
| Savant Active Spin | `uv run python sandbox/model_lab/leaderboard_sources.py fetch ...` | `cache/savant_leaderboards/active_spin/<yr>.parquet` | 2021 | Joined on `season - 1` |
| Savant Batter Custom | `uv run python sandbox/model_lab/leaderboard_sources.py fetch ...`| `cache/savant_leaderboards/batter_custom/<yr>.parquet`| 2016 | Joined on `season - 1` |

> **IMPORTANT:** Because Statcast-derived features start in 2015 and require prior-season joins (to avoid leakage), the **first usable training season is effectively 2016 or 2017**, depending on the specific feature mix.

## Build Pipelines

### Production (v1)
1.  **Fetch:** `uv run fetch/fetch_data.py` (Updates Postgres `games` and `data/mlb_<season>.csv`).
2.  **Engineer:** `model/train.py` builds `_DIFF` features and rolling averages directly from the DB/CSV state.

### Sandbox (v2)
1.  **Fetch Caches:** `uv run python sandbox/model_lab/real_sources.py fetch-all` (plus `savant_sources.py` and `leaderboard_sources.py`).
2.  **Build Master:** `uv run python sandbox/model_lab/build_master.py` combines the production frame with all sandbox caches into `output/master_sandbox_mlb.csv`.
3.  **Select Features:** `uv run python sandbox/model_lab/feature_engineer.py` evaluates candidate features, drops highly correlated pairs, checks L1 stability, and outputs a top-K list and candidate model pickle.

## Feature Engineering

### v1 (model/train.py)
*   **Scale:** ~24 to 64 columns.
*   **Logic:** Simple math `_DIFF` fields (e.g., `home_woba - away_woba`), 15-day rolling averages, pythagorean luck, and basic schedule context.

### v2 (feature_engineer.py)
*   **Scale:** Up to 306 columns (`top_k=306` by LightGBM gain).
*   **Pruning:**
    1. Coverage filter (drops columns with too many nulls).
    2. Correlation pruning (drops one feature from pairs with |r| ≥ 0.85).
    3. L1 Logistic Regression stability check.
*   **Ranking:** Permutation importance or Tree gain (LightGBM/XGBoost).
*   **Cache:** Intermediate data stored in `sandbox/model_lab/output/cache/`.

## Train/Validation Split

Models are evaluated using walk-forward validation:
*   **FOLDS:** [2022, 2023, 2024, 2025]
*   **Recency Weighting:** Training examples use a half-life of 4 seasons (`weight = 0.5 ** (age / 4.0)`).

## Leakage Rules

*   **Rolling Windows:** Shifted by 1 day (`shift(1)`).
*   **Leaderboards:** Joined on `prior_source_season = season - 1`. No current-season aggregates are used.
*   **Forbidden Columns (`LEAK_COLS`):** `home_score`, `away_score`, `home_win`, `hg`, `ag`, and near-constant availability flags.

## Operational Commands (How to Refresh)

```bash
# 1. Pull latest games (v1/v2 base)
uv run fetch/fetch_data.py

# 2. Update sandbox real sources
uv run python sandbox/model_lab/real_sources.py fetch-all

# 3. Rebuild sandbox master CSV
uv run python sandbox/model_lab/build_master.py

# 4. Re-engineer features (correlation pruning & selection)
uv run python sandbox/model_lab/feature_engineer.py

# 5. Retrain v2 model with walk-forward validation
uv run python -m sandbox.model_lab.training.train baseline --models lr,lgbm --progress
```

## Known Gaps

*   **`home_bp_xfip_30d` / `away_bp_xfip_30d`:** Schema only. xFIP requires fly-ball counts, which are missing from the current Statcast aggregate cache. FanGraphs API returns 403.
*   **`roof_closed_flag`:** Sparse. The StatsAPI feed exposes static roof types but not the daily open/closed status for retractable roofs.

## Pointers for Future Agents

*   **`docs/model_feature_catalog.md`:** The canonical planned feature list.
*   **`sandbox/model_lab/sources/data_sources.md`:** Authoritative source mappings, formulas, and coverage stats.
