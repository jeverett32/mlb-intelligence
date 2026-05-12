# Sandbox Model Lab

Isolated workspace for new MLB features, data pulls, and model experiments.

Production rules:
- Do not mutate production tables, production model artifacts, prediction outputs, or betting/order flows from this folder.
- **Nightly V2 eval + production parity** use deterministic LGBM: `uv run python -m models.model_v2.eval` and `models/model_v2/predict.py`. For CSV-based Phase 2.5–style research, use `experiments_advanced.py` / `refined_strategy.py`. `bayes_lgbm_experiment.py` is **bootstrap research only** (Phase 2.9); do not wire it into cron or ops.
- Build sandbox data into `models/model_v2/sandbox/model_lab/output/master_sandbox_mlb.csv`.
- Training and testing use completed seasons only. Default cutoff: `game_date < 2026-01-01`.
- Treat generated files in `output/` as disposable artifacts.
- Document each source and formula before promoting a feature into comparisons.
- Promote only proven features/code into `models/model_v1/`, `fetch/`, or `db.py` after time-split backtests beat the current baseline.

## Commands

Build sandbox master CSV:

```bash
uv run python models/model_v2/sandbox/model_lab/build_master.py
```

Train baseline sandbox model (original simple trainer):

```bash
uv run python models/model_v2/sandbox/model_lab/train.py
```

Show latest metrics:

```bash
uv run python models/model_v2/sandbox/model_lab/evaluate.py
```

Walk-forward training + backtests (season-by-season, expanding window):

```bash
# Baseline metrics (LR only) with a progress bar
uv run python -m models.model_v2.sandbox.model_lab.training.train baseline --models lr --progress

# Same, but restrict seasons to keep it fast
uv run python -m models.model_v2.sandbox.model_lab.training.train baseline --models lr --seasons 2022-2025 --progress

# Sweep early-season specialist cutoffs
uv run python -m models.model_v2.sandbox.model_lab.training.train cutoff --model lr --calibration platt

# Bankroll simulation (per-bet Kelly + sharpe + joint_kelly)
uv run python -m models.model_v2.sandbox.model_lab.training.train backtest --model lr --calibration platt --bankroll 10000
```

Sandbox live inference (daily recommendations; writes JSON under `output/live/`):

```bash
uv run python -m models.model_v2.sandbox.model_lab.training.live_inference --as-of 2025-09-28 --model lr --calibration platt --sizing kelly --bankroll 10000
```

## Files

```text
models/model_v2/sandbox/model_lab/
  README.md
  build_master.py
  features.py
  train.py
  evaluate.py
  feature_catalog_sandbox.md
  sources/data_sources.md
  output/
```

## Data Contract

- `master_sandbox_mlb.csv` must contain no games on or after `2026-01-01`.
- `home_win` is required for training rows.
- `game_date` must parse as datetime.
- Feature columns must preserve positive = home edge for final `_DIFF` training columns.
- Any feature unavailable before first pitch must be excluded or replaced with a pregame proxy.

## Planned Feature Schema

`planned_features.py` expands every `new` feature in `docs/model_feature_catalog.md`.

Build output includes:
- `master_sandbox_mlb.csv`: production engineered frame plus full planned sandbox schema
- `planned_feature_status.json`: per-feature coverage and status
- `master_sandbox_manifest.json`: row counts, date range, and planned feature counts

Statuses:
- `filled_real`: filled from external source cache
- `filled_internal`: filled from deterministic schedule/game metadata
- `schema_only`: column exists but source pull/feature logic still pending

Fetch real source caches:

```bash
uv run python models/model_v2/sandbox/model_lab/real_sources.py fetch-all
```

For smoke tests:

```bash
uv run python models/model_v2/sandbox/model_lab/real_sources.py fetch-all --limit 100
```

Training default uses production features plus planned numeric features with at least 60% coverage. Use production-only baseline:

```bash
uv run python models/model_v2/sandbox/model_lab/train.py --feature-set production
```
