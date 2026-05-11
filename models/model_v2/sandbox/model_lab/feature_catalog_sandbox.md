# Sandbox Feature Catalog

Tracks experimental feature status before production promotion.

Canonical planned feature list: `docs/model_feature_catalog.md`.

## Status Values

- `planned`: identified but not built
- `pulled`: source data available in sandbox
- `engineered`: feature built into sandbox master CSV
- `tested`: included in time-split backtest
- `promoted`: approved for production implementation
- `rejected`: failed coverage, leakage, or model-quality checks

## Initial Build Order

| Group | Status | Notes |
|---|---|---|
| Full planned schema | engineered | `planned_features.py` creates every planned column from `docs/model_feature_catalog.md`. |
| Hourly weather / air density | pulled | Interim proxy from current daily weather plus standard humidity/pressure assumptions. Replace with Open-Meteo hourly. |
| Bullpen team freshness and quality | planned | High-value pregame signal; needs usage history and role logic. |
| Confirmed lineup IDs and aggregate skill | planned | Strong upside; must handle late lineup availability. |
| Catcher identity plus framing / throwing | planned | Useful with starter and umpire interactions. |
| Lineup-vs-starter matchup vectors | planned | High upside; requires shrinkage for pitch-type splits. |
| Starter Statcast shape and TTTO | planned | Stuff and degradation profile. |
| Umpire, park-handedness, travel, roof | pulled | Park/roof/travel have sandbox proxies. Umpire source still pending. |
| High-leverage individual reliever layer | planned | Needs role inference. |
| Starter-catcher pair and projected bullpen composites | planned | Needs heavy shrinkage. |

## Promotion Gate

Feature groups need:
- documented source in `sources/data_sources.md`
- deterministic build code in `features.py` or source-specific sandbox module
- no leakage against `T-5min` pregame availability
- coverage report by season
- time-split lift vs current production baseline
