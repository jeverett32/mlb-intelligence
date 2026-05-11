# Handoff prompt (paste into a fresh agent)

You are picking up work on the **MLB pipeline** repo (`mlb-intelligence`). All sandbox / model-lab context below; you have no other chat history.

---

## Your job

1. **Finish the k=24 walk-forward comparison** for **XGB gain–selected features + XGB model** if it is missing or incomplete. LR and LGBM legs were already run; confirm against logs and cache.
2. **Report** a small table: mean **log_loss** (and optionally Brier) across the four validation folds (2022–2025) for:
   - `selected_by=lr_perm`, model **LR**
   - `selected_by=lgbm_gain`, model **LGBM**
   - `selected_by=xgb_gain`, model **XGB**
   vs **market** baseline from the same script output.
3. **Optional:** Compare the three **`selected`** feature lists (24 names each) from the cached `feature_sets__*.json` files — overlap count and a few notable differences.

## Background

- **Sandbox** lives under `sandbox/model_lab/`. Master table: `sandbox/model_lab/output/master_sandbox_mlb.csv` (read-only for experiments).
- Feature pipeline: coverage filter → engineer composites → **correlation prune** (~306 columns) → top‑K list.
- **Original** top‑K: L1 stability + **LR permutation** importance (`selected_by=lr_perm`).
- **New:** Top‑K can be built from **mean tree gain importances** (LGBM or XGB) averaged over **walk-forward train** folds only on the **pruned** set — `selected_by=lgbm_gain` or `xgb_gain`. Implementation is in `feature_engineer.py` (`mean_tree_gain_importances`, `load_or_build_feature_sets(..., selected_by=...)`).
- **ROI / metrics:** `sandbox/model_lab/roi_eval.py` — `--sweep-top-k`, `--selected-by`, `--model lr|lgbm|xgb|ensemble`. Note: **`ensemble` is still LR+LGBM average only**, not XGB.

## Key files

| Path | Role |
|------|------|
| `sandbox/model_lab/roi_eval.py` | Walk-forward log_loss, ROI sim, CLI |
| `sandbox/model_lab/feature_engineer.py` | Feature sets, tree gain selection, caches |
| `sandbox/model_lab/training/models.py` | `make_lgbm()`, `make_xgb()` |
| `sandbox/model_lab/output/cache/feature_sets__*.json` | `selected`, `pruned`, `selected_by` in cache key |
| `sandbox/model_lab/output/logs/roi_compare_k24_*.log` | Prior batch run (if present) |

## What we already saw (same data, k=24, recency on)

- **lr_perm + LR:** log_loss roughly **0.666–0.678** by fold; generally **better than market** on average (2024 notably strong).
- **lgbm_gain + LGBM:** log_loss roughly **0.690–0.695**; **worse** than market and worse than the LR stack on these metrics.
- **xgb_gain + XGB:** run to completion and record numbers — may have been interrupted last time.

## Commands (repo root)

```bash
uv sync   # if needed
export PYTHONUNBUFFERED=1

uv run python -u sandbox/model_lab/roi_eval.py --sweep-top-k 24 --selected-by xgb_gain --model xgb

# Reference re-runs (cached feature JSON speeds repeats):
uv run python -u sandbox/model_lab/roi_eval.py --sweep-top-k 24 --selected-by lr_perm --model lr
uv run python -u sandbox/model_lab/roi_eval.py --sweep-top-k 24 --selected-by lgbm_gain --model lgbm
```

## Rules

- Do **not** read or print `.env` or `*.pem`.
- **Postgres** is production truth; sandbox must **not** mutate production tables, model artifacts, or bet flows.
- Prefer **`uv run`** for Python. Sandbox cutoff: keep `game_date < 2026-01-01` unless the user changes it (`AGENTS.md` / sandbox rules).

## Done when

- [ ] `xgb_gain` + `xgb` k=24 completes without error.
- [ ] Table (or bullet summary) comparing the three setups + market on log_loss.
- [ ] Optional: one paragraph on feature-list overlap.

---

_End of handoff._
