#!/usr/bin/env python3
"""Phase 2.9 research: bootstrap LGBM (posterior mean + std) — not used in production.

**Operations / nightly metrics:** use `python -m models.model_v2.eval` (single
deterministic LGBM per walk-forward fold, Phase 2.5 style — same family as
`sandbox/model_lab/experiments_advanced.py` + `roi_eval.fit_predict_proba`).

This script remains for reproducing RESULTS.md Phase 2.9 and exploring std-based
filters. Do not point cron, backfills, or dashboard expectations at it.

Goal: get per-game posterior std on LGBM probabilities. Hypothesis:
  - High-variance high-edge bets are overconfident garbage (audit found
    edge>0.25 lost money).
  - Low-variance high-edge bets are real alpha.
Filter on posterior std to keep only reliable tail bets.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab import features as sf
from models.model_v2.sandbox.model_lab.feature_engineer import (
    FOLDS,
    load_or_build_engineered_frame,
    load_or_build_feature_sets,
    recency_weights,
)
from models.model_v2.sandbox.model_lab.roi_eval import ml_to_dec
from models.model_v2.sandbox.model_lab.training.models import make_lgbm

OUT_DIR = LAB_DIR / "output"
N_BOOT = 16
N_JOBS = 4
RNG = np.random.default_rng(42)


def make_lgbm_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", make_lgbm()),
    ])


def _fit_one(seed, X_tr, y_tr, sample_weight, X_va):
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    idx = rng.integers(0, n, size=n)
    sw_b = sample_weight[idx] if sample_weight is not None else None
    pipe = make_lgbm_pipe()
    pipe.fit(X_tr[idx], y_tr[idx], model__sample_weight=sw_b)
    return pipe.predict_proba(X_va)[:, 1]


def bootstrap_lgbm(train, val, cols, sample_weight, n_boot):
    X_tr = train[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    X_va = val[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y_tr = train["home_win"].astype(int).to_numpy()
    seeds = [42 + b * 1000 for b in range(n_boot)]
    out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(_fit_one)(s, X_tr, y_tr, sample_weight, X_va) for s in seeds
    )
    return np.array(out)


def walk_forward_bayes_lgbm(settled, cols, n_boot=N_BOOT):
    all_mean, all_std, all_vals = [], [], []
    for train_end, val_end in FOLDS:
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        va = settled.loc[
            (settled["game_date"] >= pd.Timestamp(train_end))
            & (settled["game_date"] < pd.Timestamp(val_end))
        ]
        if tr.empty or va.empty:
            continue
        sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)
        t0 = time.time()
        probs = bootstrap_lgbm(tr, va, cols, sw, n_boot)
        mean_p = probs.mean(axis=0)
        std_p = probs.std(axis=0)
        all_mean.append(mean_p)
        all_std.append(std_p)
        all_vals.append(va.assign(_p_mean=mean_p, _p_std=std_p))
        print(f"  fold {train_end}: n_boot={n_boot} va_n={len(va)} "
              f"std mean={std_p.mean():.4f} p95={np.percentile(std_p,95):.4f} "
              f"elapsed={time.time()-t0:.1f}s")
    full = pd.concat(all_vals)
    full["_p_mean"] = np.concatenate(all_mean)
    full["_p_std"] = np.concatenate(all_std)
    return full


def evaluate(
    full, *, edge_min, edge_max=None, std_max=None, ml_cap=None,
    kelly_factor=0.5, variance_scaled=False, label="",
):
    home_dec = ml_to_dec(full["close_home_ml"])
    away_dec = ml_to_dec(full["close_away_ml"])
    valid = np.isfinite(home_dec) & np.isfinite(away_dec)
    total_imp = (1.0 / home_dec) + (1.0 / away_dec)
    home_mkt = (1.0 / home_dec) / total_imp
    p_mean = full["_p_mean"].to_numpy()
    p_std = full["_p_std"].to_numpy()
    edge_h = p_mean - home_mkt
    abs_edge = np.abs(edge_h)
    edge_pass = abs_edge > edge_min
    if edge_max is not None:
        edge_pass &= abs_edge <= edge_max
    if std_max is not None:
        edge_pass &= p_std <= std_max
    bet_home = valid & (edge_h > 0) & edge_pass
    bet_away = valid & (edge_h < 0) & edge_pass
    ml_h = pd.to_numeric(full["close_home_ml"], errors="coerce").to_numpy()
    ml_a = pd.to_numeric(full["close_away_ml"], errors="coerce").to_numpy()
    if ml_cap is not None:
        bet_home &= ml_h <= ml_cap
        bet_away &= ml_a <= ml_cap

    outcome = full["home_win"].to_numpy()
    seasons = full["season"].to_numpy()
    by_season = {}
    total_pnl = total_staked = 0.0
    bets = wins = 0
    for i in np.where(bet_home | bet_away)[0]:
        is_home = bet_home[i]
        p = p_mean[i] if is_home else 1 - p_mean[i]
        b = (home_dec[i] if is_home else away_dec[i]) - 1
        ev = p * b - (1 - p)
        if ev <= 0:
            continue
        kf = (ev / b) * kelly_factor
        if variance_scaled:
            kf = kf / (1.0 + 8.0 * p_std[i])
        won = (outcome[i] == 1) if is_home else (outcome[i] == 0)
        pnl = kf * b if won else -kf
        total_staked += kf
        total_pnl += pnl
        bets += 1
        wins += int(won)
        s = int(seasons[i])
        rs = by_season.setdefault(s, {"bets": 0, "wins": 0, "pnl": 0.0, "staked": 0.0})
        rs["bets"] += 1
        rs["wins"] += int(won)
        rs["pnl"] += pnl
        rs["staked"] += kf

    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    print(f"\n[{label}] edge ({edge_min:.2f},{edge_max if edge_max else 'inf'}] "
          f"std<={std_max} ML<={ml_cap} kelly={kelly_factor} "
          f"var_scaled={variance_scaled}")
    print(f"  TOTAL bets={bets} wr={wins/max(bets,1)*100:.1f}% "
          f"pnl={total_pnl:+.2f}u staked={total_staked:.2f}u ROI={roi*100:+.2f}%")
    for s in sorted(by_season):
        rs = by_season[s]
        sroi = rs["pnl"] / rs["staked"] if rs["staked"] > 0 else 0
        print(f"  {s}: bets={rs['bets']:3d} wr={rs['wins']/max(rs['bets'],1)*100:5.1f}% "
              f"pnl={rs['pnl']:+.2f}u ROI={sroi*100:+.2f}%")
    return roi, bets


def main():
    t0 = time.time()
    print("loading engineered frame...")
    df = load_or_build_engineered_frame(
        input_path=OUT_DIR / "master_sandbox_mlb.csv",
        cutoff=sf.DEFAULT_CUTOFF,
        cache_dir=OUT_DIR / "cache",
        use_cache=True,
    )
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    settled["game_date"] = pd.to_datetime(settled["game_date"], errors="coerce")
    fs = load_or_build_feature_sets(
        df, min_coverage=0.6, corr_threshold=0.85, top_k=306,
        cache_dir=OUT_DIR / "cache", use_cache=True, selected_by="lgbm_gain",
    )
    cols = list(fs["selected"])
    print(f"  rows={len(settled)} feats={len(cols)} elapsed={time.time()-t0:.1f}s")

    print(f"\n=== Bootstrap LGBM walk-forward (n_boot={N_BOOT}) ===")
    full = walk_forward_bayes_lgbm(settled, cols, n_boot=N_BOOT)
    print(f"\nelapsed={time.time()-t0:.1f}s")

    # Diagnose: distribution of std among edge>0.18 bets
    home_dec = ml_to_dec(full["close_home_ml"])
    away_dec = ml_to_dec(full["close_away_ml"])
    total_imp = (1.0 / home_dec) + (1.0 / away_dec)
    home_mkt = (1.0 / home_dec) / total_imp
    edge = full["_p_mean"].to_numpy() - home_mkt
    high_edge = np.abs(edge) > 0.18
    print(f"\nedge>0.18 bet candidates: {high_edge.sum()}")
    print(f"  std percentiles among them: "
          f"p25={np.percentile(full.loc[high_edge,'_p_std'],25):.4f} "
          f"p50={np.percentile(full.loc[high_edge,'_p_std'],50):.4f} "
          f"p75={np.percentile(full.loc[high_edge,'_p_std'],75):.4f} "
          f"p90={np.percentile(full.loc[high_edge,'_p_std'],90):.4f}")

    print("\n" + "=" * 70 + "\nBASELINE (mirror Phase 2.5 combined)\n" + "=" * 70)
    evaluate(full, edge_min=0.18, edge_max=0.25, ml_cap=250,
             kelly_factor=0.5, label="combined (no std filter)")

    print("\n" + "=" * 70 + "\nWITH POSTERIOR STD CEILING\n" + "=" * 70)
    for std_max in [0.10, 0.08, 0.06, 0.05, 0.04]:
        evaluate(full, edge_min=0.18, edge_max=0.25, std_max=std_max,
                 ml_cap=250, kelly_factor=0.5, label=f"combined std<={std_max}")

    print("\n" + "=" * 70 + "\nLOOSER EDGE FLOOR + STD CEILING\n" + "=" * 70)
    for edge_min in [0.10, 0.12, 0.15]:
        evaluate(full, edge_min=edge_min, edge_max=0.25, std_max=0.05,
                 ml_cap=250, kelly_factor=0.5,
                 label=f"edge>{edge_min} std<=0.05")

    print("\n" + "=" * 70 + "\nVARIANCE-SCALED KELLY (no hard std cap)\n" + "=" * 70)
    evaluate(full, edge_min=0.18, edge_max=0.25, ml_cap=250,
             kelly_factor=0.5, variance_scaled=True,
             label="combined var-scaled Kelly")

    print(f"\nTOTAL elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
