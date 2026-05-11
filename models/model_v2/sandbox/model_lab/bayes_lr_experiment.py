#!/usr/bin/env python3
"""Bayesian-style logistic via bootstrap aggregation.

For each walk-forward fold, fit `n_boot` LR models on bootstrap samples of the
training set. For each val game, the n_boot probabilities form an approximate
posterior. Use mean as point estimate, std as uncertainty.

Bet rules tested:
  1. Vanilla mean prob (baseline)
  2. Z-edge: bet only if (|mean_edge| - k * std_edge) > floor
  3. Variance-scaled Kelly: stake = kelly_factor * f* / (1 + std)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

OUT_DIR = LAB_DIR / "output"

N_BOOT = 50
RNG = np.random.default_rng(42)


def make_lr() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
    ])


def bootstrap_predict(train, val, cols, sample_weight, n_boot):
    """Return (n_boot, n_val) array of probabilities."""
    X_tr = train[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    X_va = val[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y_tr = train["home_win"].astype(int).to_numpy()
    n = len(train)
    probs = np.zeros((n_boot, len(val)))
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        sw_b = sample_weight[idx] if sample_weight is not None else None
        pipe = make_lr()
        pipe.fit(X_tr[idx], y_tr[idx], model__sample_weight=sw_b)
        probs[b] = pipe.predict_proba(X_va)[:, 1]
    return probs


def walk_forward_bayes(settled, cols, n_boot=N_BOOT):
    all_mean, all_std, all_vals = [], [], []
    for i, (train_end, val_end) in enumerate(FOLDS):
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        va = settled.loc[
            (settled["game_date"] >= pd.Timestamp(train_end))
            & (settled["game_date"] < pd.Timestamp(val_end))
        ]
        if tr.empty or va.empty:
            continue
        sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)
        t0 = time.time()
        probs = bootstrap_predict(tr, va, cols, sw, n_boot)
        mean_p = probs.mean(axis=0)
        std_p = probs.std(axis=0)
        all_mean.append(mean_p)
        all_std.append(std_p)
        all_vals.append(va.assign(_p_mean=mean_p, _p_std=std_p))
        print(f"  fold {train_end}: n_boot={n_boot} va_n={len(va)} "
              f"mean_std={std_p.mean():.4f} elapsed={time.time()-t0:.1f}s")
    full = pd.concat(all_vals)
    full["_p_mean"] = np.concatenate(all_mean)
    full["_p_std"] = np.concatenate(all_std)
    return full


def evaluate_bayes(
    full,
    *,
    edge_floor: float,
    k_sigma: float,
    edge_max: float | None = None,
    ml_cap: float | None = None,
    kelly_factor: float = 0.5,
    variance_scaled: bool = False,
    label: str = "",
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
    # Z-edge: lower bound on edge after subtracting k*std
    z_edge = abs_edge - k_sigma * p_std

    edge_pass = (z_edge > edge_floor) & (abs_edge > edge_floor)
    if edge_max is not None:
        edge_pass &= abs_edge <= edge_max
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
            kf = kf / (1.0 + 5.0 * p_std[i])  # damp by uncertainty
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
    print(f"\n[{label}] floor={edge_floor:.2f} k_sigma={k_sigma} "
          f"edge_max={edge_max} ML<={ml_cap} kelly={kelly_factor} "
          f"var_scaled={variance_scaled}")
    print(f"  TOTAL bets={bets} wr={wins/max(bets,1)*100:.1f}% "
          f"pnl={total_pnl:+.2f}u staked={total_staked:.2f}u ROI={roi*100:+.2f}%")
    for s in sorted(by_season):
        rs = by_season[s]
        sroi = rs["pnl"] / rs["staked"] if rs["staked"] > 0 else 0
        print(f"  {s}: bets={rs['bets']:3d} wr={rs['wins']/max(rs['bets'],1)*100:5.1f}% "
              f"pnl={rs['pnl']:+.2f}u ROI={sroi*100:+.2f}%")
    return roi, bets, by_season


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

    # LR's best feature set: k=24 lr_perm
    fs = load_or_build_feature_sets(
        df, min_coverage=0.6, corr_threshold=0.85, top_k=24,
        cache_dir=OUT_DIR / "cache", use_cache=True, selected_by="lr_perm",
    )
    cols = list(fs["selected"])
    print(f"  rows={len(settled)} feats={len(cols)} elapsed={time.time()-t0:.1f}s")

    print(f"\n=== Bootstrap LR walk-forward (n_boot={N_BOOT}) ===")
    full = walk_forward_bayes(settled, cols, n_boot=N_BOOT)
    print(f"\nelapsed={time.time()-t0:.1f}s")
    print(f"posterior std summary: mean={full['_p_std'].mean():.4f} "
          f"median={full['_p_std'].median():.4f} "
          f"p95={full['_p_std'].quantile(0.95):.4f}")

    print("\n" + "=" * 70 + "\nBASELINE: Bayes-LR mean prob, edge>0.05 (LR Phase-1 cfg)\n" + "=" * 70)
    evaluate_bayes(full, edge_floor=0.05, k_sigma=0.0, kelly_factor=0.05,
                   label="bayes-LR vanilla edge>0.05 quarter-Kelly")

    print("\n" + "=" * 70 + "\nZ-EDGE: bet only when |edge| - k*std > floor\n" + "=" * 70)
    for k_sigma in [1.0, 1.5, 2.0]:
        evaluate_bayes(full, edge_floor=0.02, k_sigma=k_sigma, kelly_factor=0.5,
                       label=f"z-edge k_sigma={k_sigma}")

    print("\n" + "=" * 70 + "\nZ-EDGE + ML cap + edge band\n" + "=" * 70)
    for k_sigma in [1.0, 1.5]:
        evaluate_bayes(full, edge_floor=0.02, k_sigma=k_sigma,
                       edge_max=0.20, ml_cap=250, kelly_factor=0.5,
                       label=f"z-edge k={k_sigma} band+ML cap")

    print("\n" + "=" * 70 + "\nVARIANCE-SCALED KELLY\n" + "=" * 70)
    evaluate_bayes(full, edge_floor=0.05, k_sigma=0.0, kelly_factor=0.5,
                   variance_scaled=True,
                   label="vanilla edge>0.05 var-scaled Kelly")

    print(f"\nTOTAL elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
