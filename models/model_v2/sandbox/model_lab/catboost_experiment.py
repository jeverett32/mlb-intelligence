#!/usr/bin/env python3
"""CatBoost vs LGBM under refined strategy filters.

Walk-forward 2022-2025, k=306 (LGBM gain selection — same feature pool).
Reports raw + filtered ROI; per-season; head-to-head with LGBM.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from models.model_v2.sandbox.model_lab.roi_eval import fit_predict_proba, ml_to_dec

OUT_DIR = LAB_DIR / "output"


def make_catboost_pipeline() -> Pipeline:
    cb = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        train_dir=str(OUT_DIR / "catboost_info"),
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", cb),
        ]
    )


def fit_predict_cat(train, val, cols, sample_weight) -> np.ndarray:
    X_tr = train[cols].apply(pd.to_numeric, errors="coerce")
    X_va = val[cols].apply(pd.to_numeric, errors="coerce")
    pipe = make_catboost_pipeline()
    fit_kw = {}
    if sample_weight is not None:
        fit_kw["model__sample_weight"] = sample_weight
    pipe.fit(X_tr, train["home_win"], **fit_kw)
    return pipe.predict_proba(X_va)[:, 1]


def walk_forward(settled, cols, model: str):
    all_probs, all_vals = [], []
    for train_end, val_end in FOLDS:
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        va = settled.loc[
            (settled["game_date"] >= pd.Timestamp(train_end))
            & (settled["game_date"] < pd.Timestamp(val_end))
        ]
        if tr.empty or va.empty:
            continue
        sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)
        if model == "catboost":
            prob = fit_predict_cat(tr, va, cols, sw)
        else:
            prob = fit_predict_proba(train=tr, val=va, cols=cols, sample_weight=sw, model=model)
        all_probs.append(prob)
        all_vals.append(va.assign(_prob=prob))
        print(f"  fold {train_end}: trained {model}, va_n={len(va)}")
    full = pd.concat(all_vals)
    full["_prob"] = np.concatenate(all_probs)
    return full


def evaluate(full, edge_min, edge_max=None, ml_cap=None, kelly_factor=0.5, label=""):
    home_dec = ml_to_dec(full["close_home_ml"])
    away_dec = ml_to_dec(full["close_away_ml"])
    valid = np.isfinite(home_dec) & np.isfinite(away_dec)
    total_imp = (1.0 / home_dec) + (1.0 / away_dec)
    home_mkt = (1.0 / home_dec) / total_imp
    p_h = full["_prob"].to_numpy()
    edge_h = p_h - home_mkt
    abs_edge = np.abs(edge_h)
    edge_pass = abs_edge > edge_min
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
        p = p_h[i] if is_home else 1 - p_h[i]
        b = (home_dec[i] if is_home else away_dec[i]) - 1
        ev = p * b - (1 - p)
        if ev <= 0:
            continue
        kf = (ev / b) * kelly_factor
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
          f"ML<={ml_cap if ml_cap else 'inf'} kelly={kelly_factor}")
    print(f"  TOTAL bets={bets} wr={wins/max(bets,1)*100:.1f}% pnl={total_pnl:.2f}u "
          f"staked={total_staked:.2f}u ROI={roi*100:+.2f}%")
    for s in sorted(by_season):
        rs = by_season[s]
        sroi = rs["pnl"] / rs["staked"] if rs["staked"] > 0 else 0
        print(f"  {s}: bets={rs['bets']:3d} wr={rs['wins']/max(rs['bets'],1)*100:5.1f}% "
              f"pnl={rs['pnl']:+.2f}u ROI={sroi*100:+.2f}%")
    return {"roi": roi, "bets": bets, "by_season": by_season}


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

    # --- CatBoost walk-forward ---
    print("\n=== Training CatBoost walk-forward ===")
    full_cb = walk_forward(settled, cols, "catboost")
    print(f"\nCatBoost training done. elapsed={time.time()-t0:.1f}s")

    print("\n" + "=" * 70 + "\nCATBOOST baseline edge>0.18 half-Kelly\n" + "=" * 70)
    evaluate(full_cb, 0.18, kelly_factor=0.5, label="cb baseline")

    print("\n" + "=" * 70 + "\nCATBOOST combined band(0.18,0.25] + ML<=+250\n" + "=" * 70)
    evaluate(full_cb, 0.18, edge_max=0.25, ml_cap=250, kelly_factor=0.5,
             label="cb combined")

    print("\n" + "=" * 70 + "\nCATBOOST tight (0.18,0.22] + ML<=+200\n" + "=" * 70)
    evaluate(full_cb, 0.18, edge_max=0.22, ml_cap=200, kelly_factor=0.5,
             label="cb tight")

    # --- Sanity: also rerun LGBM at same configs (cached probs in refined_strategy log)
    # Skip rerun; cite from RESULTS.md.

    # --- Hybrid: average CB + LGBM probs ---
    print("\n=== Hybrid: CatBoost + LGBM averaged probabilities ===")
    full_lgbm = walk_forward(settled, cols, "lgbm")
    # Align on game_id
    cb_idx = full_cb.set_index(full_cb["game_id"])["_prob"]
    lg_idx = full_lgbm.set_index(full_lgbm["game_id"])["_prob"]
    common = cb_idx.index.intersection(lg_idx.index)
    blended_prob = (cb_idx.loc[common].to_numpy() + lg_idx.loc[common].to_numpy()) / 2.0
    full_blend = full_cb.set_index(full_cb["game_id"]).loc[common].copy()
    full_blend["_prob"] = blended_prob
    full_blend = full_blend.reset_index(drop=True)

    print("\n--- HYBRID combined band+ML cap ---")
    evaluate(full_blend, 0.18, edge_max=0.25, ml_cap=250, kelly_factor=0.5,
             label="cb+lgbm combined")

    print("\n--- HYBRID tight ---")
    evaluate(full_blend, 0.18, edge_max=0.22, ml_cap=200, kelly_factor=0.5,
             label="cb+lgbm tight")

    print(f"\nTOTAL elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
