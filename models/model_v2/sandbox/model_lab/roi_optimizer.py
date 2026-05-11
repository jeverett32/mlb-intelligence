#!/usr/bin/env python3
"""ROI Optimization with Kelly Criterion and Edge Threshold Grid Search."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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

INPUT = LAB_DIR / "output" / "master_sandbox_mlb.csv"
OUT_DIR = LAB_DIR / "output"

def kelly_roi_sim(val: pd.DataFrame, prob: np.ndarray, edge_thresh: float, kelly_factor: float) -> dict:
    home_dec = ml_to_dec(val["close_home_ml"])
    away_dec = ml_to_dec(val["close_away_ml"])
    valid = np.isfinite(home_dec) & np.isfinite(away_dec)
    
    total_imp = (1.0 / home_dec) + (1.0 / away_dec)
    home_mkt = (1.0 / home_dec) / total_imp

    p_h = prob
    p_a = 1.0 - p_h
    
    # Kelly fraction: f* = (p*b - 1) / (b - 1) = EV / (b - 1)
    ev_h = p_h * (home_dec - 1) - p_a
    ev_a = p_a * (away_dec - 1) - p_h
    edge_h = p_h - home_mkt

    f_h = np.where(ev_h > 0, ev_h / (home_dec - 1), 0)
    f_a = np.where(ev_a > 0, ev_a / (away_dec - 1), 0)
    
    bet_home = valid & (ev_h > 0) & (edge_h > edge_thresh)
    bet_away = valid & (ev_a > 0) & (-edge_h > edge_thresh) & (~bet_home)
    
    outcome = val["home_win"].to_numpy()
    
    total_staked = 0.0
    pnl = 0.0
    bets = 0
    wins = 0
    
    for i in np.where(bet_home)[0]:
        stake = f_h[i] * kelly_factor
        if stake <= 0: continue
        total_staked += stake
        bets += 1
        if outcome[i] == 1:
            pnl += stake * (home_dec[i] - 1)
            wins += 1
        else:
            pnl -= stake
            
    for i in np.where(bet_away)[0]:
        stake = f_a[i] * kelly_factor
        if stake <= 0: continue
        total_staked += stake
        bets += 1
        if outcome[i] == 0:
            pnl += stake * (away_dec[i] - 1)
            wins += 1
        else:
            pnl -= stake

    roi = pnl / total_staked if total_staked > 0 else 0.0
    return {
        "bets": bets,
        "pnl": pnl,
        "roi": roi,
        "staked": total_staked
    }

def optimize_model(df: pd.DataFrame, settled: pd.DataFrame, model_name: str, selected_by: str, top_k: int):
    print(f"\n=== Optimizing {model_name} (k={top_k}, by={selected_by}) ===")
    
    cache_dir = OUT_DIR / "cache"
    fs = load_or_build_feature_sets(
        df,
        min_coverage=0.6,
        corr_threshold=0.85,
        top_k=top_k,
        cache_dir=cache_dir,
        use_cache=True,
        selected_by=selected_by,
    )
    cols = list(fs["selected"])
    
    # 1. Get walk-forward probabilities
    all_probs = []
    all_vals = []
    
    for train_end, val_end in FOLDS:
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        va = settled.loc[(settled["game_date"] >= pd.Timestamp(train_end)) & (settled["game_date"] < pd.Timestamp(val_end))]
        if tr.empty or va.empty: continue
        
        sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)
        prob = fit_predict_proba(train=tr, val=va, cols=cols, sample_weight=sw, model=model_name)
        all_probs.append(prob)
        all_vals.append(va)
        
    full_prob = np.concatenate(all_probs)
    full_val = pd.concat(all_vals)
    
    # 2. Grid Search
    edges = np.arange(0.0, 0.16, 0.01)
    kellys = [0.05, 0.1, 0.2, 0.5, 1.0]
    
    best_roi = -999
    best_params = {}
    
    results = []
    for ethresh in edges:
        for kf in kellys:
            res = kelly_roi_sim(full_val, full_prob, ethresh, kf)
            if res["bets"] < 100: continue # Minimum sample size
            
            if res["roi"] > best_roi:
                best_roi = res["roi"]
                best_params = {"edge": ethresh, "kelly": kf, "bets": res["bets"], "pnl": res["pnl"]}
            
            results.append({
                "edge": ethresh,
                "kelly": kf,
                "roi": res["roi"],
                "bets": res["bets"]
            })
            
    if best_params:
        print(f"BEST: Edge > {best_params['edge']:.2f}, Kelly Factor: {best_params['kelly']:.2f}")
        print(f"ROI: {best_roi*100:.2f}% (over {best_params['bets']} bets, PnL: {best_params['pnl']:.2f} units)")
    else:
        print("No profitable configuration found with >100 bets.")

def main():
    print(f"Loading data from {INPUT}...")
    df = load_or_build_engineered_frame(input_path=INPUT, cutoff=sf.DEFAULT_CUTOFF, cache_dir=OUT_DIR / "cache", use_cache=True)
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    settled["game_date"] = pd.to_datetime(settled["game_date"], errors="coerce")

    # Top-K optimized configs
    configs = [
        ("lr", "lr_perm", 24),
        ("lgbm", "lgbm_gain", 306),
        ("xgb", "xgb_gain", 306),
    ]
    
    for m, sb, k in configs:
        optimize_model(df, settled, m, sb, k)

if __name__ == "__main__":
    main()
