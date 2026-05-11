#!/usr/bin/env python3
"""Sweep K and Edge to find the most efficient ROI-maximizing feature count."""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
from models.model_v2.sandbox.model_lab.roi_optimizer import kelly_roi_sim

INPUT = LAB_DIR / "output" / "master_sandbox_mlb.csv"
OUT_DIR = LAB_DIR / "output"

def find_best_roi_for_k(df, settled, model_name, selected_by, k):
    cache_dir = OUT_DIR / "cache"
    fs = load_or_build_feature_sets(
        df,
        min_coverage=0.6,
        corr_threshold=0.85,
        top_k=k,
        cache_dir=cache_dir,
        use_cache=True,
        selected_by=selected_by,
    )
    cols = list(fs["selected"])
    
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
    
    edges = np.arange(0.0, 0.16, 0.02)
    kellys = [0.1, 0.25, 0.5]
    
    best_roi = -1.0
    for ethresh in edges:
        for kf in kellys:
            res = kelly_roi_sim(full_val, full_prob, ethresh, kf)
            if res["bets"] < 100: continue
            if res["roi"] > best_roi:
                best_roi = res["roi"]
                
    return best_roi

def main():
    df = load_or_build_engineered_frame(input_path=INPUT, cutoff=sf.DEFAULT_CUTOFF, cache_dir=OUT_DIR / "cache", use_cache=True)
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    settled["game_date"] = pd.to_datetime(settled["game_date"], errors="coerce")

    ks = [50, 100, 150, 200, 250, 306]
    print(f"| K | LGBM ROI | XGB ROI |")
    print(f"|---|---|---|")
    for k in ks:
        lgbm_roi = find_best_roi_for_k(df, settled, "lgbm", "lgbm_gain", k)
        xgb_roi = find_best_roi_for_k(df, settled, "xgb", "xgb_gain", k)
        print(f"| {k} | {lgbm_roi*100:.2f}% | {xgb_roi*100:.2f}% |")

if __name__ == "__main__":
    main()
