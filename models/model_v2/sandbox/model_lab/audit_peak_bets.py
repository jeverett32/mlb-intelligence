#!/usr/bin/env python3
"""Audit the 626 LGBM raw edge>0.18 half-Kelly bets.

Looks at: season distribution, edge magnitude, ML range, scratched starters,
win rate by bucket. Sanity check the 16.14% ROI finding.
"""

from __future__ import annotations

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

OUT_DIR = LAB_DIR / "output"


def main():
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
        prob = fit_predict_proba(train=tr, val=va, cols=cols, sample_weight=sw, model="lgbm")
        all_probs.append(prob)
        all_vals.append(va.assign(_p_lgbm=prob))

    full = pd.concat(all_vals)
    full["_p_lgbm"] = np.concatenate(all_probs)
    full["home_dec"] = ml_to_dec(full["close_home_ml"])
    full["away_dec"] = ml_to_dec(full["close_away_ml"])
    full["valid"] = np.isfinite(full["home_dec"]) & np.isfinite(full["away_dec"])
    total_imp = (1.0 / full["home_dec"]) + (1.0 / full["away_dec"])
    full["home_mkt"] = (1.0 / full["home_dec"]) / total_imp
    full["edge_h"] = full["_p_lgbm"] - full["home_mkt"]

    EDGE = 0.18
    full["bet_home"] = full["valid"] & (full["edge_h"] > EDGE)
    full["bet_away"] = (
        full["valid"] & (-full["edge_h"] > EDGE) & (~full["bet_home"])
    )
    full["bet_side"] = np.where(
        full["bet_home"], "home",
        np.where(full["bet_away"], "away", "none"),
    )

    bets = full.loc[full["bet_side"] != "none"].copy()
    bets["won"] = np.where(
        bets["bet_side"] == "home",
        bets["home_win"] == 1,
        bets["home_win"] == 0,
    )
    bets["odds_taken"] = np.where(
        bets["bet_side"] == "home", bets["home_dec"], bets["away_dec"]
    )
    bets["ml_taken"] = np.where(
        bets["bet_side"] == "home", bets["close_home_ml"], bets["close_away_ml"]
    )

    print(f"\n=== TOTAL {len(bets)} BETS ===")
    print(f"win rate: {bets['won'].mean()*100:.2f}%")
    print(f"breakeven needed @ avg odds {bets['odds_taken'].mean():.2f}: "
          f"{(1/bets['odds_taken'].mean())*100:.2f}%")

    print("\n--- by season ---")
    season_summary = bets.groupby("season").agg(
        n=("won", "size"),
        win_rate=("won", "mean"),
        avg_odds=("odds_taken", "mean"),
        avg_edge=("edge_h", lambda s: float(np.abs(s).mean())),
    )
    print(season_summary.to_string())

    print("\n--- by side ---")
    print(bets.groupby("bet_side").agg(
        n=("won", "size"),
        win_rate=("won", "mean"),
        avg_odds=("odds_taken", "mean"),
    ).to_string())

    print("\n--- ML range distribution ---")
    bets["ml_bucket"] = pd.cut(
        pd.to_numeric(bets["ml_taken"], errors="coerce"),
        bins=[-np.inf, -300, -200, -150, -110, 110, 150, 200, 300, np.inf],
        labels=["<-300","-300..-200","-200..-150","-150..-110","-110..+110",
                "+110..+150","+150..+200","+200..+300",">+300"],
    )
    print(bets.groupby("ml_bucket", observed=True).agg(
        n=("won", "size"),
        win_rate=("won", "mean"),
    ).to_string())

    print("\n--- edge magnitude distribution ---")
    bets["abs_edge"] = bets["edge_h"].abs()
    bets["edge_bucket"] = pd.cut(
        bets["abs_edge"],
        bins=[0.18, 0.20, 0.25, 0.30, 0.40, 1.0],
    )
    print(bets.groupby("edge_bucket", observed=True).agg(
        n=("won", "size"),
        win_rate=("won", "mean"),
        avg_edge=("abs_edge", "mean"),
    ).to_string())

    # Scratched starter check
    if {"home_probable_id","away_probable_id","home_starter_id","away_starter_id"}.issubset(bets.columns):
        hp = pd.to_numeric(bets["home_probable_id"], errors="coerce")
        ap = pd.to_numeric(bets["away_probable_id"], errors="coerce")
        hs = pd.to_numeric(bets["home_starter_id"], errors="coerce")
        as_ = pd.to_numeric(bets["away_starter_id"], errors="coerce")
        scratched = ((hp.notna() & hs.notna() & (hp != hs)) |
                     (ap.notna() & as_.notna() & (ap != as_)))
        print(f"\n--- scratched starters ---")
        print(f"bets with scratched starter: {int(scratched.sum())} / {len(bets)} "
              f"({scratched.mean()*100:.1f}%)")
        if scratched.any():
            print(f"win rate w/ scratch: {bets.loc[scratched,'won'].mean()*100:.2f}%")
            print(f"win rate no scratch: {bets.loc[~scratched,'won'].mean()*100:.2f}%")

    # Recompute ROI per season at half-Kelly to mirror earlier sim
    print("\n--- ROI per season (half-Kelly) ---")
    for season, grp in bets.groupby("season"):
        f = (grp["_p_lgbm"] * (grp["home_dec"] - 1) - (1 - grp["_p_lgbm"]))
        # Side-aware Kelly
        kelly = []
        outc = []
        odds = []
        for _, r in grp.iterrows():
            p = r["_p_lgbm"] if r["bet_side"] == "home" else (1 - r["_p_lgbm"])
            b = r["odds_taken"] - 1
            ev = p * b - (1 - p)
            kf = ev / b if ev > 0 else 0
            kelly.append(kf * 0.5)
            outc.append(int(r["won"]))
            odds.append(r["odds_taken"])
        kelly = np.array(kelly)
        outc = np.array(outc)
        odds = np.array(odds)
        pnl = np.where(outc == 1, kelly * (odds - 1), -kelly).sum()
        staked = kelly.sum()
        roi = pnl / staked if staked > 0 else 0
        print(f"  {int(season)}: bets={len(grp)} pnl={pnl:.2f}u staked={staked:.2f}u ROI={roi*100:+.2f}%")


if __name__ == "__main__":
    main()
