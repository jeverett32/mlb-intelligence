#!/usr/bin/env python3
"""Refined betting strategy: edge band + ML cap + per-season check.

Audit showed:
  - edge > 0.25 wins less often than baseline
  - bets at ML > +300 win only 7.9%
  - 2025 ROI dropped to 4.6% vs ~20% prior years

This script tests strategies that filter out those bad pockets.
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


def get_lgbm_walkforward(top_k: int = 306):
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
        df, min_coverage=0.6, corr_threshold=0.85, top_k=top_k,
        cache_dir=OUT_DIR / "cache", use_cache=True, selected_by="lgbm_gain",
    )
    cols = list(fs["selected"])
    print(f"  using k={top_k} features")

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
        all_vals.append(va)

    full = pd.concat(all_vals)
    full["_p_lgbm"] = np.concatenate(all_probs)
    return full


def evaluate(
    full: pd.DataFrame,
    edge_min: float,
    edge_max: float | None = None,
    ml_cap: float | None = None,
    kelly_factor: float = 0.5,
    label: str = "",
) -> dict:
    home_dec = ml_to_dec(full["close_home_ml"])
    away_dec = ml_to_dec(full["close_away_ml"])
    valid = np.isfinite(home_dec) & np.isfinite(away_dec)
    total_imp = (1.0 / home_dec) + (1.0 / away_dec)
    home_mkt = (1.0 / home_dec) / total_imp

    p_h = full["_p_lgbm"].to_numpy()
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

    results_by_season: dict[int, dict] = {}
    total_pnl = total_staked = 0.0
    bets = 0
    wins = 0

    for i in np.where(bet_home | bet_away)[0]:
        is_home = bet_home[i]
        p = p_h[i] if is_home else 1 - p_h[i]
        b = (home_dec[i] if is_home else away_dec[i]) - 1
        ev = p * b - (1 - p)
        if ev <= 0:
            continue
        kf = (ev / b) * kelly_factor
        if kf <= 0:
            continue
        won = (outcome[i] == 1) if is_home else (outcome[i] == 0)
        pnl = kf * b if won else -kf

        total_staked += kf
        total_pnl += pnl
        bets += 1
        wins += int(won)

        s = int(seasons[i])
        rs = results_by_season.setdefault(
            s, {"bets": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}
        )
        rs["bets"] += 1
        rs["wins"] += int(won)
        rs["pnl"] += pnl
        rs["staked"] += kf

    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    print(f"\n[{label}] edge in ({edge_min:.2f}, {edge_max if edge_max else 'inf'}] "
          f"ML<={ml_cap if ml_cap else 'inf'} kelly={kelly_factor}")
    print(f"  TOTAL bets={bets} wins={wins} wr={wins/max(bets,1)*100:.1f}% "
          f"pnl={total_pnl:.2f}u staked={total_staked:.2f}u ROI={roi*100:+.2f}%")
    for s in sorted(results_by_season):
        rs = results_by_season[s]
        sroi = rs["pnl"] / rs["staked"] if rs["staked"] > 0 else 0.0
        print(f"  {s}: bets={rs['bets']:3d} wr={rs['wins']/max(rs['bets'],1)*100:5.1f}% "
              f"pnl={rs['pnl']:+.2f}u ROI={sroi*100:+.2f}%")

    return {
        "label": label, "bets": bets, "wins": wins, "pnl": total_pnl,
        "staked": total_staked, "roi": roi,
        "by_season": results_by_season,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=306)
    args = ap.parse_args()
    print(f"loading walk-forward LGBM probabilities (k={args.top_k})...")
    full = get_lgbm_walkforward(top_k=args.top_k)

    print("\n" + "=" * 70)
    print("BASELINE: edge > 0.18, half-Kelly (no filters)")
    print("=" * 70)
    evaluate(full, 0.18, kelly_factor=0.5, label="baseline")

    print("\n" + "=" * 70)
    print("STRATEGY 1: edge band 0.18 < edge <= 0.25 (cap mega-edges)")
    print("=" * 70)
    evaluate(full, 0.18, edge_max=0.25, kelly_factor=0.5, label="band 0.18-0.25")

    print("\n" + "=" * 70)
    print("STRATEGY 2: edge > 0.18 + skip dogs (ML <= +250)")
    print("=" * 70)
    evaluate(full, 0.18, ml_cap=250, kelly_factor=0.5, label="edge>0.18 ML<=+250")

    print("\n" + "=" * 70)
    print("STRATEGY 3: edge band 0.18-0.25 + ML <= +250 (combined)")
    print("=" * 70)
    evaluate(full, 0.18, edge_max=0.25, ml_cap=250,
             kelly_factor=0.5, label="combined")

    print("\n" + "=" * 70)
    print("STRATEGY 4: tighter band 0.18-0.22 + ML <= +200 (most conservative)")
    print("=" * 70)
    evaluate(full, 0.18, edge_max=0.22, ml_cap=200,
             kelly_factor=0.5, label="tight conservative")

    # Lower threshold + more bets variant
    print("\n" + "=" * 70)
    print("STRATEGY 5: edge 0.10-0.25 + ML<=+200 (more action)")
    print("=" * 70)
    evaluate(full, 0.10, edge_max=0.25, ml_cap=200,
             kelly_factor=0.25, label="more-action quarter-Kelly")


if __name__ == "__main__":
    main()
