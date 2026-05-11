#!/usr/bin/env python3
"""Advanced model experiments: calibration, market-blend, stacking.

All loads go through `load_or_build_engineered_frame` (cached). No direct CSV reads.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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
INPUT = OUT_DIR / "master_sandbox_mlb.csv"


def kelly_roi_sim(
    val: pd.DataFrame,
    prob: np.ndarray,
    edge_thresh: float,
    kelly_factor: float,
) -> dict:
    home_dec = ml_to_dec(val["close_home_ml"])
    away_dec = ml_to_dec(val["close_away_ml"])
    valid = np.isfinite(home_dec) & np.isfinite(away_dec)
    total_imp = (1.0 / home_dec) + (1.0 / away_dec)
    home_mkt = (1.0 / home_dec) / total_imp

    p_h = prob
    p_a = 1.0 - p_h
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
        if stake <= 0:
            continue
        total_staked += stake
        bets += 1
        if outcome[i] == 1:
            pnl += stake * (home_dec[i] - 1)
            wins += 1
        else:
            pnl -= stake

    for i in np.where(bet_away)[0]:
        stake = f_a[i] * kelly_factor
        if stake <= 0:
            continue
        total_staked += stake
        bets += 1
        if outcome[i] == 0:
            pnl += stake * (away_dec[i] - 1)
            wins += 1
        else:
            pnl -= stake

    roi = pnl / total_staked if total_staked > 0 else 0.0
    return {"bets": bets, "wins": wins, "pnl": pnl, "roi": roi, "staked": total_staked}


def grid_search(label: str, full_val: pd.DataFrame, full_prob: np.ndarray) -> dict:
    edges = np.arange(0.0, 0.31, 0.01)
    kellys = [0.05, 0.1, 0.2, 0.5, 1.0]
    best = {"roi": -999, "edge": None, "kelly": None, "bets": 0, "pnl": 0.0}
    for ethresh in edges:
        for kf in kellys:
            res = kelly_roi_sim(full_val, full_prob, ethresh, kf)
            if res["bets"] < 100:
                continue
            if res["roi"] > best["roi"]:
                best = {
                    "roi": res["roi"],
                    "edge": float(ethresh),
                    "kelly": kf,
                    "bets": res["bets"],
                    "pnl": res["pnl"],
                    "wins": res["wins"],
                    "staked": res["staked"],
                }
    if best["edge"] is None:
        print(f"  [{label}]  no qualifying config (>100 bets)")
        return best
    print(
        f"  [{label}]  best edge>{best['edge']:.2f}  kelly={best['kelly']}  "
        f"bets={best['bets']}  pnl={best['pnl']:.2f}u  ROI={best['roi']*100:.2f}%"
    )
    return best


def walk_forward_probs(
    settled: pd.DataFrame,
    cols: list[str],
    model: str,
    *,
    calibrate: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Get walk-forward probs. If calibrate=True, fit isotonic on a held-out slice of train."""
    all_probs = []
    all_vals = []

    for train_end, val_end in FOLDS:
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        va = settled.loc[
            (settled["game_date"] >= pd.Timestamp(train_end))
            & (settled["game_date"] < pd.Timestamp(val_end))
        ]
        if tr.empty or va.empty:
            continue

        if calibrate:
            cal_cut = pd.Timestamp(train_end) - pd.Timedelta(days=180)
            tr_fit = tr.loc[tr["game_date"] < cal_cut]
            tr_cal = tr.loc[tr["game_date"] >= cal_cut]
            if len(tr_cal) < 200 or tr_fit.empty:
                tr_fit = tr
                tr_cal = None

            sw_fit = recency_weights(
                tr_fit["season"], pd.Timestamp(train_end).year - 1, 4.0
            )
            p_va = fit_predict_proba(
                train=tr_fit, val=va, cols=cols, sample_weight=sw_fit, model=model
            )
            if tr_cal is not None and len(tr_cal) >= 200:
                p_cal = fit_predict_proba(
                    train=tr_fit, val=tr_cal, cols=cols, sample_weight=sw_fit, model=model
                )
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_cal, tr_cal["home_win"].astype(int).to_numpy())
                p_va = iso.transform(p_va)
        else:
            sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)
            p_va = fit_predict_proba(
                train=tr, val=va, cols=cols, sample_weight=sw, model=model
            )

        all_probs.append(p_va)
        all_vals.append(va)

    return np.concatenate(all_probs), pd.concat(all_vals)


def market_prob(va: pd.DataFrame) -> np.ndarray:
    home_dec = ml_to_dec(va["close_home_ml"])
    away_dec = ml_to_dec(va["close_away_ml"])
    total = (1.0 / home_dec) + (1.0 / away_dec)
    return (1.0 / home_dec) / total


def market_blend_sweep(
    label: str, full_val: pd.DataFrame, full_prob: np.ndarray
) -> list[dict]:
    """Blend model prob with market prob: p = w*model + (1-w)*market."""
    p_mkt = market_prob(full_val)
    rows = []
    for w in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blended = w * full_prob + (1.0 - w) * p_mkt
        best = grid_search(f"{label} blend w={w:.2f}", full_val, blended)
        best["blend_w"] = w
        rows.append(best)
    return rows


def stacked_meta(
    settled: pd.DataFrame, cols_lr: list[str], cols_tree: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Walk-forward meta-learner: features = [p_lr, p_lgbm, p_xgb, p_market]. Meta = LR."""
    out_probs = []
    out_vals = []

    for train_end, val_end in FOLDS:
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        va = settled.loc[
            (settled["game_date"] >= pd.Timestamp(train_end))
            & (settled["game_date"] < pd.Timestamp(val_end))
        ]
        if tr.empty or va.empty:
            continue

        cal_cut = pd.Timestamp(train_end) - pd.Timedelta(days=240)
        tr_fit = tr.loc[tr["game_date"] < cal_cut]
        tr_meta = tr.loc[tr["game_date"] >= cal_cut]
        if len(tr_meta) < 300 or tr_fit.empty:
            print(f"  [stacked] fold {train_end}: skipping (insufficient meta data)")
            continue

        sw_fit = recency_weights(
            tr_fit["season"], pd.Timestamp(train_end).year - 1, 4.0
        )

        def base_predict(va_df):
            p_lr = fit_predict_proba(
                train=tr_fit, val=va_df, cols=cols_lr,
                sample_weight=sw_fit, model="lr",
            )
            p_lg = fit_predict_proba(
                train=tr_fit, val=va_df, cols=cols_tree,
                sample_weight=sw_fit, model="lgbm",
            )
            p_xg = fit_predict_proba(
                train=tr_fit, val=va_df, cols=cols_tree,
                sample_weight=sw_fit, model="xgb",
            )
            p_mk = market_prob(va_df)
            return np.column_stack([p_lr, p_lg, p_xg, p_mk])

        X_meta = base_predict(tr_meta)
        y_meta = tr_meta["home_win"].astype(int).to_numpy()
        m_meta = np.isfinite(X_meta).all(axis=1)
        meta = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        meta.fit(X_meta[m_meta], y_meta[m_meta])

        X_val = base_predict(va)
        m_val = np.isfinite(X_val).all(axis=1)
        # fill NaN market prob with model prob to avoid dropping rows
        p_market_fill = np.where(np.isfinite(X_val[:, 3]), X_val[:, 3], X_val[:, 1])
        X_val[:, 3] = p_market_fill
        m_val = np.isfinite(X_val).all(axis=1)
        p_val = np.full(len(va), np.nan)
        p_val[m_val] = meta.predict_proba(X_val[m_val])[:, 1]
        # Fallback for rows where market still NaN: use LGBM base prob
        bad = ~np.isfinite(p_val)
        p_val[bad] = X_val[bad, 1]
        out_probs.append(p_val)
        out_vals.append(va)
        print(
            f"  [stacked] fold {train_end} meta_coef={meta.coef_[0].round(3).tolist()} "
            f"intercept={float(meta.intercept_[0]):.3f}"
        )

    return np.concatenate(out_probs), pd.concat(out_vals)


def conformal_edge_threshold(
    label: str, full_val: pd.DataFrame, full_prob: np.ndarray
) -> dict:
    """Bet only when |edge| > k * std(edge_train_residual). Approximate via percentile."""
    p_mkt = market_prob(full_val)
    edge = full_prob - p_mkt
    rows = []
    for q in [0.80, 0.85, 0.90, 0.93, 0.95]:
        thresh_h = np.nanpercentile(edge, q * 100)
        thresh_a = np.nanpercentile(-edge, q * 100)
        thresh = max(thresh_h, thresh_a)
        for kf in [0.1, 0.2, 0.5, 1.0]:
            res = kelly_roi_sim(full_val, full_prob, thresh, kf)
            if res["bets"] < 50:
                continue
            rows.append({"q": q, "thresh": thresh, "kelly": kf, **res})
    if not rows:
        print(f"  [{label}] conformal: no qualifying configs")
        return {}
    best = max(rows, key=lambda r: r["roi"])
    print(
        f"  [{label}] conformal best q={best['q']:.2f} thresh={best['thresh']:.3f} "
        f"kelly={best['kelly']} bets={best['bets']} pnl={best['pnl']:.2f}u "
        f"ROI={best['roi']*100:.2f}%"
    )
    return best


def main():
    t0 = time.time()
    print(f"loading engineered frame (cache=on)...")
    df = load_or_build_engineered_frame(
        input_path=INPUT,
        cutoff=sf.DEFAULT_CUTOFF,
        cache_dir=OUT_DIR / "cache",
        use_cache=True,
    )
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    settled["game_date"] = pd.to_datetime(settled["game_date"], errors="coerce")
    print(f"  rows={len(settled)} elapsed={time.time()-t0:.1f}s")

    print("loading feature sets (lgbm k=306, lr k=24)...")
    fs_lgbm = load_or_build_feature_sets(
        df, min_coverage=0.6, corr_threshold=0.85, top_k=306,
        cache_dir=OUT_DIR / "cache", use_cache=True, selected_by="lgbm_gain",
    )
    cols_lgbm = list(fs_lgbm["selected"])
    fs_lr = load_or_build_feature_sets(
        df, min_coverage=0.6, corr_threshold=0.85, top_k=24,
        cache_dir=OUT_DIR / "cache", use_cache=True, selected_by="lr_perm",
    )
    cols_lr = list(fs_lr["selected"])
    print(f"  lgbm feats={len(cols_lgbm)}  lr feats={len(cols_lr)}")

    findings = {}

    # ---------- Baseline reproduce ----------
    print("\n=== EXP A: LGBM baseline (uncalibrated) ===")
    p_lgbm_raw, val_lgbm = walk_forward_probs(settled, cols_lgbm, "lgbm", calibrate=False)
    findings["lgbm_baseline"] = grid_search("lgbm raw", val_lgbm, p_lgbm_raw)

    # ---------- Calibrated LGBM ----------
    print("\n=== EXP B: LGBM + isotonic calibration ===")
    p_lgbm_cal, val_lgbm_cal = walk_forward_probs(
        settled, cols_lgbm, "lgbm", calibrate=True
    )
    findings["lgbm_calibrated"] = grid_search(
        "lgbm isotonic", val_lgbm_cal, p_lgbm_cal
    )

    # ---------- Market-blend ----------
    print("\n=== EXP C: LGBM market-blend ===")
    blend_rows = market_blend_sweep("lgbm raw", val_lgbm, p_lgbm_raw)
    findings["lgbm_blend"] = max(blend_rows, key=lambda r: r["roi"])

    print("\n=== EXP C2: Calibrated LGBM market-blend ===")
    blend_rows_cal = market_blend_sweep("lgbm cal", val_lgbm_cal, p_lgbm_cal)
    findings["lgbm_cal_blend"] = max(blend_rows_cal, key=lambda r: r["roi"])

    # ---------- Conformal-style edge percentile ----------
    print("\n=== EXP D: percentile edge-threshold ===")
    findings["lgbm_pct"] = conformal_edge_threshold(
        "lgbm raw", val_lgbm, p_lgbm_raw
    )

    # ---------- Stacked meta ----------
    print("\n=== EXP E: Stacked meta-learner (LR + LGBM + XGB + market) ===")
    p_stack, val_stack = stacked_meta(settled, cols_lr, cols_lgbm)
    findings["stacked"] = grid_search("stacked", val_stack, p_stack)
    print("\n  market-blend on stacked output:")
    blend_stack = market_blend_sweep("stack", val_stack, p_stack)
    findings["stacked_blend"] = max(blend_stack, key=lambda r: r["roi"])

    # ---------- LR baseline + blend ----------
    print("\n=== EXP F: LR baseline + market-blend ===")
    p_lr_raw, val_lr = walk_forward_probs(settled, cols_lr, "lr", calibrate=False)
    findings["lr_baseline"] = grid_search("lr raw", val_lr, p_lr_raw)
    blend_lr = market_blend_sweep("lr raw", val_lr, p_lr_raw)
    findings["lr_blend"] = max(blend_lr, key=lambda r: r["roi"])

    print(f"\nTOTAL elapsed={time.time()-t0:.1f}s")

    # Write results to a JSON-ish file the next step can append to RESULTS.md.
    import json
    out_path = OUT_DIR / "experiments_advanced_results.json"

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path.write_text(json.dumps(clean(findings), indent=2, default=str))
    print(f"wrote {out_path}")
    print("\n=== SUMMARY ===")
    for name, res in findings.items():
        if not res:
            continue
        roi = res.get("roi", 0) * 100
        print(
            f"  {name:24s}  ROI={roi:+.2f}%  bets={res.get('bets', 0):4d}  "
            f"edge={res.get('edge', res.get('thresh', '-'))}  "
            f"kelly={res.get('kelly', '-')}  "
            f"blend_w={res.get('blend_w', '-')}"
        )


if __name__ == "__main__":
    main()
