"""
V2 walk-forward evaluation — Phase 2.5 style deterministic LGBM.

4 folds (2022–2025), one median-imputer + LGBM fit per fold (recency-weighted),
same Kelly ROI as config (edge band, ML cap, half-Kelly).

Run:
    uv run python -m models.model_v2.eval
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import db as DB
from models.model_v2 import config as C
from models.model_v2.feature_loader import load_or_build_engineered_frame_from_db
from models.model_v2.sandbox.model_lab.feature_engineer import (
    FOLDS,
    load_or_build_feature_sets,
    recency_weights,
)
from models.model_v2.sandbox.model_lab.roi_eval import ml_to_dec
from models.model_v2.sandbox.model_lab.training.models import make_lgbm


def _make_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", make_lgbm()),
    ])


def _fit_deterministic(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cols: list[str],
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """One LGBM fit on full train; val probabilities and feature importances (or None)."""
    pipe = _make_pipe()
    X_tr = train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    X_va = val_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y_tr = train_df["home_win"].astype(int).to_numpy()
    fit_kw: dict = {}
    if sample_weight is not None:
        fit_kw["model__sample_weight"] = sample_weight
    pipe.fit(X_tr, y_tr, **fit_kw)
    probs = pipe.predict_proba(X_va)[:, 1]
    imp = getattr(pipe.named_steps["model"], "feature_importances_", None)
    return probs, imp


def _evaluate_kelly(
    full: pd.DataFrame,
    *,
    edge_min: float,
    edge_max: float | None,
    ml_cap: float | None,
    kelly_factor: float,
) -> dict:
    home_dec = ml_to_dec(full["close_home_ml"])
    away_dec = ml_to_dec(full["close_away_ml"])
    valid = np.isfinite(home_dec) & np.isfinite(away_dec)
    total_imp = (1.0 / home_dec) + (1.0 / away_dec)
    home_mkt = (1.0 / home_dec) / total_imp
    p_mean = full["_p_mean"].to_numpy()
    edge_h = p_mean - home_mkt
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
    by_season: dict = {}
    total_pnl = total_staked = 0.0
    bets = wins = 0
    for i in np.where(bet_home | bet_away)[0]:
        is_home = bool(bet_home[i])
        p = p_mean[i] if is_home else 1 - p_mean[i]
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
    return {
        "roi": round(float(roi), 6),
        "bets": int(bets),
        "wins": int(wins),
        "pnl": round(float(total_pnl), 6),
        "staked": round(float(total_staked), 6),
        "win_rate": round(wins / max(bets, 1), 6),
        "by_season": {
            int(s): {
                "bets": int(rs["bets"]),
                "wins": int(rs["wins"]),
                "pnl": round(float(rs["pnl"]), 6),
                "staked": round(float(rs["staked"]), 6),
                "roi": round(rs["pnl"] / rs["staked"], 6) if rs["staked"] > 0 else 0.0,
                "win_rate": round(rs["wins"] / max(rs["bets"], 1), 6),
            }
            for s, rs in by_season.items()
        },
    }


def _bucketed_feature_accuracy(val_df, probs, y, feature_cols, top_n=20):
    buckets: dict = {}
    if not len(val_df):
        return buckets
    preds = (probs > 0.5).astype(int)
    correct = (preds == y.astype(int))
    for feat in feature_cols[:top_n]:
        if feat not in val_df.columns:
            continue
        vals = pd.to_numeric(val_df[feat], errors="coerce").to_numpy()
        if np.all(np.isnan(vals)):
            continue
        median = np.nanmedian(vals)
        high = ~np.isnan(vals) & (vals >= median)
        low = ~np.isnan(vals) & (vals < median)
        if high.sum() > 10 and low.sum() > 10:
            buckets[feat] = {
                "high_accuracy": round(float(correct[high].mean()), 4),
                "low_accuracy": round(float(correct[low].mean()), 4),
                "high_n": int(high.sum()),
                "low_n": int(low.sum()),
            }
    return buckets


def run_eval(cutoff: str | None = None) -> dict:
    t_start = time.time()
    cutoff = cutoff or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    cache_dir = Path(__file__).parent / "sandbox" / "model_lab" / "output" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v2-eval] Loading engineered frame (cutoff={cutoff})...")
    df = load_or_build_engineered_frame_from_db(cutoff=cutoff, cache_dir=cache_dir, use_cache=True)
    if df.empty:
        raise RuntimeError("Engineered frame is empty")

    fs = load_or_build_feature_sets(
        df,
        min_coverage=C.MIN_COVERAGE,
        corr_threshold=C.CORR_THRESHOLD,
        top_k=C.K_FEATURES,
        cache_dir=cache_dir,
        use_cache=True,
        selected_by=C.SELECTED_BY,
    )
    feature_cols = list(fs["selected"])
    print(f"[v2-eval] Selected {len(feature_cols)} features")

    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    settled["game_date"] = pd.to_datetime(settled["game_date"], errors="coerce")
    settled = settled.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)

    fold_results: list[dict] = []
    all_val_frames: list[pd.DataFrame] = []
    all_means: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_imps: list[np.ndarray] = []

    for train_end, val_end in FOLDS:
        train_ts = pd.Timestamp(train_end)
        val_ts = pd.Timestamp(val_end)
        tr = settled[settled["game_date"] < train_ts]
        va = settled[(settled["game_date"] >= train_ts) & (settled["game_date"] < val_ts)]
        if tr.empty or va.empty:
            continue
        sw = recency_weights(tr["season"], train_ts.year - 1, C.RECENCY_HALF_LIFE)
        t0 = time.time()
        mean_p, imp = _fit_deterministic(tr, va, feature_cols, sw)
        std_p = np.zeros(len(mean_p), dtype=float)
        y_vl = va["home_win"].astype(int).to_numpy()

        brier = float(brier_score_loss(y_vl, mean_p))
        try:
            ll = float(log_loss(y_vl, np.clip(mean_p, 1e-6, 1 - 1e-6)))
        except Exception:
            ll = float("nan")

        fold_results.append({
            "fold": f"{train_ts.year}",
            "train_end": train_end,
            "val_end": val_end,
            "train_rows": int(len(tr)),
            "val_rows": int(len(va)),
            "brier": round(brier, 6),
            "log_loss": round(ll, 6) if not math.isnan(ll) else None,
            "std_mean": 0.0,
            "std_p95": 0.0,
        })
        print(f"[v2-eval] fold={train_end} val_rows={len(va)} brier={brier:.4f} "
              f"elapsed={time.time()-t0:.1f}s")

        all_val_frames.append(va.assign(_p_mean=mean_p, _p_std=std_p, season=va["season"]))
        all_means.append(mean_p)
        all_y.append(y_vl)
        if imp is not None:
            all_imps.append(np.asarray(imp, dtype=np.float64))

    if not fold_results:
        raise RuntimeError("No folds produced")

    full = pd.concat(all_val_frames, ignore_index=True)
    all_probs = np.concatenate(all_means)
    all_y_concat = np.concatenate(all_y).astype(float)
    all_mkt = pd.to_numeric(full.get("market_implied_prob"), errors="coerce").to_numpy()
    all_dates = pd.to_datetime(full["game_date"])

    overall_brier = float(brier_score_loss(all_y_concat, all_probs))
    try:
        overall_ll = float(log_loss(all_y_concat, np.clip(all_probs, 1e-6, 1 - 1e-6)))
    except Exception:
        overall_ll = float("nan")

    roi_summary = _evaluate_kelly(
        full,
        edge_min=C.EDGE_MIN,
        edge_max=C.EDGE_MAX,
        ml_cap=C.ML_CAP,
        kelly_factor=C.KELLY_FACTOR,
    )

    # edge distribution (model_prob - market_implied_prob)
    edges = all_probs - all_mkt
    edges_valid = edges[~np.isnan(edges)]
    edge_distribution: dict = {}
    if edges_valid.size:
        hist, bin_edges = np.histogram(edges_valid, bins=40)
        edge_distribution = {
            "counts": hist.tolist(),
            "bin_edges": [round(float(b), 4) for b in bin_edges],
        }

    # monthly accuracy
    years = all_dates.dt.year.to_numpy()
    months = all_dates.dt.month.to_numpy()
    keys = years * 100 + months
    monthly: list[dict] = []
    for key in sorted({int(k) for k in keys}):
        mask = keys == key
        yr = int(key // 100)
        mo = int(key % 100)
        y_m = all_y_concat[mask]
        p_m = all_probs[mask]
        mk_m = all_mkt[mask]
        mk_valid = ~np.isnan(mk_m)
        accuracy = float(((p_m > 0.5) == (y_m > 0.5)).mean())
        brier_m = float(np.mean((p_m - y_m) ** 2))
        if mk_valid.any():
            market_brier = float(np.mean((mk_m[mk_valid] - y_m[mk_valid]) ** 2))
            market_accuracy = float(((mk_m[mk_valid] > 0.5) == (y_m[mk_valid] > 0.5)).mean())
        else:
            market_brier = None
            market_accuracy = None
        monthly.append({
            "year": yr,
            "month": mo,
            "year_month": f"{yr:04d}-{mo:02d}",
            "count": int(mask.sum()),
            "brier": round(brier_m, 6),
            "accuracy": round(accuracy, 6),
            "market_brier": round(market_brier, 6) if market_brier is not None else None,
            "market_accuracy": round(market_accuracy, 6) if market_accuracy is not None else None,
        })

    # feature accuracy on last fold
    last_val = all_val_frames[-1].reset_index(drop=True)
    last_probs = all_means[-1]
    last_y = all_y[-1]
    feature_accuracy = _bucketed_feature_accuracy(last_val, last_probs, last_y, feature_cols)

    feature_importances = None
    if all_imps:
        avg_imp = np.mean(np.stack(all_imps, axis=0), axis=0)
        pairs = sorted(
            ({"feature": f, "importance": float(v)} for f, v in zip(feature_cols, avg_imp)),
            key=lambda r: r["importance"],
            reverse=True,
        )
        feature_importances = pairs

    duration = time.time() - t_start
    metrics = {
        "model_type": "lgbm",
        "deterministic": True,
        "overall_brier": round(overall_brier, 6),
        "overall_log_loss": round(overall_ll, 6) if not math.isnan(overall_ll) else None,
        "mean_brier": round(float(np.mean([r["brier"] for r in fold_results])), 6),
        "mean_roi": roi_summary["roi"],
        "total_bets": roi_summary["bets"],
        "win_rate": roi_summary["win_rate"],
        "num_folds": len(fold_results),
        "num_features": len(feature_cols),
        "training_rows": int(fold_results[-1]["train_rows"]),
        "val_rows": int(fold_results[-1]["val_rows"]),
        "duration_seconds": round(duration, 1),
        "fold_results": fold_results,
        "roi_summary": roi_summary,
        "edge_distribution": edge_distribution,
        "monthly_accuracy": monthly,
        "feature_accuracy": feature_accuracy,
        "feature_importances": feature_importances,
        "config": {
            "k_features": C.K_FEATURES,
            "edge_min": C.EDGE_MIN,
            "edge_max": C.EDGE_MAX,
            "ml_cap": C.ML_CAP,
            "kelly_factor": C.KELLY_FACTOR,
            "min_coverage": C.MIN_COVERAGE,
            "corr_threshold": C.CORR_THRESHOLD,
            "selected_by": C.SELECTED_BY,
            "recency_half_life": C.RECENCY_HALF_LIFE,
            "folds": [list(f) for f in FOLDS],
        },
    }

    artifact_id = _persist_metrics(metrics)
    print(f"[v2-eval] Persisted metrics to model_artifacts_v2.id={artifact_id} "
          f"(brier={overall_brier:.4f} roi={roi_summary['roi']:.4f} "
          f"bets={roi_summary['bets']} duration={duration:.1f}s)")
    return metrics


def _persist_metrics(metrics: dict) -> int | None:
    with DB.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM model_artifacts_v2 ORDER BY created_at DESC NULLS LAST, id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                print("[v2-eval] No model_artifacts_v2 rows exist; metrics not saved.")
                return None
            artifact_id = int(row[0])
            mt = metrics.get("model_type")
            if isinstance(mt, str) and mt:
                cur.execute(
                    "UPDATE model_artifacts_v2 SET metrics = %s, model_type = %s WHERE id = %s",
                    (json.dumps(metrics), mt, artifact_id),
                )
            else:
                cur.execute(
                    "UPDATE model_artifacts_v2 SET metrics = %s WHERE id = %s",
                    (json.dumps(metrics), artifact_id),
                )
        conn.commit()
    return artifact_id


if __name__ == "__main__":
    run_eval()
