"""
V2 walk-forward evaluation.

Runs season-by-season walk-forward CV on the V2 LightGBM pipeline, computes
metrics (brier, roi, monthly accuracy, edge distribution, feature accuracy),
and persists the bundle into the latest model_artifacts_v2.metrics JSON so the
admin model-insights dashboard can render v2 the same way it does v1.

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
    load_or_build_feature_sets,
    recency_weights,
)
from models.model_v2.sandbox.model_lab.roi_eval import ml_to_dec, roi_sim
from models.model_v2.sandbox.model_lab.training.models import make_lgbm

MIN_TRAIN_SEASONS = 3


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", make_lgbm()),
    ])


def _bucketed_feature_accuracy(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    top_n: int = 20,
) -> dict:
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

    df = df[df["game_date"].notna()].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["home_win"]).copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    seasons = sorted(df["season"].dropna().unique().tolist())
    fold_results: list[dict] = []
    stitched_probs: list[np.ndarray] = []
    stitched_y: list[np.ndarray] = []
    stitched_mkt: list[np.ndarray] = []
    stitched_dates: list[np.ndarray] = []
    stitched_idx: list[np.ndarray] = []

    for s in seasons:
        train_df = df[df["season"] < s]
        val_df = df[df["season"] == s]
        if train_df.empty or val_df.empty:
            continue
        if train_df["season"].nunique() < MIN_TRAIN_SEASONS:
            continue

        sw = recency_weights(train_df["season"], int(s) - 1, C.RECENCY_HALF_LIFE)
        X_tr = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        y_tr = train_df["home_win"].astype(int)
        X_vl = val_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        y_vl = val_df["home_win"].astype(int).to_numpy()

        pipe = _build_pipeline()
        pipe.fit(X_tr, y_tr, model__sample_weight=sw)
        probs = pipe.predict_proba(X_vl)[:, 1]

        brier = float(brier_score_loss(y_vl, probs))
        try:
            ll = float(log_loss(y_vl, np.clip(probs, 1e-6, 1 - 1e-6)))
        except Exception:
            ll = float("nan")
        sim = roi_sim(val_df.reset_index(drop=True), probs, edge_thresh=C.EDGE_MIN)

        mkt = pd.to_numeric(val_df.get("market_implied_prob"), errors="coerce").to_numpy()

        fold_results.append({
            "season": int(s),
            "fold": len(fold_results) + 1,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "brier": round(brier, 6),
            "log_loss": round(ll, 6) if not math.isnan(ll) else None,
            "roi": sim["roi"],
            "n_bets": sim["bets"],
            "pnl": sim["pnl"],
            "win_rate": sim["win_rate"],
        })
        print(f"[v2-eval] season={int(s)} train_rows={len(train_df)} val_rows={len(val_df)} "
              f"brier={brier:.4f} roi={sim['roi']} bets={sim['bets']}")

        stitched_probs.append(probs)
        stitched_y.append(y_vl)
        stitched_mkt.append(mkt)
        stitched_dates.append(val_df["game_date"].to_numpy())
        stitched_idx.append(val_df.index.to_numpy())

    if not fold_results:
        raise RuntimeError("No walk-forward folds produced")

    all_probs = np.concatenate(stitched_probs)
    all_y = np.concatenate(stitched_y).astype(float)
    all_mkt = np.concatenate(stitched_mkt)
    all_dates = pd.to_datetime(np.concatenate(stitched_dates))
    all_idx = np.concatenate(stitched_idx)

    rois = [r["roi"] for r in fold_results if r["roi"] is not None]
    briers = [r["brier"] for r in fold_results if r["brier"] is not None]
    bets = [r["n_bets"] for r in fold_results]
    mean_roi = float(np.mean(rois)) if rois else None
    mean_brier = float(np.mean(briers)) if briers else None
    total_bets = int(np.sum(bets))

    # edge distribution (model_prob - market_implied_prob)
    edges = (all_probs - all_mkt)
    edges_valid = edges[~np.isnan(edges)]
    edge_distribution: dict = {}
    if edges_valid.size:
        hist, bin_edges = np.histogram(edges_valid, bins=40)
        edge_distribution = {
            "counts": hist.tolist(),
            "bin_edges": [round(float(b), 4) for b in bin_edges],
        }

    # monthly accuracy
    years = all_dates.year.values
    months = all_dates.month.values
    keys = years * 100 + months
    monthly: list[dict] = []
    for key in sorted({int(k) for k in keys}):
        mask = keys == key
        yr = int(key // 100)
        mo = int(key % 100)
        y_m = all_y[mask]
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

    # feature accuracy on last fold's val_df + its probs
    last_val_df = df.loc[stitched_idx[-1]]
    last_probs = stitched_probs[-1]
    last_y = stitched_y[-1]
    feature_accuracy = _bucketed_feature_accuracy(last_val_df, last_probs, last_y, feature_cols)

    duration = time.time() - t_start
    metrics = {
        "model_type": "lgbm_pipeline",
        "mean_brier": round(mean_brier, 6) if mean_brier is not None else None,
        "mean_roi": round(mean_roi, 6) if mean_roi is not None else None,
        "total_bets": total_bets,
        "num_folds": len(fold_results),
        "num_features": len(feature_cols),
        "training_rows": int(fold_results[-1]["train_rows"]),
        "val_rows": int(fold_results[-1]["val_rows"]),
        "duration_seconds": round(duration, 1),
        "fold_results": fold_results,
        "edge_distribution": edge_distribution,
        "monthly_accuracy": monthly,
        "feature_accuracy": feature_accuracy,
        "config": {
            "k_features": C.K_FEATURES,
            "edge_min": C.EDGE_MIN,
            "edge_max": C.EDGE_MAX,
            "kelly_factor": C.KELLY_FACTOR,
            "min_coverage": C.MIN_COVERAGE,
            "corr_threshold": C.CORR_THRESHOLD,
            "selected_by": C.SELECTED_BY,
            "recency_half_life": C.RECENCY_HALF_LIFE,
            "min_train_seasons": MIN_TRAIN_SEASONS,
        },
    }

    artifact_id = _persist_metrics(metrics)
    print(f"[v2-eval] Persisted metrics to model_artifacts_v2.id={artifact_id} "
          f"(mean_brier={mean_brier} mean_roi={mean_roi} folds={len(fold_results)} "
          f"duration={duration:.1f}s)")
    return metrics


def _persist_metrics(metrics: dict) -> int | None:
    """Update metrics JSON on the most recent model_artifacts_v2 row."""
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
            cur.execute(
                "UPDATE model_artifacts_v2 SET metrics = %s WHERE id = %s",
                (json.dumps(metrics), artifact_id),
            )
        conn.commit()
    return artifact_id


if __name__ == "__main__":
    run_eval()
