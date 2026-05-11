#!/usr/bin/env python3
"""Train sandbox-only baseline model from master_sandbox_mlb.csv."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import models.model_v1.train as production_train  # noqa: E402
from models.model_v2.sandbox.model_lab import features, planned_features  # noqa: E402


DEFAULT_INPUT = LAB_DIR / "output" / "master_sandbox_mlb.csv"
DEFAULT_METRICS = LAB_DIR / "output" / "metrics.json"
DEFAULT_MODEL = LAB_DIR / "output" / "candidate_lr.pkl"

FOLDS = [
    ("2022-01-01", "2022-01-01", "2023-01-01"),
    ("2023-01-01", "2023-01-01", "2024-01-01"),
    ("2024-01-01", "2024-01-01", "2025-01-01"),
    ("2025-01-01", "2025-01-01", "2026-01-01"),
]


def _load_frame(path: Path, cutoff: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = features.apply_sandbox_contract(df, features.SandboxContract(cutoff=cutoff))
    if (pd.to_datetime(df["game_date"]) >= pd.Timestamp(cutoff)).any():
        raise ValueError(f"training input contains rows on/after {cutoff}")
    return df


def _feature_cols(df: pd.DataFrame, feature_set: str, min_coverage: float) -> list[str]:
    production_cols = [col for col in production_train.FEATURE_COLUMNS if col in df.columns]
    if feature_set == "production":
        return production_cols
    sandbox_cols = planned_features.selectable_sandbox_features(df, min_coverage=min_coverage)
    return list(dict.fromkeys(production_cols + sandbox_cols))


def _pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
        ]
    )


def _score_market(frame: pd.DataFrame) -> dict | None:
    if "market_implied_prob" not in frame.columns:
        return None
    y = frame["home_win"].astype(int)
    p = pd.to_numeric(frame["market_implied_prob"], errors="coerce").clip(0.001, 0.999)
    mask = p.notna()
    if not mask.any():
        return None
    return {
        "log_loss": float(log_loss(y[mask], p[mask])),
        "brier": float(brier_score_loss(y[mask], p[mask])),
    }


def run_backtest(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    results: list[dict] = []

    for train_end, val_start, val_end in FOLDS:
        train_mask = settled["game_date"] < pd.Timestamp(train_end)
        val_mask = (settled["game_date"] >= pd.Timestamp(val_start)) & (settled["game_date"] < pd.Timestamp(val_end))
        train_df = settled.loc[train_mask].copy()
        val_df = settled.loc[val_mask].copy()
        if train_df.empty or val_df.empty:
            continue

        pipe = _pipeline()
        pipe.fit(train_df[feature_cols], train_df["home_win"])
        prob = pipe.predict_proba(val_df[feature_cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = None
        if val_df["home_win"].nunique() == 2:
            auc = float(roc_auc_score(val_df["home_win"], prob))
        results.append(
            {
                "fold": f"{val_start[:4]}",
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "log_loss": float(log_loss(val_df["home_win"], prob)),
                "brier": float(brier_score_loss(val_df["home_win"], prob)),
                "accuracy": float(accuracy_score(val_df["home_win"], pred)),
                "auc": auc,
                "market": _score_market(val_df),
            }
        )

    if not results:
        raise ValueError("no valid folds; build sandbox master first")

    return {
        "model": "logistic_regression_sandbox_baseline",
        "rows": int(len(settled)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "folds": results,
        "mean_log_loss": float(np.mean([r["log_loss"] for r in results])),
        "mean_brier": float(np.mean([r["brier"] for r in results])),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in results])),
    }


def fit_final(df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    pipe = _pipeline()
    pipe.fit(settled[feature_cols], settled["home_win"])
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cutoff", default=features.DEFAULT_CUTOFF)
    parser.add_argument("--feature-set", choices=["production", "sandbox"], default="sandbox")
    parser.add_argument("--min-coverage", type=float, default=0.6)
    args = parser.parse_args()

    df = _load_frame(args.input, cutoff=args.cutoff)
    feature_cols = _feature_cols(df, feature_set=args.feature_set, min_coverage=args.min_coverage)
    if not feature_cols:
        raise ValueError("no production baseline feature columns found in sandbox master")

    metrics = run_backtest(df, feature_cols)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    model = fit_final(df, feature_cols)
    with args.model_out.open("wb") as fh:
        pickle.dump({"model": model, "features": feature_cols, "metrics": metrics}, fh)

    print(f"wrote {args.metrics}")
    print(f"wrote {args.model_out}")
    print(f"mean_log_loss={metrics['mean_log_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
