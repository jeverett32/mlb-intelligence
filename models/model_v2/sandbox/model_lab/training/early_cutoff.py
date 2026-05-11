"""Sweep early-season cutoff to find best games-played threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from . import data as D
from . import walk_forward as W


DEFAULT_CUTOFFS = (0, 15, 20, 25, 30, 40)


def sweep(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    seasons: list[int] | None = None,
    model_kind: str = "lr",
    calibration_method: str = "none",
) -> pd.DataFrame:
    rows: list[dict] = []
    y_full = df["home_win"].to_numpy()

    for cut in cutoffs:
        folds = W.walk_forward(
            df, feature_cols,
            model_kind=model_kind,
            seasons=seasons,
            early_cutoff=cut if cut > 0 else None,
        )
        if not folds:
            continue
        probs, idx = W.stitched_probs(
            df, folds,
            method=calibration_method,
            use_early_specialist=cut > 0,
        )
        if probs.size == 0:
            continue
        y = y_full[idx]
        rows.append({
            "early_cutoff": cut,
            "n_val": int(len(probs)),
            "log_loss": float(log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y, probs)),
        })
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
