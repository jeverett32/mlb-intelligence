"""Walk-forward validation by season (expanding window)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from . import calibration as C
from . import data as D
from . import models as M


@dataclass
class FoldOut:
    season: int
    train_rows: int
    val_rows: int
    probs_uncal: np.ndarray
    probs_calibrated: dict[str, np.ndarray]
    probs_early_uncal: np.ndarray | None
    val_index: np.ndarray
    early_mask: np.ndarray
    metrics: dict


def _fit_train_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: list[str],
    *,
    model_kind: str,
    label: str = "home_win",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on train; return (val_probs, train_in_sample_probs)."""
    y = train_df[label].astype(int)
    if model_kind == "lr_offset":
        if "market_implied_prob" not in train_df.columns or "market_implied_prob" not in val_df.columns:
            raise ValueError("lr_offset requires market_implied_prob column")
        mp_tr = pd.to_numeric(train_df["market_implied_prob"], errors="coerce").to_numpy()
        mp_vl = pd.to_numeric(val_df["market_implied_prob"], errors="coerce").to_numpy()
        mp_tr = np.clip(mp_tr, 1e-6, 1 - 1e-6)
        mp_vl = np.clip(mp_vl, 1e-6, 1 - 1e-6)
        off_tr = np.log(mp_tr / (1 - mp_tr))
        off_vl = np.log(mp_vl / (1 - mp_vl))
        model = M.fit_offset_logit(train_df[feats].to_numpy(), y.to_numpy(), offset=off_tr, l2=1.0)
        p_vl = model.predict_proba(val_df[feats].to_numpy(), off_vl)[:, 1]
        p_tr = model.predict_proba(train_df[feats].to_numpy(), off_tr)[:, 1]
        return p_vl, p_tr
    if model_kind == "lgbm":
        model = M.make_lgbm()
        model.fit(train_df[feats], y)
        return (
            model.predict_proba(val_df[feats])[:, 1],
            model.predict_proba(train_df[feats])[:, 1],
        )
    pipe = M.make_lr()
    pipe.fit(train_df[feats], y)
    return (
        pipe.predict_proba(val_df[feats])[:, 1],
        pipe.predict_proba(train_df[feats])[:, 1],
    )


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_kind: str = "lr",
    seasons: list[int] | None = None,
    early_cutoff: int | None = 25,
    min_train_seasons: int = 3,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> list[FoldOut]:
    """Walk forward season by season; train on all prior seasons."""
    df = df.sort_values("game_date").reset_index(drop=True)
    if seasons is None:
        seasons = sorted(df["season"].unique().tolist())
    out: list[FoldOut] = []

    early_feats = D.early_features(df)

    it = seasons
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            it = tqdm(seasons, desc=progress_desc or f"walk_forward[{model_kind}]", unit="season")
        except Exception:
            it = seasons

    for s in it:
        train_df = df[df["season"] < s].copy()
        val_df = df[df["season"] == s].copy()
        if train_df.empty or val_df.empty:
            continue
        if train_df["season"].nunique() < min_train_seasons:
            continue

        probs, train_in_probs = _fit_train_val(
            train_df, val_df, feature_cols, model_kind=model_kind,
        )

        cal = C.calibrate_all(
            train_in_probs, train_df["home_win"].to_numpy(), probs,
        )

        early_tr, _ = D.split_early(train_df, early_cutoff or 0)
        early_vl, _ = D.split_early(val_df, early_cutoff or 0)
        probs_early = None
        if early_cutoff and early_feats and early_tr.any() and early_vl.any():
            sub_tr = train_df.loc[early_tr]
            sub_vl = val_df.loc[early_vl]
            if sub_tr["home_win"].nunique() == 2:
                pipe = M.make_early_lr()
                pipe.fit(sub_tr[early_feats], sub_tr["home_win"])
                early_p = pipe.predict_proba(sub_vl[early_feats])[:, 1]
                probs_early = np.full(len(val_df), np.nan)
                probs_early[np.where(early_vl)[0]] = early_p

        y = val_df["home_win"].to_numpy()
        metrics = {
            method: {
                "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
                "brier": float(brier_score_loss(y, p)),
            }
            for method, p in cal.items()
        }
        out.append(FoldOut(
            season=int(s),
            train_rows=int(len(train_df)),
            val_rows=int(len(val_df)),
            probs_uncal=probs,
            probs_calibrated=cal,
            probs_early_uncal=probs_early,
            val_index=val_df.index.to_numpy(),
            early_mask=early_vl.to_numpy(),
            metrics=metrics,
        ))
    return out


def stitched_probs(
    df: pd.DataFrame,
    folds: list[FoldOut],
    *,
    method: str = "none",
    use_early_specialist: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate val probs across folds. Returns (probs, original_index)."""
    probs_all: list[np.ndarray] = []
    idx_all: list[np.ndarray] = []
    for f in folds:
        p = f.probs_calibrated.get(method, f.probs_uncal).copy()
        if use_early_specialist and f.probs_early_uncal is not None:
            mask = ~np.isnan(f.probs_early_uncal)
            p[mask] = f.probs_early_uncal[mask]
        probs_all.append(p)
        idx_all.append(f.val_index)
    if not probs_all:
        return np.array([]), np.array([])
    return np.concatenate(probs_all), np.concatenate(idx_all)
