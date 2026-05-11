#!/usr/bin/env python3
"""Sandbox feature engineering, redundancy pruning, and selection.

Reads master_sandbox_mlb.csv read-only, performs all feature work in memory,
and writes a metrics + selected-features JSON plus a candidate model pickle.

Pipeline:
  1. Load + sandbox contract (drop legacy/live cols).
  2. Numeric coverage filter.
  3. Engineer combined/composite features (in-memory).
  4. Correlation cluster pruning (drop one per |r|>=THRESH cluster).
  5. L1 logistic for sparse selection (per-fold, count stability).
  6. Top-K feature list: either permutation importance (LR on holdout), or
     mean tree gain importances (LGBM or XGB) averaged over walk-forward train
     folds on the pruned set (``selected_by``).
  7. Time-series CV (2022/2023/2024/2025) with recency sample weights.
  8. Compare three model variants:
       - all_engineered: every numeric feature surviving step 2
       - corr_pruned: after step 4
       - selected: stable L1 + perm-importance survivors
       - market_anchored: selected + market_logit forced in via offset trick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab import features as sandbox_features  # noqa: E402

DEFAULT_INPUT = LAB_DIR / "output" / "master_sandbox_mlb.csv"
DEFAULT_OUT_DIR = LAB_DIR / "output"

ENGINEERING_CACHE_VERSION = 2

FOLDS = [
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

ID_LIKE_PATTERNS = ("_id", "umpire_id", "venue_id")
LEAK_COLS = {
    "home_score",
    "away_score",
    "home_win",
    "game_pk",
    "game_id",
    "game_time_utc",
    "season",
    "odds_source",
    "venue_name",
    "roof_type",
    "home_team",
    "away_team",
    "game_date",
    # Final-resolution metrics (post-game) we should never train on.
    "hg",
    "ag",
    # Near-constant flags caught by season_constancy_audit (always ~1.0).
    # Provide no signal; only inflate dimensionality.
    "home_bp_righty_available",
    "away_bp_righty_available",
    "home_bp_lefty_available",
    "away_bp_lefty_available",
}

LOW_VARIANCE_STD_FLOOR = 0.01  # below this std, treat as constant noise

CORR_THRESHOLD = 0.85
TOP_K_PERM = 24
RECENCY_HALF_LIFE_SEASONS = 4.0  # weight = 0.5 ** (age / half_life)

SELECTED_BY_CHOICES = ("lr_perm", "lgbm_gain", "xgb_gain")
SelectedBy = Literal["lr_perm", "lgbm_gain", "xgb_gain"]
TreeGainSelector = Literal["lgbm", "xgb"]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _fingerprint_path(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_or_build_engineered_frame(
    *,
    input_path: Path,
    cutoff: str,
    cache_dir: Path,
    use_cache: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _stable_hash(
        {
            "kind": "engineered_frame",
            "version": ENGINEERING_CACHE_VERSION,
            "input": _fingerprint_path(input_path),
            "cutoff": cutoff,
        }
    )
    pkl_path = cache_dir / f"engineered_frame__{key}.pkl"
    meta_path = cache_dir / f"engineered_frame__{key}.json"
    if use_cache and pkl_path.exists() and meta_path.exists():
        try:
            df = pd.read_pickle(pkl_path)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return df
        except Exception:
            pass

    df = pd.read_csv(input_path, low_memory=False)
    df = sandbox_features.apply_sandbox_contract(
        df, sandbox_features.SandboxContract(cutoff=cutoff)
    )
    df = engineer_combined_features(df)
    pd.to_pickle(df, pkl_path)
    meta_path.write_text(
        json.dumps(
            {
                "cache_key": key,
                "version": ENGINEERING_CACHE_VERSION,
                "input": _fingerprint_path(input_path),
                "cutoff": cutoff,
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return df


def load_or_build_feature_sets(
    df: pd.DataFrame,
    *,
    min_coverage: float,
    corr_threshold: float,
    top_k: int,
    cache_dir: Path,
    use_cache: bool,
    fold_year: int | None = None,
    perm_train_end: str = "2024-01-01",
    perm_val_end: str = "2025-01-01",
    selected_by: SelectedBy = "lr_perm",
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    game_date_min = None
    game_date_max = None
    if "game_date" in df.columns and len(df):
        gd = pd.to_datetime(df["game_date"], errors="coerce")
        game_date_min = str(gd.min()) if gd.notna().any() else None
        game_date_max = str(gd.max()) if gd.notna().any() else None
    key = _stable_hash(
        {
            "kind": "feature_sets",
            "version": ENGINEERING_CACHE_VERSION,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "game_date_min": game_date_min,
            "game_date_max": game_date_max,
            "min_coverage": min_coverage,
            "corr_threshold": corr_threshold,
            "top_k": top_k,
            "l1_C": 0.05,
            "perm_train_end": perm_train_end,
            "perm_val_end": perm_val_end,
            "fold_year": fold_year,
            "selected_by": selected_by,
        }
    )
    json_path = cache_dir / f"feature_sets__{key}.json"
    if use_cache and json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            cached_sb = payload.get("selected_by", "lr_perm")
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("cand"), list)
                and isinstance(payload.get("pruned"), list)
                and isinstance(payload.get("selected"), list)
                and cached_sb == selected_by
            ):
                return payload
        except Exception:
            pass

    if selected_by not in SELECTED_BY_CHOICES:
        raise ValueError(f"selected_by must be one of {SELECTED_BY_CHOICES}, got {selected_by!r}")

    cand = candidate_numeric_features(df, min_coverage)
    pruned, dropped_pairs = correlation_cluster_prune(df, cand, corr_threshold)
    l1_keep, l1_counts = l1_stability_select(df, pruned, C=0.05)
    if selected_by == "lr_perm":
        perm = permutation_importance_rank(df, l1_keep or pruned, perm_train_end, perm_val_end)
        perm_sorted = sorted(perm.items(), key=lambda kv: kv[1], reverse=True)
        selected = [c for c, _ in perm_sorted[:top_k]]
    else:
        tree_sel: TreeGainSelector = "lgbm" if selected_by == "lgbm_gain" else "xgb"
        imp = mean_tree_gain_importances(df, pruned, selector=tree_sel)
        perm_sorted = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        selected = [c for c, _ in perm_sorted[:top_k]]
    payload = {
        "cache_key": key,
        "version": ENGINEERING_CACHE_VERSION,
        "fold_year": fold_year,
        "selected_by": selected_by,
        "min_coverage": float(min_coverage),
        "corr_threshold": float(corr_threshold),
        "top_k": int(top_k),
        "cand": cand,
        "pruned": pruned,
        "dropped_pairs": dropped_pairs,
        "l1_keep": l1_keep,
        "l1_counts": l1_counts,
        "perm_sorted": [{"feature": c, "importance": float(v)} for c, v in perm_sorted],
        "selected": selected,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Engineering: composite features
# ---------------------------------------------------------------------------


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _logit(p: pd.Series, eps: float = 1e-4) -> pd.Series:
    p = p.clip(eps, 1 - eps)
    return np.log(p / (1 - p))


def engineer_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that combine signal already in the master CSV.

    All additions are nullable; never write back to the source CSV.
    """
    out = df.copy()

    market = _num(out, "market_implied_prob")
    out["market_logit"] = _logit(market)

    open_p = _num(out, "open_home_implied")
    out["market_open_logit"] = _logit(open_p)
    out["market_move_logit"] = out["market_logit"] - out["market_open_logit"]

    park = _num(out, "park_factor").fillna(1.0)
    wrc = _num(out, "wrc_plus_DIFF")
    woba = _num(out, "woba_DIFF")
    out["park_x_wrc_DIFF"] = park * wrc
    out["park_x_woba_DIFF"] = park * woba

    sp_fip = _num(out, "sp_fip_DIFF")
    bp_q = _num(out, "bp_quality_DIFF")
    bp_f = _num(out, "bp_freshness_DIFF")
    out["sp_x_bp_quality"] = sp_fip * bp_q
    out["sp_x_bp_freshness"] = sp_fip * bp_f
    out["bp_quality_x_freshness"] = bp_q * bp_f

    matchup = _num(out, "matchup_xwoba_DIFF")
    out["matchup_x_park"] = matchup * park
    out["matchup_x_sp"] = matchup * sp_fip

    weather_hr = _num(out, "weather_hr_boost_factor")
    out["weather_x_park"] = weather_hr * park
    out["weather_x_lineup_power"] = weather_hr * woba

    travel = _num(out, "travel_miles")
    rest = _num(out, "rest_days_DIFF")
    tz = _num(out, "travel_tz_shift")
    getaway = _num(out, "getaway_game_flag").fillna(0.0)
    out["travel_fatigue_idx"] = travel.abs() * (1.0 - rest.fillna(0.0).clip(-3, 3) / 3.0) + getaway * 50.0
    out["tz_x_getaway"] = tz.abs() * getaway

    pyth = _num(out, "pythagorean_DIFF")
    momentum = _num(out, "momentum_DIFF")
    luck = _num(out, "luck_DIFF")
    out["pyth_x_recent_form"] = pyth * momentum
    out["luck_regress"] = -luck * pyth  # regression to true talent

    rolling_era = _num(out, "rolling_era_DIFF")
    rolling_whip = _num(out, "rolling_whip_DIFF")
    out["rolling_staff_composite"] = -(rolling_era / 4.5) - rolling_whip

    closer_q = _num(out, "closer_quality_DIFF")
    setup_q = _num(out, "setup_quality_DIFF")
    out["high_lev_bp_DIFF"] = closer_q + setup_q

    catcher_fr = _num(out, "catcher_framing_DIFF")
    sp_csw = _num(out, "home_sp_csw_rate") - _num(out, "away_sp_csw_rate")
    out["sp_csw_DIFF"] = sp_csw
    out["framing_x_csw"] = catcher_fr * sp_csw

    home_ws = _num(out, "h_win_pct_15")
    away_ws = _num(out, "a_win_pct_15")
    out["recent_form_DIFF_15"] = home_ws - away_ws
    out["recent_run_diff_DIFF_15"] = _num(out, "h_run_diff_avg_15") - _num(out, "a_run_diff_avg_15")

    return out


# ---------------------------------------------------------------------------
# Coverage + redundancy
# ---------------------------------------------------------------------------


def candidate_numeric_features(df: pd.DataFrame, min_coverage: float) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in LEAK_COLS:
            continue
        if any(col.endswith(p) for p in ID_LIKE_PATTERNS):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() < min_coverage:
            continue
        if s.nunique(dropna=True) < 2:
            continue
        if s.std(skipna=True) < LOW_VARIANCE_STD_FLOOR:
            continue
        cols.append(col)
    return cols


def correlation_cluster_prune(
    df: pd.DataFrame, cols: list[str], threshold: float
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy: walk cols by descending |corr with target| if available, else by name.

    Drop later col when |r| >= threshold with kept col. Returns (kept, dropped pairs).
    """
    if "home_win" in df.columns:
        target = pd.to_numeric(df["home_win"], errors="coerce")
        priority = []
        for c in cols:
            r = pd.to_numeric(df[c], errors="coerce").corr(target)
            priority.append((abs(r) if pd.notna(r) else 0.0, c))
        priority.sort(reverse=True)
        order = [c for _, c in priority]
    else:
        order = list(cols)

    sub = df[order].apply(pd.to_numeric, errors="coerce")
    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    kept_arr: dict[str, np.ndarray] = {}
    for col in order:
        x = sub[col].values
        ok = True
        for k in kept:
            mask = ~np.isnan(x) & ~np.isnan(kept_arr[k])
            if mask.sum() < 100:
                continue
            r = np.corrcoef(x[mask], kept_arr[k][mask])[0, 1]
            if np.isfinite(r) and abs(r) >= threshold:
                dropped.append((col, k, float(r)))
                ok = False
                break
        if ok:
            kept.append(col)
            kept_arr[col] = x
    return kept, dropped


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------


def recency_weights(seasons: pd.Series, ref_season: int, half_life: float) -> np.ndarray:
    age = (ref_season - seasons.astype(float)).clip(lower=0.0)
    return np.power(0.5, age / half_life)


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------


def make_pipeline(C: float = 0.5, penalty: str = "l2") -> Pipeline:
    if penalty == "l1":
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=C, max_iter=5000, random_state=42
        )
    else:
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )


def score_market(frame: pd.DataFrame) -> dict | None:
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


@dataclass
class FoldResult:
    fold: str
    train_rows: int
    val_rows: int
    log_loss: float
    brier: float
    accuracy: float
    auc: float | None
    market: dict | None


@dataclass
class Variant:
    name: str
    features: list[str]
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "feature_count": len(self.features),
            "features": self.features,
            "folds": [f.__dict__ for f in self.folds],
            "mean_log_loss": float(np.mean([f.log_loss for f in self.folds])) if self.folds else None,
            "mean_brier": float(np.mean([f.brier for f in self.folds])) if self.folds else None,
            "mean_accuracy": float(np.mean([f.accuracy for f in self.folds])) if self.folds else None,
        }


def _fit_predict(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    sample_weight: np.ndarray | None,
    C: float = 0.5,
    penalty: str = "l2",
) -> np.ndarray:
    pipe = make_pipeline(C=C, penalty=penalty)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    X_tr = train[cols].apply(pd.to_numeric, errors="coerce")
    X_va = val[cols].apply(pd.to_numeric, errors="coerce")
    pipe.fit(X_tr, train["home_win"], **fit_kwargs)
    return pipe.predict_proba(X_va)[:, 1]


def _market_anchored_predict(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    sample_weight: np.ndarray | None,
    C: float = 0.5,
) -> np.ndarray:
    """Fit residual to logit(market): final prob = sigmoid(market_logit + adjustment).

    cols should NOT contain market_logit; we pass it as offset.
    """
    if "market_logit" not in train.columns:
        return _fit_predict(train, val, cols, sample_weight, C=C)
    train = train.copy()
    val = val.copy()
    market_train = train["market_logit"].fillna(0.0).values
    market_val = val["market_logit"].fillna(0.0).values
    # Target stays binary; we approximate offset by including market_logit as an
    # un-regularized feature with high weight via duplicating (cheap proxy).
    cols2 = [c for c in cols if c != "market_logit"] + ["market_logit"]
    pipe = make_pipeline(C=C, penalty="l2")
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    X_tr = train[cols2].apply(pd.to_numeric, errors="coerce")
    X_va = val[cols2].apply(pd.to_numeric, errors="coerce")
    pipe.fit(X_tr, train["home_win"], **fit_kwargs)
    return pipe.predict_proba(X_va)[:, 1]
    # Note: scikit's LogisticRegression doesn't support `offset` directly.
    # Including market_logit as a feature lets the model learn its weight; with
    # standardization+L2 it converges near 1.0 when other features are weak.
    # Untouched market_train/market_val retained for clarity (no-op).
    _ = market_train, market_val


def run_variant(
    df: pd.DataFrame,
    variant: Variant,
    *,
    use_recency_weights: bool,
    market_anchored: bool = False,
) -> Variant:
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    for train_end, val_end in FOLDS:
        train_mask = settled["game_date"] < pd.Timestamp(train_end)
        val_mask = (settled["game_date"] >= pd.Timestamp(train_end)) & (
            settled["game_date"] < pd.Timestamp(val_end)
        )
        train_df = settled.loc[train_mask]
        val_df = settled.loc[val_mask]
        if train_df.empty or val_df.empty:
            continue
        sw = None
        if use_recency_weights:
            ref_season = pd.Timestamp(train_end).year - 1
            sw = recency_weights(train_df["season"], ref_season, RECENCY_HALF_LIFE_SEASONS)
        if market_anchored:
            prob = _market_anchored_predict(train_df, val_df, variant.features, sw)
        else:
            prob = _fit_predict(train_df, val_df, variant.features, sw)
        pred = (prob >= 0.5).astype(int)
        auc = None
        if val_df["home_win"].nunique() == 2:
            auc = float(roc_auc_score(val_df["home_win"], prob))
        variant.folds.append(
            FoldResult(
                fold=str(pd.Timestamp(train_end).year),
                train_rows=int(len(train_df)),
                val_rows=int(len(val_df)),
                log_loss=float(log_loss(val_df["home_win"], prob)),
                brier=float(brier_score_loss(val_df["home_win"], prob)),
                accuracy=float(accuracy_score(val_df["home_win"], pred)),
                auc=auc,
                market=score_market(val_df),
            )
        )
    return variant


# ---------------------------------------------------------------------------
# Selection: L1 stability + permutation importance + tree gain ranks
# ---------------------------------------------------------------------------


def _tree_gain_pipeline(selector: TreeGainSelector) -> Pipeline:
    from models.model_v2.sandbox.model_lab.training.models import make_lgbm, make_xgb

    if selector == "lgbm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", make_lgbm()),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_xgb()),
        ]
    )


def mean_tree_gain_importances(
    df: pd.DataFrame,
    pruned_cols: list[str],
    *,
    selector: TreeGainSelector,
    folds: Iterable[tuple[str, str]] = FOLDS,
    use_recency_weights: bool = True,
) -> dict[str, float]:
    """Mean ``feature_importances_`` over walk-forward train folds (no val leakage)."""
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    settled["game_date"] = pd.to_datetime(settled["game_date"], errors="coerce")
    acc = np.zeros(len(pruned_cols), dtype=np.float64)
    n = 0
    for train_end, _val_end in folds:
        train_df = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        if train_df.empty:
            continue
        sw = None
        if use_recency_weights:
            ref_season = pd.Timestamp(train_end).year - 1
            sw = recency_weights(train_df["season"], ref_season, RECENCY_HALF_LIFE_SEASONS)
        pipe = _tree_gain_pipeline(selector)
        kwargs: dict = {}
        if sw is not None:
            kwargs["model__sample_weight"] = sw
        X_tr = train_df[pruned_cols].apply(pd.to_numeric, errors="coerce")
        pipe.fit(X_tr, train_df["home_win"], **kwargs)
        imp = pipe.named_steps["model"].feature_importances_.astype(np.float64, copy=False)
        acc += imp
        n += 1
    if n == 0:
        return {}
    mean_imp = acc / float(n)
    return {c: float(v) for c, v in zip(pruned_cols, mean_imp)}


def l1_stability_select(
    df: pd.DataFrame,
    cols: list[str],
    *,
    C: float = 0.05,
    folds: Iterable[tuple[str, str]] = FOLDS,
    use_recency_weights: bool = True,
) -> tuple[list[str], dict[str, int]]:
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    counts: dict[str, int] = {c: 0 for c in cols}
    n_folds = 0
    for train_end, _val_end in folds:
        train_df = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        if train_df.empty:
            continue
        n_folds += 1
        sw = None
        if use_recency_weights:
            ref_season = pd.Timestamp(train_end).year - 1
            sw = recency_weights(train_df["season"], ref_season, RECENCY_HALF_LIFE_SEASONS)
        pipe = make_pipeline(C=C, penalty="l1")
        kwargs = {}
        if sw is not None:
            kwargs["model__sample_weight"] = sw
        X_tr = train_df[cols].apply(pd.to_numeric, errors="coerce")
        pipe.fit(X_tr, train_df["home_win"], **kwargs)
        coef = pipe.named_steps["model"].coef_.ravel()
        for c, w in zip(cols, coef):
            if abs(w) > 1e-6:
                counts[c] += 1
    threshold = max(1, int(np.ceil(n_folds * 0.5)))
    selected = [c for c, n in counts.items() if n >= threshold]
    return selected, counts


def permutation_importance_rank(
    df: pd.DataFrame,
    cols: list[str],
    train_end: str,
    val_end: str,
    *,
    use_recency_weights: bool = True,
    n_repeats: int = 5,
) -> dict[str, float]:
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    train_df = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
    val_df = settled.loc[
        (settled["game_date"] >= pd.Timestamp(train_end))
        & (settled["game_date"] < pd.Timestamp(val_end))
    ]
    if train_df.empty or val_df.empty:
        return {}
    sw = None
    if use_recency_weights:
        ref_season = pd.Timestamp(train_end).year - 1
        sw = recency_weights(train_df["season"], ref_season, RECENCY_HALF_LIFE_SEASONS)
    pipe = make_pipeline(C=0.5, penalty="l2")
    kwargs = {}
    if sw is not None:
        kwargs["model__sample_weight"] = sw
    X_tr = train_df[cols].apply(pd.to_numeric, errors="coerce")
    X_va = val_df[cols].apply(pd.to_numeric, errors="coerce")
    pipe.fit(X_tr, train_df["home_win"], **kwargs)
    result = permutation_importance(
        pipe,
        X_va,
        val_df["home_win"],
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_log_loss",
        n_jobs=1,
    )
    return {c: float(m) for c, m in zip(cols, result.importances_mean)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cutoff", default=sandbox_features.DEFAULT_CUTOFF)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=TOP_K_PERM)
    parser.add_argument(
        "--selected-by",
        choices=SELECTED_BY_CHOICES,
        default="lr_perm",
        help="top-K list: lr_perm (L1+permutation), lgbm_gain, or xgb_gain on pruned set",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache engineered frame + feature sets under out-dir/cache/ (default: on).",
    )
    args = parser.parse_args()

    cache_dir = args.out_dir / "cache"
    print(f"loading+engineering (cache={'on' if args.use_cache else 'off'})")
    df = load_or_build_engineered_frame(
        input_path=args.input,
        cutoff=args.cutoff,
        cache_dir=cache_dir,
        use_cache=args.use_cache,
    )
    print(f"rows={len(df):,} cols={df.shape[1]}")

    print("building feature sets")
    fs = load_or_build_feature_sets(
        df,
        min_coverage=args.min_coverage,
        corr_threshold=args.corr_threshold,
        top_k=args.top_k,
        cache_dir=cache_dir,
        use_cache=args.use_cache,
        selected_by=args.selected_by,
    )
    cand = fs["cand"]
    pruned = fs["pruned"]
    dropped_pairs = fs["dropped_pairs"]
    l1_keep = fs["l1_keep"]
    l1_counts = fs["l1_counts"]
    perm_sorted = [(d["feature"], d["importance"]) for d in fs["perm_sorted"]]
    selected = fs["selected"]
    print(
        f"cand={len(cand)} pruned={len(pruned)} l1={len(l1_keep)} selected={len(selected)}"
    )

    variants: list[Variant] = []

    print("running variants")
    variants.append(
        run_variant(
            df,
            Variant(name="all_engineered_no_recency", features=cand),
            use_recency_weights=False,
        )
    )
    variants.append(
        run_variant(
            df,
            Variant(name="all_engineered_recency", features=cand),
            use_recency_weights=True,
        )
    )
    variants.append(
        run_variant(
            df,
            Variant(name="corr_pruned_recency", features=pruned),
            use_recency_weights=True,
        )
    )
    variants.append(
        run_variant(
            df,
            Variant(name="l1_stable_recency", features=l1_keep or pruned),
            use_recency_weights=True,
        )
    )
    sel_topk_name = (
        "selected_topk_recency"
        if args.selected_by == "lr_perm"
        else f"selected_topk_{args.selected_by}_recency"
    )
    variants.append(
        run_variant(
            df,
            Variant(name=sel_topk_name, features=selected),
            use_recency_weights=True,
        )
    )
    variants.append(
        run_variant(
            df,
            Variant(name=f"market_anchored_{sel_topk_name}", features=selected),
            use_recency_weights=True,
            market_anchored=True,
        )
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "input": str(args.input),
        "selected_by": args.selected_by,
        "rows": int(len(df)),
        "candidate_count": len(cand),
        "corr_pruned_count": len(pruned),
        "l1_stable_count": len(l1_keep),
        "selected_count": len(selected),
        "corr_dropped_pairs": dropped_pairs[:200],
        "l1_selection_counts": l1_counts,
        "permutation_ranking": [
            {"feature": c, "importance": float(v)} for c, v in perm_sorted
        ],
        "selected_features": selected,
        "variants": [v.summary() for v in variants],
    }
    metrics_path = out_dir / "metrics_feature_engineering.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")

    # Fit final on selected_topk + recency, save
    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    sw = recency_weights(settled["season"], settled["season"].max(), RECENCY_HALF_LIFE_SEASONS)
    final_pipe = make_pipeline(C=0.5, penalty="l2")
    final_pipe.fit(settled[selected], settled["home_win"], model__sample_weight=sw)
    model_path = out_dir / "candidate_lr_engineered.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(
            {
                "model": final_pipe,
                "features": selected,
                "metrics": [v.summary() for v in variants],
            },
            fh,
        )
    print(f"wrote {model_path}")

    print("\nvariant summary (mean log_loss):")
    for v in variants:
        s = v.summary()
        print(
            f"  {v.name:35s} feat={s['feature_count']:3d} "
            f"mean_log_loss={s['mean_log_loss']:.4f} "
            f"mean_brier={s['mean_brier']:.4f} "
            f"mean_acc={s['mean_accuracy']:.4f}"
        )
    if variants[0].folds:
        market_mean = float(np.mean([f.market["log_loss"] for f in variants[0].folds if f.market]))
        print(f"  market_baseline                    mean_log_loss={market_mean:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
