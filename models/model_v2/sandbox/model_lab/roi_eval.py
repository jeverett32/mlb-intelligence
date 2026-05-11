#!/usr/bin/env python3
"""ROI simulation across feature engineering variants.

Re-fits each variant per time-series fold and evaluates flat-stake ROI
against close moneyline odds at multiple edge thresholds.

Model backends: logistic regression (default), LightGBM, XGBoost, or averaged
LR+LGBM (`--model`). Use `--selected-by` to build the top-K list from LR
permutation importance (default) or from mean LGBM / XGB gain on the pruned set.
Use `--sweep-top-k` for a compact top-K ablation. `--lgbm-importance-pruned`
prints tree gain ranks on the correlation-pruned column set averaged over
train-side folds.

Long runs buffer stdout unless attached to a TTY; use ``python -u`` or
``PYTHONUNBUFFERED=1`` to see fold lines as they complete.
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab import features as sf
from models.model_v2.sandbox.model_lab.feature_engineer import (
    FOLDS,
    SELECTED_BY_CHOICES,
    SelectedBy,
    load_or_build_engineered_frame,
    load_or_build_feature_sets,
    make_pipeline,
    recency_weights,
)

INPUT = LAB_DIR / "output" / "master_sandbox_mlb.csv"
OUT_DIR = LAB_DIR / "output"
EDGE_THRESHOLDS = (0.00, 0.02, 0.05)

ModelName = Literal["lr", "lgbm", "xgb", "ensemble"]


def _to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce")


def _model_fit_kw(sample_weight: np.ndarray | None) -> dict:
    return {"model__sample_weight": sample_weight} if sample_weight is not None else {}


def _make_lgbm_pipeline() -> Pipeline:
    from models.model_v2.sandbox.model_lab.training.models import make_lgbm

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_lgbm()),
        ]
    )


def _make_xgb_pipeline() -> Pipeline:
    from models.model_v2.sandbox.model_lab.training.models import make_xgb

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_xgb()),
        ]
    )


def fit_predict_proba(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    sample_weight: np.ndarray | None,
    model: ModelName,
) -> np.ndarray:
    """Walk-forward probabilities for LR, LGBM, XGB, or averaged LR+LGBM."""
    X_tr = _to_numeric_cols(train, cols)
    X_va = _to_numeric_cols(val, cols)
    y_tr = train["home_win"]
    kw = _model_fit_kw(sample_weight)

    if model == "lr":
        pipe = make_pipeline(C=0.5, penalty="l2")
        pipe.fit(X_tr, y_tr, **kw)
        return pipe.predict_proba(X_va)[:, 1]

    if model == "lgbm":
        pipe = _make_lgbm_pipeline()
        pipe.fit(X_tr, y_tr, **kw)
        return pipe.predict_proba(X_va)[:, 1]

    if model == "xgb":
        pipe = _make_xgb_pipeline()
        pipe.fit(X_tr, y_tr, **kw)
        return pipe.predict_proba(X_va)[:, 1]

    pipe_lr = make_pipeline(C=0.5, penalty="l2")
    pipe_lr.fit(X_tr, y_tr, **kw)
    p_lr = pipe_lr.predict_proba(X_va)[:, 1]

    pipe_lgbm = _make_lgbm_pipeline()
    pipe_lgbm.fit(X_tr, y_tr, **kw)
    p_lgb = pipe_lgbm.predict_proba(X_va)[:, 1]
    return (p_lr + p_lgb) / 2.0


def _print_fold_line(
    val: pd.DataFrame,
    prob: np.ndarray,
    *,
    fold_year: int,
    feats: int | None = None,
) -> None:
    ll = log_loss(val["home_win"], prob)
    br = brier_score_loss(val["home_win"], prob)
    ac = accuracy_score(val["home_win"], (prob >= 0.5).astype(int))
    au = roc_auc_score(val["home_win"], prob) if val["home_win"].nunique() == 2 else float("nan")
    line = f"  {fold_year}  ll={ll:.4f}  brier={br:.4f}  acc={ac:.4f}  auc={au:.4f}"
    if feats is not None:
        line += f"  feats={feats:3d}"
    for thresh in EDGE_THRESHOLDS:
        r = roi_sim(val, prob, edge_thresh=thresh)
        roi_s = f"{r['roi']*100:+.1f}%" if r["roi"] is not None else "  n/a"
        wr = (r["win_rate"] or 0) * 100
        line += f"  |  edge>{thresh:.2f}  bets={r['bets']:4d}  wr={wr:4.1f}%  roi={roi_s}"
    print(line)


def ml_to_dec(ml: pd.Series) -> np.ndarray:
    a = pd.to_numeric(ml, errors="coerce").to_numpy()
    return np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / (-a))


def roi_sim(val: pd.DataFrame, prob: np.ndarray, edge_thresh: float, stake: float = 1.0) -> dict:
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

    outcome = val["home_win"].to_numpy()
    bet_home = valid & (ev_h > 0) & (edge_h > edge_thresh)
    bet_away = valid & (ev_a > 0) & (-edge_h > edge_thresh) & (~bet_home)
    bets = int(bet_home.sum() + bet_away.sum())
    if bets == 0:
        return {"bets": 0, "wins": 0, "win_rate": None, "pnl": 0.0, "roi": None}

    pnl = 0.0
    wins = 0
    for i in np.where(bet_home)[0]:
        if outcome[i] == 1:
            pnl += stake * (home_dec[i] - 1)
            wins += 1
        else:
            pnl -= stake
    for i in np.where(bet_away)[0]:
        if outcome[i] == 0:
            pnl += stake * (away_dec[i] - 1)
            wins += 1
        else:
            pnl -= stake

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "pnl": round(pnl, 2),
        "roi": round(pnl / (bets * stake), 4),
    }


def run_variant(
    settled: pd.DataFrame,
    name: str,
    cols: list[str],
    use_recency: bool,
    *,
    model: ModelName = "lr",
) -> None:
    label = f"{name} [{model}]" if model != "lr" else name
    print(f"\n== {label} (n={len(cols)}) ==")
    for train_end, val_end in FOLDS:
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        va = settled.loc[
            (settled["game_date"] >= pd.Timestamp(train_end))
            & (settled["game_date"] < pd.Timestamp(val_end))
        ]
        if tr.empty or va.empty:
            continue

        sw = None
        if use_recency:
            sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)

        prob = fit_predict_proba(train=tr, val=va, cols=cols, sample_weight=sw, model=model)
        _print_fold_line(va, prob, fold_year=int(pd.Timestamp(train_end).year))


def lgbm_mean_importance_pruned(
    settled: pd.DataFrame,
    pruned_cols: list[str],
    *,
    use_recency: bool = True,
    top_print: int = 40,
) -> None:
    """Average LGBM gain importances across walk-forward train folds (no leakage)."""
    print(f"\n== lgbm_mean_importance (pruned n={len(pruned_cols)}, top {top_print}) ==")
    acc = np.zeros(len(pruned_cols), dtype=np.float64)
    n = 0
    for train_end, _val_end in FOLDS:
        tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
        if tr.empty:
            continue
        sw = None
        if use_recency:
            sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)
        pipe = _make_lgbm_pipeline()
        X_tr = _to_numeric_cols(tr, pruned_cols)
        pipe.fit(X_tr, tr["home_win"], **_model_fit_kw(sw))
        acc += pipe.named_steps["model"].feature_importances_.astype(np.float64, copy=False)
        n += 1
    if n == 0:
        print("  (no folds)")
        return
    mean_imp = acc / float(n)
    order = sorted(range(len(pruned_cols)), key=lambda i: mean_imp[i], reverse=True)
    for rank, i in enumerate(order[:top_print], start=1):
        print(f"  {rank:3d}.  {pruned_cols[i]:48s}  {mean_imp[i]:.6f}")


def _null_safe_starter_changed(df: pd.DataFrame) -> pd.Series:
    """True if either side's probable pitcher differs from actual starter.

    Treat missing probables as 'unknown' (not changed) so we don't drop old seasons
    where probables aren't present.
    """
    idx = df.index
    hp = (
        pd.to_numeric(df["home_probable_id"], errors="coerce")
        if "home_probable_id" in df.columns
        else pd.Series(np.nan, index=idx)
    )
    ap = (
        pd.to_numeric(df["away_probable_id"], errors="coerce")
        if "away_probable_id" in df.columns
        else pd.Series(np.nan, index=idx)
    )
    hs = (
        pd.to_numeric(df["home_starter_id"], errors="coerce")
        if "home_starter_id" in df.columns
        else pd.Series(np.nan, index=idx)
    )
    as_ = (
        pd.to_numeric(df["away_starter_id"], errors="coerce")
        if "away_starter_id" in df.columns
        else pd.Series(np.nan, index=idx)
    )

    home_changed = hp.notna() & hs.notna() & (hp != hs)
    away_changed = ap.notna() & as_.notna() & (ap != as_)
    return (home_changed | away_changed).fillna(False)


def _nested_selected_features(
    df: pd.DataFrame,
    *,
    train_end: str,
    min_coverage: float,
    corr_threshold: float,
    top_k: int,
    cache_dir: Path,
    selected_by: SelectedBy,
) -> list[str]:
    """Fold-nested feature selection using only rows < train_end."""
    cutoff_ts = pd.Timestamp(train_end)
    train_df = df.loc[df["game_date"] < cutoff_ts].copy()
    fold_year = int(cutoff_ts.year)
    perm_train_end = f"{fold_year-1}-01-01"
    perm_val_end = f"{fold_year}-01-01"
    fs = load_or_build_feature_sets(
        train_df,
        min_coverage=min_coverage,
        corr_threshold=corr_threshold,
        top_k=top_k,
        cache_dir=cache_dir,
        use_cache=True,
        fold_year=fold_year,
        perm_train_end=perm_train_end,
        perm_val_end=perm_val_end,
        selected_by=selected_by,
    )
    return list(fs["selected"])


def market_baseline(settled: pd.DataFrame) -> None:
    print("\n== market_baseline ==")
    for train_end, val_end in FOLDS:
        va = settled.loc[
            (settled["game_date"] >= pd.Timestamp(train_end))
            & (settled["game_date"] < pd.Timestamp(val_end))
        ]
        if va.empty:
            continue
        p = pd.to_numeric(va["market_implied_prob"], errors="coerce").to_numpy()
        mask = np.isfinite(p)
        sub = va.loc[mask]
        pc = np.clip(p[mask], 1e-3, 1 - 1e-3)
        ll = log_loss(sub["home_win"], pc)
        br = brier_score_loss(sub["home_win"], pc)
        ac = accuracy_score(sub["home_win"], (pc >= 0.5).astype(int))
        print(f"  {pd.Timestamp(train_end).year}  ll={ll:.4f}  brier={br:.4f}  acc={ac:.4f}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--filter-scratches", action="store_true", help="drop games where probable starter != actual starter (listed pitcher constraint)")
    parser.add_argument("--nested-selection", action="store_true", help="select Top-K features per fold using only rows available before that fold")
    parser.add_argument(
        "--model",
        choices=("lr", "lgbm", "xgb", "ensemble"),
        default="lr",
        help="classifier backend for ROI/metrics folds (ensemble = average of LR + LGBM probs)",
    )
    parser.add_argument(
        "--selected-by",
        choices=SELECTED_BY_CHOICES,
        default="lr_perm",
        help="how to pick the top-K feature list (tree gains = mean over walk-forward train folds on pruned set)",
    )
    parser.add_argument(
        "--sweep-top-k",
        nargs="+",
        type=int,
        metavar="K",
        help="run only selected_topk_recency for each K (narrow k search); mutually exclusive with --nested-selection",
    )
    parser.add_argument(
        "--lgbm-importance-pruned",
        action="store_true",
        help="after main runs, print mean LGBM feature gain importances averaged over train-side walk-forward folds for the corr-pruned set",
    )
    parser.add_argument(
        "--importance-top",
        type=int,
        default=40,
        help="with --lgbm-importance-pruned, how many rows to print",
    )
    args = parser.parse_args()

    cache_dir = OUT_DIR / "cache"
    model_backend: ModelName = args.model
    selected_by: SelectedBy = args.selected_by

    if args.nested_selection and args.sweep_top_k:
        parser.error("--nested-selection and --sweep-top-k cannot be combined")

    print(f"loading+engineering (cache=on) {INPUT}")
    df = load_or_build_engineered_frame(
        input_path=INPUT,
        cutoff=sf.DEFAULT_CUTOFF,
        cache_dir=cache_dir,
        use_cache=True,
    )

    settled = df.dropna(subset=["home_win"]).copy()
    settled["home_win"] = settled["home_win"].astype(int)
    settled["game_date"] = pd.to_datetime(settled["game_date"], errors="coerce")
    if args.filter_scratches:
        changed = _null_safe_starter_changed(settled)
        before = len(settled)
        settled = settled.loc[~changed].copy()
        after = len(settled)
        print(f"filter-scratches: dropped {before-after:,} rows (kept {after:,})")

    pruned_for_importance: list[str] | None = None

    if args.sweep_top_k:
        ks = sorted(set(args.sweep_top_k))
        print(
            f"\n>>> sweep-top-k={ks} model={model_backend} selected_by={selected_by}"
            " (selected_topk_recency only)\n"
        )
        for k in ks:
            fs_k = load_or_build_feature_sets(
                df,
                min_coverage=args.min_coverage,
                corr_threshold=args.corr_threshold,
                top_k=k,
                cache_dir=cache_dir,
                use_cache=True,
                fold_year=None,
                selected_by=selected_by,
            )
            if pruned_for_importance is None:
                pruned_for_importance = list(fs_k["pruned"])
                print(
                    f"cand={len(fs_k['cand'])}  pruned={len(pruned_for_importance)}"
                    f"  l1={len(fs_k['l1_keep'])}  sweep k={k} -> topk_out={len(fs_k['selected'])}"
                )
            sweep_name = (
                f"selected_topk_k{k}_recency"
                if selected_by == "lr_perm"
                else f"selected_topk_k{k}_{selected_by}_recency"
            )
            run_variant(
                settled,
                sweep_name,
                list(fs_k["selected"]),
                True,
                model=model_backend,
            )
    elif not args.nested_selection:
        print("loading feature sets (cache=on)...")
        fs = load_or_build_feature_sets(
            df,
            min_coverage=args.min_coverage,
            corr_threshold=args.corr_threshold,
            top_k=args.top_k,
            cache_dir=cache_dir,
            use_cache=True,
            fold_year=None,
            selected_by=selected_by,
        )
        cand = list(fs["cand"])
        pruned_for_importance = list(fs["pruned"])
        l1_keep = list(fs["l1_keep"])
        selected_global = list(fs["selected"])

        print(
            f"cand={len(cand)}  pruned={len(pruned_for_importance)}"
            f"  l1={len(l1_keep)}  topk={len(selected_global)}  selected_by={selected_by}"
        )

        sel_variant = (
            "selected_topk_recency"
            if selected_by == "lr_perm"
            else f"selected_topk_{selected_by}_recency"
        )
        variants = [
            ("all_engineered_no_recency", cand, False),
            ("all_engineered_recency", cand, True),
            ("corr_pruned_recency", pruned_for_importance, True),
            ("l1_stable_recency", l1_keep, True),
            (sel_variant, selected_global, True),
        ]
        for name, cols, rec in variants:
            run_variant(settled, name, cols, rec, model=model_backend)
    else:
        print("loading feature sets (reference sizes, cache=on)...")
        fs = load_or_build_feature_sets(
            df,
            min_coverage=args.min_coverage,
            corr_threshold=args.corr_threshold,
            top_k=args.top_k,
            cache_dir=cache_dir,
            use_cache=True,
            fold_year=None,
            selected_by=selected_by,
        )
        pruned_for_importance = list(fs["pruned"])
        print(
            f"cand={len(fs['cand'])}  pruned={len(pruned_for_importance)}"
            f"  l1={len(fs['l1_keep'])}  nested k={args.top_k}  selected_by={selected_by}"
        )
        print(
            f"\n== nested_selected_topk (k={args.top_k}, selected_by={selected_by})"
            f" [{model_backend}] =="
        )
        for train_end, val_end in FOLDS:
            tr = settled.loc[settled["game_date"] < pd.Timestamp(train_end)]
            va = settled.loc[
                (settled["game_date"] >= pd.Timestamp(train_end))
                & (settled["game_date"] < pd.Timestamp(val_end))
            ]
            if tr.empty or va.empty:
                continue
            cols = _nested_selected_features(
                settled,
                train_end=train_end,
                min_coverage=args.min_coverage,
                corr_threshold=args.corr_threshold,
                top_k=args.top_k,
                cache_dir=cache_dir,
                selected_by=selected_by,
            )
            sw = recency_weights(tr["season"], pd.Timestamp(train_end).year - 1, 4.0)
            prob = fit_predict_proba(train=tr, val=va, cols=cols, sample_weight=sw, model=model_backend)
            _print_fold_line(
                va,
                prob,
                fold_year=int(pd.Timestamp(train_end).year),
                feats=len(cols),
            )

    market_baseline(settled)
    if args.lgbm_importance_pruned:
        if not pruned_for_importance:
            print("--lgbm-importance-pruned: no pruned feature list (unexpected); skipping")
        else:
            lgbm_mean_importance_pruned(
                settled,
                pruned_for_importance,
                use_recency=True,
                top_print=args.importance_top,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
