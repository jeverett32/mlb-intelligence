"""Sandbox training orchestrator.

Subcommands:
  baseline    walk-forward LR + LGBM, all calibrations, summary metrics
  cutoff      sweep early-season cutoff
  backtest    bankroll simulation w/ kelly + sharpe sizing
  all         run baseline + cutoff + backtest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

REPO_ROOT = Path(__file__).resolve().parents[5]
LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab.training import data as D  # noqa: E402
from models.model_v2.sandbox.model_lab.training import early_cutoff as EC  # noqa: E402
from models.model_v2.sandbox.model_lab.training import simulate as S  # noqa: E402
from models.model_v2.sandbox.model_lab.training import walk_forward as W  # noqa: E402

OUTDIR = LAB_DIR / "output" / "training"


def _save(name: str, payload: dict) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def _market_baseline(df: pd.DataFrame) -> dict:
    if "market_implied_prob" not in df.columns:
        return {}
    p = pd.to_numeric(df["market_implied_prob"], errors="coerce").clip(1e-6, 1 - 1e-6)
    y = df["home_win"].astype(int)
    mask = p.notna()
    if not mask.any():
        return {}
    return {
        "log_loss": float(log_loss(y[mask], p[mask])),
        "brier": float(brier_score_loss(y[mask], p[mask])),
        "n": int(mask.sum()),
    }


def _apply_min_season(df: pd.DataFrame, min_season: int | None) -> pd.DataFrame:
    if not min_season:
        return df.reset_index(drop=True)
    if "season" not in df.columns:
        return df.reset_index(drop=True)
    out = df[df["season"] >= int(min_season)].copy()
    # walk_forward resets index internally; keep our outer df aligned.
    return out.reset_index(drop=True)


def cmd_baseline(args) -> None:
    df = _apply_min_season(D.load_master(args.input), args.min_season)
    extra_drop = tuple(args.drop or [])
    feats = D.select_numeric_features(df, min_coverage=args.min_coverage, extra_drop=extra_drop)
    market = _market_baseline(df)

    summary = {
        "n_rows": int(len(df)),
        "n_features": len(feats),
        "min_coverage": args.min_coverage,
        "early_cutoff": args.early_cutoff,
        "min_season": args.min_season,
        "drop": list(args.drop or []),
        "market_baseline": market,
        "models": {},
    }

    for kind in args.models:
        try:
            kind_fe = feats
            if kind == "lr_offset":
                # Residual model learns correction; market is provided via offset, not as a feature.
                kind_fe = [c for c in feats if c != "market_implied_prob"]
            folds = W.walk_forward(
                df,
                kind_fe,
                model_kind=kind,
                early_cutoff=args.early_cutoff,
                seasons=args.seasons,
                show_progress=args.progress,
                progress_desc=f"baseline[{kind}]",
            )
        except ImportError as exc:
            summary["models"][kind] = {"error": str(exc)}
            continue
        if not folds:
            continue
        per_method = {}
        for method in ["none", "platt", "isotonic"]:
            # Main model only (no early-season specialist override).
            probs0, idx0 = W.stitched_probs(df, folds, method=method, use_early_specialist=False)
            if probs0.size:
                y0 = df.loc[idx0, "home_win"].to_numpy()
                per_method[method] = {
                    "log_loss": float(log_loss(y0, np.clip(probs0, 1e-6, 1 - 1e-6))),
                    "brier": float(brier_score_loss(y0, probs0)),
                    "n": int(len(probs0)),
                }

            # Main + early-season specialist override (if available in folds).
            probs1, idx1 = W.stitched_probs(df, folds, method=method, use_early_specialist=True)
            if probs1.size:
                y1 = df.loc[idx1, "home_win"].to_numpy()
                per_method[f"{method}+early"] = {
                    "log_loss": float(log_loss(y1, np.clip(probs1, 1e-6, 1 - 1e-6))),
                    "brier": float(brier_score_loss(y1, probs1)),
                    "n": int(len(probs1)),
                }
        per_fold = [
            {"season": f.season, "train_rows": f.train_rows, "val_rows": f.val_rows,
             "metrics": f.metrics}
            for f in folds
        ]
        summary["models"][kind] = {"per_method": per_method, "per_fold": per_fold}

    out = _save(args.output_name or "baseline.json", summary)
    print(f"wrote {out}")


def cmd_cutoff(args) -> None:
    df = _apply_min_season(D.load_master(args.input), args.min_season)
    feats = D.select_numeric_features(df, min_coverage=args.min_coverage, extra_drop=tuple(args.drop or []))
    res = EC.sweep(
        df, feats,
        cutoffs=tuple(args.cutoffs),
        model_kind=args.model,
        calibration_method=args.calibration,
    )
    payload = {
        "model": args.model,
        "calibration": args.calibration,
        "min_season": args.min_season,
        "drop": list(args.drop or []),
        "results": res.to_dict(orient="records"),
    }
    out = _save(f"cutoff_{args.model}.json", payload)
    print(f"wrote {out}")
    if not res.empty:
        print(res.to_string(index=False))


def cmd_backtest(args) -> None:
    df = _apply_min_season(D.load_master(args.input), args.min_season)
    feats = D.select_numeric_features(df, min_coverage=args.min_coverage, extra_drop=tuple(args.drop or []))
    folds = W.walk_forward(
        df, feats, model_kind=args.model, early_cutoff=args.early_cutoff,
    )
    if not folds:
        print("no folds; check data")
        return
    probs, idx = W.stitched_probs(df, folds, method=args.calibration)
    sub = df.loc[idx].copy().reset_index(drop=True)

    results: dict[str, dict] = {}
    for sizing in ["kelly", "sharpe", "joint_kelly"]:
        sim = S.simulate(
            sub, probs,
            sizing=sizing,
            initial_bankroll=args.bankroll,
            threshold=args.threshold,
        )
        results[sizing] = sim.summary()

    payload = {
        "model": args.model,
        "calibration": args.calibration,
        "early_cutoff": args.early_cutoff,
        "threshold": args.threshold,
        "initial_bankroll": args.bankroll,
        "n_val_rows": int(len(sub)),
        "drop": list(args.drop or []),
        "sizing": results,
    }
    out = _save(f"backtest_{args.model}.json", payload)
    print(f"wrote {out}")
    print(json.dumps(results, indent=2, default=float))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common_input = lambda p: p.add_argument(
        "--input", type=Path, default=D.DEFAULT_MASTER,
    )
    common_cov = lambda p: p.add_argument("--min-coverage", type=float, default=0.5)

    p_base = sub.add_parser("baseline")
    common_input(p_base); common_cov(p_base)
    p_base.add_argument("--early-cutoff", type=int, default=25)
    p_base.add_argument("--min-season", type=int, default=None, help="drop rows before this season")
    p_base.add_argument("--drop", action="append", default=[], help="drop a feature column (repeatable)")
    p_base.add_argument(
        "--models",
        default="lr,lgbm",
        help="comma-separated: lr,lr_offset,lgbm (default: lr,lgbm)",
    )
    p_base.add_argument(
        "--seasons",
        default=None,
        help="optional seasons like '2017-2025' or '2019,2020,2021'",
    )
    p_base.add_argument("--progress", action="store_true", help="show tqdm progress bar")
    p_base.add_argument(
        "--output-name",
        default=None,
        help="output json filename (default: baseline.json)",
    )
    p_base.set_defaults(func=cmd_baseline)

    p_cut = sub.add_parser("cutoff")
    common_input(p_cut); common_cov(p_cut)
    p_cut.add_argument("--model", choices=["lr", "lgbm"], default="lr")
    p_cut.add_argument("--calibration", default="none")
    p_cut.add_argument("--min-season", type=int, default=None, help="drop rows before this season")
    p_cut.add_argument("--drop", action="append", default=[], help="drop a feature column (repeatable)")
    p_cut.add_argument("--cutoffs", type=int, nargs="+", default=list(EC.DEFAULT_CUTOFFS))
    p_cut.set_defaults(func=cmd_cutoff)

    p_bt = sub.add_parser("backtest")
    common_input(p_bt); common_cov(p_bt)
    p_bt.add_argument("--model", choices=["lr", "lr_offset", "lgbm"], default="lr")
    p_bt.add_argument("--calibration", default="none")
    p_bt.add_argument("--early-cutoff", type=int, default=25)
    p_bt.add_argument("--min-season", type=int, default=None, help="drop rows before this season")
    p_bt.add_argument("--drop", action="append", default=[], help="drop a feature column (repeatable)")
    p_bt.add_argument("--threshold", type=float, default=0.04)
    p_bt.add_argument("--bankroll", type=float, default=1000.0)
    p_bt.set_defaults(func=cmd_backtest)

    p_all = sub.add_parser("all")
    common_input(p_all); common_cov(p_all)
    p_all.add_argument("--early-cutoff", type=int, default=25)
    p_all.add_argument("--min-season", type=int, default=None, help="drop rows before this season")
    p_all.add_argument("--drop", action="append", default=[], help="drop a feature column (repeatable)")
    p_all.add_argument("--threshold", type=float, default=0.04)
    p_all.add_argument("--bankroll", type=float, default=1000.0)
    def _all(args):
        cmd_baseline(args)
        for kind in ("lr", "lgbm"):
            ns = argparse.Namespace(
                input=args.input, min_coverage=args.min_coverage,
                model=kind, calibration="none",
                cutoffs=list(EC.DEFAULT_CUTOFFS),
            )
            try:
                cmd_cutoff(ns)
            except ImportError as exc:
                print(f"skip cutoff {kind}: {exc}")
            ns2 = argparse.Namespace(
                input=args.input, min_coverage=args.min_coverage,
                model=kind, calibration="none",
                early_cutoff=args.early_cutoff,
                threshold=args.threshold, bankroll=args.bankroll,
            )
            try:
                cmd_backtest(ns2)
            except ImportError as exc:
                print(f"skip backtest {kind}: {exc}")
    p_all.set_defaults(func=_all)

    args = parser.parse_args()
    if getattr(args, "models", None):
        args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if getattr(args, "seasons", None):
        raw = args.seasons
        if isinstance(raw, str):
            raw = raw.strip()
            if "-" in raw and "," not in raw:
                a, b = raw.split("-", 1)
                args.seasons = list(range(int(a), int(b) + 1))
            else:
                args.seasons = [int(x.strip()) for x in raw.split(",") if x.strip()]
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
