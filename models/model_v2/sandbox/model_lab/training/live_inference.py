"""Sandbox live-inference: daily portfolio recommendations from master CSV.

Reads `sandbox/model_lab/output/master_sandbox_mlb.csv`, fits LR (or LGBM) on
all rows strictly before --as-of, predicts probabilities for the slice on
--as-of, runs the same calibration + portfolio sizing pipeline used in
backtests, and writes a JSON file with recommended bets.

No production tables, model artifacts, or order flows are touched. All output
goes under `sandbox/model_lab/output/live/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab.training import calibration as C  # noqa: E402
from models.model_v2.sandbox.model_lab.training import data as D  # noqa: E402
from models.model_v2.sandbox.model_lab.training import models as M  # noqa: E402
from models.model_v2.sandbox.model_lab.training import portfolio as P  # noqa: E402

OUTDIR = LAB_DIR / "output" / "live"


def _parse_as_of(raw: str | None) -> pd.Timestamp:
    if not raw:
        return pd.Timestamp(date.today())
    return pd.Timestamp(raw)


def _fit_predict(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feats: list[str],
    *,
    model_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on train, return (target_probs, train_in_probs) for calibrator fit."""
    y = train_df["home_win"].astype(int)
    if model_kind == "lgbm":
        model = M.make_lgbm()
        model.fit(train_df[feats], y)
        target_p = model.predict_proba(target_df[feats])[:, 1]
        train_p = model.predict_proba(train_df[feats])[:, 1]
    else:
        pipe = M.make_lr()
        pipe.fit(train_df[feats], y)
        target_p = pipe.predict_proba(target_df[feats])[:, 1]
        train_p = pipe.predict_proba(train_df[feats])[:, 1]
    return target_p, train_p


def _early_specialist_predict(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    early_feats: list[str],
    cutoff: int,
) -> np.ndarray | None:
    """Return early-spec probs aligned to target_df, NaN for non-early rows."""
    early_tr, _ = D.split_early(train_df, cutoff)
    early_vl, _ = D.split_early(target_df, cutoff)
    if not early_feats or not early_tr.any() or not early_vl.any():
        return None
    sub_tr = train_df.loc[early_tr]
    if sub_tr["home_win"].nunique() != 2:
        return None
    pipe = M.make_early_lr()
    pipe.fit(sub_tr[early_feats], sub_tr["home_win"].astype(int))
    sub_vl = target_df.loc[early_vl]
    early_p = pipe.predict_proba(sub_vl[early_feats])[:, 1]
    out = np.full(len(target_df), np.nan, dtype=float)
    out[np.where(early_vl.to_numpy())[0]] = early_p
    return out


def recommend(
    *,
    as_of: pd.Timestamp,
    input_path: Path,
    model_kind: str = "lr",
    calibration_method: str = "platt",
    early_cutoff: int | None = 25,
    sizing: str = "kelly",
    threshold: float = P.CONFIDENCE_THRESHOLD,
    bankroll: float = 10_000.0,
    min_coverage: float = 0.5,
) -> dict:
    df = D.load_master(input_path)
    train_df = df[df["game_date"] < as_of].copy()
    target_df = df[df["game_date"] == as_of].copy()
    if train_df.empty:
        raise ValueError(f"no training rows before {as_of.date()}")
    if target_df.empty:
        raise ValueError(f"no rows for game_date == {as_of.date()}")

    feats = D.select_numeric_features(df, min_coverage=min_coverage)
    if not feats:
        raise ValueError("no usable feature columns")
    early_feats = D.early_features(df)

    target_p, train_p = _fit_predict(train_df, target_df, feats, model_kind=model_kind)
    cal = C.calibrate_all(train_p, train_df["home_win"].to_numpy(), target_p)
    probs = cal.get(calibration_method, target_p)

    if early_cutoff:
        early_probs = _early_specialist_predict(train_df, target_df, early_feats, early_cutoff)
        if early_probs is not None:
            mask = ~np.isnan(early_probs)
            probs = probs.copy()
            probs[mask] = early_probs[mask]

    market = pd.to_numeric(target_df.get("market_implied_prob"), errors="coerce").to_numpy()
    cands = P.candidate_bets(probs, market, threshold=threshold)

    flat: list[tuple[int, P.Bet]] = []
    for game_idx, bets in enumerate(cands):
        for b in bets:
            flat.append((game_idx, b))

    if not flat:
        recs: list[dict] = []
    else:
        bets_only = [fb[1] for fb in flat]
        if sizing == "sharpe":
            fracs = P.sharpe_optimize(bets_only)
        elif sizing == "joint_kelly":
            fracs = P.joint_kelly(bets_only)
        else:
            fracs = P.kelly_stakes(bets_only)
        recs = []
        for (gi, bet), frac in zip(flat, fracs):
            row = target_df.iloc[gi]
            stake = float(frac) * bankroll
            recs.append({
                "game_id": _scalar(row.get("game_id")),
                "game_pk": _scalar(row.get("game_pk")),
                "home_team": _scalar(row.get("home_team")),
                "away_team": _scalar(row.get("away_team")),
                "side": bet.side,
                "model_prob": float(bet.prob),
                "market_prob": float(bet.market_prob),
                "edge": float(bet.edge),
                "decimal_odds": float(bet.decimal_odds),
                "stake_frac": float(frac),
                "stake_usd": stake,
            })

    return {
        "as_of": str(as_of.date()),
        "input": str(input_path),
        "model": model_kind,
        "calibration": calibration_method,
        "sizing": sizing,
        "early_cutoff": early_cutoff,
        "threshold": threshold,
        "bankroll_assumed": bankroll,
        "min_coverage": min_coverage,
        "n_features": len(feats),
        "n_train_rows": int(len(train_df)),
        "n_games_today": int(len(target_df)),
        "n_candidate_bets": len(recs),
        "total_stake_frac": float(sum(r["stake_frac"] for r in recs)),
        "total_stake_usd": float(sum(r["stake_usd"] for r in recs)),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "recommendations": recs,
    }


def _scalar(v):
    if v is None:
        return None
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def _save(payload: dict) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    name = f"recommendations_{payload['as_of']}.json"
    path = OUTDIR / name
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=D.DEFAULT_MASTER)
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; default = today UTC")
    parser.add_argument("--model", choices=["lr", "lgbm"], default="lr")
    parser.add_argument(
        "--calibration", choices=["none", "platt", "isotonic"], default="platt",
    )
    parser.add_argument(
        "--sizing", choices=["kelly", "sharpe", "joint_kelly"], default="kelly",
    )
    parser.add_argument("--early-cutoff", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=P.CONFIDENCE_THRESHOLD)
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--print", action="store_true", help="print payload to stdout")
    args = parser.parse_args()

    payload = recommend(
        as_of=_parse_as_of(args.as_of),
        input_path=args.input,
        model_kind=args.model,
        calibration_method=args.calibration,
        early_cutoff=args.early_cutoff if args.early_cutoff > 0 else None,
        sizing=args.sizing,
        threshold=args.threshold,
        bankroll=args.bankroll,
        min_coverage=args.min_coverage,
    )
    out = _save(payload)
    print(f"wrote {out}")
    if args.print:
        print(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
