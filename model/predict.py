"""
MLB betting model prediction script.
Trains on all historical data with current season context, then predicts
edge + Kelly stake for one or more target games.

Library API:
    shared = prepare_shared(game_pks, game_date)
    result = predict_one(game_pk, shared)   # writes to DB, returns dict

CLI (for debug/one-off use):
    uv run model/predict.py --game_pk 12345
    uv run model/predict.py --game_date 2026-04-01 --home_team NYY --away_team BOS
"""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import warnings
from datetime import datetime, timezone

os.environ.setdefault("PYTHONHASHSEED", "42")
# Keep BLAS / OpenMP single-threaded so concurrent workers don't oversubscribe.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import train as T
import db as DB
from config import ACTIVE_SEASON, CURRENT_CSV

HISTORICAL_CSV = "data/master_mlb.csv"
ARTIFACT_SCHEMA_VERSION = 1


class PredictError(RuntimeError):
    pass


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _source_hash() -> str | None:
    try:
        h = hashlib.sha256()
        for path in (getattr(T, "__file__", None), __file__):
            if not path:
                continue
            with open(path, "rb") as f:
                h.update(f.read())
        return h.hexdigest()
    except Exception:
        return None


def _feature_config(active_feats: list[str], early_feats: list[str]) -> dict:
    return _jsonable({
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": T.MODEL,
        "calibrate": T.CALIBRATE,
        "early_cutoff": T.EARLY_CUTOFF,
        "best_w": T.BEST_W,
        "momentum_w": T.MOMENTUM_W,
        "train_window_years": T.TRAIN_WINDOW_YEARS,
        "early_season_games": T.EARLY_SEASON_GAMES,
        "drop_substr": sorted(T.DROP_SUBSTR),
        "active_feats": active_feats,
        "early_feats": early_feats,
        "feature_columns_config": list(T.FEATURE_COLUMNS),
        "early_feature_columns_config": list(T.EARLY_FEATURE_COLUMNS),
        "lr_params": getattr(T, "LR_PARAMS", {}),
        "lgb_params": getattr(T, "LGB_PARAMS", {}),
        "xgb_params": getattr(T, "XGB_PARAMS", {}),
        "mlp_params": {
            "epochs": getattr(T, "MLP_EPOCHS", None),
            "lr": getattr(T, "MLP_LR", None),
        },
        "git_commit": _git_commit(),
        "source_hash": _source_hash(),
    })


def _training_fingerprint(train_df: pd.DataFrame, active_feats: list[str], early_feats: list[str]) -> tuple[str, dict]:
    config = _feature_config(active_feats, early_feats)
    cols = [
        c for c in (
            ["game_pk", "game_date", "home_win", "market_implied_prob",
             "home_games_played", "away_games_played"] + active_feats + early_feats
        )
        if c in train_df.columns
    ]
    cols = list(dict.fromkeys(cols))
    frame = train_df[cols].copy()
    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sort_cols = [c for c in ["game_date", "game_pk"] if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="mergesort")
    frame = frame.reset_index(drop=True)

    h = hashlib.sha256()
    h.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(b"\n")
    h.update(frame.to_json(orient="split", date_format="iso", double_precision=12).encode("utf-8"))

    max_date = None
    if "game_date" in train_df.columns:
        max_raw = pd.to_datetime(train_df["game_date"], errors="coerce").max()
        max_date = max_raw.date().isoformat() if pd.notna(max_raw) else None
    meta = {
        "feature_config": config,
        "settled_row_count": int(len(train_df)),
        "max_settled_game_date": max_date,
        "num_features": int(len(active_feats)),
        "git_commit": config.get("git_commit"),
    }
    return h.hexdigest(), meta


def _serialize_artifact(bundle: dict) -> tuple[bytes, str]:
    blob = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    return blob, hashlib.sha256(blob).hexdigest()


def _deserialize_artifact(row: dict, expected_fingerprint: str) -> dict:
    blob = bytes(row["artifact_bytes"])
    expected_sha = row.get("artifact_sha256")
    actual_sha = hashlib.sha256(blob).hexdigest()
    if expected_sha and expected_sha != actual_sha:
        raise PredictError("Model artifact checksum mismatch.")
    bundle = pickle.loads(blob)
    if bundle.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise PredictError("Model artifact schema version mismatch.")
    if bundle.get("training_fingerprint") != expected_fingerprint:
        raise PredictError("Model artifact fingerprint mismatch.")
    if bundle.get("model_type") != T.MODEL:
        raise PredictError("Model artifact type mismatch.")
    return bundle


# ---------------------------------------------------------------------------
# Explainability (Logistic Regression) - no LLM, no SHAP.
# We compute per-feature contributions in log-odds space using:
#   logit = intercept + sum_j coef_j * x_scaled_j
# where x_scaled is post-impute + post-standardize.
# For away-side signals we negate contributions (away win == not home win).
# ---------------------------------------------------------------------------

def _artifact_feature_importances(clf, active_feats: list[str]) -> list[dict]:
    """Best-effort feature importances for the production model artifact.
    Returns [] if shape unsupported."""
    try:
        pipe = _extract_lr_pipeline_for_explainability(clf)
        if pipe is None or not hasattr(pipe, "named_steps"):
            return []
        mdl = pipe.named_steps.get("mdl")
        imp = pipe.named_steps.get("imp")
        if mdl is None:
            return []
        feats = list(active_feats)
        stats = getattr(imp, "statistics_", None) if imp is not None else None
        if stats is not None:
            survived = np.isfinite(np.asarray(stats))
            n_in = getattr(mdl, "n_features_in_", None)
            if len(survived) == len(feats) and (n_in is None or int(survived.sum()) == int(n_in)):
                feats = [f for f, keep in zip(feats, survived) if keep]
        from train import extract_feature_importance
        return extract_feature_importance(mdl, feats) or []
    except Exception:
        return []


def _extract_lr_pipeline_for_explainability(clf):
    """
    Return a fitted sklearn Pipeline with steps: imp, scl, mdl (LogisticRegression),
    or None if the classifier shape isn't supported.
    """
    # Early specialist is a plain Pipeline.
    if hasattr(clf, "named_steps"):
        return clf
    # Regular-season LR is wrapped in CalibratedClassifierCV; take one estimator as a representative.
    calibrated = getattr(clf, "calibrated_classifiers_", None)
    if calibrated:
        est = getattr(calibrated[0], "estimator", None)
        if est is not None and hasattr(est, "named_steps"):
            return est
    return None


def _lr_feature_contributions(pipe, feature_names: list[str], x_raw: np.ndarray) -> dict | None:
    try:
        imp = pipe.named_steps.get("imp")
        scl = pipe.named_steps.get("scl")
        mdl = pipe.named_steps.get("mdl")
    except Exception:
        return None
    if imp is None or scl is None or mdl is None:
        return None
    if not hasattr(mdl, "coef_") or not hasattr(mdl, "intercept_"):
        return None

    x_imp = imp.transform(x_raw)
    x_sc = scl.transform(x_imp)
    coef = np.asarray(mdl.coef_).reshape(-1)
    intercept = float(np.asarray(mdl.intercept_).reshape(-1)[0])
    contrib = (coef * x_sc.reshape(-1)).astype(float)

    stats = getattr(imp, "statistics_", None)
    if stats is not None and len(feature_names) != len(contrib):
        survived = np.isfinite(np.asarray(stats))
        if len(survived) == len(feature_names) and int(survived.sum()) == len(contrib):
            feature_names = [f for f, keep in zip(feature_names, survived) if keep]

    if len(contrib) != len(feature_names):
        return None

    logit = float(intercept + float(np.sum(contrib)))
    # Raw (uncalibrated) prob from the LR layer. Note: final output may be isotonic-calibrated.
    raw_prob = float(1.0 / (1.0 + np.exp(-logit)))

    def _humanize_feature(name: str) -> str:
        n = (name or "").strip()
        if not n:
            return "-"
        SPECIAL = {
            "market_implied_prob": "Market implied prob (home)",
            "home_implied_prob": "Market implied prob (home)",
            "away_implied_prob": "Market implied prob (away)",
            "edge": "Edge",
            "bet_frac": "Recommended stake (fraction)",
            "momentum_DIFF": "Momentum (home - away)",
            "season_win_pct_DIFF": "Season win% (home - away)",
            "streak_DIFF": "Streak (home - away)",
            "sp_fip_DIFF": "SP FIP (away - home)",
            "sp_era_DIFF": "SP ERA (away - home)",
            "sp_k9_DIFF": "SP K/9 (home - away)",
            "sp_bb9_DIFF": "SP BB/9 (away - home)",
            "wrc_plus_DIFF": "wRC+ (home - away)",
            "woba_DIFF": "wOBA (home - away)",
            "k_pct_DIFF": "K% (away - home)",
            "bb_pct_DIFF": "BB% (home - away)",
            "hr_per_9_DIFF": "HR/9 (away - home)",
            "run_diff_avg_W_DIFF": "Avg run diff (home - away)",
            "runs_scored_avg_W_DIFF": "Avg runs scored (home - away)",
            "runs_allowed_avg_W_DIFF": "Avg runs allowed (home - away)",
            "win_pct_W_DIFF": "Win% (home - away)",
            "pitcher_handedness_diff": "SP handedness (home - away; L=1)",
            "sharp_move_flag": "Sharp move flag",
            "sharp_x_fip": "Sharp flag x SP FIP (away - home)",
            "early_season_flag": "Early season flag",
            "home_games_played": "Home games played",
            "away_games_played": "Away games played",
        }
        if n in SPECIAL:
            return SPECIAL[n]

        # Generic transformations.
        # Common suffixes.
        if n.endswith("_DIFF"):
            base = n[:-5]
            n = f"{base} (diff)"
        n = n.replace("_lag1", " (lag 1)")
        n = n.replace("_pct", "%")
        n = n.replace("_per_9", "/9")
        n = n.replace("_", " ")

        # Preserve known acronyms.
        for a in ("wOBA", "wRC+", "xFIP", "FIP", "ERA", "WHIP", "K/9", "BB/9", "HR/9"):
            pass

        # Title-case but keep all-caps tokens as-is.
        parts = []
        for tok in n.split():
            if tok.upper() in {"ERA", "FIP", "WHIP"}:
                parts.append(tok.upper())
            elif tok.lower() in {"woba", "wrc+", "xfip"}:
                parts.append({"woba": "wOBA", "wrc+": "wRC+", "xfip": "xFIP"}[tok.lower()])
            else:
                parts.append(tok[:1].upper() + tok[1:])
        return " ".join(parts)

    items = [
        {"feature": feature_names[i], "label": _humanize_feature(feature_names[i]), "value": float(contrib[i])}
        for i in range(len(feature_names))
    ]
    items.sort(key=lambda r: r["value"], reverse=True)
    top_pos = [r for r in items if r["value"] > 0][:6]
    top_neg = sorted([r for r in items if r["value"] < 0], key=lambda r: r["value"])[:6]

    return {
        "type": "lr_logodds_contrib",
        "raw_logit_home": logit,
        "raw_prob_home": raw_prob,
        "top_positive": top_pos,
        "top_negative": top_neg,
    }


def _build_plaintext_explanation(explanation: dict) -> str:
    """
    Build a short plaintext summary from top contributions.
    ASCII-only output to avoid DB encoding pitfalls.
    """
    if not explanation:
        return ""
    side = str(explanation.get("bet_side") or "").lower()
    side_label = "HOME" if side == "home" else ("AWAY" if side == "away" else "N/A")
    contrib = explanation.get("contributions") or {}
    pos = contrib.get("top_positive") or []
    neg = contrib.get("top_negative") or []

    def _names(rows: list[dict], n: int = 3) -> list[str]:
        out = []
        for r in rows[:n]:
            lbl = str(r.get("label") or r.get("feature") or "").strip() or "?"
            lbl = lbl.replace("-", "-")
            out.append(lbl)
        return out

    for_list = _names(pos, 3)
    against_list = _names(neg, 2)

    lines = []
    lines.append(f"Signal: {side_label}.")
    if for_list:
        lines.append("Biggest drivers FOR this side: " + "; ".join(for_list) + ".")
    if against_list:
        lines.append("Biggest drivers AGAINST this side: " + "; ".join(against_list) + ".")
    if explanation.get("recomputed") is True:
        lines.append("Note: recomputed later using current pipeline code/data; may differ from original-day explanation.")
    lines.append("Contributions are logistic-regression log-odds components (post-impute/scale), ranked by magnitude.")
    return " ".join(lines)[:900]


def explain_one(
    game_pk: str,
    shared: dict,
    *,
    predicted_prob_home: float | None,
    market_implied_prob_home: float | None,
    edge: float | None,
    bet_side: str,
    bet_frac: float,
    recomputed: bool = False,
    recomputed_reason: str = "",
    write_db: bool = True,
) -> dict | None:
    """
    Compute and store explainability for a specific bet signal without re-running prediction logic.
    Returns explanation dict or None if not applicable/available.
    """
    game_pk = str(game_pk)
    if bet_side not in {"home", "away"} or float(bet_frac or 0.0) <= 0:
        return None

    target_df = shared["target_rows"].get(game_pk)
    if target_df is None:
        return None

    clf = shared["clf"]
    early_clf = shared["early_clf"]
    active_feats = shared["active_feats"]
    early_feats = shared["early_feats"]

    is_early = False
    if T.EARLY_CUTOFF is not None:
        hgp = float(target_df["home_games_played"].iloc[0]) if "home_games_played" in target_df else 999
        agp = float(target_df["away_games_played"].iloc[0]) if "away_games_played" in target_df else 999
        is_early = (hgp < T.EARLY_CUTOFF) or (agp < T.EARLY_CUTOFF)

    if is_early and early_clf is not None:
        pipe = _extract_lr_pipeline_for_explainability(early_clf)
        feats = early_feats
        x_raw = target_df[early_feats].values.astype(np.float32)
        is_early_specialist = True
    else:
        pipe = _extract_lr_pipeline_for_explainability(clf)
        feats = active_feats
        x_raw = target_df[active_feats].values.astype(np.float32)
        is_early_specialist = False

    if pipe is None or not feats:
        return None

    base = _lr_feature_contributions(pipe, feats, x_raw)
    if base is None:
        return None

    if bet_side == "away":
        old_pos = list(base.get("top_positive", []) or [])
        old_neg = list(base.get("top_negative", []) or [])
        base = dict(base)
        base["raw_logit_side"] = -float(base.get("raw_logit_home", 0.0))
        base["raw_prob_side"] = float(1.0 - float(base.get("raw_prob_home", 0.5)))
        base["top_positive"] = [{"feature": r.get("feature"), "label": r.get("label"), "value": float(-r["value"])} for r in old_neg]
        base["top_negative"] = [{"feature": r.get("feature"), "label": r.get("label"), "value": float(-r["value"])} for r in old_pos]
    else:
        base["raw_logit_side"] = float(base.get("raw_logit_home", 0.0))
        base["raw_prob_side"] = float(base.get("raw_prob_home", 0.5))

    explanation = {
        "game_pk": game_pk,
        "bet_side": bet_side,
        "model": str(T.MODEL),
        "is_early_specialist": bool(is_early_specialist),
        "predicted_prob_home": float(predicted_prob_home) if predicted_prob_home is not None else None,
        "market_implied_prob_home": float(market_implied_prob_home) if market_implied_prob_home is not None else None,
        "edge": float(edge) if edge is not None else None,
        "bet_frac": float(bet_frac or 0.0),
        "contributions": base,
        "recomputed": bool(recomputed),
        "recomputed_reason": (recomputed_reason or "")[:200],
        "recomputed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat() if recomputed else None,
        "plain_text": "",
        "version": 2,
    }
    explanation["plain_text"] = _build_plaintext_explanation(explanation)
    if write_db:
        DB.init_bets_explainability()
        DB.update_bet_explanation(game_pk, explanation)
    return explanation


# ---------------------------------------------------------------------------
# Feature engineering (same logic as train.py but accepts a pre-loaded df)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["game_date"].dt.year

    FG_TEAM_COLS = [c for c in [
        "home_avg", "home_obp", "home_slg", "home_woba", "home_wrc_plus",
        "home_war", "home_k_pct", "home_bb_pct", "home_k_per_9",
        "home_bb_per_9", "home_hr_per_9", "home_era", "home_fip", "home_owar",
        "away_avg", "away_obp", "away_slg", "away_woba", "away_wrc_plus",
        "away_war", "away_k_pct", "away_bb_pct", "away_k_per_9",
        "away_bb_per_9", "away_hr_per_9", "away_era", "away_fip", "away_owar",
    ] if c in df.columns]

    for side in ["home", "away"]:
        team_col = f"{side}_team"
        side_fg  = [c for c in FG_TEAM_COLS if c.startswith(side)]
        df = df.sort_values([team_col, "season", "game_date"])
        for col in side_fg:
            df[col] = df.groupby([team_col, "season"])[col].shift(1)
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    df = T.add_market_columns(df)
    df = T.add_schedule_context(df)

    long_games = T.build_long_format(df)
    roll_main  = T.add_rolling_features(long_games, T.BEST_W)
    roll_short = T.add_rolling_features(long_games, T.MOMENTUM_W)

    df_feat = df.merge(roll_main, on="game_id", how="left")
    df_feat = df_feat.merge(
        roll_short[["game_id",
                    f"h_win_pct_{T.MOMENTUM_W}", f"a_win_pct_{T.MOMENTUM_W}",
                    f"h_runs_scored_avg_{T.MOMENTUM_W}", f"h_runs_allowed_avg_{T.MOMENTUM_W}",
                    f"a_runs_scored_avg_{T.MOMENTUM_W}", f"a_runs_allowed_avg_{T.MOMENTUM_W}"]],
        on="game_id", how="left")

    ptch = T.build_pitcher_features(df)
    df_feat = df_feat.merge(ptch, on=["game_id", "game_date"], how="left")

    W = T.BEST_W
    df_feat["win_pct_W_DIFF"]          = T._gcol(df_feat, f"h_win_pct_{W}")         - T._gcol(df_feat, f"a_win_pct_{W}")
    df_feat["run_diff_avg_W_DIFF"]     = T._gcol(df_feat, f"h_run_diff_avg_{W}")    - T._gcol(df_feat, f"a_run_diff_avg_{W}")
    df_feat["run_diff_std_W_DIFF"]     = T._gcol(df_feat, f"h_run_diff_std_{W}")    - T._gcol(df_feat, f"a_run_diff_std_{W}")
    df_feat["runs_scored_avg_W_DIFF"]  = T._gcol(df_feat, f"h_runs_scored_avg_{W}") - T._gcol(df_feat, f"a_runs_scored_avg_{W}")
    df_feat["runs_allowed_avg_W_DIFF"] = T._gcol(df_feat, f"h_runs_allowed_avg_{W}")- T._gcol(df_feat, f"a_runs_allowed_avg_{W}")
    df_feat["season_win_pct_DIFF"]     = T._gcol(df_feat, "h_season_win_pct")       - T._gcol(df_feat, "a_season_win_pct")
    df_feat["season_run_diff_avg_DIFF"]= T._gcol(df_feat, "h_season_run_diff_avg")  - T._gcol(df_feat, "a_season_run_diff_avg")
    df_feat["streak_DIFF"]             = T._gcol(df_feat, "h_streak")               - T._gcol(df_feat, "a_streak")

    h_momentum = T._gcol(df_feat, f"h_win_pct_{T.MOMENTUM_W}") - T._gcol(df_feat, f"h_win_pct_{W}")
    a_momentum = T._gcol(df_feat, f"a_win_pct_{T.MOMENTUM_W}") - T._gcol(df_feat, f"a_win_pct_{W}")
    df_feat["momentum_DIFF"]           = h_momentum - a_momentum
    h_fip = T._gcol(df_feat, "h_fip_lag1").combine_first(T._gcol(df_feat, "home_sp_fip")).combine_first(T._gcol(df_feat, "home_starter_fip"))
    a_fip = T._gcol(df_feat, "a_fip_lag1").combine_first(T._gcol(df_feat, "away_sp_fip")).combine_first(T._gcol(df_feat, "away_starter_fip"))
    h_era = T._gcol(df_feat, "h_sp_era_lag1").combine_first(T._gcol(df_feat, "home_sp_era")).combine_first(T._gcol(df_feat, "home_starter_era"))
    a_era = T._gcol(df_feat, "a_sp_era_lag1").combine_first(T._gcol(df_feat, "away_sp_era")).combine_first(T._gcol(df_feat, "away_starter_era"))
    h_k9 = T._gcol(df_feat, "h_sp_k9_lag1").combine_first(T._gcol(df_feat, "home_sp_k9")).combine_first(T._gcol(df_feat, "home_starter_k9"))
    a_k9 = T._gcol(df_feat, "a_sp_k9_lag1").combine_first(T._gcol(df_feat, "away_sp_k9")).combine_first(T._gcol(df_feat, "away_starter_k9"))
    h_bb9 = T._gcol(df_feat, "h_sp_bb9_lag1").combine_first(T._gcol(df_feat, "home_sp_bb9")).combine_first(T._gcol(df_feat, "home_starter_bb9"))
    a_bb9 = T._gcol(df_feat, "a_sp_bb9_lag1").combine_first(T._gcol(df_feat, "away_sp_bb9")).combine_first(T._gcol(df_feat, "away_starter_bb9"))
    df_feat["sp_fip_DIFF"]             = pd.to_numeric(a_fip, errors="coerce") - pd.to_numeric(h_fip, errors="coerce")
    df_feat["sp_era_DIFF"]             = pd.to_numeric(a_era, errors="coerce") - pd.to_numeric(h_era, errors="coerce")
    df_feat["sp_k9_DIFF"]              = pd.to_numeric(h_k9, errors="coerce")  - pd.to_numeric(a_k9, errors="coerce")
    df_feat["sp_bb9_DIFF"]             = pd.to_numeric(a_bb9, errors="coerce") - pd.to_numeric(h_bb9, errors="coerce")
    df_feat["wrc_plus_DIFF"]           = T._gcol(df_feat, "home_wrc_plus")          - T._gcol(df_feat, "away_wrc_plus")
    df_feat["woba_DIFF"]               = T._gcol(df_feat, "home_woba")              - T._gcol(df_feat, "away_woba")
    df_feat["avg_DIFF"]                = T._gcol(df_feat, "home_avg")               - T._gcol(df_feat, "away_avg")
    df_feat["obp_DIFF"]                = T._gcol(df_feat, "home_obp")               - T._gcol(df_feat, "away_obp")
    df_feat["slg_DIFF"]                = T._gcol(df_feat, "home_slg")               - T._gcol(df_feat, "away_slg")
    df_feat["k_pct_DIFF"]              = T._gcol(df_feat, "away_k_pct")             - T._gcol(df_feat, "home_k_pct")
    df_feat["bb_pct_DIFF"]             = T._gcol(df_feat, "home_bb_pct")            - T._gcol(df_feat, "away_bb_pct")
    df_feat["k_per_9_DIFF"]            = T._gcol(df_feat, "away_k_per_9")           - T._gcol(df_feat, "home_k_per_9")
    df_feat["bb_per_9_DIFF"]           = T._gcol(df_feat, "home_bb_per_9")          - T._gcol(df_feat, "away_bb_per_9")
    df_feat["hr_per_9_DIFF"]           = T._gcol(df_feat, "away_hr_per_9")          - T._gcol(df_feat, "home_hr_per_9")
    df_feat["era_DIFF"]                = T._gcol(df_feat, "away_era")               - T._gcol(df_feat, "home_era")
    df_feat["fip_DIFF"]                = T._gcol(df_feat, "away_fip")               - T._gcol(df_feat, "home_fip")
    df_feat["owar_DIFF"]               = T._gcol(df_feat, "home_owar")              - T._gcol(df_feat, "away_owar")
    df_feat["war_DIFF"]                = T._gcol(df_feat, "h_war_lag1")             - T._gcol(df_feat, "a_war_lag1")
    df_feat["pitcher_handedness_diff"] = T._gcol(df_feat, "home_pitcher_is_lefty")  - T._gcol(df_feat, "away_pitcher_is_lefty")
    df_feat["sharp_x_fip"]             = df_feat["sharp_move_flag"] * df_feat["sp_fip_DIFF"]

    df_feat = df_feat.sort_values("game_date").reset_index(drop=True)
    df_feat["hg"] = T._gcol(df_feat, "h_games_played")
    df_feat["ag"] = T._gcol(df_feat, "a_games_played")
    df_feat["early_season_flag"] = (
        df_feat[["hg", "ag"]].min(axis=1) < T.EARLY_SEASON_GAMES).astype(float)
    df_feat["home_games_played"] = df_feat["hg"]
    df_feat["away_games_played"] = df_feat["ag"]

    df_feat = T.engineer_new_features(df_feat)
    return df_feat


# ---------------------------------------------------------------------------
# Shared preparation - load data, engineer features, train model(s).
# Returns a cache of fitted artifacts + only the target rows (not full df_feat).
# ---------------------------------------------------------------------------

def _load_historical() -> pd.DataFrame:
    try:
        hist = DB.get_games_df()
        hist = hist[hist["season"].notna() & (hist["season"].astype(float) < ACTIVE_SEASON)]
        if hist.empty:
            raise ValueError("No historical rows in DB")
        return hist
    except Exception as e:
        print(f"  WARNING: DB load failed ({e}), falling back to CSV.")
        return pd.read_csv(HISTORICAL_CSV, low_memory=False)


def _load_current() -> pd.DataFrame:
    try:
        curr = DB.get_games_df(season=ACTIVE_SEASON)
        if curr.empty:
            raise ValueError(f"No {ACTIVE_SEASON} rows in DB")
        return curr
    except Exception as e:
        print(f"  WARNING: DB load failed ({e}), falling back to CSV.")
        if not os.path.exists(CURRENT_CSV):
            raise PredictError(f"{CURRENT_CSV} not found. Run fetch/fetch_data.py first.")
        return pd.read_csv(CURRENT_CSV, low_memory=False)


def prepare_shared(game_pks: list[str], game_date: str) -> dict:
    """
    Load data, engineer features, load or train model artifact. Returns a cache dict:
        clf, early_clf, imputer, active_feats, early_feats,
        target_rows: {game_pk_str: one-row DataFrame}.
    The full df_feat is not retained - only rows for the requested game_pks.
    """
    game_pks = [str(pk) for pk in game_pks]
    print("  Loading historical data...")
    hist = _load_historical()
    curr = _load_current()

    curr["game_date"] = pd.to_datetime(curr["game_date"])
    target_date = pd.to_datetime(game_date)
    curr_subset = curr[curr["game_date"] <= target_date].copy()

    all_cols = list(dict.fromkeys(list(hist.columns) + list(curr_subset.columns)))
    combined = pd.concat(
        [hist.reindex(columns=all_cols), curr_subset.reindex(columns=all_cols)],
        ignore_index=True,
    )
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    combined = combined.sort_values("game_date").reset_index(drop=True)

    print(f"  Engineering features ({len(combined):,} rows)...")
    df_feat = engineer_features(combined)

    train_df = df_feat[df_feat["home_win"].notna()].copy()
    active_feats = [c for c in T.FEATURE_COLUMNS       if c in df_feat.columns]
    early_feats  = [c for c in T.EARLY_FEATURE_COLUMNS if c in df_feat.columns]

    req = ["home_win", "market_implied_prob", "game_date",
           "home_games_played", "away_games_played"]
    train_df = train_df.dropna(subset=[c for c in req if c in train_df.columns])
    if len(train_df) < 100:
        raise PredictError("Insufficient training data after filtering.")
    print(f"  Training rows: {len(train_df):,}")

    fingerprint, fp_meta = _training_fingerprint(train_df, active_feats, early_feats)
    artifact_id = None
    artifact_loaded = False
    artifact_table_ready = True

    try:
        DB.init_model_artifacts_table()
        artifact_row = DB.get_model_artifact_by_fingerprint(fingerprint)
    except Exception as e:
        artifact_table_ready = False
        artifact_row = None
        print(f"  WARNING: model artifact lookup failed ({e}); training in-process.")

    if artifact_row:
        try:
            bundle = _deserialize_artifact(artifact_row, fingerprint)
            clf = bundle["clf"]
            early_clf = bundle.get("early_clf")
            imputer = bundle.get("imputer")
            active_feats = list(bundle.get("active_feats") or active_feats)
            early_feats = list(bundle.get("early_feats") or early_feats)
            artifact_id = int(artifact_row["id"])
            artifact_loaded = True
            print(f"  Loaded model artifact id={artifact_id} ({T.MODEL}, fingerprint={fingerprint[:12]}).")
        except Exception as e:
            print(f"  WARNING: model artifact load failed ({e}); retraining.")

    if not artifact_loaded:
        X_train = train_df[active_feats].values.astype(np.float32)
        y_train = train_df["home_win"].values.astype(np.float32)

        early_mask = (
            (train_df["home_games_played"] < T.EARLY_CUTOFF) |
            (train_df["away_games_played"] < T.EARLY_CUTOFF)
        ) if T.EARLY_CUTOFF else pd.Series(False, index=train_df.index)

        early_clf = None
        if T.EARLY_CUTOFF and early_mask.sum() > 50 and early_feats:
            X_early = train_df.loc[early_mask, early_feats].values.astype(np.float32)
            y_early = y_train[early_mask.values]
            early_clf = T.build_early_lr(X_early, y_early)

        reg_mask = ~early_mask.values if T.EARLY_CUTOFF else np.ones(len(train_df), dtype=bool)
        X_tr_reg = X_train[reg_mask]
        y_tr_reg = y_train[reg_mask]

        print(f"  Training {T.MODEL}...")
        if T.MODEL == "lgb":
            clf = T.build_lgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
        elif T.MODEL == "xgb":
            clf = T.build_xgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
        elif T.MODEL == "lr":
            clf = T.build_lr(X_tr_reg, y_tr_reg)
        elif T.MODEL == "mlp":
            clf = T.build_mlp(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
        elif T.MODEL in ("ensemble_avg", "ensemble_stack"):
            clf = (
                T.build_lgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg),
                T.build_xgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg),
                T.build_lr(X_tr_reg, y_tr_reg),
            )
        else:
            raise PredictError(f"Unknown MODEL: {T.MODEL!r}")

        # Fit ensemble imputer once - consumed by the ensemble LR branch at predict time.
        imputer = None
        if T.MODEL in ("ensemble_avg", "ensemble_stack"):
            imputer = SimpleImputer(strategy="median").fit(X_tr_reg)

        if artifact_table_ready:
            try:
                bundle = {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "training_fingerprint": fingerprint,
                    "model_type": T.MODEL,
                    "clf": clf,
                    "early_clf": early_clf,
                    "imputer": imputer,
                    "active_feats": active_feats,
                    "early_feats": early_feats,
                    "feature_config": fp_meta["feature_config"],
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                artifact_bytes, artifact_sha256 = _serialize_artifact(bundle)
                fi_list = _artifact_feature_importances(clf, active_feats)
                artifact_id = DB.save_model_artifact({
                    "model_type": T.MODEL,
                    "training_fingerprint": fingerprint,
                    "artifact_format": "pickle",
                    "artifact_bytes": artifact_bytes,
                    "artifact_sha256": artifact_sha256,
                    "settled_row_count": fp_meta["settled_row_count"],
                    "max_settled_game_date": fp_meta["max_settled_game_date"],
                    "active_season": ACTIVE_SEASON,
                    "target_train_cutoff": str(target_date.date()),
                    "num_features": fp_meta["num_features"],
                    "feature_columns": active_feats,
                    "early_feature_columns": early_feats,
                    "feature_config": fp_meta["feature_config"],
                    "metrics": {"source": "prediction_batch"},
                    "feature_importances": fi_list,
                    "git_commit": fp_meta["git_commit"],
                })
                print(f"  Saved model artifact id={artifact_id} ({T.MODEL}, fingerprint={fingerprint[:12]}).")
            except Exception as e:
                artifact_id = None
                print(f"  WARNING: model artifact save failed ({e}); continuing with in-memory model.")

    # Keep only target rows; drop the full df_feat to shrink the cache.
    tgt_mask = df_feat["game_pk"].astype(str).isin(set(game_pks))
    targets = df_feat[tgt_mask].copy()
    target_rows = {}
    for pk in game_pks:
        rows = targets[targets["game_pk"].astype(str) == pk]
        if not rows.empty:
            target_rows[pk] = rows.iloc[[0]].reset_index(drop=True)

    return {
        "clf":          clf,
        "early_clf":    early_clf,
        "imputer":      imputer,
        "active_feats": active_feats,
        "early_feats":  early_feats,
        "target_rows":  target_rows,
        "model_artifact_id": artifact_id,
        "training_fingerprint": fingerprint,
    }


# ---------------------------------------------------------------------------
# Target-game lookup
# ---------------------------------------------------------------------------

def find_target_game(game_pk: str | None = None, game_date: str | None = None,
                     home_team: str | None = None, away_team: str | None = None) -> dict:
    if game_pk:
        row = DB.get_bet(game_pk)
        if row:
            return {
                "game_pk":   str(row["game_pk"]),
                "game_date": str(row["game_date"])[:10],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
        df = DB.get_games_df(season=ACTIVE_SEASON)
        if not df.empty:
            df["game_pk"] = df["game_pk"].astype(str)
            rows = df[df["game_pk"] == str(game_pk)]
            if not rows.empty:
                r = rows.iloc[0]
                return {
                    "game_pk":   str(r["game_pk"]),
                    "game_date": str(r["game_date"])[:10],
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                }
        raise PredictError(f"game_pk={game_pk} not found in DB")

    df = DB.get_games_df(season=ACTIVE_SEASON)
    if df.empty:
        raise PredictError(f"No {ACTIVE_SEASON} games in DB. Run fetch/fetch_data.py first.")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    mask = (
        (df["game_date"] == str(game_date)) &
        (df["home_team"].str.upper() == home_team.upper()) &
        (df["away_team"].str.upper() == away_team.upper())
    )
    rows = df[mask]
    if rows.empty:
        raise PredictError(
            f"No game on {game_date} with home={home_team} away={away_team} in DB"
        )
    r = rows.iloc[0]
    return {
        "game_pk":   str(r["game_pk"]),
        "game_date": str(r["game_date"])[:10],
        "home_team": r["home_team"],
        "away_team": r["away_team"],
    }


# ---------------------------------------------------------------------------
# Per-game prediction
# ---------------------------------------------------------------------------

def predict_one(game_pk: str, shared: dict) -> dict:
    """Predict a single game using a shared cache. Writes to DB and returns a result dict."""
    game_pk = str(game_pk)
    target_df = shared["target_rows"].get(game_pk)
    if target_df is None:
        raise PredictError(f"Target game {game_pk} missing from shared cache.")

    clf          = shared["clf"]
    early_clf    = shared["early_clf"]
    imputer      = shared["imputer"]
    active_feats = shared["active_feats"]
    early_feats  = shared["early_feats"]
    model_artifact_id = shared.get("model_artifact_id")

    home_team = target_df["home_team"].iloc[0]
    away_team = target_df["away_team"].iloc[0]
    game_date = str(target_df["game_date"].iloc[0])[:10]
    print(f"Predicting: {away_team} @ {home_team} on {game_date} (game_pk={game_pk})")

    is_early = False
    if T.EARLY_CUTOFF is not None:
        hgp = float(target_df["home_games_played"].iloc[0]) if "home_games_played" in target_df else 999
        agp = float(target_df["away_games_played"].iloc[0]) if "away_games_played" in target_df else 999
        is_early = (hgp < T.EARLY_CUTOFF) or (agp < T.EARLY_CUTOFF)
    if is_early and early_clf is not None:
        print("  Using early-season specialist model.")

    X_target = target_df[active_feats].values.astype(np.float32)

    if is_early and early_clf is not None:
        X_tgt_early = target_df[early_feats].values.astype(np.float32)
        prob = float(T.get_proba(early_clf, X_tgt_early)[0])
    elif T.MODEL in ("ensemble_avg", "ensemble_stack"):
        clf_lgb, clf_xgb, clf_lr = clf
        X_tgt_sc = imputer.transform(X_target) if imputer is not None else X_target
        p_lgb = T.get_proba(clf_lgb, X_target)[0]
        p_xgb = T.get_proba(clf_xgb, X_target)[0]
        p_lr  = T.get_proba(clf_lr,  X_tgt_sc)[0]
        prob  = float((p_lgb + p_xgb + p_lr) / 3.0)
    else:
        prob = float(T.get_proba(clf, X_target)[0])

    mkt_prob_raw = target_df["market_implied_prob"].iloc[0]
    if pd.isna(mkt_prob_raw):
        print("WARNING: No market odds available. Cannot calculate edge or bet size.")
        edge, bet_frac, bet_side = float("nan"), 0.0, "none"
    else:
        mp = float(mkt_prob_raw)
        pp = float(np.clip(prob, T.PROB_CAP[0], T.PROB_CAP[1]))
        thresh = T.CONFIDENCE_THRESHOLD + (0.02 * abs(mp - 0.5) if T.DYNAMIC_THRESHOLD else 0.0)
        edge_home = pp - mp
        edge_away = mp - pp
        if edge_home >= thresh and mp > 1e-6:
            bet_frac = T.kelly_stake(pp, 1.0 / mp, is_warmup=is_early)
            bet_side, edge = "home", edge_home
        elif edge_away >= thresh and (1.0 - mp) > 1e-6:
            bet_frac = T.kelly_stake(1.0 - pp, 1.0 / (1.0 - mp), is_warmup=is_early)
            bet_side, edge = "away", edge_away
        else:
            bet_frac, bet_side, edge = 0.0, "none", max(edge_home, edge_away)

    # Explainability: store top LR contributions for bet signals.
    explanation = None
    try:
        if bet_side in {"home", "away"} and float(bet_frac or 0.0) > 0:
            DB.init_bets_explainability()
            if is_early and early_clf is not None:
                pipe = _extract_lr_pipeline_for_explainability(early_clf)
                feats = early_feats
                x_raw = target_df[early_feats].values.astype(np.float32)
            else:
                pipe = _extract_lr_pipeline_for_explainability(clf)
                feats = active_feats
                x_raw = X_target

            if pipe is not None and feats:
                base = _lr_feature_contributions(pipe, feats, x_raw)
                if base is not None:
                    if bet_side == "away":
                        # Away win attribution is the negation of home win log-odds contributions.
                        base = dict(base)
                        old_pos = list(base.get("top_positive", []) or [])
                        old_neg = list(base.get("top_negative", []) or [])
                        base["raw_logit_side"] = -float(base.get("raw_logit_home", 0.0))
                        base["raw_prob_side"] = float(1.0 - float(base.get("raw_prob_home", 0.5)))
                        base["top_positive"] = [{"feature": r["feature"], "value": float(-r["value"])} for r in old_neg]
                        base["top_negative"] = [{"feature": r["feature"], "value": float(-r["value"])} for r in old_pos]
                    else:
                        base["raw_logit_side"] = float(base.get("raw_logit_home", 0.0))
                        base["raw_prob_side"] = float(base.get("raw_prob_home", 0.5))

                    explanation = {
                        "game_pk": game_pk,
                        "bet_side": bet_side,
                        "model": str(T.MODEL),
                        "is_early_specialist": bool(is_early and early_clf is not None),
                        "predicted_prob_home": float(prob),
                        "market_implied_prob_home": (float(mkt_prob_raw) if not pd.isna(mkt_prob_raw) else None),
                        "edge": float(edge) if edge is not None and edge == edge else None,
                        "bet_frac": float(bet_frac or 0.0),
                        "contributions": base,
                        "recomputed": False,
                        "recomputed_reason": "",
                        "recomputed_at_utc": None,
                        "plain_text": "",
                        "version": 1,
                    }
                    explanation["plain_text"] = _build_plaintext_explanation(explanation)
                    DB.update_bet_explanation(game_pk, explanation)
    except Exception:
        # Explainability is best-effort; prediction + DB write should still succeed.
        explanation = None

    print(f"\n{'='*50}")
    print(f"  Model prob (home win): {prob:.4f}")
    print(f"  Market implied prob:   {float(mkt_prob_raw) if not pd.isna(mkt_prob_raw) else 'N/A'}")
    if isinstance(edge, float) and np.isnan(edge):
        print("  Edge: N/A")
    else:
        print(f"  Edge:                  {edge:.4f}")
    print(f"  Bet side:              {bet_side}")
    print(f"  Kelly fraction:        {bet_frac:.4f}")
    print(f"  >> {'BET ' + bet_side.upper() + f' at {bet_frac*100:.2f}% of bankroll' if bet_frac > 0 else 'NO BET'}")
    print(f"{'='*50}")

    mkt_prob_val = float(mkt_prob_raw) if not pd.isna(mkt_prob_raw) else None
    edge_val = float(edge) if not (isinstance(edge, float) and np.isnan(edge)) else None
    DB.update_bet_prediction(game_pk, prob, edge_val, bet_side, bet_frac, mkt_prob_val, model_artifact_id)
    print("Results written to bets table")
    return {
        "game_pk":  game_pk,
        "prob":     prob,
        "edge":     edge_val,
        "bet_side": bet_side,
        "bet_frac": bet_frac,
        "explanation": explanation,
        "model_artifact_id": model_artifact_id,
    }


# ---------------------------------------------------------------------------
# CLI entry - single game, builds shared cache for just that game.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MLB game outcome and bet size.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--game_pk",   type=str, help="MLB Stats API game_pk")
    group.add_argument("--game_date", type=str, help="Game date YYYY-MM-DD (with --home_team / --away_team)")
    parser.add_argument("--home_team", type=str, default=None)
    parser.add_argument("--away_team", type=str, default=None)
    args = parser.parse_args()

    if args.game_date and (not args.home_team or not args.away_team):
        parser.error("--game_date requires --home_team and --away_team")

    try:
        info = find_target_game(
            game_pk=args.game_pk, game_date=args.game_date,
            home_team=args.home_team, away_team=args.away_team,
        )
        shared = prepare_shared([info["game_pk"]], info["game_date"])
        predict_one(info["game_pk"], shared)
    except PredictError as e:
        sys.exit(f"ERROR: {e}")
