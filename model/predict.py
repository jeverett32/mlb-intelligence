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
import os
import sys
import warnings

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


class PredictError(RuntimeError):
    pass


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
    df_feat["hg"] = df_feat.groupby(["home_team", "season"]).cumcount()
    df_feat["ag"] = df_feat.groupby(["away_team", "season"]).cumcount()
    df_feat["early_season_flag"] = (
        df_feat[["hg", "ag"]].min(axis=1) < T.EARLY_SEASON_GAMES).astype(float)
    df_feat["home_games_played"] = df_feat["hg"]
    df_feat["away_games_played"] = df_feat["ag"]

    df_feat = T.engineer_new_features(df_feat)
    return df_feat


# ---------------------------------------------------------------------------
# Shared preparation — load data, engineer features, train model(s).
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
    Load data, engineer features, train models. Returns a cache dict:
        clf, early_clf, imputer, active_feats, early_feats,
        target_rows: {game_pk_str: one-row DataFrame}.
    The full df_feat is not retained — only rows for the requested game_pks.
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

    # Fit ensemble imputer once — consumed by the ensemble LR branch at predict time.
    imputer = None
    if T.MODEL in ("ensemble_avg", "ensemble_stack"):
        imputer = SimpleImputer(strategy="median").fit(X_tr_reg)

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
    DB.update_bet_prediction(game_pk, prob, edge_val, bet_side, bet_frac, mkt_prob_val)
    print("Results written to bets table")
    return {
        "game_pk":  game_pk,
        "prob":     prob,
        "edge":     edge_val,
        "bet_side": bet_side,
        "bet_frac": bet_frac,
    }


# ---------------------------------------------------------------------------
# CLI entry — single game, builds shared cache for just that game.
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
