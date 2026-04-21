"""
MLB betting model prediction script.
Trains on all historical data (master_mlb.csv) with current season context
(mlb_2026.csv), then predicts edge + Kelly stake for a single target game.

Usage:
    uv run model/predict.py --game_pk 12345
    uv run model/predict.py --game_date 2026-04-01 --home_team NYY --away_team BOS

Output: writes predicted_prob, edge, bet_frac, bet_side to games.csv for the
        target game row (row must already exist, created by the scheduler).
"""

import argparse
import os
import sys
import warnings

os.environ["PYTHONHASHSEED"] = "42"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Import model config + feature engineering from train.py.
# train.py's main block is guarded by __name__ == "__main__", so importing
# it here only loads functions and constants — no training runs on import.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import train as T
import db as DB

# CSV fallback paths (used when DB is unavailable)
HISTORICAL_CSV = "data/master_mlb.csv"
CURRENT_CSV    = "data/mlb_2026.csv"


# ---------------------------------------------------------------------------
# Feature engineering on an arbitrary dataframe (same logic as train.py's
# load_and_engineer_features, but accepts a pre-loaded dataframe).
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline. df must have the same schema as master_mlb.csv."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["game_date"].dt.year

    # Lag FanGraphs season-to-date team stats by 1 game
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
        df = df.sort_values([team_col, "season", "game_date"]).copy()
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
    df_feat["sharp_x_fip"]            = df_feat["sharp_move_flag"] * df_feat["sp_fip_DIFF"]

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
# Shared model preparation (called once per batch to avoid redundant training)
# ---------------------------------------------------------------------------

def prepare_shared_data(game_date: str, output_path: str):
    """
    Load data, engineer features, and train models for all games on game_date.
    Serializes artifacts to output_path so parallel predict workers can load
    them without repeating the expensive data load + train step.
    """
    print(f"  Loading historical data...")
    try:
        hist = DB.get_games_df()
        hist = hist[hist["season"].notna() & (hist["season"].astype(float) < 2026)]
        if hist.empty:
            raise ValueError("No historical rows in DB")
    except Exception as e:
        print(f"  WARNING: DB load failed ({e}), falling back to CSV.")
        hist = pd.read_csv(HISTORICAL_CSV, low_memory=False)

    try:
        curr = DB.get_games_df(season=2026)
        if curr.empty:
            raise ValueError("No 2026 rows in DB")
    except Exception as e:
        print(f"  WARNING: DB load failed ({e}), falling back to CSV.")
        curr = pd.read_csv(CURRENT_CSV, low_memory=False)

    # Include all 2026 rows up to and including game_date
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

    req = ["home_win", "market_implied_prob", "game_date", "home_games_played", "away_games_played"]
    train_df = train_df.dropna(subset=[c for c in req if c in train_df.columns])
    print(f"  Training rows: {len(train_df):,}")

    X_train = train_df[active_feats].values.astype(np.float32)
    y_train = train_df["home_win"].values.astype(np.float32)

    early_mask = (
        (train_df["home_games_played"] < T.EARLY_CUTOFF) |
        (train_df["away_games_played"] < T.EARLY_CUTOFF)
    ) if T.EARLY_CUTOFF else pd.Series(False, index=train_df.index)

    # Always train early specialist (each game decides whether to use it)
    early_clf = None
    if T.EARLY_CUTOFF and early_mask.sum() > 50 and early_feats:
        ef = [f for f in early_feats if f in train_df.columns]
        X_early = train_df.loc[early_mask, ef].values.astype(np.float32)
        y_early = y_train[early_mask.values]
        early_clf = T.build_early_lr(X_early, y_early)

    reg_mask = ~early_mask.values if T.EARLY_CUTOFF else np.ones(len(train_df), dtype=bool)
    X_tr_reg = X_train[reg_mask]
    y_tr_reg  = y_train[reg_mask]

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
        sys.exit(f"Unknown MODEL: {T.MODEL!r}")

    cache = {
        "df_feat":     df_feat,
        "clf":         clf,
        "early_clf":   early_clf,
        "active_feats": active_feats,
        "early_feats": early_feats,
    }
    joblib.dump(cache, output_path, compress=3)
    print(f"  Shared model cache written → {output_path}")


# ---------------------------------------------------------------------------
# Locate the target game in games.csv (and mlb_2026.csv for features)
# ---------------------------------------------------------------------------

def find_target_game(args) -> dict:
    """Return a dict with game_pk, game_date, home_team, away_team from the DB."""
    if args.game_pk:
        # Try bets table first (init_bet is always called before predict)
        row = DB.get_bet(args.game_pk)
        if row:
            return {
                "game_pk":   str(row["game_pk"]),
                "game_date": str(row["game_date"])[:10],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
        # Fall back to games table
        df = DB.get_games_df(season=2026)
        if not df.empty:
            df["game_pk"] = df["game_pk"].astype(str)
            rows = df[df["game_pk"] == str(args.game_pk)]
            if not rows.empty:
                r = rows.iloc[0]
                return {
                    "game_pk":   str(r["game_pk"]),
                    "game_date": str(r["game_date"])[:10],
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                }
        sys.exit(f"ERROR: game_pk={args.game_pk} not found in DB")

    # Lookup by date + teams
    df = DB.get_games_df(season=2026)
    if df.empty:
        sys.exit("ERROR: No 2026 games in DB. Run fetch/fetch_data.py first.")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    mask = (
        (df["game_date"] == str(args.game_date)) &
        (df["home_team"].str.upper() == args.home_team.upper()) &
        (df["away_team"].str.upper() == args.away_team.upper())
    )
    rows = df[mask]
    if rows.empty:
        sys.exit(
            f"ERROR: no game on {args.game_date} with "
            f"home={args.home_team} away={args.away_team} in DB"
        )
    r = rows.iloc[0]
    return {
        "game_pk":   str(r["game_pk"]),
        "game_date": str(r["game_date"])[:10],
        "home_team": r["home_team"],
        "away_team": r["away_team"],
    }


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def predict(game_info: dict, shared_data: dict | None = None):
    game_pk    = str(game_info["game_pk"])
    game_date  = str(game_info["game_date"])
    home_team  = str(game_info["home_team"])
    away_team  = str(game_info["away_team"])

    print(f"Predicting: {away_team} @ {home_team} on {game_date} (game_pk={game_pk})")

    if shared_data is not None:
        # --- Fast path: use pre-trained model from batch preparation ----------
        df_feat      = shared_data["df_feat"]
        clf          = shared_data["clf"]
        early_clf    = shared_data["early_clf"]
        active_feats = shared_data["active_feats"]
        early_feats  = shared_data["early_feats"]

        tgt_mask  = df_feat["game_pk"].astype(str) == game_pk
        target_df = df_feat[tgt_mask].copy()
        if target_df.empty:
            sys.exit("ERROR: Target game row missing in shared feature data.")

        is_early = False
        if T.EARLY_CUTOFF is not None:
            hgp = float(target_df["home_games_played"].iloc[0]) if "home_games_played" in target_df else 999
            agp = float(target_df["away_games_played"].iloc[0]) if "away_games_played" in target_df else 999
            is_early = (hgp < T.EARLY_CUTOFF) or (agp < T.EARLY_CUTOFF)
        if is_early and early_clf is not None:
            print("  Using early-season specialist model (shared).")

    else:
        # --- Standard path: load data, engineer features, train ---------------
        print("Loading historical data...")
        try:
            hist = DB.get_games_df()
            hist = hist[hist["season"].notna() & (hist["season"].astype(float) < 2026)]
            if hist.empty:
                raise ValueError("No historical rows in DB")
        except Exception as e:
            print(f"  WARNING: DB load failed ({e}), falling back to CSV.")
            hist = pd.read_csv(HISTORICAL_CSV, low_memory=False)
        print(f"  {len(hist):,} historical rows")

        try:
            curr = DB.get_games_df(season=2026)
            if curr.empty:
                raise ValueError("No 2026 rows in DB")
        except Exception as e:
            print(f"  WARNING: DB load failed ({e}), falling back to CSV.")
            if not os.path.exists(CURRENT_CSV):
                sys.exit(f"ERROR: {CURRENT_CSV} not found. Run fetch/fetch_data.py first.")
            curr = pd.read_csv(CURRENT_CSV, low_memory=False)
        print(f"  {len(curr):,} 2026 rows")

        # Locate target game in current season data
        curr["game_pk"] = curr["game_pk"].astype(str)
        target_mask = curr["game_pk"] == game_pk
        if not target_mask.any():
            curr["game_date"] = pd.to_datetime(curr["game_date"]).dt.strftime("%Y-%m-%d")
            target_mask = (
                (curr["game_date"] == game_date) &
                (curr["home_team"].str.upper() == home_team.upper()) &
                (curr["away_team"].str.upper() == away_team.upper())
            )
        if not target_mask.any():
            sys.exit(
                f"ERROR: Target game not found in {CURRENT_CSV}. "
                "Run fetch_2026_data.py to refresh."
            )

        target_game_pk = curr.loc[target_mask, "game_pk"].iloc[0]

        curr["game_date"] = pd.to_datetime(curr["game_date"])
        target_date       = curr.loc[target_mask, "game_date"].iloc[0]
        curr_subset       = curr[curr["game_date"] <= target_date].copy()

        all_cols = list(dict.fromkeys(list(hist.columns) + list(curr_subset.columns)))
        combined = pd.concat(
            [hist.reindex(columns=all_cols), curr_subset.reindex(columns=all_cols)],
            ignore_index=True,
        )
        combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
        combined = combined.sort_values("game_date").reset_index(drop=True)

        print("Engineering features...")
        df_feat = engineer_features(combined)

        train_df  = df_feat[df_feat["home_win"].notna()].copy()
        tgt_mask  = df_feat["game_pk"].astype(str) == str(target_game_pk)
        target_df = df_feat[tgt_mask].copy()

        if target_df.empty:
            sys.exit("ERROR: Target game row missing after feature engineering.")

        print(f"  Training rows: {len(train_df):,}")

        active_feats = [c for c in T.FEATURE_COLUMNS       if c in df_feat.columns]
        early_feats  = [c for c in T.EARLY_FEATURE_COLUMNS if c in df_feat.columns]

        req = ["home_win", "market_implied_prob", "game_date",
               "home_games_played", "away_games_played"]
        train_df = train_df.dropna(subset=[c for c in req if c in train_df.columns])

        if len(train_df) < 100:
            sys.exit("ERROR: Insufficient training data after filtering.")

        is_early = False
        if T.EARLY_CUTOFF is not None:
            hgp = float(target_df["home_games_played"].iloc[0]) if "home_games_played" in target_df else 999
            agp = float(target_df["away_games_played"].iloc[0]) if "away_games_played" in target_df else 999
            is_early = (hgp < T.EARLY_CUTOFF) or (agp < T.EARLY_CUTOFF)

        print(f"Training {T.MODEL} on {len(train_df):,} rows...")
        X_train = train_df[active_feats].values.astype(np.float32)
        y_train = train_df["home_win"].values.astype(np.float32)

        early_mask = (
            (train_df["home_games_played"] < T.EARLY_CUTOFF) |
            (train_df["away_games_played"] < T.EARLY_CUTOFF)
        ) if T.EARLY_CUTOFF else pd.Series(False, index=train_df.index)

        early_clf = None
        if is_early and T.EARLY_CUTOFF and early_mask.sum() > 50 and early_feats:
            ef = [f for f in early_feats if f in train_df.columns]
            X_early = train_df.loc[early_mask, ef].values.astype(np.float32)
            y_early = y_train[early_mask.values]
            early_clf = T.build_early_lr(X_early, y_early)
            print("  Using early-season specialist model.")

        reg_mask  = ~early_mask.values if T.EARLY_CUTOFF else np.ones(len(train_df), dtype=bool)
        X_tr_reg  = X_train[reg_mask]
        y_tr_reg  = y_train[reg_mask]

        if T.MODEL == "lgb":
            clf = T.build_lgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
        elif T.MODEL == "xgb":
            clf = T.build_xgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
        elif T.MODEL == "lr":
            clf = T.build_lr(X_tr_reg, y_tr_reg)
        elif T.MODEL == "mlp":
            clf = T.build_mlp(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
        elif T.MODEL in ("ensemble_avg", "ensemble_stack"):
            clf_lgb = T.build_lgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
            clf_xgb = T.build_xgb(X_tr_reg, y_tr_reg, X_tr_reg, y_tr_reg)
            clf_lr  = T.build_lr(X_tr_reg, y_tr_reg)
            clf = (clf_lgb, clf_xgb, clf_lr)
        else:
            sys.exit(f"Unknown MODEL: {T.MODEL!r}")

    # --- Predict -------------------------------------------------------------
    X_target = target_df[active_feats].values.astype(np.float32)

    if is_early and early_clf is not None:
        ef = [f for f in early_feats if f in target_df.columns]
        X_tgt_early = target_df[ef].values.astype(np.float32)
        prob = float(T.get_proba(early_clf, X_tgt_early)[0])
    elif T.MODEL in ("ensemble_avg", "ensemble_stack"):
        clf_lgb, clf_xgb, clf_lr = clf
        imp = SimpleImputer(strategy="median").fit(X_tr_reg)
        X_tgt_sc = imp.transform(X_target)
        p_lgb = T.get_proba(clf_lgb, X_target)[0]
        p_xgb = T.get_proba(clf_xgb, X_target)[0]
        p_lr  = T.get_proba(clf_lr,  X_tgt_sc)[0]
        prob  = float((p_lgb + p_xgb + p_lr) / 3.0)
    else:
        prob = float(T.get_proba(clf, X_target)[0])

    # --- Edge + Kelly sizing -------------------------------------------------
    mkt_prob_raw = target_df["market_implied_prob"].iloc[0]
    if pd.isna(mkt_prob_raw):
        print("WARNING: No market odds available. Cannot calculate edge or bet size.")
        edge       = float("nan")
        bet_frac   = 0.0
        bet_side   = "none"
    else:
        mp   = float(mkt_prob_raw)
        pp   = float(np.clip(prob, T.PROB_CAP[0], T.PROB_CAP[1]))
        thresh = T.CONFIDENCE_THRESHOLD + (0.02 * abs(mp - 0.5) if T.DYNAMIC_THRESHOLD else 0.0)

        edge_home = pp - mp
        edge_away = mp - pp

        if edge_home >= thresh and mp > 1e-6:
            dec_odds = 1.0 / mp
            bet_frac = T.kelly_stake(pp, dec_odds, is_warmup=is_early)
            bet_side = "home"
            edge     = edge_home
        elif edge_away >= thresh and (1.0 - mp) > 1e-6:
            dec_odds = 1.0 / (1.0 - mp)
            bet_frac = T.kelly_stake(1.0 - pp, dec_odds, is_warmup=is_early)
            bet_side = "away"
            edge     = edge_away
        else:
            bet_frac = 0.0
            bet_side = "none"
            edge     = max(edge_home, edge_away)

    print(f"\n{'='*50}")
    print(f"  Model prob (home win): {prob:.4f}")
    print(f"  Market implied prob:   {float(mkt_prob_raw) if not pd.isna(mkt_prob_raw) else 'N/A'}")
    print(f"  Edge:                  {edge:.4f}" if not (isinstance(edge, float) and np.isnan(edge)) else "  Edge: N/A")
    print(f"  Bet side:              {bet_side}")
    print(f"  Kelly fraction:        {bet_frac:.4f}")
    if bet_frac > 0:
        print(f"  >> BET {bet_side.upper()} at {bet_frac*100:.2f}% of bankroll")
    else:
        print("  >> NO BET")
    print(f"{'='*50}")

    # --- Write prediction to bets table --------------------------------------
    mkt_prob_val = float(mkt_prob_raw) if not pd.isna(mkt_prob_raw) else None
    edge_val     = float(edge) if not (isinstance(edge, float) and np.isnan(edge)) else None
    DB.update_bet_prediction(game_pk, prob, edge_val, bet_side, bet_frac, mkt_prob_val)
    print("Results written to bets table")
    return prob, edge, bet_side, bet_frac


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MLB game outcome and bet size.")
    # --prepare_only is a separate mode; --game_pk and --game_date are predict modes
    parser.add_argument("--prepare_only", action="store_true", help="Pre-train and cache model artifacts for a batch")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--game_pk",      type=str, help="MLB Stats API game_pk")
    group.add_argument("--game_date",    type=str, help="Game date YYYY-MM-DD (with --home_team / --away_team)")
    parser.add_argument("--home_team",   type=str, default=None)
    parser.add_argument("--away_team",   type=str, default=None)
    parser.add_argument("--output",      type=str, default=None, help="Output path for --prepare_only cache")
    parser.add_argument("--shared_data", type=str, default=None, help="Path to pre-trained model cache")
    args = parser.parse_args()

    if args.prepare_only:
        if not args.output:
            parser.error("--prepare_only requires --output PATH")
        if not args.game_date:
            parser.error("--prepare_only requires --game_date YYYY-MM-DD")
        prepare_shared_data(args.game_date, args.output)
        sys.exit(0)

    if not args.game_pk and not args.game_date:
        parser.error("one of --game_pk or --game_date is required")

    if args.game_date and (not args.home_team or not args.away_team):
        parser.error("--game_date requires --home_team and --away_team")

    game_info = find_target_game(args)

    shared = None
    if args.shared_data and os.path.exists(args.shared_data):
        print(f"Loading shared model cache from {args.shared_data}...")
        shared = joblib.load(args.shared_data)

    predict(game_info, shared_data=shared)
