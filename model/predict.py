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

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Import model config + feature engineering from train.py.
# train.py's main block is guarded by __name__ == "__main__", so importing
# it here only loads functions and constants — no training runs on import.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
import train as T

# Paths (relative to project root; script is called from project root)
HISTORICAL_CSV  = "data/master_mlb.csv"   # full historical data used for training
CURRENT_CSV     = "data/mlb_2026.csv"
GAMES_CSV       = "data/games.csv"
GAMES_CSV_COLS  = [
    "game_pk", "game_date", "home_team", "away_team",
    "predicted_prob", "edge", "bet_side", "bet_frac",
    "market_implied_prob", "result",
]


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
    df_feat["sp_fip_DIFF"]             = T._gcol(df_feat, "a_fip_lag1")             - T._gcol(df_feat, "h_fip_lag1")
    df_feat["sp_era_DIFF"]             = T._gcol(df_feat, "a_sp_era_lag1")          - T._gcol(df_feat, "h_sp_era_lag1")
    df_feat["sp_k9_DIFF"]              = T._gcol(df_feat, "h_sp_k9_lag1")           - T._gcol(df_feat, "a_sp_k9_lag1")
    df_feat["sp_bb9_DIFF"]             = T._gcol(df_feat, "a_sp_bb9_lag1")          - T._gcol(df_feat, "h_sp_bb9_lag1")
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
# Locate the target game in games.csv (and mlb_2026.csv for features)
# ---------------------------------------------------------------------------

def find_target_game(args) -> dict:
    """Return a dict with game_pk, game_date, home_team, away_team from games.csv."""
    if not os.path.exists(GAMES_CSV):
        sys.exit(f"ERROR: {GAMES_CSV} not found. Initialize the game row first.")

    games = pd.read_csv(GAMES_CSV, dtype=str)

    if args.game_pk:
        mask = games["game_pk"].astype(str) == str(args.game_pk)
        rows = games[mask]
        if rows.empty:
            sys.exit(f"ERROR: game_pk={args.game_pk} not found in {GAMES_CSV}")
        return rows.iloc[0].to_dict()

    # Lookup by date + teams
    mask = (
        (games["game_date"] == str(args.game_date)) &
        (games["home_team"].str.upper() == args.home_team.upper()) &
        (games["away_team"].str.upper() == args.away_team.upper())
    )
    rows = games[mask]
    if rows.empty:
        sys.exit(
            f"ERROR: no game on {args.game_date} with "
            f"home={args.home_team} away={args.away_team} in {GAMES_CSV}"
        )
    return rows.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def predict(game_info: dict):
    game_pk    = str(game_info["game_pk"])
    game_date  = str(game_info["game_date"])
    home_team  = str(game_info["home_team"])
    away_team  = str(game_info["away_team"])

    print(f"Predicting: {away_team} @ {home_team} on {game_date} (game_pk={game_pk})")

    # --- Load data -----------------------------------------------------------
    print("Loading historical data...")
    hist = pd.read_csv(HISTORICAL_CSV, low_memory=False)
    print(f"  {len(hist):,} historical rows")

    if not os.path.exists(CURRENT_CSV):
        sys.exit(f"ERROR: {CURRENT_CSV} not found. Run fetch_2026_data.py first.")
    curr = pd.read_csv(CURRENT_CSV, low_memory=False)
    print(f"  {len(curr):,} 2026 rows")

    # Locate target game in current season data
    curr["game_pk"] = curr["game_pk"].astype(str)
    target_mask = curr["game_pk"] == game_pk
    if not target_mask.any():
        # Fall back to date + teams
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

    # Include all 2026 rows up to + including the target game (for rolling context)
    curr["game_date"] = pd.to_datetime(curr["game_date"])
    target_date       = curr.loc[target_mask, "game_date"].iloc[0]
    curr_subset       = curr[curr["game_date"] <= target_date].copy()

    # Combine: historical (has home_win) + current season rows (may not have home_win)
    # Align columns — fill missing with NaN
    all_cols = list(dict.fromkeys(list(hist.columns) + list(curr_subset.columns)))
    hist_aligned = hist.reindex(columns=all_cols)
    curr_aligned = curr_subset.reindex(columns=all_cols)
    combined = pd.concat([hist_aligned, curr_aligned], ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    combined = combined.sort_values("game_date").reset_index(drop=True)

    # --- Feature engineering -------------------------------------------------
    print("Engineering features...")
    df_feat = engineer_features(combined)

    # --- Split: train = rows with known outcome; target = our game -----------
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

    # --- Determine if target game is early-season ----------------------------
    is_early = False
    if T.EARLY_CUTOFF is not None:
        hgp = float(target_df["home_games_played"].iloc[0]) if "home_games_played" in target_df else 999
        agp = float(target_df["away_games_played"].iloc[0]) if "away_games_played" in target_df else 999
        is_early = (hgp < T.EARLY_CUTOFF) or (agp < T.EARLY_CUTOFF)

    # --- Train model ---------------------------------------------------------
    print(f"Training {T.MODEL} on {len(train_df):,} rows...")
    X_train = train_df[active_feats].values.astype(np.float32)
    y_train = train_df["home_win"].values.astype(np.float32)

    early_mask = (
        (train_df["home_games_played"] < T.EARLY_CUTOFF) |
        (train_df["away_games_played"] < T.EARLY_CUTOFF)
    ) if T.EARLY_CUTOFF else pd.Series(False, index=train_df.index)

    # Early specialist
    early_clf = None
    if is_early and T.EARLY_CUTOFF and early_mask.sum() > 50 and early_feats:
        ef = [f for f in early_feats if f in train_df.columns]
        X_early = train_df.loc[early_mask, ef].values.astype(np.float32)
        y_early = y_train[early_mask.values]
        early_clf = T.build_early_lr(X_early, y_early)
        print("  Using early-season specialist model.")

    # Main model (trained on non-early rows when specialist is active)
    reg_mask  = ~early_mask.values if T.EARLY_CUTOFF else np.ones(len(train_df), dtype=bool)
    X_tr_reg  = X_train[reg_mask]
    y_tr_reg  = y_train[reg_mask]

    # Dummy val set (same as train — we're deploying, not validating)
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
        clf = (clf_lgb, clf_xgb, clf_lr)  # tuple — handled below
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

    # --- Write to games.csv --------------------------------------------------
    games = pd.read_csv(GAMES_CSV, dtype=str)
    for col in ["predicted_prob", "edge", "bet_side", "bet_frac", "market_implied_prob"]:
        if col not in games.columns:
            games[col] = ""

    mask = games["game_pk"].astype(str) == str(game_pk)
    if not mask.any():
        # Match by date + teams
        mask = (
            (games["game_date"] == game_date) &
            (games["home_team"].str.upper() == home_team.upper()) &
            (games["away_team"].str.upper() == away_team.upper())
        )

    games.loc[mask, "predicted_prob"]      = f"{prob:.4f}"
    games.loc[mask, "edge"]                = f"{edge:.4f}" if not (isinstance(edge, float) and np.isnan(edge)) else ""
    games.loc[mask, "bet_side"]            = bet_side
    games.loc[mask, "bet_frac"]            = f"{bet_frac:.4f}"
    games.loc[mask, "market_implied_prob"] = f"{float(mkt_prob_raw):.4f}" if not pd.isna(mkt_prob_raw) else ""

    games.to_csv(GAMES_CSV, index=False)
    print(f"Results written to {GAMES_CSV}")
    return prob, edge, bet_side, bet_frac


# ---------------------------------------------------------------------------
# Entry point
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

    game_info = find_target_game(args)
    predict(game_info)
