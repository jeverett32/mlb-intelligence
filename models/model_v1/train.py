"""
MLB betting model training script — autoresearch edition.
Supports: LightGBM, XGBoost, Logistic Regression, MLP, Ensemble
Walk-forward validation across multiple seasons.
Usage: uv run train.py
"""

import os
import math
import time
import random
import warnings
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
os.environ['PYTHONHASHSEED'] = '42'
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

import asb

from config import ACTIVE_SEASON, CURRENT_CSV

# ---------------------------------------------------------------------------
# Hyperparameters — the agent modifies these between experiments
# ---------------------------------------------------------------------------

CSV_INPUT_PATH = "data/master_mlb.csv"

# Walk-forward folds: (train_end, val_start, val_end).
# Trains on all data < train_end, validates on [val_start, val_end).
WALK_FORWARD_FOLDS = [
    ("2022-01-01", "2022-01-01", "2023-01-01"),   # train ≤2021, val=2022
    ("2023-01-01", "2023-01-01", "2024-01-01"),   # train ≤2022, val=2023
    ("2024-01-01", "2024-01-01", "2025-01-01"),   # train ≤2023, val=2024
    ("2025-01-01", "2025-01-01", "2026-01-01"),   # train ≤2024, val=2025
]

# ---------------------------------------------------------------------------
# MODEL SELECTOR — set to one of:
#   "lgb"            LightGBM classifier
#   "xgb"            XGBoost classifier
#   "lr"             Logistic Regression (with Platt calibration)
#   "mlp"            PyTorch MLP (imported lazily)
#   "ensemble_avg"   simple average of lgb + xgb + lr probs
#   "ensemble_stack" LR meta-learner stacked on lgb + xgb + lr
# ---------------------------------------------------------------------------
MODEL = "lr"

# Calibration: apply isotonic regression post-hoc to GBDT/MLP model probs.
# LR already calibrates internally via CalibratedClassifierCV.
CALIBRATE = True

# Early season specialist: separate LR model for games where either team has
# played fewer than EARLY_CUTOFF games. Set to None to disable.
EARLY_CUTOFF = 25   # games played threshold; None to disable specialist

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
BEST_W = 15           # rolling window (games) for win%, run-diff, etc.
MOMENTUM_W = 10       # short window for hot/cold streak signal
TRAIN_WINDOW_YEARS = None    # limit training to last N years; None = use all history
EARLY_SEASON_GAMES = 15  # used for early_season_flag feature

DROP_SUBSTR = {"game_pk", "odds_source", "starter_id"}

# 2024 park factors (FanGraphs; 100 = league average; stored as value/100)
PARK_FACTORS = {
    "COL": 113, "BOS": 106, "CIN": 105, "TEX": 105, "PHI": 104,
    "CHW": 103, "NYY": 103, "TOR": 103, "MIL": 102, "HOU": 102,
    "ATL": 101, "BAL": 101, "LAA": 100, "MIN": 100, "STL": 100,
    "DET": 100, "KCR": 100, "ARI": 99,  "WSN": 99,  "NYM": 99,
    "PIT": 98,  "LAD": 98,  "CLE": 98,  "CHC": 98,  "SEA": 97,
    "TBR": 97,  "SFG": 96,  "SDP": 96,  "MIA": 95,  "ATH": 94,
}

ASB_DATES = asb.ASB_DATES

# ---------------------------------------------------------------------------
# FEATURE COLUMNS — the agent prunes/adds column names here.
# Only names present in the DataFrame will be used (missing are skipped).
# See data_dictionary.md for directional definitions.
#
# Key design rule: DIFF features are positive = home advantage.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # --- Market ---
    "market_implied_prob",
    "open_home_implied",
    "line_move_delta",          # closing - opening (sharp signal)
    "sharp_move_flag",          # 1 if |line_move_delta| >= 0.03
    "total_move_delta",         # o/u line movement

    # --- Pitcher quality ---
    "sp_fip_DIFF",              # away FIP - home FIP (+ = home SP better)
    "sp_era_DIFF",              # away ERA - home ERA
    "sp_k9_DIFF",               # home K/9 - away K/9
    "sp_bb9_DIFF",              # away BB/9 - home BB/9
    "sp_whip_DIFF",             # away WHIP - home WHIP
    "rolling_k9_DIFF",          # home rolling K/9 - away rolling K/9
    "rolling_era_DIFF",         # away rolling ERA - home rolling ERA
    "rolling_whip_DIFF",        # away rolling WHIP - home rolling WHIP
    "rookie_DIFF",              # away is_rookie - home is_rookie (+ = home more experienced)
    "sp_rest_DIFF",             # home SP rest days - away SP rest days

    # --- Batting (FanGraphs season-to-date, lagged 1 game) ---
    "wrc_plus_DIFF",            # home wRC+ - away wRC+
    "woba_DIFF",                # home wOBA - away wOBA
    "avg_DIFF",                 # home AVG - away AVG
    "obp_DIFF",                 # home OBP - away OBP
    "slg_DIFF",                 # home SLG - away SLG
    "k_pct_DIFF",               # away K% - home K% (+ = home strikes out less)
    "bb_pct_DIFF",              # home BB% - away BB%
    "k_per_9_DIFF",             # away K/9 (team) - home K/9 (team)
    "bb_per_9_DIFF",            # home BB/9 (team) - away BB/9 (team)
    "hr_per_9_DIFF",            # away HR/9 - home HR/9
    "era_DIFF",                 # away team ERA - home team ERA
    "fip_DIFF",                 # away team FIP - home team FIP
    "owar_DIFF",                # home oWAR - away oWAR
    "war_DIFF",                 # home WAR - away WAR

    # --- Handedness ---
    "pitcher_handedness_diff",  # home_is_lefty - away_is_lefty
    "home_pitcher_is_lefty",
    "away_pitcher_is_lefty",

    # --- Rolling form (W = BEST_W games) ---
    "win_pct_W_DIFF",           # home - away W-game win%
    "run_diff_avg_W_DIFF",      # home - away W-game run differential avg
    "run_diff_std_W_DIFF",      # home - away W-game run differential std (consistency)
    "runs_scored_avg_W_DIFF",   # home - away W-game runs scored avg
    "runs_allowed_avg_W_DIFF",  # home - away W-game runs allowed avg

    # --- Season-to-date (expanding mean) ---
    "season_win_pct_DIFF",      # home - away season win%
    "season_run_diff_avg_DIFF", # home - away season run diff avg

    # --- Streak ---
    "streak_DIFF",              # home streak - away streak (+ = home on winning streak)

    # --- Weather / venue ---
    "temp_c",
    "wind_speed_kmh",
    "wind_dir_sin",             # circular encoding of wind direction
    "wind_dir_cos",
    "park_factor",
    "is_night_game",

    # --- Schedule context ---
    "home_rest_days",           # individual rest (not just diff)
    "away_rest_days",
    "rest_days_DIFF",           # home - away rest days
    "is_series_finale",
    "early_season_flag",        # binary: either team < EARLY_SEASON_GAMES games played

    # --- Calendar ---
    "month",                    # 1-12; season structure matters
    "days_since_asb",           # 0 pre/during ASB; days elapsed post-break
    "day_of_week",              # 0=Mon...6=Sun; travel/fatigue patterns
    "post_rule_change",         # 1 if season >= 2023 (pitch clock, shift ban, bigger bases)
    "post_dh_era",              # 1 if season >= 2022 (universal DH: NL pitchers no longer bat)
    "covid_era",                # 1 if season == 2020 (60-game season, no fans, unusual stats)

    # --- Interactions ---
    "sharp_x_fip",              # sharp_move_flag * sp_fip_DIFF
    "momentum_DIFF",            # (home 5g win% - home 15g win%) - same for away

    # --- Pythagorean / luck ---
    "luck_DIFF",                # (season_win_pct - pythagorean_win_pct) DIFF
    "pythagorean_DIFF",         # expected win% from run scoring efficiency DIFF
    "luck_x_momentum",          # lucky teams trending up → due for regression
    "park_x_pythagorean",       # park factor amplifies pythagorean edge
    "pythagorean_short_DIFF",   # short-window (MOMENTUM_W) pythagorean DIFF
]

# Early specialist feature set: used when EARLY_CUTOFF is set and either team
# has fewer than EARLY_CUTOFF games. Pitcher + market + context only — rolling
# team stats are unreliable this early.
EARLY_FEATURE_COLUMNS = [
    "sp_era_DIFF",
    "sp_whip_DIFF",
    "sp_k9_DIFF",
    "sp_bb9_DIFF",
    "rolling_era_DIFF",
    "rolling_whip_DIFF",
    "rolling_k9_DIFF",
    "market_implied_prob",
    "park_factor",
    "home_rest_days",
    "away_rest_days",
    "month",
    "is_night_game",
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "n_estimators":      2000,
    "learning_rate":     0.02,
    "num_leaves":        31,
    "max_depth":         -1,
    "min_child_samples": 30,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "n_jobs":            1,
    "verbose":           -1,
}

XGB_PARAMS = {
    "objective":             "binary:logistic",
    "eval_metric":           "logloss",
    "n_estimators":          2000,
    "learning_rate":         0.02,
    "max_depth":             4,
    "min_child_weight":      10,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "random_state":          42,
    "n_jobs":                1,
    "verbosity":             0,
    "early_stopping_rounds": 50,
}

LR_PARAMS = {
    "C":            0.04,
    "penalty":      "l1",
    "max_iter":     2000,
    "solver":       "saga",
    "random_state": 42,
    "n_jobs":       1,
}

# Betting
CONFIDENCE_THRESHOLD = 0.14   # minimum edge (model_prob - market_implied_prob)
DYNAMIC_THRESHOLD    = False  # if True: threshold = BASE + 0.02 * abs(mp - 0.5)
KELLY_FRACTION       = 0.25   # fractional Kelly multiplier
MAX_BET_FRAC         = 0.25   # hard cap on any single bet (fraction of unit stake)
PROB_CAP             = (0.20, 0.80)  # clip model probs before edge cap (reduces overconfidence)
WARMUP_KELLY_MULT    = 0.5    # multiply Kelly stake by this for early-season games

# MLP params (only used when MODEL="mlp")
MLP_TIME_BUDGET    = 300
MLP_HIDDEN_DIMS    = (256, 128, 64)
MLP_DROPOUT        = 0.2
MLP_LR             = 1e-3
MLP_WEIGHT_DECAY   = 3e-4
MLP_BATCH_SIZE     = 512
MLP_WARMUP_RATIO   = 0.05
MLP_WARMDOWN_RATIO = 0.4
MLP_FINAL_LR_FRAC  = 0.01

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def american_to_raw_implied(ml):
    try:
        if pd.isna(ml):
            return np.nan
    except TypeError:
        pass
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    return abs(ml) / (abs(ml) + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def american_to_decimal(ml):
    try:
        if pd.isna(ml):
            return np.nan
    except TypeError:
        pass
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    return 1.0 + 100.0 / abs(ml) if ml < 0 else 1.0 + ml / 100.0


def _devig_home(p_h_raw, p_a_raw):
    s     = p_h_raw + p_a_raw
    out_h = p_h_raw / s
    bad   = (s <= 0) | pd.isna(s)
    return out_h.where(~bad, np.nan)


def _gcol(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_market_columns(df):
    df    = df.copy()
    oh_ml = pd.to_numeric(df["open_home_ml"],  errors="coerce")
    oa_ml = pd.to_numeric(df["open_away_ml"],  errors="coerce")
    ch_ml = pd.to_numeric(df["close_home_ml"], errors="coerce")
    ca_ml = pd.to_numeric(df["close_away_ml"], errors="coerce")

    oh = oh_ml.map(american_to_raw_implied)
    oa = oa_ml.map(american_to_raw_implied)
    ch = ch_ml.map(american_to_raw_implied)
    ca = ca_ml.map(american_to_raw_implied)

    oh = oh.where(oh.notna(), ch)
    oa = oa.where(oa.notna(), ca)

    df["open_home_implied"]   = _devig_home(oh, oa)
    df["market_implied_prob"] = _devig_home(ch, ca)
    df["line_move_delta"]     = df["market_implied_prob"] - df["open_home_implied"]
    df["sharp_move_flag"]     = (df["line_move_delta"].abs() >= 0.03).astype(float)

    ot = pd.to_numeric(df["open_total"],  errors="coerce")
    ct = pd.to_numeric(df["close_total"], errors="coerce")
    df["total_move_delta"] = (ct - ot).fillna(0.0)
    df["dec_close_home"]   = ch_ml.map(american_to_decimal)
    return df


def add_schedule_context(df):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # Park factor
    df["park_factor"] = df["home_team"].map(
        lambda t: float(PARK_FACTORS.get(t, 100)) / 100.0)

    # Wind direction — circular encoding (raw degrees are not linear)
    deg = np.deg2rad(pd.to_numeric(df["wind_dir_deg"], errors="coerce").fillna(0))
    df["wind_dir_sin"] = np.sin(deg)
    df["wind_dir_cos"] = np.cos(deg)
    df["wind_speed_kmh"] = pd.to_numeric(df["wind_speed_kmh"], errors="coerce").fillna(0.0)

    # Month and day-of-week
    df["month"] = df["game_date"].dt.month
    df["day_of_week"] = df["game_date"].dt.dayofweek  # 0=Mon, 6=Sun

    # Days since All-Star break (0 before/during ASB)
    def _days_asb(row):
        yr = row["season"]
        if yr not in ASB_DATES:
            return 0
        return max((row["game_date"] - pd.to_datetime(ASB_DATES[yr])).days, 0)
    df["days_since_asb"] = df.apply(_days_asb, axis=1)

    # Rest days (per team per season)
    rows = []
    for _, r in df.iterrows():
        for side, team in [("home", r["home_team"]), ("away", r["away_team"])]:
            rows.append({"game_id": r["game_id"], "game_date": r["game_date"],
                         "season": r["season"], "team": team, "side": side})
    g = pd.DataFrame(rows).sort_values(["team", "season", "game_date"])
    g["prev_date"]  = g.groupby(["team", "season"], sort=False)["game_date"].shift(1)
    g["rest_days"]  = (g["game_date"] - g["prev_date"]).dt.days.fillna(4).clip(upper=7)

    rh = g[g["side"] == "home"][["game_id", "rest_days"]].rename(columns={"rest_days": "home_rest_days"})
    ra = g[g["side"] == "away"][["game_id", "rest_days"]].rename(columns={"rest_days": "away_rest_days"})
    df = df.merge(rh, on="game_id", how="left").merge(ra, on="game_id", how="left")
    df["rest_days_DIFF"] = df["home_rest_days"] - df["away_rest_days"]

    # Series finale flag
    tmp = df.sort_values(["home_team", "season", "game_date"]).copy()
    nxt = tmp.groupby(["home_team", "season"])["away_team"].shift(-1)
    tmp["is_series_finale"] = ((nxt != tmp["away_team"]) | nxt.isna()).astype(float)
    df  = df.drop(columns=["is_series_finale"], errors="ignore")
    df  = df.merge(tmp[["game_id", "is_series_finale"]], on="game_id", how="left")

    return df


def compute_streak(win_series):
    """Current win/loss streak: positive = winning, negative = losing."""
    streak = [0]
    for i in range(1, len(win_series)):
        prev, last = win_series.iloc[i - 1], streak[-1]
        streak.append(
            (last + 1 if last >= 0 else 1) if prev == 1
            else (last - 1 if last <= 0 else -1)
        )
    return streak


def build_long_format(df):
    """Build per-team game log (long format) — does NOT lag yet."""
    base_ids  = ["game_id", "game_date", "season", "home_team", "away_team",
                 "home_win", "home_score", "away_score"]

    home_cols = [
        c for c in df.columns
        if c.startswith("home_")
        and c not in {"home_team", "home_win", "home_score", "home_starter_id",
                      "home_pitcher_is_lefty"}
        and not any(s in c for s in DROP_SUBSTR)
    ]
    away_cols = [
        c for c in df.columns
        if c.startswith("away_")
        and c not in {"away_team", "away_score", "away_starter_id",
                      "away_pitcher_is_lefty"}
        and not any(s in c for s in DROP_SUBSTR)
    ]

    h = df[base_ids + ["home_pitcher_is_lefty"] + home_cols].copy()
    h = h.rename(columns={c: c.replace("home_", "", 1) for c in home_cols})
    h["team"], h["opp"], h["is_home"] = df["home_team"], df["away_team"], 1.0
    h["win"]          = df["home_win"].astype(float)
    h["runs_for"]     = df["home_score"].astype(float)
    h["runs_against"] = df["away_score"].astype(float)

    a = df[base_ids + ["away_pitcher_is_lefty"] + away_cols].copy()
    a = a.rename(columns={c: c.replace("away_", "", 1) for c in away_cols})
    a["team"], a["opp"], a["is_home"] = df["away_team"], df["home_team"], 0.0
    a["win"]          = 1.0 - df["home_win"].astype(float)
    a["runs_for"]     = df["away_score"].astype(float)
    a["runs_against"] = df["home_score"].astype(float)

    long = pd.concat([h, a], ignore_index=True)
    long = long.sort_values(["team", "season", "game_date", "game_id"])

    # Lag all stat columns by 1 to prevent lookahead
    exclude   = {"game_id", "game_date", "season", "home_team", "away_team",
                 "home_win", "home_score", "away_score", "team", "opp",
                 "is_home", "win", "runs_for", "runs_against"}
    stat_cols = [c for c in long.columns if c not in exclude]
    for c in stat_cols:
        long[c] = pd.to_numeric(long[c], errors="coerce")
    for c in stat_cols:
        long[c + "_lag1"] = long.groupby(["team", "season"], sort=False)[c].shift(1)
    long = long.drop(columns=stat_cols, errors="ignore")

    return long


def add_rolling_features(long_games, W):
    """
    Compute rolling (W-game) and expanding (season-to-date) stats.
    All based on shift(1) — the current game's result is excluded.
    """
    g = long_games.sort_values(["team", "season", "game_date", "game_id"]).copy()
    g["runs_for"]     = pd.to_numeric(g.get("runs_for",     pd.Series(dtype=float)), errors="coerce")
    g["runs_against"] = pd.to_numeric(g.get("runs_against", pd.Series(dtype=float)), errors="coerce")
    g["win"]          = pd.to_numeric(g.get("win",          pd.Series(dtype=float)), errors="coerce")

    def rmean(ser, window):
        return ser.shift(1).rolling(window, min_periods=max(1, window // 3)).mean()
    def rstd(ser, window):
        return ser.shift(1).rolling(window, min_periods=max(2, window // 2)).std()
    def expnd(ser):
        return ser.shift(1).expanding(min_periods=1).mean()

    grp = g.groupby(["team", "season"], sort=False)

    g[f"win_pct_{W}"]           = grp["win"].transform(lambda s: rmean(s, W))
    g["run_margin"]              = g["runs_for"] - g["runs_against"]
    g[f"run_diff_avg_{W}"]      = grp["run_margin"].transform(lambda s: rmean(s, W))
    g[f"run_diff_std_{W}"]      = grp["run_margin"].transform(lambda s: rstd(s, W))
    g[f"runs_scored_avg_{W}"]   = grp["runs_for"].transform(lambda s: rmean(s, W))
    g[f"runs_allowed_avg_{W}"]  = grp["runs_against"].transform(lambda s: rmean(s, W))
    g["season_win_pct"]          = grp["win"].transform(expnd)
    g["season_run_diff_avg"]     = grp["run_margin"].transform(expnd)
    g["games_played"]            = grp["win"].transform(lambda s: s.shift(1).expanding().count())
    g["streak"]                  = g.groupby(["team", "season"])["win"].transform(
        lambda s: pd.Series(compute_streak(s), index=s.index))

    roll_cols = [f"win_pct_{W}", f"run_diff_avg_{W}", f"run_diff_std_{W}",
                 f"runs_scored_avg_{W}", f"runs_allowed_avg_{W}",
                 "season_win_pct", "season_run_diff_avg", "games_played", "streak"]

    mask_home = g["is_home"] == 1
    mask_away = g["is_home"] == 0
    hm = g[mask_home][["game_id"] + roll_cols].rename(columns={c: f"h_{c}" for c in roll_cols})
    aw = g[mask_away][["game_id"] + roll_cols].rename(columns={c: f"a_{c}" for c in roll_cols})
    return hm.merge(aw, on="game_id")


def build_pitcher_features(df):
    """
    Pitcher DIFF features + rookie flag + SP rest.
    Uses pre-lagged columns already present in df (sp_* columns are game-day values
    from the CSV, not rolling — they reflect the starting pitcher announced pre-game,
    so no additional lag is needed for the pitcher identity features).
    """
    out = df[["game_id", "game_date"]].copy()

    # WHIP diff (away - home → positive = home SP better)
    for col in ["sp_whip"]:
        h, a = f"home_{col}", f"away_{col}"
        if h in df.columns and a in df.columns:
            out[f"{col}_DIFF"] = pd.to_numeric(df[a], errors="coerce") - \
                                  pd.to_numeric(df[h], errors="coerce")

    # Rolling pitcher stats (ERA, WHIP, K9)
    for col, fb in [("rolling_era", "sp_era"), ("rolling_whip", "sp_whip")]:
        h_r  = df.get(f"home_{col}",  pd.Series(np.nan, index=df.index))
        a_r  = df.get(f"away_{col}",  pd.Series(np.nan, index=df.index))
        h_fb = df.get(f"home_{fb}",   pd.Series(np.nan, index=df.index))
        a_fb = df.get(f"away_{fb}",   pd.Series(np.nan, index=df.index))
        out[f"{col}_DIFF"] = (pd.to_numeric(a_r, errors="coerce").fillna(
                               pd.to_numeric(a_fb, errors="coerce")) -
                              pd.to_numeric(h_r, errors="coerce").fillna(
                               pd.to_numeric(h_fb, errors="coerce")))

    if "home_rolling_k9" in df.columns:
        h_k9 = pd.to_numeric(df["home_rolling_k9"], errors="coerce").fillna(
               pd.to_numeric(df.get("home_sp_k9", np.nan), errors="coerce"))
        a_k9 = pd.to_numeric(df["away_rolling_k9"], errors="coerce").fillna(
               pd.to_numeric(df.get("away_sp_k9", np.nan), errors="coerce"))
        out["rolling_k9_DIFF"] = h_k9 - a_k9

    # SP rest (actual per-starter rest, clipped to 14 days)
    out2 = df[["game_id", "game_date"]].copy()
    for side in ["home", "away"]:
        id_col = f"{side}_starter_id"
        if id_col in df.columns:
            temp = df[["game_date", id_col]].copy().sort_values([id_col, "game_date"])
            rest = temp.groupby(id_col)["game_date"].diff().dt.days.clip(upper=14)
            rest = rest.reindex(df.index)
            out2[f"{side}_starter_rest"] = rest.values
    if "home_starter_rest" in out2.columns and "away_starter_rest" in out2.columns:
        out["sp_rest_DIFF"] = (out2["home_starter_rest"] - out2["away_starter_rest"]).fillna(0)

    # Rookie flag: pitcher with <= 5 prior career starts in this dataset.
    if {"home_starter_id", "away_starter_id"} <= set(df.columns):
        row_id = pd.Series(np.arange(len(df)), index=df.index)
        starts = []
        for side in ["home", "away"]:
            id_col = f"{side}_starter_id"
            side_starts = df[["game_date", id_col]].copy()
            side_starts["_row_id"] = row_id.loc[side_starts.index].values
            side_starts["side"] = side
            side_starts = side_starts.rename(columns={id_col: "p_id"})
            starts.append(side_starts)

        all_starts = (
            pd.concat(starts, ignore_index=True)
            .dropna(subset=["p_id"])
            .sort_values(["p_id", "game_date", "_row_id"])
        )
        all_starts["career_starts"] = all_starts.groupby("p_id").cumcount()

        for side in ["home", "away"]:
            side_starts = all_starts[all_starts["side"] == side].set_index("_row_id")
            out[f"{side}_career_starts"] = row_id.map(side_starts["career_starts"])
            out[f"{side}_is_rookie"] = np.where(
                out[f"{side}_career_starts"].notna(),
                (out[f"{side}_career_starts"] <= 5).astype(float),
                np.nan,
            )
    if "home_is_rookie" in out.columns and "away_is_rookie" in out.columns:
        out["rookie_DIFF"] = out["away_is_rookie"] - out["home_is_rookie"]

    return out


def _sanitize_training_df(df: pd.DataFrame) -> pd.DataFrame:
    if "home_win" in df.columns:
        df["home_win"] = df["home_win"].map(
            {"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0, True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}
        ).where(df["home_win"].notna())
    return df


def load_training_frame(*, allow_csv_fallback: bool = True):
    try:
        import db
        hist = db.get_games_df()
        hist = hist[hist["season"].notna() & (hist["season"].astype(float) < ACTIVE_SEASON)]
        curr = db.get_games_df(season=ACTIVE_SEASON)
        if hist.empty or curr.empty:
            raise ValueError("DB returned empty historical or current rows")
        all_cols = list(dict.fromkeys(list(hist.columns) + list(curr.columns)))
        df = pd.concat(
            [hist.reindex(columns=all_cols), curr.reindex(columns=all_cols)],
            ignore_index=True,
        )
        print("Loading DB games...")
        print(f"  {len(hist):,} historical rows + {len(curr):,} current rows loaded")
        return _sanitize_training_df(df)
    except Exception as e:
        if not allow_csv_fallback:
            raise
        print(f"Loading CSV... DB unavailable ({e})")
        hist = pd.read_csv(CSV_INPUT_PATH, low_memory=False)
        if os.path.exists(CURRENT_CSV):
            curr = pd.read_csv(CURRENT_CSV, low_memory=False)
            all_cols = list(dict.fromkeys(list(hist.columns) + list(curr.columns)))
            df = pd.concat(
                [hist.reindex(columns=all_cols), curr.reindex(columns=all_cols)],
                ignore_index=True,
            )
            print(f"  {len(hist):,} historical rows + {len(curr):,} current rows loaded")
            return _sanitize_training_df(df)
        print(f"  {len(hist):,} rows loaded")
        return _sanitize_training_df(hist)


def load_and_engineer_features(*, allow_csv_fallback: bool = True):
    df = (
        load_training_frame()
        if allow_csv_fallback
        else load_training_frame(allow_csv_fallback=False)
    )

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"]    = df["game_date"].dt.year

    # ── Lag FanGraphs season-to-date team stats by 1 game ────────────────────
    # These are cumulative season stats scraped as of game day — the current
    # game's result is baked in, so we must shift within each team-season.
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
    print(f"  FanGraphs stats lagged. NaN rate: {df[FG_TEAM_COLS].isna().mean().mean():.1%}")

    print("Adding market columns...")
    df = add_market_columns(df)

    print("Adding schedule context...")
    df = add_schedule_context(df)

    print("Building long-format game log + lagging team stats...")
    long_games = build_long_format(df)

    print(f"Adding rolling form (W={BEST_W}) + season-to-date + streaks...")
    roll_main  = add_rolling_features(long_games, BEST_W)
    roll_short = add_rolling_features(long_games, MOMENTUM_W)

    df_feat = df.merge(roll_main,  on="game_id", how="left")
    df_feat = df_feat.merge(
        roll_short[["game_id", f"h_win_pct_{MOMENTUM_W}", f"a_win_pct_{MOMENTUM_W}",
                    f"h_runs_scored_avg_{MOMENTUM_W}", f"h_runs_allowed_avg_{MOMENTUM_W}",
                    f"a_runs_scored_avg_{MOMENTUM_W}", f"a_runs_allowed_avg_{MOMENTUM_W}"]],
        on="game_id", how="left")

    print("Building pitcher features...")
    ptch = build_pitcher_features(df)
    df_feat = df_feat.merge(ptch, on=["game_id", "game_date"], how="left")

    # ── DIFF features ─────────────────────────────────────────────────────────
    W = BEST_W

    # Rolling form DIFFs
    df_feat["win_pct_W_DIFF"]          = _gcol(df_feat, f"h_win_pct_{W}")              - _gcol(df_feat, f"a_win_pct_{W}")
    df_feat["run_diff_avg_W_DIFF"]     = _gcol(df_feat, f"h_run_diff_avg_{W}")         - _gcol(df_feat, f"a_run_diff_avg_{W}")
    df_feat["run_diff_std_W_DIFF"]     = _gcol(df_feat, f"h_run_diff_std_{W}")         - _gcol(df_feat, f"a_run_diff_std_{W}")
    df_feat["runs_scored_avg_W_DIFF"]  = _gcol(df_feat, f"h_runs_scored_avg_{W}")      - _gcol(df_feat, f"a_runs_scored_avg_{W}")
    df_feat["runs_allowed_avg_W_DIFF"] = _gcol(df_feat, f"h_runs_allowed_avg_{W}")     - _gcol(df_feat, f"a_runs_allowed_avg_{W}")
    df_feat["season_win_pct_DIFF"]     = _gcol(df_feat, "h_season_win_pct")            - _gcol(df_feat, "a_season_win_pct")
    df_feat["season_run_diff_avg_DIFF"]= _gcol(df_feat, "h_season_run_diff_avg")       - _gcol(df_feat, "a_season_run_diff_avg")
    df_feat["streak_DIFF"]             = _gcol(df_feat, "h_streak")                    - _gcol(df_feat, "a_streak")

    # Momentum
    h_momentum = _gcol(df_feat, f"h_win_pct_{MOMENTUM_W}") - _gcol(df_feat, f"h_win_pct_{W}")
    a_momentum = _gcol(df_feat, f"a_win_pct_{MOMENTUM_W}") - _gcol(df_feat, f"a_win_pct_{W}")
    df_feat["momentum_DIFF"]           = h_momentum - a_momentum

    # Pitcher DIFFs. Historical rows use lagged long-format columns; live rows
    # use announced-starter columns from fetch_data.py.
    h_fip = _gcol(df_feat, "h_fip_lag1").combine_first(_gcol(df_feat, "home_sp_fip")).combine_first(_gcol(df_feat, "home_starter_fip"))
    a_fip = _gcol(df_feat, "a_fip_lag1").combine_first(_gcol(df_feat, "away_sp_fip")).combine_first(_gcol(df_feat, "away_starter_fip"))
    h_era = _gcol(df_feat, "h_sp_era_lag1").combine_first(_gcol(df_feat, "home_sp_era")).combine_first(_gcol(df_feat, "home_starter_era"))
    a_era = _gcol(df_feat, "a_sp_era_lag1").combine_first(_gcol(df_feat, "away_sp_era")).combine_first(_gcol(df_feat, "away_starter_era"))
    h_k9 = _gcol(df_feat, "h_sp_k9_lag1").combine_first(_gcol(df_feat, "home_sp_k9")).combine_first(_gcol(df_feat, "home_starter_k9"))
    a_k9 = _gcol(df_feat, "a_sp_k9_lag1").combine_first(_gcol(df_feat, "away_sp_k9")).combine_first(_gcol(df_feat, "away_starter_k9"))
    h_bb9 = _gcol(df_feat, "h_sp_bb9_lag1").combine_first(_gcol(df_feat, "home_sp_bb9")).combine_first(_gcol(df_feat, "home_starter_bb9"))
    a_bb9 = _gcol(df_feat, "a_sp_bb9_lag1").combine_first(_gcol(df_feat, "away_sp_bb9")).combine_first(_gcol(df_feat, "away_starter_bb9"))
    df_feat["sp_fip_DIFF"]             = pd.to_numeric(a_fip, errors="coerce") - pd.to_numeric(h_fip, errors="coerce")
    df_feat["sp_era_DIFF"]             = pd.to_numeric(a_era, errors="coerce") - pd.to_numeric(h_era, errors="coerce")
    df_feat["sp_k9_DIFF"]              = pd.to_numeric(h_k9, errors="coerce")  - pd.to_numeric(a_k9, errors="coerce")
    df_feat["sp_bb9_DIFF"]             = pd.to_numeric(a_bb9, errors="coerce") - pd.to_numeric(h_bb9, errors="coerce")

    # FanGraphs season-to-date batting DIFFs (already lagged above)
    df_feat["wrc_plus_DIFF"]           = _gcol(df_feat, "home_wrc_plus")               - _gcol(df_feat, "away_wrc_plus")
    df_feat["woba_DIFF"]               = _gcol(df_feat, "home_woba")                   - _gcol(df_feat, "away_woba")
    df_feat["avg_DIFF"]                = _gcol(df_feat, "home_avg")                    - _gcol(df_feat, "away_avg")
    df_feat["obp_DIFF"]                = _gcol(df_feat, "home_obp")                    - _gcol(df_feat, "away_obp")
    df_feat["slg_DIFF"]                = _gcol(df_feat, "home_slg")                    - _gcol(df_feat, "away_slg")
    df_feat["k_pct_DIFF"]              = _gcol(df_feat, "away_k_pct")                  - _gcol(df_feat, "home_k_pct")
    df_feat["bb_pct_DIFF"]             = _gcol(df_feat, "home_bb_pct")                 - _gcol(df_feat, "away_bb_pct")
    df_feat["k_per_9_DIFF"]            = _gcol(df_feat, "away_k_per_9")                - _gcol(df_feat, "home_k_per_9")
    df_feat["bb_per_9_DIFF"]           = _gcol(df_feat, "home_bb_per_9")               - _gcol(df_feat, "away_bb_per_9")
    df_feat["hr_per_9_DIFF"]           = _gcol(df_feat, "away_hr_per_9")               - _gcol(df_feat, "home_hr_per_9")
    df_feat["era_DIFF"]                = _gcol(df_feat, "away_era")                    - _gcol(df_feat, "home_era")
    df_feat["fip_DIFF"]                = _gcol(df_feat, "away_fip")                    - _gcol(df_feat, "home_fip")
    df_feat["owar_DIFF"]               = _gcol(df_feat, "home_owar")                   - _gcol(df_feat, "away_owar")
    df_feat["war_DIFF"]                = _gcol(df_feat, "home_war")                    - _gcol(df_feat, "away_war")

    # Handedness
    df_feat["pitcher_handedness_diff"] = _gcol(df_feat, "home_pitcher_is_lefty")       - _gcol(df_feat, "away_pitcher_is_lefty")

    # Interaction
    df_feat["sharp_x_fip"]             = df_feat["sharp_move_flag"] * df_feat["sp_fip_DIFF"]

    # Early season flag (game count-based)
    df_feat = df_feat.sort_values("game_date").reset_index(drop=True)
    df_feat["hg"] = _gcol(df_feat, "h_games_played")
    df_feat["ag"] = _gcol(df_feat, "a_games_played")
    df_feat["early_season_flag"] = (
        df_feat[["hg", "ag"]].min(axis=1) < EARLY_SEASON_GAMES).astype(float)

    # Games played (for early specialist routing)
    df_feat["home_games_played"] = df_feat["hg"]
    df_feat["away_games_played"] = df_feat["ag"]

    # Agent-modifiable extra features
    df_feat = engineer_new_features(df_feat)

    return df_feat


def engineer_new_features(df_feat):
    """
    Agent-modifiable zone. Add new computed features here.
    All upstream lag/rolling/FanGraphs features are already available.

    Every new feature must:
      1. Pass feature_engineering.md Phase 1 screening before a full training run.
      2. Be documented in data_dictionary.md Section 8.
      3. Be added to FEATURE_COLUMNS to take effect.

    Seeded candidates (not yet in FEATURE_COLUMNS — add after screening):
      fip_x_line   : sp_fip_DIFF * line_move_delta
      form_x_fip   : run_diff_avg_W_DIFF * sp_fip_DIFF
      temp_x_wrc   : temp_c * wrc_plus_DIFF
      woba_x_sharp : woba_DIFF * sharp_move_flag
      early_x_fip  : early_season_flag * sp_fip_DIFF
    """
    if "sp_fip_DIFF" in df_feat.columns and "line_move_delta" in df_feat.columns:
        df_feat["fip_x_line"] = df_feat["sp_fip_DIFF"] * df_feat["line_move_delta"]

    if "run_diff_avg_W_DIFF" in df_feat.columns and "sp_fip_DIFF" in df_feat.columns:
        df_feat["form_x_fip"] = df_feat["run_diff_avg_W_DIFF"] * df_feat["sp_fip_DIFF"]

    if "temp_c" in df_feat.columns and "wrc_plus_DIFF" in df_feat.columns:
        df_feat["temp_x_wrc"] = pd.to_numeric(df_feat["temp_c"], errors="coerce") * df_feat["wrc_plus_DIFF"]

    if "woba_DIFF" in df_feat.columns and "sharp_move_flag" in df_feat.columns:
        df_feat["woba_x_sharp"] = df_feat["woba_DIFF"] * df_feat["sharp_move_flag"]

    if "early_season_flag" in df_feat.columns and "sp_fip_DIFF" in df_feat.columns:
        df_feat["early_x_fip"] = df_feat["early_season_flag"] * df_feat["sp_fip_DIFF"]

    # Pythagorean luck: actual season win% minus Pythagorean expectation (RS^2/(RS^2+RA^2))
    # Positive = home team winning more than run differential suggests (luck regression coming)
    h_rs = _gcol(df_feat, f"h_runs_scored_avg_{BEST_W}").replace(0, np.nan)
    h_ra = _gcol(df_feat, f"h_runs_allowed_avg_{BEST_W}").replace(0, np.nan)
    a_rs = _gcol(df_feat, f"a_runs_scored_avg_{BEST_W}").replace(0, np.nan)
    a_ra = _gcol(df_feat, f"a_runs_allowed_avg_{BEST_W}").replace(0, np.nan)
    if not (h_rs.isna().all() or h_ra.isna().all()):
        h_pyth = h_rs ** 2 / (h_rs ** 2 + h_ra ** 2)
        a_pyth = a_rs ** 2 / (a_rs ** 2 + a_ra ** 2)
        pyth_diff = h_pyth - a_pyth
        h_swp = _gcol(df_feat, "h_season_win_pct")
        a_swp = _gcol(df_feat, "a_season_win_pct")
        df_feat["luck_DIFF"] = (h_swp - a_swp) - pyth_diff
        df_feat["pythagorean_DIFF"] = pyth_diff

    # Luck x momentum: lucky teams trending up are especially due for regression
    if "luck_DIFF" in df_feat.columns and "momentum_DIFF" in df_feat.columns:
        df_feat["luck_x_momentum"] = df_feat["luck_DIFF"] * df_feat["momentum_DIFF"]

    # Park x pythagorean: high-scoring parks amplify run-efficiency edge
    if "pythagorean_DIFF" in df_feat.columns and "park_factor" in df_feat.columns:
        df_feat["park_x_pythagorean"] = df_feat["park_factor"] * df_feat["pythagorean_DIFF"]

    # Short-window pythagorean
    h_rs_s = _gcol(df_feat, f"h_runs_scored_avg_{MOMENTUM_W}").replace(0, np.nan)
    h_ra_s = _gcol(df_feat, f"h_runs_allowed_avg_{MOMENTUM_W}").replace(0, np.nan)
    a_rs_s = _gcol(df_feat, f"a_runs_scored_avg_{MOMENTUM_W}").replace(0, np.nan)
    a_ra_s = _gcol(df_feat, f"a_runs_allowed_avg_{MOMENTUM_W}").replace(0, np.nan)
    if not (h_rs_s.isna().all() or h_ra_s.isna().all()):
        h_pyth_s = h_rs_s ** 2 / (h_rs_s ** 2 + h_ra_s ** 2)
        a_pyth_s = a_rs_s ** 2 / (a_rs_s ** 2 + a_ra_s ** 2)
        df_feat["pythagorean_short_DIFF"] = h_pyth_s - a_pyth_s

    # Post rule-change flag: 2023 changes (pitch clock, shift ban, bigger bases) may alter patterns
    if "game_date" in df_feat.columns:
        year_series = pd.to_datetime(df_feat["game_date"]).dt.year
        df_feat["post_rule_change"] = (year_series >= 2023).astype(int)
        df_feat["post_dh_era"] = (year_series >= 2022).astype(int)
        df_feat["covid_era"] = (year_series == 2020).astype(int)
    elif "season" in df_feat.columns:
        df_feat["post_rule_change"] = (df_feat["season"] >= 2023).astype(int)
        df_feat["post_dh_era"] = (df_feat["season"] >= 2022).astype(int)
        df_feat["covid_era"] = (df_feat["season"] == 2020).astype(int)

    return df_feat


# ---------------------------------------------------------------------------
# Imputer-wrapped model builders
# GBDTs handle NaN natively; LR/MLP use SimpleImputer so no rows are dropped.
# ---------------------------------------------------------------------------

def build_lgb(X_train, y_train, X_val, y_val):
    params = dict(LGB_PARAMS)
    n_est  = params.pop("n_estimators")
    clf    = lgb.LGBMClassifier(n_estimators=n_est, **params)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return clf


def build_xgb(X_train, y_train, X_val, y_val):
    params = dict(XGB_PARAMS)
    params.pop("early_stopping_rounds", None)
    clf = xgb.XGBClassifier(early_stopping_rounds=50, **params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def build_lr(X_train, y_train):
    # Pipeline: impute → scale → calibrated LR
    base = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", LogisticRegression(**LR_PARAMS)),
    ])
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)
    return clf


def build_early_lr(X_train, y_train):
    """Simpler LR for early-season specialist — fewer features, no calibration wrapper."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs",
                                    random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def build_mlp(X_train, y_train, X_val, y_val):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Impute before passing to MLP
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_val   = imp.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class BettingMLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            layers, in_dim = [], n_in
            for h in MLP_HIDDEN_DIMS:
                layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(MLP_DROPOUT)]
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = BettingMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    def get_lr_mul(progress):
        if progress < MLP_WARMUP_RATIO:
            return progress / MLP_WARMUP_RATIO if MLP_WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - MLP_WARMDOWN_RATIO:
            return 1.0
        cool = (1.0 - progress) / MLP_WARMDOWN_RATIO
        return cool + (1 - cool) * MLP_FINAL_LR_FRAC

    total_time, step, smooth_loss = 0.0, 0, 0.0
    while True:
        te = time.time()
        model.train()
        perm = torch.randperm(len(X_t), device=device)
        el, nb = 0.0, 0
        for i in range(0, len(X_t), MLP_BATCH_SIZE):
            idx    = perm[i:i + MLP_BATCH_SIZE]
            logits = model(X_t[idx])
            loss   = F.binary_cross_entropy_with_logits(logits, y_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            el += loss.item(); nb += 1
        if step > 2:
            total_time += time.time() - te
        progress    = min(total_time / MLP_TIME_BUDGET, 1.0)
        lrm         = get_lr_mul(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        smooth_loss = 0.9 * smooth_loss + 0.1 * (el / max(nb, 1))
        debiased    = smooth_loss / (1 - 0.9 ** (step + 1))
        print(f"\r  mlp epoch {step:04d} ({100*progress:.1f}%) loss={debiased:.4f}  ", end="", flush=True)
        step += 1
        if step > 2 and total_time >= MLP_TIME_BUDGET:
            break
    print()

    class MLPWrapper:
        def __init__(self, m, dev, imp_):
            self.m = m; self.dev = dev; self.imp = imp_
        def predict_proba(self, X):
            X = self.imp.transform(X)
            self.m.eval()
            Xt = torch.tensor(X, dtype=torch.float32, device=self.dev)
            with torch.no_grad():
                p = torch.sigmoid(self.m(Xt)).cpu().numpy()
            return np.column_stack([1 - p, p])
    return MLPWrapper(model, device, imp)


def get_proba(clf, X):
    return clf.predict_proba(X)[:, 1]


def calibrate_probs(probs_train, y_train, probs_val):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs_train, y_train)
    return ir.transform(probs_val)


def extract_feature_importance(clf, feature_names) -> list[dict] | None:
    fi = None
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        fi = np.abs(clf.coef_[0])
    elif hasattr(clf, "calibrated_classifiers_"):
        try:
            coefs = [c.estimator[-1].coef_[0] for c in clf.calibrated_classifiers_]
            fi = np.abs(np.mean(coefs, axis=0))
        except Exception:
            pass
    if fi is None or len(fi) != len(feature_names):
        return None
    pairs = sorted(zip(feature_names, fi), key=lambda x: -x[1])
    return [{"feature": name, "importance": round(float(imp), 6)} for name, imp in pairs]


def print_feature_importance(clf, feature_names, top_n=20):
    pairs = extract_feature_importance(clf, feature_names)
    if pairs is None:
        return
    print(f"\n  Top-{top_n} feature importances (last fold):")
    scale = pairs[0]["importance"] + 1e-9
    for p in pairs[:top_n]:
        bar = "█" * int(40 * p["importance"] / scale)
        print(f"    {p['feature']:<32s} {p['importance']:8.4f}  {bar}")
    print(f"\n  Bottom-5 (pruning candidates):")
    for p in pairs[-5:]:
        print(f"    {p['feature']:<32s} {p['importance']:8.4f}")

# ---------------------------------------------------------------------------
# Betting evaluation — DO NOT MODIFY THIS FUNCTION
# ---------------------------------------------------------------------------

def kelly_stake(prob, decimal_odds, fraction=KELLY_FRACTION,
                max_frac=MAX_BET_FRAC, is_warmup=False):
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (b * prob - (1.0 - prob)) / b
    stake = kelly * fraction
    if is_warmup:
        stake *= WARMUP_KELLY_MULT
    return float(max(0.0, min(stake, max_frac)))


def _ml_to_dec(ml):
    """American moneyline -> decimal odds. NaN-safe."""
    try:
        a = float(ml)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(a) or a == 0:
        return float("nan")
    return 1.0 + (100.0 / abs(a) if a < 0 else a / 100.0)


def evaluate(probs, y, mkt_probs, home_ml=None, away_ml=None, is_warmup=None):
    """
    Returns (brier_score, roi, n_bets).

    Edge is computed against the de-vigged ``market_implied_prob``, but
    stake sizing and PnL use the real closing decimal odds derived from
    ``close_home_ml`` / ``close_away_ml`` so that ROI reflects what a book
    would actually pay out (no-vig odds overstate ROI by the vig).
    """
    brier        = float(np.mean((probs - y) ** 2))
    total_staked = 0.0
    total_profit = 0.0
    n_bets       = 0

    if is_warmup is None:
        is_warmup = np.zeros(len(probs), dtype=bool)
    n = len(probs)
    if home_ml is None:
        home_ml = np.full(n, np.nan, dtype=float)
    if away_ml is None:
        away_ml = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if np.isnan(mkt_probs[i]):
            continue
        mp        = float(mkt_probs[i])
        pp        = float(np.clip(probs[i], PROB_CAP[0], PROB_CAP[1]))
        warmup_i  = bool(is_warmup[i])
        edge_home = pp - mp
        edge_away = mp - pp
        thresh_i  = CONFIDENCE_THRESHOLD + (0.02 * abs(mp - 0.5) if DYNAMIC_THRESHOLD else 0.0)

        h_dec = _ml_to_dec(home_ml[i]) if not np.isnan(home_ml[i]) else (1.0 / mp if mp > 1e-6 else float("nan"))
        a_dec = _ml_to_dec(away_ml[i]) if not np.isnan(away_ml[i]) else (1.0 / (1.0 - mp) if (1.0 - mp) > 1e-6 else float("nan"))

        if edge_home >= thresh_i and np.isfinite(h_dec) and h_dec > 1.0:
            stake = kelly_stake(pp, h_dec, is_warmup=warmup_i)
            if stake > 0:
                n_bets       += 1
                total_staked += stake
                total_profit += stake * (h_dec - 1.0) if y[i] == 1.0 else -stake

        elif edge_away >= thresh_i and np.isfinite(a_dec) and a_dec > 1.0:
            prob_away = 1.0 - pp
            stake     = kelly_stake(prob_away, a_dec, is_warmup=warmup_i)
            if stake > 0:
                n_bets       += 1
                total_staked += stake
                total_profit += stake * (a_dec - 1.0) if y[i] == 0.0 else -stake

    roi = total_profit / total_staked if total_staked > 0 else float('nan')
    return brier, roi, n_bets

# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def build_metric_folds(df):
    folds = list(WALK_FORWARD_FOLDS)
    if "game_date" not in df.columns or df.empty:
        return folds

    max_date = pd.to_datetime(df["game_date"]).max()
    if pd.isna(max_date):
        return folds

    current_year = int(max_date.year)
    current_start = f"{current_year}-01-01"
    current_end = (max_date.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_static_end = pd.to_datetime(folds[-1][2]) if folds else pd.Timestamp.min
    if pd.to_datetime(current_end) > last_static_end:
        folds.append((current_start, current_start, current_end))
    return folds


def run_walk_forward(df, active_feats, early_feats, folds=None):
    folds = folds or WALK_FORWARD_FOLDS
    fold_results = []
    last_fi = None
    last_edges = None
    last_train_rows = None
    last_val_rows = None
    last_probs = None
    last_y = None
    last_mkt = None
    monthly_dates = []
    monthly_probs = []
    monthly_y = []
    monthly_mkt = []

    for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(folds)}: "
              f"train < {train_end}  |  val [{val_start}, {val_end})")

        if TRAIN_WINDOW_YEARS is not None:
            train_start = str(int(train_end[:4]) - TRAIN_WINDOW_YEARS) + train_end[4:]
            train_mask = (df["game_date"] >= train_start) & (df["game_date"] < train_end)
        else:
            train_mask = df["game_date"] < train_end
        val_mask   = (df["game_date"] >= val_start) & (df["game_date"] < val_end)
        train_df   = df[train_mask].copy()
        val_df     = df[val_mask].copy()

        if len(train_df) < 100 or len(val_df) < 50:
            print(f"  Skipping — insufficient data (train={len(train_df)}, val={len(val_df)})")
            continue

        # Early-season mask
        use_early = (EARLY_CUTOFF is not None) and len(early_feats) > 0
        if use_early:
            early_tr = ((train_df["home_games_played"] < EARLY_CUTOFF) |
                        (train_df["away_games_played"] < EARLY_CUTOFF))
            early_vl = ((val_df["home_games_played"] < EARLY_CUTOFF) |
                        (val_df["away_games_played"] < EARLY_CUTOFF))
        else:
            early_tr = pd.Series(False, index=train_df.index)
            early_vl = pd.Series(False, index=val_df.index)

        print(f"  Train: {len(train_df):,} rows  (early: {early_tr.sum():,})")
        print(f"  Val:   {len(val_df):,} rows  (early: {early_vl.sum():,})")

        X_train  = train_df[active_feats].values.astype(np.float32)
        y_train  = train_df["home_win"].values.astype(np.float32)
        X_val    = val_df[active_feats].values.astype(np.float32)
        y_val    = val_df["home_win"].values.astype(np.float32)
        mkt_val  = val_df["market_implied_prob"].values.astype(np.float32)
        hml_val  = pd.to_numeric(val_df.get("close_home_ml"), errors="coerce").to_numpy(dtype=np.float64)
        aml_val  = pd.to_numeric(val_df.get("close_away_ml"), errors="coerce").to_numpy(dtype=np.float64)
        wu_val   = early_vl.values.astype(bool)

        # Scale (fit on train only — after imputation for non-tree models)
        imp_outer   = SimpleImputer(strategy="median")
        scaler      = StandardScaler()
        X_train_sc  = scaler.fit_transform(imp_outer.fit_transform(X_train))
        imp_outer2  = SimpleImputer(strategy="median").fit(X_train)
        X_val_sc    = scaler.transform(imp_outer2.transform(X_val))
        survived_mask = np.isfinite(imp_outer.statistics_)
        survived_feats = [f for f, s in zip(active_feats, survived_mask) if s]

        probs_val   = np.zeros(len(val_df))
        probs_train = np.zeros(len(train_df))
        clf         = None

        print(f"  Training: {MODEL}")

        # ── Early specialist ───────────────────────────────────────────────
        early_clf    = None
        has_early    = False
        if use_early and early_tr.sum() > 50:
            ef_present = [f for f in early_feats if f in val_df.columns]
            X_early_tr = train_df.loc[early_tr, ef_present].values.astype(np.float32)
            y_early_tr = train_df.loc[early_tr, "home_win"].values.astype(np.float32)
            early_clf  = build_early_lr(X_early_tr, y_early_tr)
            has_early  = True

        # Fill early-season val probs from early specialist
        if has_early and early_vl.sum() > 0:
            ef_present = [f for f in early_feats if f in val_df.columns]
            X_early_vl = val_df.loc[early_vl, ef_present].values.astype(np.float32)
            probs_val[early_vl.values] = get_proba(early_clf, X_early_vl)

        # ── Regular model (non-early rows) ────────────────────────────────
        X_tr_reg = X_train_sc[~early_tr.values]
        y_tr_reg = y_train[~early_tr.values]
        X_vl_reg = X_val_sc[~early_vl.values]

        if MODEL == "lgb":
            # LGB handles NaN natively — use raw arrays not scaled
            X_tr_r = train_df.loc[~early_tr, active_feats].values.astype(np.float32)
            X_vl_r = val_df.loc[~early_vl, active_feats].values.astype(np.float32)
            X_eval = X_vl_r if len(X_vl_r) else X_tr_r
            y_eval = y_val[~early_vl.values] if len(X_vl_r) else y_tr_reg
            clf = build_lgb(X_tr_r, y_tr_reg,
                            X_eval, y_eval)
            p_tr = get_proba(clf, X_tr_r)
            p_vl = get_proba(clf, X_vl_r) if len(X_vl_r) else np.array([])
            if CALIBRATE and len(p_vl):
                p_vl = calibrate_probs(p_tr, y_tr_reg, p_vl)

        elif MODEL == "xgb":
            X_tr_r = train_df.loc[~early_tr, active_feats].values.astype(np.float32)
            X_vl_r = val_df.loc[~early_vl, active_feats].values.astype(np.float32)
            X_eval = X_vl_r if len(X_vl_r) else X_tr_r
            y_eval = y_val[~early_vl.values] if len(X_vl_r) else y_tr_reg
            clf = build_xgb(X_tr_r, y_tr_reg, X_eval, y_eval)
            p_tr = get_proba(clf, X_tr_r)
            p_vl = get_proba(clf, X_vl_r) if len(X_vl_r) else np.array([])
            if CALIBRATE and len(p_vl):
                p_vl = calibrate_probs(p_tr, y_tr_reg, p_vl)

        elif MODEL == "lr":
            clf  = build_lr(X_tr_reg, y_tr_reg)
            p_tr = get_proba(clf, X_tr_reg)
            p_vl = get_proba(clf, X_vl_reg) if len(X_vl_reg) else np.array([])

        elif MODEL == "mlp":
            X_tr_r = train_df.loc[~early_tr, active_feats].values.astype(np.float32)
            X_vl_r = val_df.loc[~early_vl, active_feats].values.astype(np.float32)
            X_eval = X_vl_r if len(X_vl_r) else X_tr_r
            y_eval = y_val[~early_vl.values] if len(X_vl_r) else y_tr_reg
            clf  = build_mlp(X_tr_r, y_tr_reg, X_eval, y_eval)
            p_tr = get_proba(clf, X_tr_r)
            p_vl = get_proba(clf, X_vl_r) if len(X_vl_r) else np.array([])
            if CALIBRATE and len(p_vl):
                p_vl = calibrate_probs(p_tr, y_tr_reg, p_vl)

        elif MODEL in ("ensemble_avg", "ensemble_stack"):
            X_tr_r = train_df.loc[~early_tr, active_feats].values.astype(np.float32)
            X_vl_r = val_df.loc[~early_vl, active_feats].values.astype(np.float32)

            X_eval = X_vl_r if len(X_vl_r) else X_tr_r
            y_eval = y_val[~early_vl.values] if len(X_vl_r) else y_tr_reg
            clf_lgb = build_lgb(X_tr_r, y_tr_reg, X_eval, y_eval)
            clf_xgb = build_xgb(X_tr_r, y_tr_reg, X_eval, y_eval)
            clf_lr  = build_lr(X_tr_reg, y_tr_reg)

            p_lgb_tr = get_proba(clf_lgb, X_tr_r)
            p_xgb_tr = get_proba(clf_xgb, X_tr_r)
            p_lr_tr  = get_proba(clf_lr,  X_tr_reg)
            p_lgb_vl = get_proba(clf_lgb, X_vl_r) if len(X_vl_r) else np.array([])
            p_xgb_vl = get_proba(clf_xgb, X_vl_r) if len(X_vl_r) else np.array([])
            p_lr_vl  = get_proba(clf_lr,  X_vl_reg) if len(X_vl_reg) else np.array([])

            if CALIBRATE and len(p_lgb_vl):
                p_lgb_vl = calibrate_probs(p_lgb_tr, y_tr_reg, p_lgb_vl)
            if CALIBRATE and len(p_xgb_vl):
                p_xgb_vl = calibrate_probs(p_xgb_tr, y_tr_reg, p_xgb_vl)

            if MODEL == "ensemble_avg":
                p_tr = (p_lgb_tr + p_xgb_tr + p_lr_tr) / 3.0
                p_vl = (p_lgb_vl + p_xgb_vl + p_lr_vl) / 3.0
            else:
                # Two-stage stack: also include disagreement as a feature
                # (the signal that stats and market diverge)
                mkt_tr  = train_df.loc[~early_tr, "market_implied_prob"].fillna(0.54).values
                mkt_vl  = val_df.loc[~early_vl,   "market_implied_prob"].fillna(0.54).values
                meta_tr = np.column_stack([p_lgb_tr, p_xgb_tr, p_lr_tr,
                                           p_lr_tr - mkt_tr])
                meta_vl = np.column_stack([p_lgb_vl, p_xgb_vl, p_lr_vl,
                                           p_lr_vl - mkt_vl])
                meta    = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
                meta.fit(meta_tr, y_tr_reg)
                p_tr = meta.predict_proba(meta_tr)[:, 1]
                p_vl = meta.predict_proba(meta_vl)[:, 1]
                clf  = meta

        else:
            raise ValueError(f"Unknown MODEL: {MODEL!r}. "
                             "Choose: lgb, xgb, lr, mlp, ensemble_avg, ensemble_stack")

        probs_val[~early_vl.values] = p_vl

        monthly_dates.append(val_df["game_date"].values)
        monthly_probs.append(probs_val.copy())
        monthly_y.append(y_val.copy())
        monthly_mkt.append(mkt_val.copy())

        brier, roi, n_bets = evaluate(
            probs_val, y_val, mkt_val,
            home_ml=hml_val, away_ml=aml_val, is_warmup=wu_val,
        )
        print(f"  brier={brier:.4f}  roi={roi:.4f}  n_bets={n_bets}")

        if clf is not None and fold_idx == len(folds) - 1:
            fi_feats = survived_feats if MODEL in ("lr", "mlp") else active_feats
            print_feature_importance(clf, fi_feats)
            last_fi = extract_feature_importance(clf, fi_feats)
            last_edges = (np.clip(probs_val, PROB_CAP[0], PROB_CAP[1]) - mkt_val).tolist()
            last_train_rows = len(train_df)
            last_val_rows = len(val_df)
            last_probs = probs_val.copy()
            last_y = y_val.copy()
            last_mkt = mkt_val.copy()

        fold_results.append({
            "fold": fold_idx + 1, "brier": brier, "roi": roi, "n_bets": n_bets,
            "train_rows": len(train_df), "val_rows": len(val_df),
        })

    extras = {}
    if last_fi is not None:
        extras["feature_importances"] = last_fi
    if last_edges is not None:
        hist, bin_edges = np.histogram(last_edges, bins=40)
        extras["edge_distribution"] = {
            "counts": hist.tolist(),
            "bin_edges": [round(float(b), 4) for b in bin_edges],
        }
    if last_train_rows is not None:
        extras["training_rows"] = last_train_rows
        extras["val_rows"] = last_val_rows
    if last_probs is not None:
        buckets = {}
        for feat in active_feats:
            if feat not in df.columns:
                continue
            last_fold = folds[-1]
            val_start, val_end = last_fold[1], last_fold[2]
            val_mask = (df["game_date"] >= val_start) & (df["game_date"] < val_end)
            val_df_last = df[val_mask]
            if feat in val_df_last.columns and len(val_df_last) == len(last_y):
                vals = val_df_last[feat].values
                median = np.nanmedian(vals)
                high = ~np.isnan(vals) & (vals >= median)
                low = ~np.isnan(vals) & (vals < median)
                if high.sum() > 10 and low.sum() > 10:
                    acc_high = float(np.mean((last_probs[high] > 0.5) == last_y[high]))
                    acc_low = float(np.mean((last_probs[low] > 0.5) == last_y[low]))
                    buckets[feat] = {
                        "high_accuracy": round(acc_high, 4),
                        "low_accuracy": round(acc_low, 4),
                        "high_n": int(high.sum()),
                        "low_n": int(low.sum()),
                    }
        extras["feature_accuracy"] = buckets

    if monthly_dates:
        all_dates = pd.to_datetime(np.concatenate(monthly_dates))
        all_probs = np.concatenate(monthly_probs).astype(float)
        all_y = np.concatenate(monthly_y).astype(float)
        all_mkt = np.concatenate(monthly_mkt).astype(float)
        try:
            extras["overall_log_loss"] = float(
                log_loss(
                    all_y.astype(np.int32, copy=False),
                    np.clip(all_probs, 1e-6, 1.0 - 1e-6),
                    labels=[0, 1],
                )
            )
        except Exception:
            extras["overall_log_loss"] = None
        years = all_dates.year.values
        months = all_dates.month.values
        keys = years * 100 + months
        monthly = []
        for key in sorted(set(int(k) for k in keys)):
            mask = keys == key
            yr = int(key // 100)
            mo = int(key % 100)
            n = int(mask.sum())
            y = all_y[mask]
            p = all_probs[mask]
            mk = all_mkt[mask]
            mk_valid = ~np.isnan(mk)
            brier = float(np.mean((p - y) ** 2))
            accuracy = float(np.mean((p > 0.5) == (y > 0.5)))
            try:
                ll = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
            except Exception:
                ll = None
            if mk_valid.any():
                market_brier = float(np.mean((mk[mk_valid] - y[mk_valid]) ** 2))
                market_accuracy = float(np.mean((mk[mk_valid] > 0.5) == (y[mk_valid] > 0.5)))
            else:
                market_brier = None
                market_accuracy = None
            monthly.append({
                "year": yr,
                "month": mo,
                "year_month": f"{yr:04d}-{mo:02d}",
                "count": n,
                "brier": round(brier, 6),
                "log_loss": round(ll, 6) if ll is not None else None,
                "accuracy": round(accuracy, 6),
                "market_brier": round(market_brier, 6) if market_brier is not None else None,
                "market_accuracy": round(market_accuracy, 6) if market_accuracy is not None else None,
            })
        extras["monthly_accuracy"] = monthly

    return fold_results, extras

# ---------------------------------------------------------------------------
# Main  (walk-forward validation — run directly: uv run train.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    df = load_and_engineer_features()

    active_feats = [c for c in FEATURE_COLUMNS      if c in df.columns]
    early_feats  = [c for c in EARLY_FEATURE_COLUMNS if c in df.columns]

    print(f"\nActive features ({len(active_feats)}): {active_feats}")
    print(f"Early features  ({len(early_feats)}): {early_feats}")

    # Use imputer-aware dropna: only drop rows missing market_implied_prob or target
    # (GBDTs handle NaN natively; LR/MLP impute internally)
    req_always = ["home_win", "market_implied_prob", "game_date",
                  "home_games_played", "away_games_played"]
    df = df.dropna(subset=[c for c in req_always if c in df.columns])
    print(f"Rows after dropna (required cols only): {len(df):,}")

    metric_folds = build_metric_folds(df)
    fold_results, extras = run_walk_forward(df, active_feats, early_feats, metric_folds)

    if fold_results:
        rois       = [r["roi"]    for r in fold_results if not math.isnan(r["roi"])]
        briers     = [r["brier"]  for r in fold_results]
        bets       = [r["n_bets"] for r in fold_results]
        mean_roi   = float(np.mean(rois))   if rois   else float("nan")
        mean_brier = float(np.mean(briers)) if briers else float("nan")
        total_bets = int(np.sum(bets))
    else:
        mean_roi = mean_brier = float("nan")
        total_bets = 0

    print(f"\n{'='*60}")
    print(f"Walk-forward summary ({len(fold_results)} folds):")
    for r in fold_results:
        print(f"  Fold {r['fold']}: brier={r['brier']:.4f}  roi={r['roi']:.4f}  n_bets={r['n_bets']}")
    print(f"  Mean ROI:   {mean_roi:.6f}")
    print(f"  Mean Brier: {mean_brier:.6f}")
    print(f"  Total bets: {total_bets}")
    print(f"  total_seconds: {time.time() - t_start:.1f}")

    # Save results
    results_file = "data/results.tsv"
    header = "commit\tval_roi\tval_brier\tstatus\tdescription\n"
    commit  = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "HEAD"
    status  = "ok" if not math.isnan(mean_roi) else "fail"
    n_folds = len(fold_results)
    desc    = (f"model={MODEL} calibrate={CALIBRATE} folds={n_folds} "
               f"early_cutoff={EARLY_CUTOFF} W={BEST_W} momentum_W={MOMENTUM_W} "
               f"threshold={CONFIDENCE_THRESHOLD} feats={len(active_feats)}")
    row = f"{commit}\t{mean_roi:.6f}\t{mean_brier:.6f}\t{status}\t{desc}\n"

    try:
        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                f.write(header)
        with open(results_file, "a") as f:
            f.write(row)
        print(f"Results saved to {results_file}")
    except OSError as e:
        print(f"WARNING: Could not write {results_file}: {e}")

    # Save to DB
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
        import db
        duration = time.time() - t_start
        run_data = {
            "model_type": MODEL,
            "git_commit": commit,
            "num_features": len(active_feats),
            "training_rows": extras.get("training_rows"),
            "val_rows": extras.get("val_rows"),
            "num_folds": n_folds,
            "mean_brier": mean_brier if not math.isnan(mean_brier) else None,
            "mean_roi": mean_roi if not math.isnan(mean_roi) else None,
            "overall_log_loss": extras.get("overall_log_loss"),
            "total_bets": total_bets,
            "duration_seconds": round(duration, 1),
            "feature_importances": extras.get("feature_importances", []),
            "fold_results": fold_results,
            "edge_distribution": extras.get("edge_distribution", {}),
            "config": {
                "calibrate": CALIBRATE,
                "early_cutoff": EARLY_CUTOFF,
                "best_w": BEST_W,
                "momentum_w": MOMENTUM_W,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "kelly_fraction": KELLY_FRACTION,
                "max_bet_frac": MAX_BET_FRAC,
                "prob_cap": list(PROB_CAP),
                "metric_folds": metric_folds,
            },
            "feature_accuracy": extras.get("feature_accuracy", {}),
            "monthly_accuracy": extras.get("monthly_accuracy", []),
        }
        run_id = db.save_model_metric_snapshot(run_data)
        print(f"Model metric snapshot saved to DB (id={run_id})")
    except Exception as e:
        print(f"WARNING: Could not save model metric snapshot to DB: {e}")
