import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import hashlib

# Absolute imports to reach sandbox and main repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import db as DB
from model_v2 import config as C
from sandbox.model_lab.feature_engineer import (
    load_or_build_engineered_frame,
    load_or_build_feature_sets,
    recency_weights
)
from sandbox.model_lab.training.models import make_lgbm
from sandbox.model_lab.roi_eval import ml_to_dec

class PredictV2Error(RuntimeError):
    pass

def prepare_shared(game_pks: list[str], game_date: str) -> dict:
    """
    Loads data, fits LGBM on history before game_date, 
    and extracts target rows for specified PKs.
    """
    try:
        cache_dir = os.path.join(ROOT, "sandbox/model_lab/output/cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 1. Load engineered frame (sandbox master)
        # We pass game_date as the cutoff to prevent leakage.
        df = load_or_build_engineered_frame(
            input_path=os.path.join(ROOT, "sandbox/model_lab/output/master_sandbox_mlb.csv"),
            cutoff=game_date, 
            cache_dir=cache_dir,
            use_cache=True
        )
        
        # 2. Get top-K features
        fs = load_or_build_feature_sets(
            df,
            min_coverage=C.MIN_COVERAGE,
            corr_threshold=C.CORR_THRESHOLD,
            top_k=C.K_FEATURES,
            cache_dir=cache_dir,
            use_cache=True,
            selected_by=C.SELECTED_BY
        )
        feature_cols = list(fs["selected"])
        
        # 3. Filter for training (date < target_date)
        cutoff_ts = pd.Timestamp(game_date)
        train_df = df[df["game_date"].notna()].copy()
        train_df["game_date"] = pd.to_datetime(train_df["game_date"])
        
        history = train_df[train_df["game_date"] < cutoff_ts].dropna(subset=["home_win"]).copy()
        if history.empty:
            raise PredictV2Error(f"No training data found before {game_date}")
            
        # 4. Prepare target rows
        target_rows = {}
        lookup_pks = [int(pk) for pk in game_pks]
        target_pool = train_df[train_df["game_pk"].isin(lookup_pks)].copy()
        for pk in game_pks:
            row = target_pool[target_pool["game_pk"] == int(pk)]
            if not row.empty:
                target_rows[str(pk)] = row.iloc[0:1]
            else:
                # Silently skip missing rows for now, predict_one will handle it
                pass

        # 5. Fit LGBM Pipeline
        # Anchored on the year before game_date
        sw = recency_weights(history["season"], cutoff_ts.year - 1, C.RECENCY_HALF_LIFE)
        
        X_tr = history[feature_cols].apply(pd.to_numeric, errors="coerce")
        y_tr = history["home_win"].astype(int)
        
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_lgbm()),
        ])
        
        pipe.fit(X_tr, y_tr, model__sample_weight=sw)
        
        # Create a training fingerprint
        fingerprint_raw = ",".join(sorted(feature_cols)) + str(len(history))
        training_fingerprint = hashlib.sha256(fingerprint_raw.encode()).hexdigest()[:12]
        
        return {
            "clf": pipe,
            "feature_cols": feature_cols,
            "target_rows": target_rows,
            "training_fingerprint": training_fingerprint
        }
        
    except Exception as e:
        if isinstance(e, PredictV2Error):
            raise
        raise PredictV2Error(f"Failed to prepare shared V2 model: {e}")

def predict_one(game_pk: str, shared: dict, dry_run: bool = False) -> dict:
    """
    Predicts probability and edge for one game using the shared model.
    Applies Phase 2.5 filters and half-Kelly sizing.
    """
    game_pk = str(game_pk)
    target_row = shared["target_rows"].get(game_pk)
    if target_row is None:
        raise PredictV2Error(f"Target game {game_pk} missing from shared cache.")
        
    clf = shared["clf"]
    feature_cols = shared["feature_cols"]
    
    # 1. Probabilities
    X_va = target_row[feature_cols].apply(pd.to_numeric, errors="coerce")
    prob = float(clf.predict_proba(X_va)[0, 1])
    
    # 2. Market/Edge
    h_ml = target_row.get("close_home_ml")
    a_ml = target_row.get("close_away_ml")
    
    if h_ml is None or a_ml is None or pd.isna(h_ml.iloc[0]) or pd.isna(a_ml.iloc[0]):
        h_imp = target_row.get("market_implied_prob")
        if h_imp is not None and not pd.isna(h_imp.iloc[0]):
            market_implied_prob = float(h_imp.iloc[0])
            h_dec = 1.0 / market_implied_prob if market_implied_prob > 0 else 100.0
            a_dec = 1.0 / (1.0 - market_implied_prob) if market_implied_prob < 1 else 100.0
        else:
             raise PredictV2Error(f"Market odds missing for game {game_pk}")
    else:
        h_ml_val = float(h_ml.iloc[0])
        a_ml_val = float(a_ml.iloc[0])
        h_dec = float(ml_to_dec(pd.Series([h_ml_val]))[0])
        a_dec = float(ml_to_dec(pd.Series([a_ml_val]))[0])
        
        total_imp = (1.0 / h_dec) + (1.0 / a_dec)
        market_implied_prob = (1.0 / h_dec) / total_imp
        
    edge = prob - market_implied_prob
    
    # 3. Phase 2.5 Filter & Sizing
    if edge > 0:
        side = "home"
        edge_mag = edge
        odds_on_side = h_dec
        ev = prob * (h_dec - 1) - (1 - prob)
    else:
        side = "away"
        edge_mag = -edge
        odds_on_side = a_dec
        ev = (1 - prob) * (a_dec - 1) - prob
        
    if C.EDGE_MIN < edge_mag <= C.EDGE_MAX and odds_on_side <= 3.5 and ev > 0:
        bet_side = side
        bet_frac = (ev / (odds_on_side - 1)) * C.KELLY_FACTOR
    else:
        bet_side = "none"
        bet_frac = 0.0
        
    res = {
        "game_pk": game_pk,
        "prob": prob,
        "market_implied_prob": market_implied_prob,
        "edge": edge,
        "bet_side": bet_side,
        "bet_frac": bet_frac,
        "kelly_factor_used": C.KELLY_FACTOR,
        "model_version": C.MODEL_VERSION,
        "training_fingerprint": shared["training_fingerprint"]
    }
    
    if not dry_run:
        if hasattr(DB, "update_bet_v2_prediction"):
            try:
                DB.update_bet_v2_prediction(
                    game_pk=int(game_pk),
                    predicted_prob=prob,
                    edge=edge,
                    bet_side=bet_side,
                    bet_frac=bet_frac,
                    market_implied_prob=market_implied_prob,
                    model_artifact_id=None
                )
            except Exception as e:
                print(f"  V2 DB Update Error (bet): {e}")
                
        if hasattr(DB, "upsert_paper_order_v2") and bet_side != "none":
            try:
                DB.upsert_paper_order_v2(
                    game_pk=int(game_pk),
                    bet_side=bet_side,
                    bet_frac=bet_frac,
                    predicted_prob=prob,
                    market_implied_prob=market_implied_prob,
                    edge=edge
                )
            except Exception as e:
                print(f"  V2 DB Update Error (order): {e}")
    
    return res
