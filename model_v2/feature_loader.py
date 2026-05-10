import os
import sys
import pandas as pd
import hashlib
from pathlib import Path

# Absolute imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import db as DB
from sandbox.model_lab.feature_engineer import load_or_build_feature_sets
from model_v2 import config as C

_FRAME_CACHE: dict = {"df": None}


def _load_frame() -> pd.DataFrame:
    if _FRAME_CACHE["df"] is None:
        _FRAME_CACHE["df"] = DB.load_games_v2_frame(cutoff=None)
    return _FRAME_CACHE["df"]


def load_or_build_engineered_frame_from_db(cutoff: str, cache_dir: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Pulls rows from games_v2, explodes features, and applies sandbox feature sets.
    Caches the results to avoid repeated DB lookups and heavy processing.
    """
    cache_path = Path(cache_dir)
    os.makedirs(cache_path, exist_ok=True)

    df = _load_frame()
    if df.empty:
        return df
        
    # We don't really want to cache the full frame here because it might change frequently.
    # But we want to ensure feature columns are consistent.
    
    # 2. Get top-K features using existing sandbox logic
    # This function expects the full engineered frame to identify stable features.
    fs = load_or_build_feature_sets(
        df,
        min_coverage=C.MIN_COVERAGE,
        corr_threshold=C.CORR_THRESHOLD,
        top_k=C.K_FEATURES,
        cache_dir=cache_path,
        use_cache=use_cache,
        selected_by=C.SELECTED_BY
    )
    
    # We return the whole frame, but prepare_shared will use the 'selected' cols.
    return df

