import os
import sys
import pandas as pd
from pathlib import Path

# Absolute imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import db as DB

_FRAME_CACHE: dict = {"df": None}


def _load_frame() -> pd.DataFrame:
    if _FRAME_CACHE["df"] is None:
        _FRAME_CACHE["df"] = DB.load_games_v2_frame(cutoff=None)
    return _FRAME_CACHE["df"]


def load_or_build_engineered_frame_from_db(cutoff: str, cache_dir: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Pulls rows from games_v2 and explodes features.
    Caches the results to avoid repeated DB lookups and heavy processing.
    """
    cache_path = Path(cache_dir)
    os.makedirs(cache_path, exist_ok=True)

    df = _load_frame()
    if df.empty:
        return df

    return df
