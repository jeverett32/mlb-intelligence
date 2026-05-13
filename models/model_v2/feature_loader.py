import os
import sys
import pandas as pd
import hashlib
import pickle
from pathlib import Path

# Absolute imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import db as DB

_FRAME_CACHE: dict = {"df": None, "key": None}


def invalidate_games_v2_frame_cache() -> None:
    """Clear in-process / on-disk frame cache after games_v2 bulk updates."""
    _FRAME_CACHE["df"] = None
    _FRAME_CACHE["key"] = None


def _frame_cache_key() -> str:
    with DB.pooled_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), MAX(updated_at) FROM games_v2")
            row_count, max_updated_at = cur.fetchone()
    raw = f"{int(row_count or 0)}:{max_updated_at}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_frame(cache_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    key = _frame_cache_key()
    if _FRAME_CACHE["df"] is not None and _FRAME_CACHE.get("key") == key:
        return _FRAME_CACHE["df"]

    cache_file = cache_dir / f"games_v2_frame__{key}.pkl"
    if use_cache and cache_file.exists():
        with cache_file.open("rb") as f:
            _FRAME_CACHE["df"] = pickle.load(f)
        _FRAME_CACHE["key"] = key
        return _FRAME_CACHE["df"]

    _FRAME_CACHE["df"] = DB.load_games_v2_frame(cutoff=None)
    _FRAME_CACHE["key"] = key
    if use_cache:
        try:
            with cache_file.open("wb") as f:
                pickle.dump(_FRAME_CACHE["df"], f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"  V2 frame cache save skipped: {exc}")
    return _FRAME_CACHE["df"]


def load_or_build_engineered_frame_from_db(cutoff: str, cache_dir: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Pulls rows from games_v2 and explodes features.
    Caches the results to avoid repeated DB lookups and heavy processing.
    """
    cache_path = Path(cache_dir)
    os.makedirs(cache_path, exist_ok=True)

    df = _load_frame(cache_path, use_cache=use_cache)
    if df.empty:
        return df

    return df
