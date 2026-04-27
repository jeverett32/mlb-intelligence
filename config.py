"""Shared runtime configuration for the MLB pipeline."""

import os
from datetime import datetime


def get_active_season() -> int:
    """Return the MLB season to operate on, overridable via MLB_SEASON."""
    raw = os.environ.get("MLB_SEASON", "").strip()
    if raw:
        return int(raw)
    return datetime.now().year


ACTIVE_SEASON = get_active_season()
CURRENT_CSV = f"data/mlb_{ACTIVE_SEASON}.csv"
ODDS_CACHE_CSV = f"odds_{ACTIVE_SEASON}.csv"
