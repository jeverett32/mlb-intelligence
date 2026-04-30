"""
All-Star break helpers.

Training should remain deterministic/offline, so `ASB_DATES` is the canonical
mapping used by feature engineering. The fetch helpers are for maintenance
and verification (e.g., scripts/tests), not for use in training-time feature
computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


# All-Star Game dates by season (used as the anchor for days_since_asb).
# Stored as YYYY-MM-DD (UTC date).
ASB_DATES: dict[int, str] = {
    2010: "2010-07-13",
    2011: "2011-07-12",
    2012: "2012-07-10",
    2013: "2013-07-16",
    2014: "2014-07-15",
    2015: "2015-07-14",
    2016: "2016-07-12",
    2017: "2017-07-11",
    2018: "2018-07-17",
    2019: "2019-07-09",
    # 2020 omitted (COVID season)
    2021: "2021-07-13",
    2022: "2022-07-19",
    2023: "2023-07-11",
    2024: "2024-07-16",
    2025: "2025-07-15",
    2026: "2026-07-14",
}


@dataclass(frozen=True)
class AsgLookupResult:
    season: int
    asg_date: str | None
    source: str


def fetch_asg_date_from_mlb_stats_api(season: int, *, timeout_s: float = 20.0) -> AsgLookupResult:
    """
    Fetch the All-Star Game date for a season using MLB Stats API schedule.

    We query a July window and look for a game with gameType == 'A'
    (All-Star Game).

    Returns AsgLookupResult(season, asg_date|None, source).
    """
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "startDate": f"{season}-07-01",
        "endDate": f"{season}-07-31",
    }
    resp = requests.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()

    for d in data.get("dates", []) or []:
        date = d.get("date")
        for g in d.get("games", []) or []:
            if g.get("gameType") == "A":
                return AsgLookupResult(season=season, asg_date=str(date), source="mlb_stats_api")
    return AsgLookupResult(season=season, asg_date=None, source="mlb_stats_api")

