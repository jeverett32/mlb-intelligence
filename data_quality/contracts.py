"""Single source of truth for data quality expectations on `games` table columns.

Each FeatureContract describes the allowed shape of one column: dtype, plausible
numeric range, and the maximum tolerable share of NaN rows in a recent window.
The same contracts drive ingest validation, the audit job, the SQL migration
that adds CHECK constraints, and the unit tests.

Ranges are intentionally generous: they catch schema drift (a wOBA value of 12,
a probability of -3) without rejecting legitimate outliers (a 162 wRC+ team).
Tighten only if false-positive rate stays at zero in production for a season.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class FeatureContract:
    name: str
    dtype: str
    min_value: float | None = None
    max_value: float | None = None
    max_nan_pct: float = 1.0
    source: str = ""
    notes: str = ""
    sql_type: str = "DOUBLE PRECISION"
    nullable: bool = True
    constraint_safe: bool = True

    def in_range(self, value: float) -> bool:
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


_PCT = dict(min_value=0.0, max_value=1.0, sql_type="DOUBLE PRECISION")
_PROB = dict(min_value=0.0, max_value=1.0, sql_type="DOUBLE PRECISION")


CONTRACTS: tuple[FeatureContract, ...] = (
    # --- Identifiers / dates ---
    FeatureContract("game_pk", "int", min_value=1, source="MLB StatsAPI",
                    sql_type="BIGINT", nullable=False, constraint_safe=False),
    FeatureContract("season", "int", min_value=1900, max_value=2100,
                    source="MLB StatsAPI", sql_type="INTEGER"),

    # --- Scores ---
    FeatureContract("home_score", "int", min_value=0, max_value=50,
                    source="MLB StatsAPI", sql_type="INTEGER",
                    notes="Hard cap >50 = parser bug, not real game"),
    FeatureContract("away_score", "int", min_value=0, max_value=50,
                    source="MLB StatsAPI", sql_type="INTEGER"),

    # --- Odds (American moneyline) ---
    FeatureContract("open_home_ml", "float", min_value=-10000, max_value=10000,
                    source="Odds API / SBR"),
    FeatureContract("open_away_ml", "float", min_value=-10000, max_value=10000,
                    source="Odds API / SBR"),
    FeatureContract("close_home_ml", "float", min_value=-10000, max_value=10000,
                    source="Odds API / SBR"),
    FeatureContract("close_away_ml", "float", min_value=-10000, max_value=10000,
                    source="Odds API / SBR"),
    FeatureContract("home_implied_prob", "float", **_PROB,
                    source="Derived from close ML"),
    FeatureContract("away_implied_prob", "float", **_PROB,
                    source="Derived from close ML"),
    FeatureContract("over_under", "float", min_value=0, max_value=30,
                    source="Odds API"),

    # --- Starter stats ---
    FeatureContract("home_starter_era", "float", min_value=0, max_value=30,
                    source="MLB StatsAPI / FanGraphs"),
    FeatureContract("away_starter_era", "float", min_value=0, max_value=30),
    FeatureContract("home_starter_whip", "float", min_value=0, max_value=5),
    FeatureContract("away_starter_whip", "float", min_value=0, max_value=5),
    FeatureContract("home_starter_k9", "float", min_value=0, max_value=20),
    FeatureContract("away_starter_k9", "float", min_value=0, max_value=20),
    FeatureContract("home_starter_bb9", "float", min_value=0, max_value=15),
    FeatureContract("away_starter_bb9", "float", min_value=0, max_value=15),
    FeatureContract("home_starter_fip", "float", min_value=0, max_value=15,
                    notes="FIP can be negative in tiny samples; cap at 0 for prod"),
    FeatureContract("away_starter_fip", "float", min_value=0, max_value=15),

    # --- Weather ---
    FeatureContract("temp_c", "float", min_value=-30, max_value=50,
                    source="OpenWeather"),
    FeatureContract("wind_speed_kph", "float", min_value=0, max_value=200),
    FeatureContract("wind_dir_deg", "float", min_value=0, max_value=360),
    FeatureContract("precip_mm", "float", min_value=0, max_value=500),

    # --- Team batting (FanGraphs season-to-date) ---
    FeatureContract("home_wrc_plus", "float", min_value=10, max_value=300,
                    source="FanGraphs",
                    notes="Catches schema drift; team season-level realistic ~60-160"),
    FeatureContract("away_wrc_plus", "float", min_value=10, max_value=300,
                    source="FanGraphs"),
    FeatureContract("home_woba", "float", min_value=0.150, max_value=0.500,
                    source="FanGraphs"),
    FeatureContract("away_woba", "float", min_value=0.150, max_value=0.500),
    FeatureContract("home_avg", "float", min_value=0.100, max_value=0.400),
    FeatureContract("away_avg", "float", min_value=0.100, max_value=0.400),
    FeatureContract("home_obp", "float", min_value=0.150, max_value=0.500),
    FeatureContract("away_obp", "float", min_value=0.150, max_value=0.500),
    FeatureContract("home_slg", "float", min_value=0.150, max_value=0.700),
    FeatureContract("away_slg", "float", min_value=0.150, max_value=0.700),

    # --- Team pitching (FanGraphs season-to-date) ---
    FeatureContract("home_era", "float", min_value=0, max_value=15,
                    source="FanGraphs"),
    FeatureContract("away_era", "float", min_value=0, max_value=15),
    FeatureContract("home_fip", "float", min_value=0, max_value=10),
    FeatureContract("away_fip", "float", min_value=0, max_value=10),
    FeatureContract("home_k9", "float", min_value=0, max_value=20),
    FeatureContract("away_k9", "float", min_value=0, max_value=20),
    FeatureContract("home_bb9", "float", min_value=0, max_value=15),
    FeatureContract("away_bb9", "float", min_value=0, max_value=15),
)


_BY_NAME: dict[str, FeatureContract] = {c.name: c for c in CONTRACTS}


def get_contract(name: str) -> FeatureContract | None:
    return _BY_NAME.get(name)


def contract_names() -> list[str]:
    return [c.name for c in CONTRACTS]


def constraint_eligible() -> Iterable[FeatureContract]:
    """Contracts safe to enforce as DB CHECK constraints (numeric + bounded)."""
    for c in CONTRACTS:
        if not c.constraint_safe:
            continue
        if c.min_value is None and c.max_value is None:
            continue
        yield c
