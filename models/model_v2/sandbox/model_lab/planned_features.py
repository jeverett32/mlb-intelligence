"""Planned sandbox features from docs/model_feature_catalog.md.

This module adds the full planned schema to the sandbox master. Features that
need new external source pulls are created as nullable columns so downstream
work can target one stable contract. A small subset is filled with documented
proxies from current production data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


SIDES = ("home", "away")


def _side_cols(prefix: str, metrics: list[str]) -> list[str]:
    return [f"{side}_{prefix}_{metric}" for side in SIDES for metric in metrics]


SP_WORKLOAD = _side_cols("sp", ["days_rest", "pitch_count_l3", "bf_l3", "ip_l3"])

SP_STATCAST_SHAPE = _side_cols(
    "sp",
    [
        "fastball_velo",
        "fastball_spin",
        "extension",
        "active_spin",
        "whiff_rate",
        "chase_rate",
        "cs_rate",
        "csw_rate",
        "groundball_rate",
        "barrel_allowed_rate",
        "hard_hit_allowed_rate",
        "avg_ev_allowed",
        "xwoba_allowed",
        "xera",
    ],
)

SP_PITCH_MIX = _side_cols(
    "sp",
    [
        "pitch_mix_ff",
        "pitch_mix_si",
        "pitch_mix_fc",
        "pitch_mix_sl",
        "pitch_mix_st",
        "pitch_mix_cu",
        "pitch_mix_ch",
    ],
)

SP_TTTO_PLATOON = (
    _side_cols(
        "sp",
        [
            "times_through_order_avg",
            "tto3_penalty",
            "first_time_woba",
            "second_time_woba",
            "third_time_woba",
            "platoon_split_vs_lhb",
            "platoon_split_vs_rhb",
            "ttto_penalty_expected",
        ],
    )
    + ["sp_tto_avg_DIFF", "sp_tto3_penalty_DIFF", "sp_third_minus_first_woba_DIFF", "sp_platoon_split_DIFF"]
)

BULLPEN_QUALITY = _side_cols(
    "bp",
    ["era_30d", "fip_30d", "xfip_30d", "whiff_rate_30d", "k_minus_bb_30d", "xwoba_allowed_30d"],
) + ["bp_quality_DIFF"]

BULLPEN_FRESHNESS = (
    _side_cols(
        "bp",
        [
            "pitches_1d",
            "pitches_2d",
            "pitches_3d",
            "bf_2d",
            "ip_2d",
            "high_leverage_pitches_2d",
            "lefty_available",
            "righty_available",
            "freshness_score",
            "fatigue_expected",
            "projected_first_outs_quality",
        ],
    )
    + _side_cols("", ["closer_used_yesterday", "top3_rp_used_2d"])
    + ["bp_pitches_2d_DIFF", "bp_freshness_DIFF"]
)

# _side_cols("", ...) creates home__x; keep explicit names for role flags.
BULLPEN_FRESHNESS = [
    c.replace("home__", "home_").replace("away__", "away_") for c in BULLPEN_FRESHNESS
]

HIGH_LEVERAGE_RP = (
    _side_cols("closer", ["xwoba_allowed_30d", "k_minus_bb_30d", "whiff_rate_30d"])
    + _side_cols("closer", ["pitches_yesterday", "pitches_2d", "available_flag"])
    + _side_cols("setup", ["xwoba_allowed_30d", "k_minus_bb_30d", "whiff_rate_30d"])
    + _side_cols("setup", ["pitches_yesterday", "pitches_2d", "available_flag"])
    + _side_cols("top_lhrp", ["xwoba_allowed_30d", "pitches_yesterday"])
    + _side_cols("top_rhrp", ["xwoba_allowed_30d", "pitches_yesterday"])
    + [
        "closer_quality_DIFF",
        "closer_freshness_DIFF",
        "setup_quality_DIFF",
        "setup_freshness_DIFF",
        "top_lhrp_quality_DIFF",
        "top_rhrp_quality_DIFF",
    ]
)

LINEUP_SKILL = _side_cols(
    "lineup",
    [
        "avg_woba",
        "avg_wrc_plus",
        "avg_xwoba",
        "avg_xba",
        "avg_slg",
        "avg_obp",
        "avg_iso",
        "avg_k_rate",
        "avg_bb_rate",
        "hard_hit_rate",
        "barrel_rate",
        "avg_ev",
        "avg_la",
        "sweet_spot_rate",
        "sprint_speed",
    ],
) + ["lineup_quality_DIFF", "lineup_contact_quality_DIFF"]

LINEUP_RECENT = _side_cols(
    "lineup",
    [
        "recent_xwoba_14d",
        "recent_hard_hit_14d",
        "recent_barrel_14d",
        "recent_k_rate_14d",
        "recent_xwoba_30d",
        "recent_hard_hit_30d",
        "recent_barrel_30d",
        "recent_k_rate_30d",
    ],
) + ["lineup_recent_form_DIFF", "home_contact_quality_recent", "away_contact_quality_recent"]

LINEUP_COMPOSITION = _side_cols(
    "lineup",
    ["lhb_count", "rhb_count", "top4_xwoba", "bottom5_xwoba", "depth_score"],
) + ["lineup_handedness_balance_DIFF"]

CATCHER = (
    _side_cols("catcher", ["framing_runs", "strike_rate", "shadow_strike_rate", "framing_runs_1000"])
    + ["catcher_framing_DIFF"]
    + _side_cols("catcher", ["blocking_runs", "blocks_above_avg"])
    + _side_cols("catcher", ["pop_time", "arm_strength", "caught_stealing_rate"])
    + ["catcher_throwing_DIFF"]
)

SP_CATCHER_PAIR = _side_cols("sp_catcher_pair", ["framing", "cs_rate"])

MATCHUP_VECTORS = [
    "home_lineup_vs_away_sp_xwoba",
    "away_lineup_vs_home_sp_xwoba",
    "matchup_xwoba_DIFF",
    "home_lineup_vs_away_sp_pitch_mix_fit",
    "away_lineup_vs_home_sp_pitch_mix_fit",
    "pitch_mix_fit_DIFF",
    "home_lineup_vs_away_sp_platoon_fit",
    "away_lineup_vs_home_sp_platoon_fit",
    "platoon_fit_DIFF",
    "home_lineup_whiff_risk_vs_away_sp",
    "away_lineup_whiff_risk_vs_home_sp",
    "home_lineup_gb_fit_vs_away_sp",
    "away_lineup_gb_fit_vs_home_sp",
    "home_lineup_power_vs_away_sp",
    "away_lineup_power_vs_home_sp",
    "home_lineup_vs_away_sp_pitch_type_ff",
    "away_lineup_vs_home_sp_pitch_type_ff",
    "home_lineup_vs_away_sp_pitch_type_sl",
    "away_lineup_vs_home_sp_pitch_type_sl",
    "home_lineup_vs_away_sp_pitch_type_ch",
    "away_lineup_vs_home_sp_pitch_type_ch",
]

WEATHER_AIR = [
    "temp_c_game_time",
    "relative_humidity_game_time",
    "dew_point_c_game_time",
    "surface_pressure_hpa_game_time",
    "air_density_game_time",
    "density_altitude_game_time",
    "wind_speed_kmh_game_time",
    "wind_dir_deg_game_time",
    "wind_out_to_center_kmh",
    "wind_out_to_left_kmh",
    "wind_out_to_right_kmh",
    "weather_hr_boost_factor",
    "weather_run_env_factor",
]

UMPIRE = [
    "umpire_id",
    "umpire_zone_size",
    "umpire_zone_tightness",
    "umpire_called_strike_rate",
    "umpire_consistency",
]

PARK_HANDEDNESS = ["park_hr_factor_lhb", "park_hr_factor_rhb", "park_run_factor_lhb", "park_run_factor_rhb"]

TRAVEL_BODY_CLOCK = ["travel_miles", "travel_tz_shift", "getaway_game_flag"]

ROOF_STATE = ["roof_closed_flag", "roof_possible_flag"]

PLANNED_FEATURE_GROUPS: dict[str, list[str]] = {
    "starting_pitcher_workload": SP_WORKLOAD,
    "starting_pitcher_statcast_shape": SP_STATCAST_SHAPE,
    "starting_pitcher_pitch_mix": SP_PITCH_MIX,
    "starting_pitcher_ttto_platoon": SP_TTTO_PLATOON,
    "bullpen_quality": BULLPEN_QUALITY,
    "bullpen_freshness": BULLPEN_FRESHNESS,
    "high_leverage_relievers": HIGH_LEVERAGE_RP,
    "lineup_skill": LINEUP_SKILL,
    "lineup_recent": LINEUP_RECENT,
    "lineup_composition": LINEUP_COMPOSITION,
    "catcher": CATCHER,
    "starter_catcher_pair": SP_CATCHER_PAIR,
    "matchup_vectors": MATCHUP_VECTORS,
    "weather_air": WEATHER_AIR,
    "umpire": UMPIRE,
    "park_handedness": PARK_HANDEDNESS,
    "travel_body_clock": TRAVEL_BODY_CLOCK,
    "roof_state": ROOF_STATE,
}

PLANNED_FEATURE_COLUMNS = list(dict.fromkeys(c for cols in PLANNED_FEATURE_GROUPS.values() for c in cols))

OBJECT_FEATURES = {"umpire_id"}

RETRACTABLE_ROOF_TEAMS = {"ARI", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR", "TBR"}

TEAM_PARKS = {
    "ARI": (33.4455, -112.0667, -7),
    "ATL": (33.8908, -84.4678, -5),
    "BAL": (39.2840, -76.6217, -5),
    "BOS": (42.3467, -71.0972, -5),
    "CHC": (41.9484, -87.6553, -6),
    "CHW": (41.8300, -87.6338, -6),
    "CIN": (39.0979, -84.5082, -5),
    "CLE": (41.4962, -81.6852, -5),
    "COL": (39.7561, -104.9942, -7),
    "DET": (42.3390, -83.0485, -5),
    "HOU": (29.7573, -95.3555, -6),
    "KCR": (39.0517, -94.4803, -6),
    "LAA": (33.8003, -117.8827, -8),
    "LAD": (34.0739, -118.2400, -8),
    "MIA": (25.7781, -80.2197, -5),
    "MIL": (43.0280, -87.9712, -6),
    "MIN": (44.9817, -93.2776, -6),
    "NYM": (40.7571, -73.8458, -5),
    "NYY": (40.8296, -73.9262, -5),
    "ATH": (38.5804, -121.5132, -8),
    "OAK": (37.7516, -122.2005, -8),
    "PHI": (39.9061, -75.1665, -5),
    "PIT": (40.4469, -80.0057, -5),
    "SDP": (32.7073, -117.1566, -8),
    "SEA": (47.5914, -122.3325, -8),
    "SFG": (37.7786, -122.3893, -8),
    "STL": (38.6226, -90.1928, -6),
    "TBR": (27.7682, -82.6534, -5),
    "TEX": (32.7473, -97.0842, -6),
    "TOR": (43.6414, -79.3894, -5),
    "WSN": (38.8730, -77.0074, -5),
}


@dataclass(frozen=True)
class FeatureStatus:
    feature: str
    group: str
    status: str
    source: str
    notes: str


def _gcol(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PLANNED_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="object" if col in OBJECT_FEATURES else "Float64")
    return out


def _haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def _derive_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    temp = out["temp_c_game_time"]
    rh = _gcol(out, "relative_humidity_game_time") / 100.0
    pressure_pa = out["surface_pressure_hpa_game_time"] * 100.0
    sat_vapor_pa = 610.94 * np.exp((17.625 * temp) / (temp + 243.04))
    vapor_pa = rh * sat_vapor_pa
    dry_pa = pressure_pa - vapor_pa
    out["air_density_game_time"] = (dry_pa / (287.05 * (temp + 273.15))) + (vapor_pa / (461.495 * (temp + 273.15)))
    out["density_altitude_game_time"] = (1.225 - out["air_density_game_time"]) * 9000.0

    wind = _gcol(out, "wind_speed_kmh_game_time")
    wind_from = np.deg2rad(_gcol(out, "wind_dir_deg_game_time"))
    wind_to = wind_from + math.pi
    center = np.deg2rad(_gcol(out, "venue_azimuth_angle"))
    out["wind_out_to_center_kmh"] = wind * np.cos(wind_to - center)
    out["wind_out_to_left_kmh"] = wind * np.cos(wind_to - (center - math.radians(35)))
    out["wind_out_to_right_kmh"] = wind * np.cos(wind_to - (center + math.radians(35)))
    out["weather_hr_boost_factor"] = (
        ((temp - 20.0) / 30.0).fillna(0.0)
        + (out["wind_out_to_center_kmh"] / 50.0).fillna(0.0)
        + (_gcol(out, "park_factor").fillna(1.0) - 1.0)
    )
    out["weather_run_env_factor"] = out["weather_hr_boost_factor"] + (_gcol(out, "park_factor").fillna(1.0) - 1.0)
    has_weather = temp.notna() & _gcol(out, "wind_speed_kmh_game_time").notna()
    out.loc[~has_weather, ["weather_hr_boost_factor", "weather_run_env_factor"]] = np.nan
    return out


def _add_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["travel_miles"] = np.nan
    out["travel_tz_shift"] = np.nan
    out["getaway_game_flag"] = 0.0

    if not {"home_team", "away_team", "game_date"}.issubset(out.columns):
        return out

    work = out.sort_values(["season", "game_date", "game_pk" if "game_pk" in out.columns else "game_id"]).copy()
    previous: dict[tuple[int, str], tuple[str, pd.Timestamp]] = {}

    for idx, row in work.iterrows():
        season = int(row["season"]) if pd.notna(row.get("season")) else int(pd.Timestamp(row["game_date"]).year)
        game_date = pd.Timestamp(row["game_date"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        current_park = home
        team_travel: dict[str, float] = {}
        team_tz_shift: dict[str, float] = {}

        for team in (home, away):
            prev = previous.get((season, team))
            if prev and prev[0] in TEAM_PARKS and current_park in TEAM_PARKS:
                prev_park, prev_date = prev
                team_travel[team] = _haversine_miles(TEAM_PARKS[prev_park][:2], TEAM_PARKS[current_park][:2])
                team_tz_shift[team] = float(TEAM_PARKS[current_park][2] - TEAM_PARKS[prev_park][2])
                if (game_date.normalize() - prev_date.normalize()).days <= 1 and team_travel[team] > 100:
                    work.at[idx, "getaway_game_flag"] = 1.0
            else:
                team_travel[team] = 0.0
                team_tz_shift[team] = 0.0

        # Positive value means away team traveled more than home team.
        work.at[idx, "travel_miles"] = team_travel.get(away, 0.0) - team_travel.get(home, 0.0)
        work.at[idx, "travel_tz_shift"] = team_tz_shift.get(away, 0.0) - team_tz_shift.get(home, 0.0)

        previous[(season, home)] = (current_park, game_date)
        previous[(season, away)] = (current_park, game_date)

    out.loc[work.index, ["travel_miles", "travel_tz_shift", "getaway_game_flag"]] = work[
        ["travel_miles", "travel_tz_shift", "getaway_game_flag"]
    ]
    return out


def _add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in ("home", "away"):
        if f"{side}_sp_times_through_order_avg" in out.columns:
            out[f"{side}_sp_times_through_order_avg"] = _gcol(out, f"{side}_sp_times_through_order_avg").fillna(
                _gcol(out, f"{side}_sp_bf_l3") / 9.0
            )
        if f"{side}_sp_ttto_penalty_expected" in out.columns:
            out[f"{side}_sp_ttto_penalty_expected"] = _gcol(out, f"{side}_sp_ttto_penalty_expected").fillna(
                _gcol(out, f"{side}_sp_tto3_penalty")
            )
        if f"{side}_bp_projected_first_outs_quality" in out.columns:
            out[f"{side}_bp_projected_first_outs_quality"] = _gcol(out, f"{side}_bp_projected_first_outs_quality").fillna(
                pd.concat(
                    [
                        _gcol(out, f"{side}_closer_xwoba_allowed_30d"),
                        _gcol(out, f"{side}_setup_xwoba_allowed_30d"),
                    ],
                    axis=1,
                ).mean(axis=1)
            )

    if {"home_lineup_lhb_count", "home_lineup_rhb_count", "away_sp_platoon_split_vs_lhb", "away_sp_platoon_split_vs_rhb"} <= set(out.columns):
        home_total = (_gcol(out, "home_lineup_lhb_count") + _gcol(out, "home_lineup_rhb_count")).replace(0, np.nan)
        out["home_lineup_vs_away_sp_platoon_fit"] = (
            (_gcol(out, "home_lineup_lhb_count") * _gcol(out, "away_sp_platoon_split_vs_lhb"))
            + (_gcol(out, "home_lineup_rhb_count") * _gcol(out, "away_sp_platoon_split_vs_rhb"))
        ) / home_total
    if {"away_lineup_lhb_count", "away_lineup_rhb_count", "home_sp_platoon_split_vs_lhb", "home_sp_platoon_split_vs_rhb"} <= set(out.columns):
        away_total = (_gcol(out, "away_lineup_lhb_count") + _gcol(out, "away_lineup_rhb_count")).replace(0, np.nan)
        out["away_lineup_vs_home_sp_platoon_fit"] = (
            (_gcol(out, "away_lineup_lhb_count") * _gcol(out, "home_sp_platoon_split_vs_lhb"))
            + (_gcol(out, "away_lineup_rhb_count") * _gcol(out, "home_sp_platoon_split_vs_rhb"))
        ) / away_total

    pairs = {
        "sp_tto_avg_DIFF": ("home_sp_times_through_order_avg", "away_sp_times_through_order_avg"),
        "sp_tto3_penalty_DIFF": ("away_sp_tto3_penalty", "home_sp_tto3_penalty"),
        "sp_third_minus_first_woba_DIFF": ("away_sp_third_time_woba", "home_sp_third_time_woba"),
        "sp_platoon_split_DIFF": ("away_sp_platoon_split_vs_lhb", "home_sp_platoon_split_vs_lhb"),
        "bp_pitches_2d_DIFF": ("away_bp_pitches_2d", "home_bp_pitches_2d"),
        "bp_freshness_DIFF": ("home_bp_freshness_score", "away_bp_freshness_score"),
        "closer_freshness_DIFF": ("home_closer_available_flag", "away_closer_available_flag"),
        "setup_freshness_DIFF": ("home_setup_available_flag", "away_setup_available_flag"),
        "lineup_quality_DIFF": ("home_lineup_avg_wrc_plus", "away_lineup_avg_wrc_plus"),
        "lineup_contact_quality_DIFF": ("home_lineup_avg_xwoba", "away_lineup_avg_xwoba"),
        "lineup_recent_form_DIFF": ("home_lineup_recent_xwoba_14d", "away_lineup_recent_xwoba_14d"),
        "lineup_handedness_balance_DIFF": ("home_lineup_lhb_count", "away_lineup_lhb_count"),
        "catcher_framing_DIFF": ("home_catcher_framing_runs", "away_catcher_framing_runs"),
        "catcher_throwing_DIFF": ("home_catcher_caught_stealing_rate", "away_catcher_caught_stealing_rate"),
        "matchup_xwoba_DIFF": ("home_lineup_vs_away_sp_xwoba", "away_lineup_vs_home_sp_xwoba"),
        "pitch_mix_fit_DIFF": ("home_lineup_vs_away_sp_pitch_mix_fit", "away_lineup_vs_home_sp_pitch_mix_fit"),
        "platoon_fit_DIFF": ("home_lineup_vs_away_sp_platoon_fit", "away_lineup_vs_home_sp_platoon_fit"),
    }
    for dest, (home_col, away_col) in pairs.items():
        out[dest] = _gcol(out, home_col) - _gcol(out, away_col)
    if "lineup_quality_DIFF" in out.columns:
        out["lineup_quality_DIFF"] = _gcol(out, "lineup_quality_DIFF").fillna(
            _gcol(out, "home_lineup_avg_xwoba") - _gcol(out, "away_lineup_avg_xwoba")
        )
    return out


def add_planned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all planned feature columns and deterministic internal features."""
    out = _ensure_schema(df)
    out = _add_travel_features(out)
    out = _add_diff_features(out)
    return out


def derive_after_real_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Derive composites after real source columns have been merged."""
    out = _ensure_schema(df)
    out = _derive_weather_features(out)
    out = _zero_fill_pitch_mix(out)
    out = _add_diff_features(out)
    return out


def _zero_fill_pitch_mix(df: pd.DataFrame) -> pd.DataFrame:
    """For pitchers with any pitch-mix data, fill NaN usage rates with 0."""
    out = df
    for side in ("home", "away"):
        cols = [c for c in out.columns if c.startswith(f"{side}_sp_pitch_mix_") and not c.endswith("_fit")]
        if not cols:
            continue
        has_any = out[cols].notna().any(axis=1)
        out.loc[has_any, cols] = out.loc[has_any, cols].fillna(0.0)
    return out


def planned_feature_status(df: pd.DataFrame) -> list[dict]:
    """Return coverage/status metadata for planned features."""
    internal_features = set(TRAVEL_BODY_CLOCK)
    rows: list[dict] = []
    for group, cols in PLANNED_FEATURE_GROUPS.items():
        for col in cols:
            non_null = float(df[col].notna().mean()) if col in df.columns and len(df) else 0.0
            status = "filled_real" if non_null > 0 else "schema_only"
            if col in internal_features and non_null > 0:
                status = "filled_internal"
            if status == "filled_internal":
                source = "schedule_internal"
            elif status == "filled_real":
                source = "real_source_cache"
            else:
                source = "external_source_pending"
            rows.append(
                {
                    "feature": col,
                    "group": group,
                    "status": status,
                    "coverage": round(non_null, 6),
                    "source": source,
                }
            )
    return rows


def selectable_sandbox_features(df: pd.DataFrame, min_coverage: float = 0.6) -> list[str]:
    """Numeric planned features with enough coverage for sandbox model trials."""
    selected: list[str] = []
    for col in PLANNED_FEATURE_COLUMNS:
        if col in OBJECT_FEATURES or col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if float(values.notna().mean()) < min_coverage:
            continue
        if values.nunique(dropna=True) < 2:
            continue
        selected.append(col)
    return selected
