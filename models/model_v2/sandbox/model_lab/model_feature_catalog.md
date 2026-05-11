# Model Feature Catalog

Reference catalog for current and planned model features.

Purpose:
- keep one canonical list of model inputs
- record source of truth for each feature
- define formulas for derived features
- explain why each feature exists

Notation:
- `home_/away_` means one feature for each side
- `*_DIFF` means `home_value - away_value` unless noted otherwise
- lineup-vs-starter matchup features may flip sign convention where noted
- `current` means already implemented in repo
- `new` means planned

## Source Systems

- Odds: Odds API / cached odds rows
- MLB schedule + boxscore + live feed: MLB Stats API
- Player and team Statcast: Baseball Savant
- Team season leaderboards: FanGraphs
- Weather: Open-Meteo hourly / archive APIs
- Internal: derived from `games` table history, static maps, or feature-engineering code

## Current Features

### Market / Odds

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `market_implied_prob` | current | Odds API | consensus implied home win probability from closing/home price | market baseline; strongest prior |
| `open_home_implied` | current | Odds API | implied home win probability from opening line | captures opener vs close drift |
| `line_move_delta` | current | Odds API | `market_implied_prob - open_home_implied` | sharp/late move signal |
| `sharp_move_flag` | current | Odds API | `1 if abs(line_move_delta) >= threshold else 0` | discrete sharp action proxy |
| `total_move_delta` | current | Odds API | `close_total - open_total` | game environment / information move |

### Starting Pitcher Core

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `sp_fip_DIFF` | current | MLB Stats API + internal | `away_fip - home_fip` | cleaner talent signal than ERA |
| `sp_era_DIFF` | current | MLB Stats API + internal | `away_era - home_era` | captures runs actually allowed |
| `sp_k9_DIFF` | current | MLB Stats API + internal | `home_k9 - away_k9` | strikeout edge |
| `sp_bb9_DIFF` | current | MLB Stats API + internal | `away_bb9 - home_bb9` | command edge |
| `sp_whip_DIFF` | current | MLB Stats API + internal | `away_whip - home_whip` | baserunner suppression |
| `rolling_k9_DIFF` | current | MLB Stats API game logs | `home_rolling_k9 - away_rolling_k9` | recent form |
| `rolling_era_DIFF` | current | MLB Stats API game logs | `away_rolling_era - home_rolling_era` | recent run prevention |
| `rolling_whip_DIFF` | current | MLB Stats API game logs | `away_rolling_whip - home_rolling_whip` | recent traffic allowed |
| `rookie_DIFF` | current | MLB Stats API game history | `away_is_rookie - home_is_rookie` | inexperience proxy |
| `sp_rest_DIFF` | current | MLB Stats API game history | `home_sp_rest_days - away_sp_rest_days` | fatigue / routine |

### Team Offense and Team Pitching

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `wrc_plus_DIFF`, `woba_DIFF`, `avg_DIFF`, `obp_DIFF`, `slg_DIFF`, `k_pct_DIFF`, `bb_pct_DIFF`, `owar_DIFF`, `war_DIFF` | current | FanGraphs | home minus away except strikeout-style fields use positive = home edge | team hitting quality |
| `k_per_9_DIFF`, `bb_per_9_DIFF`, `hr_per_9_DIFF`, `era_DIFF`, `fip_DIFF` | current | FanGraphs | sign chosen so positive = home edge | team pitching quality |

### Handedness

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_pitcher_is_lefty`, `away_pitcher_is_lefty` | current | MLB Stats API | binary from starter handedness | platoon context |
| `pitcher_handedness_diff` | current | MLB Stats API + internal | `home_pitcher_is_lefty - away_pitcher_is_lefty` | simplified handedness contrast |

### Rolling and Season Team Form

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `win_pct_W_DIFF`, `run_diff_avg_W_DIFF`, `run_diff_std_W_DIFF`, `runs_scored_avg_W_DIFF`, `runs_allowed_avg_W_DIFF` | current | Internal from `games` | home rolling window minus away | recent team form |
| `season_win_pct_DIFF`, `season_run_diff_avg_DIFF` | current | Internal from `games` | home season-to-date minus away | season baseline |
| `streak_DIFF` | current | Internal from `games` | `home_streak - away_streak` | momentum / confidence proxy |

### Weather / Venue

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `temp_c`, `wind_speed_kmh` | current | Open-Meteo daily | daily weather value for park/date | rough run environment |
| `wind_dir_sin`, `wind_dir_cos` | current | Open-Meteo daily + internal | circular encoding of wind direction | avoids angle discontinuity |
| `park_factor` | current | Static map | normalized park scalar | venue run-scoring bias |
| `is_night_game` | current | MLB schedule | binary from start time | environment / rest split |

### Schedule / Calendar / Era

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_rest_days`, `away_rest_days`, `rest_days_DIFF`, `is_series_finale`, `early_season_flag` | current | Internal from schedule history | direct schedule-derived fields | fatigue / context |
| `month`, `days_since_asb`, `day_of_week`, `post_rule_change`, `post_dh_era`, `covid_era` | current | Internal date logic | deterministic from date/season | calendar and regime shifts |

### Pythagorean / Luck / Interaction

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `luck_DIFF` | current | Internal | `(home_win_pct - home_pythag) - (away_win_pct - away_pythag)` | regression candidate |
| `pythagorean_DIFF` | current | Internal | `home_pythag - away_pythag` | expected talent from runs |
| `pythagorean_short_DIFF` | current | Internal | short-window pythagorean home minus away | short-term underlying form |
| `luck_x_momentum` | current | Internal | `luck_DIFF * momentum_DIFF` | regression plus trend |
| `park_x_pythagorean` | current | Internal | `park_factor * pythagorean_DIFF` | venue-amplified edge |
| `sharp_x_fip` | current | Internal | `sharp_move_flag * sp_fip_DIFF` | market move aligned with pitcher edge |
| `momentum_DIFF` | current | Internal | short-form minus long-form differential | acceleration / deceleration |

## Planned Features

### Starting Pitcher Workload

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_sp_days_rest` | new | MLB Stats API | days since prior start | direct fatigue / routine |
| `home_/away_sp_pitch_count_l3` | new | MLB Stats API game logs | avg or sum of last 3 start pitch counts | recent workload |
| `home_/away_sp_bf_l3` | new | MLB Stats API game logs | avg or sum batters faced last 3 starts | workload and leash proxy |
| `home_/away_sp_ip_l3` | new | MLB Stats API game logs | avg innings pitched last 3 starts | endurance / manager trust |

### Starting Pitcher Statcast Shape

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_sp_fastball_velo`, `fastball_spin`, `extension`, `active_spin` | new | Baseball Savant | season-to-date or rolling metric | raw stuff quality |
| `home_/away_sp_whiff_rate`, `chase_rate`, `cs_rate`, `csw_rate` | new | Baseball Savant | rate metrics from pitch outcomes | swing-and-miss / deception |
| `home_/away_sp_groundball_rate` | new | Baseball Savant | GB / balls in play | contact profile |
| `home_/away_sp_barrel_allowed_rate`, `hard_hit_allowed_rate`, `avg_ev_allowed` | new | Baseball Savant | allowed-contact quality | damage suppression |
| `home_/away_sp_xwoba_allowed`, `xera` | new | Baseball Savant | expected run-prevention metrics | defense-neutral quality |

### Starting Pitcher Pitch Mix

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_sp_pitch_mix_ff`, `si`, `fc`, `sl`, `st`, `cu`, `ch` | new | Baseball Savant | usage share by pitch type; normalize to 1.0 | matchup fit vs lineup skill by pitch type |

### Starting Pitcher TTTO and Platoon Splits

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_sp_times_through_order_avg` | new | MLB live feed / Savant | average completed trips through lineup per start | leash / stamina |
| `home_/away_sp_tto3_penalty` | new | MLB live feed / Savant | `third_time_woba - first_time_woba` | late-start degradation |
| `home_/away_sp_first_time_woba`, `second_time_woba`, `third_time_woba` | new | MLB live feed / Savant | wOBA allowed by trip number | direct TTTO profile |
| `sp_tto_avg_DIFF`, `sp_tto3_penalty_DIFF`, `sp_third_minus_first_woba_DIFF` | new | Internal | home minus away derived values | compare starter degradation |
| `home_/away_sp_platoon_split_vs_lhb`, `vs_rhb` | new | Baseball Savant / MLB splits | allowed metric vs LHB and RHB | lineup-handedness interaction |
| `sp_platoon_split_DIFF` | new | Internal | compare home and away split severity | platoon mismatch signal |
| `home_/away_sp_ttto_penalty_expected` | new | Internal | expected TTTO cost from historical usage + lineup depth | pregame late-inning starter risk |

### Bullpen Team Aggregate Quality

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_bp_era_30d`, `fip_30d`, `xfip_30d`, `whiff_rate_30d`, `k_minus_bb_30d`, `xwoba_allowed_30d` | new | MLB Stats API + Savant | aggregate over relievers last 30 days | bullpen quality baseline |
| `bp_quality_DIFF` | new | Internal | home aggregate quality minus away | compare pen strength |

### Bullpen Team Aggregate Freshness

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_bp_pitches_1d`, `pitches_2d`, `pitches_3d` | new | MLB boxscores | total reliever pitches over trailing windows | fatigue |
| `home_/away_bp_bf_2d`, `ip_2d`, `high_leverage_pitches_2d` | new | MLB boxscores | total bullpen workload, with leverage weighting where available | better than raw pitches alone |
| `home_/away_closer_used_yesterday`, `home_/away_top3_rp_used_2d` | new | MLB boxscores | binary or count flags | late-inning availability |
| `home_/away_bp_lefty_available`, `bp_righty_available` | new | Internal from reliever usage | availability estimate by hand | matchup flexibility |
| `home_/away_bp_freshness_score` | new | Internal | weighted score from recency, leverage, role availability | single bullpen fatigue summary |
| `bp_pitches_2d_DIFF`, `bp_freshness_DIFF` | new | Internal | home minus away | relative freshness |
| `home_/away_bp_fatigue_expected` | new | Internal | forecast fatigue cost based on prior workload | pregame bullpen drag |
| `home_/away_bp_projected_first_outs_quality` | new | Internal | inferred quality of first likely bullpen outs | captures bridge-to-late-innings strength |

### Bullpen High-Leverage Individual Arms

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_closer_xwoba_allowed_30d`, `closer_k_minus_bb_30d`, `closer_whiff_rate_30d` | new | Savant + MLB role inference | trailing 30-day metric for inferred closer | high-leverage outs matter more than average RP |
| `home_/away_closer_pitches_yesterday`, `closer_pitches_2d`, `closer_available_flag` | new | MLB boxscores + role inference | workload and availability for inferred closer | key late-inning availability |
| `home_/away_setup_xwoba_allowed_30d`, `setup_k_minus_bb_30d`, `setup_whiff_rate_30d` | new | Savant + role inference | trailing 30-day setup man quality | likely 8th inning arm quality |
| `home_/away_setup_pitches_yesterday`, `setup_pitches_2d`, `setup_available_flag` | new | MLB boxscores + role inference | workload and availability | bridge inning risk |
| `home_/away_top_lhrp_xwoba_allowed_30d`, `top_lhrp_pitches_yesterday` | new | Savant + role inference | top recent lefty reliever quality and freshness | lineup handedness matchup |
| `home_/away_top_rhrp_xwoba_allowed_30d`, `top_rhrp_pitches_yesterday` | new | Savant + role inference | top recent righty reliever quality and freshness | lineup handedness matchup |
| `closer_quality_DIFF`, `closer_freshness_DIFF`, `setup_quality_DIFF`, `setup_freshness_DIFF`, `top_lhrp_quality_DIFF`, `top_rhrp_quality_DIFF` | new | Internal | home minus away | compare most important relief arms |

### Confirmed Lineup Aggregate Skill

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_lineup_avg_woba`, `avg_wrc_plus`, `avg_xwoba`, `avg_xba`, `avg_slg`, `avg_obp`, `avg_iso` | new | MLB lineup + Savant/FanGraphs | average across confirmed 9 hitters | upgrade from team-average to actual lineup |
| `home_/away_lineup_avg_k_rate`, `avg_bb_rate`, `hard_hit_rate`, `barrel_rate`, `avg_ev`, `avg_la`, `sweet_spot_rate`, `sprint_speed` | new | MLB lineup + Savant | average across confirmed 9 hitters | contact, discipline, power, speed |
| `lineup_quality_DIFF`, `lineup_contact_quality_DIFF` | new | Internal | home minus away summary values | clean lineup comparison |

### Confirmed Lineup Recent Form

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_lineup_recent_xwoba_14d`, `recent_hard_hit_14d`, `recent_barrel_14d`, `recent_k_rate_14d` | new | MLB lineup + Savant | average across confirmed hitters using trailing 14-day splits | short-term contact-quality form |
| `home_/away_lineup_recent_xwoba_30d`, `recent_hard_hit_30d`, `recent_barrel_30d`, `recent_k_rate_30d` | new | MLB lineup + Savant | average across confirmed hitters using trailing 30-day splits | stabilizer for 14-day noise |
| `lineup_recent_form_DIFF` | new | Internal | home minus away | relative hot/cold bats |
| `home_/away_contact_quality_recent` | new | Internal/Savant | composite of recent xwOBA, hard-hit, barrel | concise recent form score |

### Confirmed Lineup Composition and Depth

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_lineup_lhb_count`, `rhb_count` | new | MLB lineup + MLB player metadata | count by batting handedness | platoon structure |
| `lineup_handedness_balance_DIFF` | new | Internal | compare hand mix | starter split interaction |
| `home_/away_lineup_top4_xwoba`, `bottom5_xwoba` | new | MLB lineup + Savant | average xwOBA for top 4 and bottom 5 lineup spots | star power vs depth |
| `home_/away_lineup_depth_score` | new | Internal | weighted lineup-slot quality score | full-order strength |

### Catcher Features

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_catcher_framing_runs`, `strike_rate`, `shadow_strike_rate`, `framing_runs_1000` | new | MLB lineup + Savant catcher leaderboards | season or rolling catcher framing metrics | catcher defense impacts called strikes |
| `catcher_framing_DIFF` | new | Internal | home minus away | framing edge |
| `home_/away_catcher_blocking_runs`, `blocks_above_avg` | new | Savant | passed-ball / blocking quality | run-prevention detail |
| `home_/away_catcher_pop_time`, `arm_strength`, `caught_stealing_rate` | new | Savant | throwing metrics | running-game control |
| `catcher_throwing_DIFF` | new | Internal | home minus away | arm / steal deterrence edge |

### Starter-Catcher Pair Effects

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_/away_sp_catcher_pair_framing`, `sp_catcher_pair_cs_rate` | new | Internal from historical joins | prior performance of announced starter with announced catcher, heavily shrunk | some pairs work materially better together |

### Lineup-vs-Starter Matchup Vectors

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `home_lineup_vs_away_sp_xwoba`, `away_lineup_vs_home_sp_xwoba` | new | MLB lineup + Savant | lineup batter expected performance aggregated against starter profile | direct matchup quality |
| `matchup_xwoba_DIFF` | new | Internal | `home_lineup_vs_away_sp_xwoba - away_lineup_vs_home_sp_xwoba` | compare matchup edge |
| `home_lineup_vs_away_sp_pitch_mix_fit`, `away_lineup_vs_home_sp_pitch_mix_fit` | new | Internal from lineup + Savant | weighted sum of lineup performance by pitch type using starter pitch usage weights | explicit pitch-mix fit |
| `pitch_mix_fit_DIFF` | new | Internal | home minus away | compare pitch-type matchup edge |
| `home_lineup_vs_away_sp_platoon_fit`, `away_lineup_vs_home_sp_platoon_fit` | new | Internal | lineup handedness mix weighted by starter platoon splits | left/right matchup |
| `platoon_fit_DIFF` | new | Internal | home minus away | compare platoon edge |
| `home_lineup_whiff_risk_vs_away_sp`, `away_lineup_whiff_risk_vs_home_sp` | new | Internal from Savant | lineup whiff tendency weighted by starter bat-missing skill | strikeout-risk profile |
| `home_lineup_gb_fit_vs_away_sp`, `away_lineup_gb_fit_vs_home_sp` | new | Internal from Savant | lineup launch/contact profile against starter GB profile | batted-ball fit |
| `home_lineup_power_vs_away_sp`, `away_lineup_power_vs_home_sp` | new | Internal from Savant | lineup barrel/EV/pull power versus starter contact-allowed profile | slugging matchup |
| `home_lineup_vs_away_sp_pitch_type_ff`, `pitch_type_sl`, `pitch_type_ch`, and away equivalents | new | Internal from Savant | selected pitch-type matchup subscores | interpretable matchup components |

### Weather and Air Density Upgrade

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `temp_c_game_time`, `relative_humidity_game_time`, `dew_point_c_game_time`, `surface_pressure_hpa_game_time` | new | Open-Meteo hourly | nearest game-time hourly reading or forecast | actual game-time environment |
| `air_density_game_time` | new | Internal from weather inputs | air-density equation using temp, humidity, pressure | drag / carry proxy |
| `density_altitude_game_time` | new | Internal from weather + park elevation | density altitude formula | home-run environment proxy |
| `wind_speed_kmh_game_time`, `wind_dir_deg_game_time` | new | Open-Meteo hourly | hourly weather | directional wind strength |
| `wind_out_to_center_kmh`, `wind_out_to_left_kmh`, `wind_out_to_right_kmh` | new | Internal from wind + stadium orientation | projected wind component toward each field sector | better than raw direction |
| `weather_hr_boost_factor` | new | Internal | composite of density, temp, wind, park | home-run carry signal |
| `weather_run_env_factor` | new | Internal | broader run-scoring environment composite | total-scoring context |

### Umpire

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `umpire_id` | new | MLB Stats API officials | raw plate ump identifier | join key |
| `umpire_zone_size`, `umpire_zone_tightness`, `umpire_called_strike_rate`, `umpire_consistency` | new | MLB officials + Statcast/umpire cache | historical ump metrics | cheap strike-zone context; interacts with framing |

### Park-Handedness Splits

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `park_hr_factor_lhb`, `park_hr_factor_rhb`, `park_run_factor_lhb`, `park_run_factor_rhb` | new | Static park table from Savant/FanGraphs | handedness-specific park values | more precise than one scalar park factor |

### Travel and Body Clock

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `travel_miles` | new | Internal from schedule + coordinates | miles from previous city to current city | travel fatigue |
| `travel_tz_shift` | new | Internal from schedule + timezone map | time-zone delta from prior game city | circadian effect |
| `getaway_game_flag` | new | Internal from schedule | prior game ended/travel occurred under getaway pattern | fatigue / bullpen usage context |

### Roof State

| Feature(s) | Status | Source | Formula | Why |
|---|---|---|---|---|
| `roof_closed_flag`, `roof_possible_flag` | new | MLB feed + static stadium metadata | binary roof state / retractable-park flag | weather relevance changes with roof |

## Formula Notes

### Sign Convention

Most `_DIFF` features use:

```text
home_value - away_value
```

Exception:
- some current pitcher/team features are already defined in code with sign chosen so positive means home edge, even if that uses `away - home` internally for “bad” stats like ERA/FIP/WHIP

Rule for new work:
- preserve “positive = home edge” in final training column

### Pitch Mix Fit

Intended definition:

```text
lineup_vs_starter_pitch_mix_fit
= sum over pitch types (
    starter_pitch_usage[pitch_type]
    * lineup_expected_value_vs_pitch_type[pitch_type]
)
```

Where lineup expected value may be:
- average xwOBA vs pitch type across confirmed hitters
- or weighted batter run value / 100 vs pitch type

### Platoon Fit

Intended definition:

```text
lineup_vs_starter_platoon_fit
= weighted lineup handedness exposure against starter split severity
```

Example:
- if starter much weaker vs LHB and lineup starts 6 LHB, fit score increases

### Bullpen Freshness Score

Intended definition:

```text
bp_freshness_score
= w1 * recent_pitch_volume
+ w2 * recent_high_leverage_volume
+ w3 * closer_availability
+ w4 * setup_availability
+ w5 * hand-specific_availability
```

Higher score should mean fresher / more available bullpen.

### Weather Composite Scores

Intended definitions:

```text
weather_hr_boost_factor
= f(air_density, temp, wind_out_components, park_hr_factor)

weather_run_env_factor
= g(weather_hr_boost_factor, humidity, wind, roof_state, park_run_factor)
```

### Shrinkage Needed

Use shrinkage or priors for:
- starter-catcher pair metrics
- 14-day lineup recent-form features
- pitch-type matchup splits on small samples
- umpire metrics with limited called-pitch samples

## Modeling Notes

- This catalog is a superset, not a promise every feature ships into every model.
- Logistic regression will likely need feature pruning, regularization, or grouped selection.
- Tree models can tolerate more breadth but still benefit from pruning noisy or redundant columns.
- Any feature unavailable at `T-5min` must be excluded from training or replaced with a pregame-estimable proxy to avoid leakage.

## Build Order

Recommended implementation order:

1. hourly weather / air density
2. bullpen team freshness and quality
3. confirmed lineup IDs and lineup aggregate skill
4. catcher identity plus framing / throwing
5. lineup-vs-starter matchup vectors
6. starter Statcast shape and TTTO
7. umpire, park-handedness, travel, roof
8. high-leverage individual reliever layer
9. starter-catcher pair and projected bullpen composites
