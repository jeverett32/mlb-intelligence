// Shared feature glossary used across dashboard + admin.
// Exposed as a global so templates can consume without bundling.
window.FEATURE_DICT = {
  // Market
  market_implied_prob: "Closing market-implied home win probability derived from sportsbook moneylines (DraftKings via SBR/Odds API)",
  open_home_implied: "Opening home win probability before line movement",
  line_move_delta: "Closing minus opening implied probability — positive means money moved toward home (sharp signal)",
  sharp_move_flag: "Binary: 1 if |line_move_delta| ≥ 3% (indicates sharp bettor action)",
  total_move_delta: "Over/under line movement from open to close",
  // Pitcher quality
  sp_fip_DIFF: "Starting pitcher FIP difference (away − home). Positive = home SP better",
  sp_era_DIFF: "Starting pitcher ERA difference (away − home)",
  sp_k9_DIFF: "Starting pitcher K/9 difference (home − away)",
  sp_bb9_DIFF: "Starting pitcher BB/9 difference (away − home). Positive = home SP walks fewer",
  sp_whip_DIFF: "Starting pitcher WHIP difference (away − home)",
  rolling_k9_DIFF: "Recent rolling K/9 difference (home − away)",
  rolling_era_DIFF: "Recent rolling ERA difference (away − home)",
  rolling_whip_DIFF: "Recent rolling WHIP difference (away − home)",
  rookie_DIFF: "Rookie flag difference (away − home). Positive = home pitcher more experienced",
  sp_rest_DIFF: "Starting pitcher rest days difference (home − away)",
  // Batting (FanGraphs)
  wrc_plus_DIFF: "Weighted Runs Created Plus difference (home − away). 100 = league average",
  woba_DIFF: "Weighted On-Base Average difference (home − away)",
  avg_DIFF: "Batting average difference (home − away)",
  obp_DIFF: "On-base percentage difference (home − away)",
  slg_DIFF: "Slugging percentage difference (home − away)",
  k_pct_DIFF: "Strikeout rate difference (away − home). Positive = home strikes out less",
  bb_pct_DIFF: "Walk rate difference (home − away)",
  k_per_9_DIFF: "Team K/9 difference (away − home)",
  bb_per_9_DIFF: "Team BB/9 difference (home − away)",
  hr_per_9_DIFF: "HR/9 difference (away − home). Positive = home allows fewer HRs",
  era_DIFF: "Team ERA difference (away − home)",
  fip_DIFF: "Team FIP difference (away − home)",
  owar_DIFF: "Offensive WAR difference (home − away)",
  war_DIFF: "Total WAR difference (home − away)",
  // Handedness
  pitcher_handedness_diff: "Home pitcher is lefty minus away pitcher is lefty",
  home_pitcher_is_lefty: "Binary: 1 if home starting pitcher throws left-handed",
  away_pitcher_is_lefty: "Binary: 1 if away starting pitcher throws left-handed",
  // Rolling form
  win_pct_W_DIFF: "Rolling win% difference over best window (home − away)",
  run_diff_avg_W_DIFF: "Rolling run differential average difference (home − away)",
  run_diff_std_W_DIFF: "Rolling run differential consistency difference (home − away)",
  runs_scored_avg_W_DIFF: "Rolling runs scored average difference (home − away)",
  runs_allowed_avg_W_DIFF: "Rolling runs allowed average difference (home − away)",
  // Season-to-date
  season_win_pct_DIFF: "Season win% difference (home − away)",
  season_run_diff_avg_DIFF: "Season run differential average difference (home − away)",
  // Streak
  streak_DIFF: "Win streak difference (home − away). Positive = home on better streak",
  // Weather / venue
  temp_c: "Game-time temperature in Celsius",
  wind_speed_kmh: "Wind speed in km/h",
  wind_dir_sin: "Sine of wind direction (circular encoding)",
  wind_dir_cos: "Cosine of wind direction (circular encoding)",
  park_factor: "Ballpark run-scoring factor. Above 1 = hitter-friendly",
  is_night_game: "Binary: 1 if game starts after 5 PM local time",
  // Schedule context
  home_rest_days: "Days since home team's last game",
  away_rest_days: "Days since away team's last game",
  rest_days_DIFF: "Rest days difference (home − away)",
  is_series_finale: "Binary: 1 if final game of the series",
  early_season_flag: "Binary: 1 if either team has played fewer than 15 games",
  // Calendar
  month: "Month of the season (1–12)",
  days_since_asb: "Days since All-Star break (0 before/during break)",
  day_of_week: "Day of week (0=Mon, 6=Sun) — captures travel/fatigue patterns",
  post_rule_change: "Binary: 1 if season ≥ 2023 (pitch clock, shift ban, bigger bases)",
  post_dh_era: "Binary: 1 if season ≥ 2022 (universal DH)",
  covid_era: "Binary: 1 if 2020 season (60 games, no fans, unusual stats)",
  // Interactions
  sharp_x_fip: "Interaction: sharp_move_flag × sp_fip_DIFF",
  momentum_DIFF: "Short-term vs long-term win% trend difference between teams",
  // Pythagorean / luck
  luck_DIFF: "Over/underperformance vs expected win% (actual − pythagorean) difference",
  pythagorean_DIFF: "Expected win% from run scoring efficiency difference (home − away)",
  luck_x_momentum: "Interaction: lucky teams trending up may be due for regression",
  park_x_pythagorean: "Interaction: park factor amplifies pythagorean edge",
  pythagorean_short_DIFF: "Short-window pythagorean win% difference (home − away)",
};

