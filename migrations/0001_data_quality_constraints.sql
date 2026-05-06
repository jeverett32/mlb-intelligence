-- Auto-generated from data_quality/contracts.py
-- Regenerate with: uv run scripts/generate_constraint_migration.py
--
-- Constraints are added NOT VALID so existing rows do not block apply.
-- Run audit + repair, then promote each constraint with:
--   ALTER TABLE games VALIDATE CONSTRAINT <name>;

BEGIN;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_season_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_season_range_chk
    CHECK (season IS NULL OR (season >= 1900 AND season <= 2100)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_score_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_score_range_chk
    CHECK (home_score IS NULL OR (home_score >= 0 AND home_score <= 50)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_score_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_score_range_chk
    CHECK (away_score IS NULL OR (away_score >= 0 AND away_score <= 50)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_open_home_ml_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_open_home_ml_range_chk
    CHECK (open_home_ml IS NULL OR (open_home_ml >= -10000 AND open_home_ml <= 10000)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_open_away_ml_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_open_away_ml_range_chk
    CHECK (open_away_ml IS NULL OR (open_away_ml >= -10000 AND open_away_ml <= 10000)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_close_home_ml_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_close_home_ml_range_chk
    CHECK (close_home_ml IS NULL OR (close_home_ml >= -10000 AND close_home_ml <= 10000)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_close_away_ml_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_close_away_ml_range_chk
    CHECK (close_away_ml IS NULL OR (close_away_ml >= -10000 AND close_away_ml <= 10000)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_implied_prob_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_implied_prob_range_chk
    CHECK (home_implied_prob IS NULL OR (home_implied_prob >= 0.0 AND home_implied_prob <= 1.0)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_implied_prob_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_implied_prob_range_chk
    CHECK (away_implied_prob IS NULL OR (away_implied_prob >= 0.0 AND away_implied_prob <= 1.0)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_over_under_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_over_under_range_chk
    CHECK (over_under IS NULL OR (over_under >= 0 AND over_under <= 30)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_starter_era_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_starter_era_range_chk
    CHECK (home_starter_era IS NULL OR (home_starter_era >= 0 AND home_starter_era <= 30)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_starter_era_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_starter_era_range_chk
    CHECK (away_starter_era IS NULL OR (away_starter_era >= 0 AND away_starter_era <= 30)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_starter_whip_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_starter_whip_range_chk
    CHECK (home_starter_whip IS NULL OR (home_starter_whip >= 0 AND home_starter_whip <= 5)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_starter_whip_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_starter_whip_range_chk
    CHECK (away_starter_whip IS NULL OR (away_starter_whip >= 0 AND away_starter_whip <= 5)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_starter_k9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_starter_k9_range_chk
    CHECK (home_starter_k9 IS NULL OR (home_starter_k9 >= 0 AND home_starter_k9 <= 20)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_starter_k9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_starter_k9_range_chk
    CHECK (away_starter_k9 IS NULL OR (away_starter_k9 >= 0 AND away_starter_k9 <= 20)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_starter_bb9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_starter_bb9_range_chk
    CHECK (home_starter_bb9 IS NULL OR (home_starter_bb9 >= 0 AND home_starter_bb9 <= 15)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_starter_bb9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_starter_bb9_range_chk
    CHECK (away_starter_bb9 IS NULL OR (away_starter_bb9 >= 0 AND away_starter_bb9 <= 15)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_starter_fip_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_starter_fip_range_chk
    CHECK (home_starter_fip IS NULL OR (home_starter_fip >= 0 AND home_starter_fip <= 15)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_starter_fip_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_starter_fip_range_chk
    CHECK (away_starter_fip IS NULL OR (away_starter_fip >= 0 AND away_starter_fip <= 15)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_temp_c_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_temp_c_range_chk
    CHECK (temp_c IS NULL OR (temp_c >= -30 AND temp_c <= 50)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_wind_speed_kph_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_wind_speed_kph_range_chk
    CHECK (wind_speed_kph IS NULL OR (wind_speed_kph >= 0 AND wind_speed_kph <= 200)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_wind_dir_deg_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_wind_dir_deg_range_chk
    CHECK (wind_dir_deg IS NULL OR (wind_dir_deg >= 0 AND wind_dir_deg <= 360)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_precip_mm_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_precip_mm_range_chk
    CHECK (precip_mm IS NULL OR (precip_mm >= 0 AND precip_mm <= 500)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_wrc_plus_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_wrc_plus_range_chk
    CHECK (home_wrc_plus IS NULL OR (home_wrc_plus >= 10 AND home_wrc_plus <= 300)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_wrc_plus_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_wrc_plus_range_chk
    CHECK (away_wrc_plus IS NULL OR (away_wrc_plus >= 10 AND away_wrc_plus <= 300)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_woba_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_woba_range_chk
    CHECK (home_woba IS NULL OR (home_woba >= 0.15 AND home_woba <= 0.5)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_woba_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_woba_range_chk
    CHECK (away_woba IS NULL OR (away_woba >= 0.15 AND away_woba <= 0.5)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_avg_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_avg_range_chk
    CHECK (home_avg IS NULL OR (home_avg >= 0.1 AND home_avg <= 0.4)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_avg_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_avg_range_chk
    CHECK (away_avg IS NULL OR (away_avg >= 0.1 AND away_avg <= 0.4)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_obp_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_obp_range_chk
    CHECK (home_obp IS NULL OR (home_obp >= 0.15 AND home_obp <= 0.5)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_obp_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_obp_range_chk
    CHECK (away_obp IS NULL OR (away_obp >= 0.15 AND away_obp <= 0.5)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_slg_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_slg_range_chk
    CHECK (home_slg IS NULL OR (home_slg >= 0.15 AND home_slg <= 0.7)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_slg_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_slg_range_chk
    CHECK (away_slg IS NULL OR (away_slg >= 0.15 AND away_slg <= 0.7)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_era_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_era_range_chk
    CHECK (home_era IS NULL OR (home_era >= 0 AND home_era <= 15)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_era_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_era_range_chk
    CHECK (away_era IS NULL OR (away_era >= 0 AND away_era <= 15)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_fip_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_fip_range_chk
    CHECK (home_fip IS NULL OR (home_fip >= 0 AND home_fip <= 10)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_fip_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_fip_range_chk
    CHECK (away_fip IS NULL OR (away_fip >= 0 AND away_fip <= 10)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_k9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_k9_range_chk
    CHECK (home_k9 IS NULL OR (home_k9 >= 0 AND home_k9 <= 20)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_k9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_k9_range_chk
    CHECK (away_k9 IS NULL OR (away_k9 >= 0 AND away_k9 <= 20)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_home_bb9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_home_bb9_range_chk
    CHECK (home_bb9 IS NULL OR (home_bb9 >= 0 AND home_bb9 <= 15)) NOT VALID;

ALTER TABLE games
    DROP CONSTRAINT IF EXISTS games_away_bb9_range_chk;
ALTER TABLE games
    ADD CONSTRAINT games_away_bb9_range_chk
    CHECK (away_bb9 IS NULL OR (away_bb9 >= 0 AND away_bb9 <= 15)) NOT VALID;


COMMIT;
