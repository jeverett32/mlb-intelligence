#!/usr/bin/env python3
"""
migrate_to_postgres.py
Imports master_mlb.csv and mlb_2026.csv into the games table in PostgreSQL.
Run from project root: uv run migrate_to_postgres.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DB = dict(
    host=os.environ["DB_HOST"],
    port=int(os.environ.get("DB_PORT", 5432)),
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
)

BOOL_COLS = {"home_win"}

DIRECT_COLS = {
    "game_pk", "game_date", "season", "game_time_et",
    "home_team", "away_team", "home_score", "away_score", "home_win",
    "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml",
    "home_implied_prob", "away_implied_prob", "over_under", "odds_source",
    "home_starter_id", "away_starter_id",
    "home_starter_era", "home_starter_whip", "home_starter_k9",
    "home_starter_bb9", "home_starter_fip", "home_starter_hand",
    "away_starter_era", "away_starter_whip", "away_starter_k9",
    "away_starter_bb9", "away_starter_fip", "away_starter_hand",
    "temp_c", "wind_speed_kph", "wind_dir_deg", "precip_mm",
    "home_wrc_plus", "home_woba", "home_avg", "home_obp", "home_slg",
    "home_era", "home_fip", "home_k9", "home_bb9",
    "away_wrc_plus", "away_woba", "away_avg", "away_obp", "away_slg",
    "away_era", "away_fip", "away_k9", "away_bb9",
}


REAL_COLS = {
    "home_score", "away_score",
    "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml",
    "home_implied_prob", "away_implied_prob", "over_under",
    "home_starter_era", "home_starter_whip", "home_starter_k9",
    "home_starter_bb9", "home_starter_fip",
    "away_starter_era", "away_starter_whip", "away_starter_k9",
    "away_starter_bb9", "away_starter_fip",
    "temp_c", "wind_speed_kph", "wind_dir_deg", "precip_mm",
    "home_wrc_plus", "home_woba", "home_avg", "home_obp", "home_slg",
    "home_era", "home_fip", "home_k9", "home_bb9",
    "away_wrc_plus", "away_woba", "away_avg", "away_obp", "away_slg",
    "away_era", "away_fip", "away_k9", "away_bb9",
}


def coerce(val, col):
    try:
        is_na = pd.isna(val)
    except Exception:
        is_na = False
    if is_na:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if col in BOOL_COLS:
        try:
            return bool(int(val))
        except (ValueError, TypeError):
            return None
    if col in REAL_COLS:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return val


def df_to_rows(df):
    rows = []
    for _, row in df.iterrows():
        direct = {}
        extra = {}
        for col in df.columns:
            val = coerce(row[col], col)
            if col in DIRECT_COLS:
                direct[col] = val
            elif val is not None:
                extra[col] = val
        direct["extra"] = json.dumps(extra) if extra else None
        rows.append(direct)
    return rows


def insert_rows(conn, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    values = [[r.get(c) for c in cols] for r in rows]
    col_list = ", ".join(cols)
    update_list = ", ".join(
        c + " = EXCLUDED." + c for c in cols if c != "game_pk"
    )
    sql = (
        "INSERT INTO games (" + col_list + ") VALUES %s "
        "ON CONFLICT (game_pk) DO UPDATE SET "
        + update_list
        + ", updated_at = NOW()"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()


def main():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB)
    print("Connected.")
    data_dir = Path(__file__).parent / "data"
    for fname in ["master_mlb.csv", "mlb_2026.csv"]:
        path = data_dir / fname
        if not path.exists():
            print(fname + " not found, skipping.")
            continue
        print("Importing " + fname + "...")
        df = pd.read_csv(path, low_memory=False)
        print("  " + str(len(df)) + " rows loaded")
        rows = df_to_rows(df)
        for i in range(0, len(rows), 5000):
            insert_rows(conn, rows[i : i + 5000])
            done = min(i + 5000, len(rows))
            print("  " + str(done) + "/" + str(len(rows)) + " inserted")
        print(fname + " done.")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season"
        )
        print("\nGames per season:")
        for r in cur.fetchall():
            print("  " + str(r[0]) + ": " + str(r[1]) + " games")
    conn.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    main()
