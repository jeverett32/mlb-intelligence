"""
db.py — Shared PostgreSQL helpers for the MLB pipeline.
All scripts import from this module instead of reading/writing CSVs.

Connection is configured via .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Direct columns that map 1:1 between the games DataFrame and the DB table.
# All other columns are stored in the `extra` JSONB column.
# ---------------------------------------------------------------------------
DIRECT_COLS = {
    "game_pk", "game_date", "season", "game_time_utc",
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


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


# ---------------------------------------------------------------------------
# games table
# ---------------------------------------------------------------------------

def _coerce(val, col):
    try:
        is_na = pd.isna(val)
    except Exception:
        is_na = False
    if is_na:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if col == "home_win":
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


def _row_to_db(row: dict) -> dict:
    """Convert a DataFrame row dict into a DB-ready dict (direct cols + extra JSONB)."""
    direct = {}
    extra = {}
    for col, val in row.items():
        v = _coerce(val, col)
        if col in DIRECT_COLS:
            direct[col] = v
        elif v is not None:
            extra[col] = v
    direct["extra"] = json.dumps(extra) if extra else None
    return direct


def upsert_games(df: pd.DataFrame):
    """Upsert a DataFrame of game rows into the games table."""
    if df.empty:
        return
    rows = [_row_to_db(r) for _, r in df.iterrows()]
    cols = list(rows[0].keys())
    values = [[r.get(c) for c in cols] for r in rows]
    col_list = ", ".join(cols)
    update_parts = []
    for c in cols:
        if c == "game_pk":
            continue
        if c == "extra":
            update_parts.append(
                "extra = COALESCE(games.extra, '{}'::jsonb) || COALESCE(EXCLUDED.extra, '{}'::jsonb)"
            )
        else:
            update_parts.append(c + " = EXCLUDED." + c)
    update_list = ", ".join(update_parts)
    sql = (
        "INSERT INTO games (" + col_list + ") VALUES %s "
        "ON CONFLICT (game_pk) DO UPDATE SET "
        + update_list
        + ", updated_at = NOW()"
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, values)
        conn.commit()
    finally:
        conn.close()


def get_games_df(season: int = None, upcoming_only: bool = False) -> pd.DataFrame:
    """
    Load games from DB into a DataFrame matching the CSV schema.
    Extra JSONB columns are exploded back into DataFrame columns.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            conditions = []
            params = []
            if season is not None:
                conditions.append("season = %s")
                params.append(season)
            if upcoming_only:
                conditions.append("home_win IS NULL")
                conditions.append(
                    "COALESCE(extra->>'game_status', '') NOT IN ('postponed', 'cancelled')"
                )
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            cur.execute(f"SELECT * FROM games {where} ORDER BY game_date, game_pk", params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        r = dict(row)
        extra = r.pop("extra", None) or {}
        if isinstance(extra, str):
            extra = json.loads(extra)
        r.update(extra)
        # Drop internal DB timestamps
        r.pop("created_at", None)
        r.pop("updated_at", None)
        records.append(r)

    return pd.DataFrame(records)


def get_complete_game_pks(season: int) -> set:
    """Return set of game_pks that are fully complete (all key cols non-null)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT game_pk FROM games
                   WHERE season = %s
                     AND home_score IS NOT NULL
                     AND away_score IS NOT NULL
                     AND home_win IS NOT NULL
                     AND home_starter_id IS NOT NULL
                     AND temp_c IS NOT NULL
                     AND home_avg IS NOT NULL""",
                (season,),
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# bets table
# ---------------------------------------------------------------------------

def get_processed_game_pks() -> set:
    """Return set of game_pks already in the bets table."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT game_pk FROM bets")
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def bet_exists(game_pk) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM bets WHERE game_pk = %s", (int(game_pk),))
            return cur.fetchone() is not None
    finally:
        conn.close()


def init_bet(game_pk, game_date, home_team, away_team):
    """Insert a bare bet row if it doesn't already exist."""
    if bet_exists(game_pk):
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO bets (game_pk, game_date, home_team, away_team)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (game_pk) DO NOTHING""",
                (int(game_pk), str(game_date)[:10], home_team, away_team),
            )
        conn.commit()
    finally:
        conn.close()


def update_bet_prediction(game_pk, predicted_prob, edge, bet_side, bet_frac, market_implied_prob):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE bets SET
                       predicted_prob = %s,
                       edge = %s,
                       bet_side = %s,
                       bet_frac = %s,
                       market_implied_prob = %s,
                       updated_at = NOW()
                   WHERE game_pk = %s""",
                (
                    float(predicted_prob) if predicted_prob is not None else None,
                    float(edge) if edge is not None and edge == edge else None,  # NaN → None
                    bet_side,
                    float(bet_frac),
                    float(market_implied_prob) if market_implied_prob is not None else None,
                    int(game_pk),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def update_bet_order(game_pk, kalshi_order_id, bet_dollars, n_contracts):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE bets SET
                       kalshi_order_id = %s,
                       bet_dollars = %s,
                       n_contracts = %s,
                       updated_at = NOW()
                   WHERE game_pk = %s""",
                (kalshi_order_id, float(bet_dollars), int(n_contracts), int(game_pk)),
            )
        conn.commit()
    finally:
        conn.close()


def update_bet_result(game_pk, home_win):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE bets SET
                       result = %s,
                       profit_loss = CASE
                           WHEN bet_side IS NULL OR bet_side = 'none'
                                OR bet_dollars IS NULL THEN NULL
                           WHEN (%s AND bet_side = 'home') OR (NOT %s AND bet_side = 'away') THEN
                               ROUND((
                                   CASE
                                       WHEN n_contracts IS NOT NULL THEN n_contracts::numeric - bet_dollars
                                       WHEN market_implied_prob IS NOT NULL AND bet_side = 'home' THEN bet_dollars * (1.0 / NULLIF(market_implied_prob, 0) - 1)
                                       WHEN market_implied_prob IS NOT NULL AND bet_side = 'away' THEN bet_dollars * (1.0 / NULLIF(1.0 - market_implied_prob, 0) - 1)
                                       ELSE NULL
                                   END
                               )::numeric, 2)
                           ELSE -bet_dollars
                       END,
                       updated_at = NOW()
                   WHERE game_pk = %s""",
                (bool(home_win), bool(home_win), bool(home_win), int(game_pk)),
            )
        conn.commit()
    finally:
        conn.close()


def backfill_bet_results():
    """Copy settled game outcomes into bets.result and compute profit_loss."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bets b
                SET result = g.home_win,
                    profit_loss = CASE
                        WHEN b.bet_side IS NULL OR b.bet_side = 'none'
                             OR b.bet_dollars IS NULL THEN NULL
                        WHEN (g.home_win AND b.bet_side = 'home') OR (NOT g.home_win AND b.bet_side = 'away') THEN
                            ROUND((
                                CASE
                                    WHEN b.n_contracts IS NOT NULL THEN b.n_contracts::numeric - b.bet_dollars
                                    WHEN b.market_implied_prob IS NOT NULL AND b.bet_side = 'home' THEN b.bet_dollars * (1.0 / NULLIF(b.market_implied_prob, 0) - 1)
                                    WHEN b.market_implied_prob IS NOT NULL AND b.bet_side = 'away' THEN b.bet_dollars * (1.0 / NULLIF(1.0 - b.market_implied_prob, 0) - 1)
                                    ELSE NULL
                                END
                            )::numeric, 2)
                        ELSE -b.bet_dollars
                    END,
                    updated_at = NOW()
                FROM games g
                WHERE b.game_pk = g.game_pk
                  AND g.home_win IS NOT NULL
                  AND (b.result IS DISTINCT FROM g.home_win OR b.profit_loss IS NULL)
                """
            )
            count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def get_bet(game_pk) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM bets WHERE game_pk = %s", (int(game_pk),))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_all_bets() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM bets ORDER BY game_date DESC, game_pk")
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.drop(columns=["created_at", "updated_at"], errors="ignore")
    return df


def get_model_picks() -> pd.DataFrame:
    """
    Return all predicted games that have a known result (home_win).
    Joins bets (for predictions) with games (for actual outcome).
    Used to measure model accuracy independent of whether a bet was placed.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT b.game_pk, b.game_date, b.home_team, b.away_team,
                       b.predicted_prob, b.bet_side, b.bet_frac, b.edge,
                       b.market_implied_prob,
                       g.home_win
                FROM bets b
                JOIN games g ON b.game_pk = g.game_pk
                WHERE b.predicted_prob IS NOT NULL
                  AND g.home_win IS NOT NULL
                ORDER BY b.game_date DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# balance table
# ---------------------------------------------------------------------------

def insert_balance(balance_cents: int):
    balance_dollars = balance_cents / 100.0
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO balance (recorded_at, balance_cents, balance_dollars) VALUES (%s, %s, %s)",
                (datetime.now(timezone.utc), balance_cents, balance_dollars),
            )
        conn.commit()
    finally:
        conn.close()


def get_last_balance_cents() -> int | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT balance_cents FROM balance ORDER BY recorded_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_balance_history() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT recorded_at, balance_cents, balance_dollars FROM balance ORDER BY recorded_at"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# settings table
# ---------------------------------------------------------------------------

def get_setting(key: str, default: str = "") -> str:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = cur.fetchone()
            return row[0] if row else default
    finally:
        conn.close()


def set_setting(key: str, value: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO settings (key, value, updated_at)
                   VALUES (%s, %s, NOW())
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
                (key, value),
            )
        conn.commit()
    finally:
        conn.close()


def is_live_betting() -> bool:
    return get_setting("live_betting", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Generic table browser (for dashboard DB viewer)
# ---------------------------------------------------------------------------

BROWSABLE_TABLES = {"games", "bets", "balance", "settings"}


def browse_table(table: str, limit: int = 100, offset: int = 0) -> dict:
    if table not in BROWSABLE_TABLES:
        raise ValueError(f"Table '{table}' is not browsable")
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total row count
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total = cur.fetchone()["count"]
            # Page of rows (exclude extra JSONB for games to keep it readable)
            if table == "games":
                cols = "game_pk, game_date, season, home_team, away_team, home_score, away_score, home_win, home_implied_prob, away_implied_prob, close_home_ml, close_away_ml, odds_source"
                cur.execute(f"SELECT {cols} FROM {table} ORDER BY game_date DESC, game_pk LIMIT %s OFFSET %s", (limit, offset))
            else:
                cur.execute(f"SELECT * FROM {table} ORDER BY 1 DESC LIMIT %s OFFSET %s", (limit, offset))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    return {"columns": columns, "rows": [dict(r) for r in rows], "total": int(total)}


# ---------------------------------------------------------------------------
# Auth — users & sessions
# ---------------------------------------------------------------------------

import secrets as _secrets


def init_auth_tables():
    """Create users and sessions tables if they don't exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email         TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    email      TEXT REFERENCES users(email) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        conn.close()


def upsert_user(email: str, password_hash: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, password_hash)
                VALUES (%s, %s)
                ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash
            """, (email, password_hash))
        conn.commit()
    finally:
        conn.close()


def get_user_hash(email: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def create_session(email: str, expires_days: int = 30) -> str:
    session_id = _secrets.token_urlsafe(32)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (session_id, email, expires_at)
                VALUES (%s, %s, NOW() + (%s || ' days')::INTERVAL)
            """, (session_id, email, str(expires_days)))
        conn.commit()
    finally:
        conn.close()
    return session_id


def get_session_email(session_id: str):
    if not session_id:
        return None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT email FROM sessions
                WHERE session_id = %s AND expires_at > NOW()
            """, (session_id,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def delete_session(session_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        conn.commit()
    finally:
        conn.close()
