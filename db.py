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

import threading
from contextlib import contextmanager

import pandas as pd
import psycopg2
from psycopg2 import pool as _pg_pool
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

def _conn_kwargs() -> dict:
    return dict(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def get_connection():
    return psycopg2.connect(**_conn_kwargs())


_POOL: _pg_pool.ThreadedConnectionPool | None = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> _pg_pool.ThreadedConnectionPool:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = _pg_pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=int(os.environ.get("DB_POOL_MAX", "8")),
                    **_conn_kwargs(),
                )
    return _POOL


@contextmanager
def pooled_connection():
    """Checkout a pooled connection; reconnect if the pool returns a dead one."""
    pool = _get_pool()
    conn = pool.getconn()
    pooled = True
    try:
        # Cheap liveness probe; on failure, return dead conn to pool with close=True
        # and check out a fresh one. ThreadedConnectionPool requires putconn to
        # receive the same object that was checked out, so we cannot just swap in
        # a raw psycopg2.connect(...) and putconn() that later.
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            try:
                pool.putconn(conn, close=True)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
            conn = pool.getconn()
        yield conn
    except Exception:
        try:
            pool.putconn(conn, close=True)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
        pooled = False
        raise
    finally:
        if pooled:
            try:
                pool.putconn(conn)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass


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


def get_upcoming_needing_prediction(season: int = 2026) -> pd.DataFrame:
    """
    Single-query version of the old run_pipeline filter:
    games that haven't started (home_win IS NULL), not postponed/cancelled,
    and either not in bets or have no predicted_prob yet.
    """
    sql = """
        SELECT g.*
        FROM games g
        LEFT JOIN bets b ON b.game_pk = g.game_pk
        WHERE g.season = %s
          AND g.home_win IS NULL
          AND COALESCE(g.extra->>'game_status', '') NOT IN ('postponed', 'cancelled')
          AND (b.game_pk IS NULL OR b.predicted_prob IS NULL)
        ORDER BY g.game_date, g.game_pk
    """
    with pooled_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (season,))
            rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        r = dict(row)
        extra = r.pop("extra", None) or {}
        if isinstance(extra, str):
            extra = json.loads(extra)
        r.update(extra)
        r.pop("created_at", None)
        r.pop("updated_at", None)
        records.append(r)
    return pd.DataFrame(records)


def get_settleable_games(season: int, cutoff_utc: datetime) -> list[dict]:
    """Rows from games∩bets that started before cutoff and still lack a result."""
    with pooled_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT g.game_pk, g.game_date, g.home_team, g.away_team,
                       g.game_time_utc
                FROM games g
                JOIN bets b ON g.game_pk = b.game_pk
                WHERE g.season = %s
                  AND g.home_win IS NULL
                  AND g.game_time_utc IS NOT NULL
                  AND g.game_time_utc::timestamptz < %s
                  AND COALESCE(g.extra->>'game_status', '') NOT IN ('postponed', 'cancelled')
                """,
                (season, cutoff_utc),
            )
            return [dict(r) for r in cur.fetchall()]


def apply_settlements(finals: list[dict], postponed_pks: list[int]) -> None:
    """
    Batch-apply settlement results in a single transaction.
    finals: [{"game_pk", "home_score", "away_score", "home_win"}]
    postponed_pks: list of game_pks to mark postponed in extra JSONB.
    Also refreshes bets.result / profit_loss via the standard backfill query.
    """
    if not finals and not postponed_pks:
        return
    with pooled_connection() as conn:
        with conn.cursor() as cur:
            if finals:
                execute_values(
                    cur,
                    """
                    UPDATE games AS g SET
                        home_score = v.home_score,
                        away_score = v.away_score,
                        home_win   = v.home_win,
                        updated_at = NOW()
                    FROM (VALUES %s) AS v(game_pk, home_score, away_score, home_win)
                    WHERE g.game_pk = v.game_pk
                    """,
                    [(int(f["game_pk"]), int(f["home_score"]),
                      int(f["away_score"]), bool(f["home_win"])) for f in finals],
                    template="(%s, %s, %s, %s)",
                )
            if postponed_pks:
                execute_values(
                    cur,
                    """
                    UPDATE games AS g SET
                        extra = COALESCE(g.extra, '{}'::jsonb)
                                || jsonb_build_object('game_status', 'postponed'),
                        updated_at = NOW()
                    FROM (VALUES %s) AS v(game_pk)
                    WHERE g.game_pk = v.game_pk
                    """,
                    [(int(pk),) for pk in postponed_pks],
                    template="(%s)",
                )
        conn.commit()
    # Roll game outcomes forward into bets in one shot.
    if finals:
        backfill_bet_results()


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
            cur.execute(
                """SELECT b.*, g.game_time_utc
                   FROM bets b
                   LEFT JOIN games g ON b.game_pk = g.game_pk
                   WHERE b.game_pk = %s""",
                (int(game_pk),),
            )
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

BROWSABLE_TABLES = {
    "games",
    "bets",
    "balance",
    "settings",
    "users",
    "sessions",
    "app_users",
    "app_sessions",
    "user_settings",
    "kalshi_accounts",
    "user_balance",
    "user_orders",
}


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

USER_STATUS_PENDING = "pending"
USER_STATUS_APPROVED = "approved"
USER_STATUS_REJECTED = "rejected"
USER_STATUS_DISABLED = "disabled"
USER_STATUSES = {
    USER_STATUS_PENDING,
    USER_STATUS_APPROVED,
    USER_STATUS_REJECTED,
    USER_STATUS_DISABLED,
}
APP_USERS_TABLE = "app_users"
APP_SESSIONS_TABLE = "app_sessions"


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def init_auth_tables():
    """Create auth/execution tables owned by the app DB user and import legacy rows."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {APP_USERS_TABLE} (
                    email TEXT PRIMARY KEY,
                    full_name TEXT,
                    password_hash TEXT NOT NULL,
                    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                    approval_status TEXT NOT NULL DEFAULT 'approved',
                    approved_at TIMESTAMPTZ,
                    approved_by TEXT,
                    last_login_at TIMESTAMPTZ,
                    rejection_reason TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT {APP_USERS_TABLE}_approval_status_check
                    CHECK (approval_status IN ('pending', 'approved', 'rejected', 'disabled'))
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {APP_SESSIONS_TABLE} (
                    session_id TEXT PRIMARY KEY,
                    email TEXT REFERENCES {APP_USERS_TABLE}(email) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    created_by_ip INET,
                    user_agent TEXT,
                    revoked_at TIMESTAMPTZ
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS user_settings (
                    email TEXT NOT NULL REFERENCES {APP_USERS_TABLE}(email) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (email, key)
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS kalshi_accounts (
                    email TEXT PRIMARY KEY REFERENCES {APP_USERS_TABLE}(email) ON DELETE CASCADE,
                    label TEXT NOT NULL DEFAULT 'Primary account',
                    key_id TEXT NOT NULL,
                    key_path TEXT NOT NULL,
                    kalshi_env TEXT NOT NULL DEFAULT 'prod',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    last_verified_at TIMESTAMPTZ,
                    last_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS user_balance (
                    id BIGSERIAL PRIMARY KEY,
                    email TEXT NOT NULL REFERENCES {APP_USERS_TABLE}(email) ON DELETE CASCADE,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    balance_cents BIGINT NOT NULL,
                    balance_dollars DOUBLE PRECISION NOT NULL,
                    source TEXT NOT NULL DEFAULT 'kalshi'
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS user_orders (
                    id BIGSERIAL PRIMARY KEY,
                    email TEXT NOT NULL REFERENCES {APP_USERS_TABLE}(email) ON DELETE CASCADE,
                    game_pk BIGINT NOT NULL,
                    game_date DATE,
                    home_team TEXT,
                    away_team TEXT,
                    predicted_prob DOUBLE PRECISION,
                    market_implied_prob DOUBLE PRECISION,
                    edge DOUBLE PRECISION,
                    bet_side TEXT,
                    bet_frac DOUBLE PRECISION,
                    bet_dollars DOUBLE PRECISION,
                    n_contracts INTEGER,
                    kalshi_order_id TEXT,
                    live_price DOUBLE PRECISION,
                    live_edge DOUBLE PRECISION,
                    status TEXT NOT NULL DEFAULT 'pending',
                    dry_run BOOLEAN NOT NULL DEFAULT TRUE,
                    result BOOLEAN,
                    profit_loss DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (email, game_pk)
                )
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{APP_USERS_TABLE}_approval_status ON {APP_USERS_TABLE} (approval_status)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{APP_USERS_TABLE}_is_admin ON {APP_USERS_TABLE} (is_admin)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{APP_SESSIONS_TABLE}_email ON {APP_SESSIONS_TABLE} (email)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_balance_email_recorded ON user_balance (email, recorded_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_orders_email_status ON user_orders (email, status, game_date)")

            cur.execute(
                f"""
                INSERT INTO {APP_USERS_TABLE} (
                    email, password_hash, created_at, updated_at, approval_status, approved_at, is_admin
                )
                SELECT u.email, u.password_hash, COALESCE(u.created_at, NOW()), NOW(), 'approved',
                       COALESCE(u.created_at, NOW()), FALSE
                FROM users u
                ON CONFLICT (email) DO NOTHING
                """
            )
            cur.execute(
                f"""
                INSERT INTO {APP_SESSIONS_TABLE} (session_id, email, expires_at, created_at)
                SELECT s.session_id, s.email, s.expires_at, COALESCE(s.created_at, NOW())
                FROM sessions s
                JOIN {APP_USERS_TABLE} u ON u.email = s.email
                ON CONFLICT (session_id) DO NOTHING
                """
            )

            bootstrap_admin = _norm_email(os.environ.get("MLB_BOOTSTRAP_ADMIN_EMAIL", ""))
            if bootstrap_admin:
                cur.execute(
                    f"""
                    UPDATE {APP_USERS_TABLE}
                    SET is_admin = TRUE,
                        approval_status = 'approved',
                        approved_at = COALESCE(approved_at, NOW()),
                        updated_at = NOW()
                    WHERE email = %s
                    """,
                    (bootstrap_admin,),
                )
            else:
                cur.execute(f"SELECT COUNT(*) FROM {APP_USERS_TABLE}")
                user_count = cur.fetchone()[0]
                if user_count == 1:
                    cur.execute(
                        f"""
                        UPDATE {APP_USERS_TABLE}
                        SET is_admin = TRUE,
                            approval_status = 'approved',
                            approved_at = COALESCE(approved_at, created_at, NOW()),
                            updated_at = NOW()
                        WHERE is_admin = FALSE
                        """
                    )

            cur.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (
                    'global_live_betting',
                    COALESCE((SELECT value FROM settings WHERE key = 'live_betting'), 'false'),
                    NOW()
                )
                ON CONFLICT (key) DO NOTHING
                """
            )
            cur.execute(
                f"""
                INSERT INTO user_settings (email, key, value, updated_at)
                SELECT email, 'live_betting',
                       COALESCE((SELECT value FROM settings WHERE key = 'global_live_betting'), 'false'),
                       NOW()
                FROM {APP_USERS_TABLE}
                ON CONFLICT (email, key) DO NOTHING
                """
            )
        conn.commit()
    finally:
        conn.close()


def upsert_user(
    email: str,
    password_hash: str,
    *,
    full_name: str = "",
    is_admin: bool = False,
    approval_status: str = USER_STATUS_APPROVED,
    approved_by: str | None = None,
):
    email = _norm_email(email)
    approved_by = _norm_email(approved_by or "") or None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO app_users (
                    email, password_hash, full_name, is_admin, approval_status,
                    approved_at, approved_by, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    CASE WHEN %s = 'approved' THEN NOW() ELSE NULL END,
                    %s, NOW()
                )
                ON CONFLICT (email) DO UPDATE SET
                    password_hash = EXCLUDED.password_hash,
                    full_name = EXCLUDED.full_name,
                    is_admin = EXCLUDED.is_admin,
                    approval_status = EXCLUDED.approval_status,
                    approved_at = CASE
                        WHEN EXCLUDED.approval_status = 'approved'
                        THEN COALESCE(app_users.approved_at, NOW())
                        ELSE app_users.approved_at
                    END,
                    approved_by = COALESCE(EXCLUDED.approved_by, app_users.approved_by),
                    updated_at = NOW()
            """, (email, password_hash, full_name, is_admin, approval_status, approval_status, approved_by))
        conn.commit()
    finally:
        conn.close()


def create_pending_user(email: str, password_hash: str, full_name: str = "") -> bool:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_users (email, password_hash, full_name, is_admin, approval_status, updated_at)
                VALUES (%s, %s, %s, FALSE, 'pending', NOW())
                ON CONFLICT (email) DO NOTHING
                """,
                (email, password_hash, full_name.strip()),
            )
            created = cur.rowcount > 0
            if created:
                cur.execute(
                    """
                    INSERT INTO user_settings (email, key, value, updated_at)
                    VALUES (%s, 'live_betting', 'false', NOW())
                    ON CONFLICT (email, key) DO NOTHING
                    """,
                    (email,),
                )
        conn.commit()
        return created
    finally:
        conn.close()


def get_user_hash(email: str):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash FROM app_users WHERE email = %s", (email,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_user(email: str) -> dict | None:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT email, full_name, is_admin, approval_status, approved_at,
                       approved_by, created_at, updated_at, last_login_at,
                       rejection_reason
                FROM app_users
                WHERE email = %s
                """,
                (email,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def list_users(status: str = "all") -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            params = []
            where = ""
            if status != "all":
                where = "WHERE approval_status = %s"
                params.append(status)
            cur.execute(
                f"""
                SELECT email, full_name, is_admin, approval_status, approved_at,
                       approved_by, created_at, updated_at, last_login_at,
                       rejection_reason
                FROM app_users
                {where}
                ORDER BY created_at DESC, email
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def set_user_approval_status(
    email: str,
    status: str,
    *,
    actor_email: str | None = None,
    rejection_reason: str = "",
):
    if status not in USER_STATUSES:
        raise ValueError(f"Invalid approval status: {status}")
    email = _norm_email(email)
    actor_email = _norm_email(actor_email or "") or None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE app_users
                SET approval_status = %s,
                    approved_at = CASE WHEN %s = 'approved' THEN NOW() ELSE approved_at END,
                    approved_by = CASE WHEN %s = 'approved' THEN %s ELSE approved_by END,
                    rejection_reason = CASE
                        WHEN %s IN ('rejected', 'disabled') THEN NULLIF(%s, '')
                        ELSE NULL
                    END,
                    updated_at = NOW()
                WHERE email = %s
                """,
                (status, status, status, actor_email, status, rejection_reason.strip(), email),
            )
        conn.commit()
    finally:
        conn.close()


def set_user_admin(email: str, is_admin: bool):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE app_users
                SET is_admin = %s,
                    updated_at = NOW()
                WHERE email = %s
                """,
                (bool(is_admin), email),
            )
        conn.commit()
    finally:
        conn.close()


def mark_user_login(email: str):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE app_users SET last_login_at = NOW(), updated_at = NOW() WHERE email = %s",
                (email,),
            )
        conn.commit()
    finally:
        conn.close()


def create_session(
    email: str,
    expires_days: int = 30,
    *,
    created_by_ip: str | None = None,
    user_agent: str = "",
) -> str:
    email = _norm_email(email)
    session_id = _secrets.token_urlsafe(32)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO app_sessions (
                    session_id, email, expires_at, created_by_ip, user_agent
                )
                VALUES (%s, %s, NOW() + (%s || ' days')::INTERVAL, %s, %s)
            """, (session_id, email, str(expires_days), created_by_ip, user_agent[:512]))
        conn.commit()
    finally:
        conn.close()
    return session_id


def get_session_user(session_id: str):
    if not session_id:
        return None
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT u.email, u.full_name, u.is_admin, u.approval_status,
                       u.approved_at, u.approved_by, u.created_at, u.updated_at,
                       u.last_login_at, u.rejection_reason
                FROM app_sessions s
                JOIN app_users u ON u.email = s.email
                WHERE s.session_id = %s
                  AND s.expires_at > NOW()
                  AND s.revoked_at IS NULL
            """, (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_session_email(session_id: str):
    user = get_session_user(session_id)
    return user["email"] if user else None


def delete_session(session_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM app_sessions WHERE session_id = %s", (session_id,))
        conn.commit()
    finally:
        conn.close()


def revoke_sessions_for_email(email: str):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE app_sessions
                SET revoked_at = NOW()
                WHERE email = %s AND revoked_at IS NULL
                """,
                (email,),
            )
        conn.commit()
    finally:
        conn.close()


def purge_expired_sessions() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM app_sessions WHERE expires_at < NOW()")
            n = cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def get_user_setting(email: str, key: str, default: str = "") -> str:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM user_settings WHERE email = %s AND key = %s",
                (email, key),
            )
            row = cur.fetchone()
            return row[0] if row else default
    finally:
        conn.close()


def set_user_setting(email: str, key: str, value: str):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_settings (email, key, value, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (email, key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (email, key, value),
            )
        conn.commit()
    finally:
        conn.close()


def is_global_live_betting() -> bool:
    return get_setting("global_live_betting", get_setting("live_betting", "false")).lower() == "true"


def is_user_live_betting(email: str) -> bool:
    return get_user_setting(email, "live_betting", "true").lower() == "true"


def upsert_kalshi_account(
    email: str,
    *,
    key_id: str,
    key_path: str,
    kalshi_env: str,
    label: str = "Primary account",
    is_active: bool = True,
    last_verified: bool = False,
    last_error: str = "",
):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kalshi_accounts (
                    email, label, key_id, key_path, kalshi_env, is_active,
                    last_verified_at, last_error, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s THEN NOW() ELSE NULL END,
                    NULLIF(%s, ''),
                    NOW()
                )
                ON CONFLICT (email) DO UPDATE SET
                    label = EXCLUDED.label,
                    key_id = EXCLUDED.key_id,
                    key_path = EXCLUDED.key_path,
                    kalshi_env = EXCLUDED.kalshi_env,
                    is_active = EXCLUDED.is_active,
                    last_verified_at = CASE
                        WHEN %s THEN NOW() ELSE kalshi_accounts.last_verified_at
                    END,
                    last_error = NULLIF(%s, ''),
                    updated_at = NOW()
                """,
                (
                    email,
                    label,
                    key_id.strip(),
                    key_path,
                    kalshi_env.strip().lower() or "prod",
                    bool(is_active),
                    bool(last_verified),
                    last_error,
                    bool(last_verified),
                    last_error,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_kalshi_account(email: str) -> dict | None:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT email, label, key_id, key_path, kalshi_env, is_active,
                       last_verified_at, last_error, created_at, updated_at
                FROM kalshi_accounts
                WHERE email = %s
                """,
                (email,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def delete_kalshi_account(email: str):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kalshi_accounts WHERE email = %s", (email,))
        conn.commit()
    finally:
        conn.close()


def list_approved_users_with_accounts() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT u.email, u.is_admin, u.approval_status,
                       a.key_id, a.key_path, a.kalshi_env, a.is_active
                FROM app_users u
                JOIN kalshi_accounts a ON a.email = u.email
                WHERE u.approval_status = 'approved'
                  AND a.is_active = TRUE
                ORDER BY u.email
                """
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def insert_user_balance(email: str, balance_cents: int, source: str = "kalshi"):
    email = _norm_email(email)
    balance_dollars = balance_cents / 100.0
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_balance (
                    email, recorded_at, balance_cents, balance_dollars, source
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (email, datetime.now(timezone.utc), int(balance_cents), balance_dollars, source),
            )
        conn.commit()
    finally:
        conn.close()


def get_last_user_balance_cents(email: str) -> int | None:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT balance_cents
                FROM user_balance
                WHERE email = %s
                ORDER BY recorded_at DESC, id DESC
                LIMIT 1
                """,
                (email,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None
    finally:
        conn.close()


def get_user_balance_history(email: str) -> pd.DataFrame:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT recorded_at, balance_cents, balance_dollars, source
                FROM user_balance
                WHERE email = %s
                ORDER BY recorded_at
                """,
                (email,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def upsert_user_order(
    email: str,
    game_pk: str | int,
    *,
    game_date: str = "",
    home_team: str = "",
    away_team: str = "",
    predicted_prob=None,
    market_implied_prob=None,
    edge=None,
    bet_side: str = "none",
    bet_frac: float = 0.0,
    bet_dollars=None,
    n_contracts=None,
    kalshi_order_id: str | None = None,
    live_price=None,
    live_edge=None,
    status: str = "pending",
    dry_run: bool = True,
):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_orders (
                    email, game_pk, game_date, home_team, away_team,
                    predicted_prob, market_implied_prob, edge,
                    bet_side, bet_frac, bet_dollars, n_contracts,
                    kalshi_order_id, live_price, live_edge, status, dry_run, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (email, game_pk) DO UPDATE SET
                    game_date = EXCLUDED.game_date,
                    home_team = EXCLUDED.home_team,
                    away_team = EXCLUDED.away_team,
                    predicted_prob = EXCLUDED.predicted_prob,
                    market_implied_prob = EXCLUDED.market_implied_prob,
                    edge = EXCLUDED.edge,
                    bet_side = EXCLUDED.bet_side,
                    bet_frac = EXCLUDED.bet_frac,
                    bet_dollars = COALESCE(EXCLUDED.bet_dollars, user_orders.bet_dollars),
                    n_contracts = COALESCE(EXCLUDED.n_contracts, user_orders.n_contracts),
                    kalshi_order_id = COALESCE(EXCLUDED.kalshi_order_id, user_orders.kalshi_order_id),
                    live_price = EXCLUDED.live_price,
                    live_edge = EXCLUDED.live_edge,
                    status = EXCLUDED.status,
                    dry_run = EXCLUDED.dry_run,
                    updated_at = NOW()
                """,
                (
                    email,
                    int(game_pk),
                    str(game_date)[:10] or None,
                    home_team,
                    away_team,
                    float(predicted_prob) if predicted_prob is not None else None,
                    float(market_implied_prob) if market_implied_prob is not None else None,
                    float(edge) if edge is not None and edge == edge else None,
                    bet_side,
                    float(bet_frac or 0.0),
                    float(bet_dollars) if bet_dollars is not None else None,
                    int(n_contracts) if n_contracts is not None else None,
                    kalshi_order_id,
                    float(live_price) if live_price is not None else None,
                    float(live_edge) if live_edge is not None else None,
                    status,
                    bool(dry_run),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_user_orders(email: str) -> pd.DataFrame:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM user_orders
                WHERE email = %s
                ORDER BY game_date DESC NULLS LAST, game_pk DESC
                """,
                (email,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.drop(columns=["created_at", "updated_at"], errors="ignore")
    return df


def get_user_order(email: str, game_pk: str | int) -> dict | None:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM user_orders WHERE email = %s AND game_pk = %s",
                (email, int(game_pk)),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def backfill_user_order_results():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_orders uo
                SET result = g.home_win,
                    profit_loss = CASE
                        WHEN uo.bet_side IS NULL OR uo.bet_side = 'none'
                             OR uo.bet_dollars IS NULL THEN NULL
                        WHEN (g.home_win AND uo.bet_side = 'home') OR (NOT g.home_win AND uo.bet_side = 'away') THEN
                            ROUND((
                                CASE
                                    WHEN uo.n_contracts IS NOT NULL THEN uo.n_contracts::numeric - uo.bet_dollars
                                    WHEN uo.market_implied_prob IS NOT NULL AND uo.bet_side = 'home' THEN uo.bet_dollars * (1.0 / NULLIF(uo.market_implied_prob, 0) - 1)
                                    WHEN uo.market_implied_prob IS NOT NULL AND uo.bet_side = 'away' THEN uo.bet_dollars * (1.0 / NULLIF(1.0 - uo.market_implied_prob, 0) - 1)
                                    ELSE NULL
                                END
                            )::numeric, 2)
                        ELSE -uo.bet_dollars
                    END,
                    updated_at = NOW()
                FROM games g
                WHERE uo.game_pk = g.game_pk
                  AND g.home_win IS NOT NULL
                  AND (uo.result IS DISTINCT FROM g.home_win OR uo.profit_loss IS NULL)
                """
            )
            count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def ping() -> bool:
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        finally:
            conn.close()
    except Exception:
        return False
