"""
db.py — Shared PostgreSQL helpers for the MLB pipeline.
All scripts import from this module instead of reading/writing CSVs.

Connection is configured via .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import json
import os
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path

import threading
from contextlib import contextmanager

import pandas as pd
import psycopg2
from psycopg2 import pool as _pg_pool
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv

from cryptography.fernet import Fernet, InvalidToken

from config import ACTIVE_SEASON

load_dotenv()

# ---------------------------------------------------------------------------
# Field-level encryption for sensitive credentials (Kalshi API keys, etc.)
# Set ENCRYPTION_KEY in .env — generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# ---------------------------------------------------------------------------
_ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "")
_fernet = Fernet(_ENCRYPTION_KEY.encode()) if _ENCRYPTION_KEY else None
_ENCRYPTED_PREFIX = "enc:"


def encrypt_field(plaintext: str) -> str:
    if not _fernet or not plaintext:
        return plaintext
    return _ENCRYPTED_PREFIX + _fernet.encrypt(plaintext.encode()).decode()


def decrypt_field(stored: str) -> str:
    if not _fernet or not stored or not stored.startswith(_ENCRYPTED_PREFIX):
        return stored
    try:
        return _fernet.decrypt(stored[len(_ENCRYPTED_PREFIX):].encode()).decode()
    except InvalidToken:
        return stored

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

ODDS_OPEN_COLS = {"open_home_ml", "open_away_ml"}
ODDS_CLOSE_COLS = {"close_home_ml", "close_away_ml"}
ODDS_IMPLIED_PROB_COLS = {"home_implied_prob", "away_implied_prob"}


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


def _is_real_line_sql(side: str) -> str:
    return (
        f"EXCLUDED.close_{side}_ml IS NOT NULL "
        f"AND EXCLUDED.open_{side}_ml IS NOT NULL "
        f"AND EXCLUDED.close_{side}_ml != EXCLUDED.open_{side}_ml"
    )


def _upsert_assignment(col: str, cols: set[str]) -> str:
    """Build a safe ON CONFLICT assignment for one games-table column."""
    if col == "extra":
        return "extra = COALESCE(games.extra, '{}'::jsonb) || COALESCE(EXCLUDED.extra, '{}'::jsonb)"
    if col in {"home_score", "away_score", "home_win"}:
        return f"{col} = COALESCE(EXCLUDED.{col}, games.{col})"
    if col in ODDS_OPEN_COLS:
        side = "home" if col == "open_home_ml" else "away"
        if f"close_{side}_ml" not in cols:
            return f"{col} = COALESCE(EXCLUDED.{col}, games.{col})"
        return (
            f"{col} = CASE WHEN {_is_real_line_sql(side)} "
            f"THEN EXCLUDED.{col} "
            f"ELSE COALESCE(games.{col}, EXCLUDED.{col}) END"
        )
    if col in ODDS_CLOSE_COLS:
        side = "home" if col == "close_home_ml" else "away"
        if f"open_{side}_ml" not in cols:
            return f"{col} = COALESCE(EXCLUDED.{col}, games.{col})"
        return (
            f"{col} = CASE WHEN {_is_real_line_sql(side)} "
            f"THEN EXCLUDED.{col} "
            f"ELSE COALESCE(games.{col}, EXCLUDED.{col}) END"
        )
    if col in ODDS_IMPLIED_PROB_COLS:
        quality_checks = []
        if {"open_home_ml", "close_home_ml"} <= cols:
            quality_checks.append(f"({_is_real_line_sql('home')})")
        if {"open_away_ml", "close_away_ml"} <= cols:
            quality_checks.append(f"({_is_real_line_sql('away')})")
        if not quality_checks:
            return f"{col} = COALESCE(EXCLUDED.{col}, games.{col})"
        return (
            f"{col} = CASE WHEN {' OR '.join(quality_checks)} "
            f"THEN COALESCE(EXCLUDED.{col}, games.{col}) "
            f"ELSE COALESCE(games.{col}, EXCLUDED.{col}) END"
        )
    return col + " = EXCLUDED." + col


def _build_upsert_assignments(cols: list[str]) -> list[str]:
    col_set = set(cols)
    return [_upsert_assignment(c, col_set) for c in cols if c != "game_pk"]


def upsert_games(df: pd.DataFrame):
    """Upsert a DataFrame of game rows into the games table."""
    if df.empty:
        return
    rows = [_row_to_db(r) for _, r in df.iterrows()]
    cols = list(rows[0].keys())
    values = [[r.get(c) for c in cols] for r in rows]
    col_list = ", ".join(cols)
    update_parts = _build_upsert_assignments(cols)
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


RETRYABLE_ORDER_STATUSES = (
    "error",
    "unfilled",
    "skipped_no_market",
    "skipped_no_live_price",
)

PAPER_STARTING_BANKROLL_DOLLARS = 10_000.0
PAPER_UNIVERSAL_EMAIL = "__paper_universal__"


def get_upcoming_needing_prediction(season: int = ACTIVE_SEASON) -> pd.DataFrame:
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
          AND (
              b.game_pk IS NULL
              OR b.predicted_prob IS NULL
              OR EXISTS (
                  SELECT 1
                  FROM user_orders uo
                  WHERE uo.game_pk = g.game_pk
                    AND uo.status = ANY(%s)
              )
          )
        ORDER BY g.game_date, g.game_pk
    """
    with pooled_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (season, list(RETRYABLE_ORDER_STATUSES)))
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
    """Games that started before cutoff and still lack a result.

    Postponed games are intentionally included. MLB can later attach a final
    result to the same game_pk after the make-up date, so settlement must
    revisit them until they either become final or are explicitly cancelled.
    """
    with pooled_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT g.game_pk, g.game_date, g.home_team, g.away_team,
                       g.game_time_utc
                FROM games g
                WHERE g.season = %s
                  AND g.home_win IS NULL
                  AND g.game_time_utc IS NOT NULL
                  AND g.game_time_utc::timestamptz < %s
                  AND COALESCE(g.extra->>'game_status', '') <> 'cancelled'
                ORDER BY g.game_date, g.game_pk
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
                        extra = COALESCE(g.extra, '{}'::jsonb)
                                || jsonb_build_object('game_status', 'final'),
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


def apply_live_game_updates(updates: list[dict]) -> int:
    """Update scores/status/inning metadata from MLB live schedule data."""
    if not updates:
        return 0
    values = []
    for item in updates:
        extra = {
            k: v
            for k, v in {
                "game_status": item.get("game_status"),
                "game_state": item.get("game_state"),
                "detailed_state": item.get("detailed_state"),
                "inning": item.get("inning"),
                "inning_ordinal": item.get("inning_ordinal"),
                "inning_state": item.get("inning_state"),
                "is_top_inning": item.get("is_top_inning"),
                "outs": item.get("outs"),
                "balls": item.get("balls"),
                "strikes": item.get("strikes"),
                "live_status_text": item.get("live_status_text"),
                "live_updated_at": item.get("live_updated_at"),
            }.items()
            if v is not None
        }
        values.append(
            (
                int(item["game_pk"]),
                item.get("home_score"),
                item.get("away_score"),
                item.get("home_win"),
                json.dumps(extra),
            )
        )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                UPDATE games g
                SET home_score = COALESCE(v.home_score::double precision, g.home_score),
                    away_score = COALESCE(v.away_score::double precision, g.away_score),
                    home_win = COALESCE(v.home_win::boolean, g.home_win),
                    extra = COALESCE(g.extra, '{}'::jsonb) || v.extra::jsonb,
                    updated_at = NOW()
                FROM (VALUES %s) AS v(game_pk, home_score, away_score, home_win, extra)
                WHERE g.game_pk = v.game_pk::bigint
                """,
                values,
            )
            count = cur.rowcount
        conn.commit()
        return count
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


def update_bet_prediction(game_pk, predicted_prob, edge, bet_side, bet_frac, market_implied_prob, model_artifact_id=None):
    values = (
        float(predicted_prob) if predicted_prob is not None else None,
        float(edge) if edge is not None and edge == edge else None,  # NaN → None
        bet_side,
        float(bet_frac),
        float(market_implied_prob) if market_implied_prob is not None else None,
        int(model_artifact_id) if model_artifact_id is not None else None,
        int(game_pk),
    )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """UPDATE bets SET
                           predicted_prob = %s,
                           edge = %s,
                           bet_side = %s,
                           bet_frac = %s,
                           market_implied_prob = %s,
                           model_artifact_id = %s,
                           updated_at = NOW()
                       WHERE game_pk = %s""",
                    values,
                )
            except psycopg2.errors.UndefinedColumn:
                conn.rollback()
                cur.execute(
                    """UPDATE bets SET
                           predicted_prob = %s,
                           edge = %s,
                           bet_side = %s,
                           bet_frac = %s,
                           market_implied_prob = %s,
                           updated_at = NOW()
                       WHERE game_pk = %s""",
                    values[:5] + (values[-1],),
                )
        conn.commit()
    finally:
        conn.close()


def update_bet_explanation(game_pk, explanation: dict | None) -> None:
    """Set bets.explanation JSON payload for a game_pk (best-effort)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bets
                SET explanation = %s,
                    updated_at = NOW()
                WHERE game_pk = %s
                """,
                (json.dumps(explanation) if explanation is not None else None, int(game_pk)),
            )
        conn.commit()
    finally:
        conn.close()


def init_bets_explainability() -> None:
    """Ensure the bets table has explainability columns."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # First check if the column already exists so non-owner roles can proceed.
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'bets'
                  AND column_name = 'explanation'
                LIMIT 1
                """
            )
            if cur.fetchone():
                return
            cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS explanation JSONB")
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


def get_upcoming_bet_signals(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT b.game_pk, b.game_date, b.home_team, b.away_team,
                       b.predicted_prob, b.market_implied_prob, b.edge,
                       b.bet_side, b.bet_frac, b.updated_at,
                       g.game_time_utc
                FROM bets b
                JOIN games g ON g.game_pk = b.game_pk
                WHERE b.predicted_prob IS NOT NULL
                  AND COALESCE(b.bet_frac, 0) > 0
                  AND b.bet_side IN ('home', 'away')
                  AND g.home_win IS NULL
                  AND COALESCE(g.extra->>'game_status', '') NOT IN ('postponed', 'cancelled')
                ORDER BY g.game_date, g.game_time_utc NULLS LAST, b.game_pk
                LIMIT %s
                """,
                (int(limit),),
            )
            return [dict(row) for row in cur.fetchall()]
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


# ---------------------------------------------------------------------------
# Generic table browser (for dashboard DB viewer)
# ---------------------------------------------------------------------------

BROWSABLE_TABLES = {
    "games",
    "bets",
    "settings",
    "app_users",
    "app_sessions",
    "user_settings",
    "kalshi_accounts",
    "user_balance",
    "user_orders",
    "paper_orders",
    "model_metric_snapshots",
    "model_artifacts",
    "admin_notes",
}

BROWSE_TABLE_ORDER_BY = {
    "games": "game_date DESC NULLS LAST, game_time_utc DESC NULLS LAST, game_pk DESC",
    "bets": "game_date DESC NULLS LAST, updated_at DESC NULLS LAST, game_pk DESC",
    "settings": "updated_at DESC NULLS LAST, key ASC",
    "app_users": "created_at DESC NULLS LAST, email ASC",
    "app_sessions": "created_at DESC NULLS LAST, session_id DESC",
    "user_settings": "updated_at DESC NULLS LAST, email ASC, key ASC",
    "kalshi_accounts": "updated_at DESC NULLS LAST, created_at DESC NULLS LAST, email ASC",
    "user_balance": "recorded_at DESC NULLS LAST, id DESC",
    "user_orders": "created_at DESC NULLS LAST, updated_at DESC NULLS LAST, id DESC",
    "paper_orders": "created_at DESC NULLS LAST, updated_at DESC NULLS LAST, id DESC",
    "model_metric_snapshots": "trained_at DESC NULLS LAST, id DESC",
    "model_artifacts": "created_at DESC NULLS LAST, id DESC",
    "admin_notes": "sort_order ASC NULLS LAST, updated_at DESC NULLS LAST, id DESC",
}


def _browse_table_order_by(table: str) -> str:
    if table not in BROWSABLE_TABLES:
        raise ValueError(f"Table '{table}' is not browsable")
    return BROWSE_TABLE_ORDER_BY[table]


def browse_table(table: str, limit: int = 100, offset: int = 0) -> dict:
    if table not in BROWSABLE_TABLES:
        raise ValueError(f"Table '{table}' is not browsable")
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total row count
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total = cur.fetchone()["count"]
            # Page of rows (exclude bulky/internal columns for readability)
            if table == "games":
                cols = "game_pk, game_date, season, home_team, away_team, home_score, away_score, home_win, home_implied_prob, away_implied_prob, close_home_ml, close_away_ml, odds_source"
                cur.execute(f"SELECT {cols} FROM {table} ORDER BY {_browse_table_order_by(table)} LIMIT %s OFFSET %s", (limit, offset))
            elif table == "model_artifacts":
                cols = """
                    id, created_at, model_type, training_fingerprint,
                    octet_length(artifact_bytes) AS artifact_bytes_size,
                    artifact_format, artifact_sha256, settled_row_count,
                    max_settled_game_date, active_season, target_train_cutoff,
                    num_features, git_commit, is_active, feature_columns,
                    early_feature_columns, feature_config, metrics
                """
                cur.execute(f"SELECT {cols} FROM {table} ORDER BY {_browse_table_order_by(table)} LIMIT %s OFFSET %s", (limit, offset))
            else:
                cur.execute(f"SELECT * FROM {table} ORDER BY {_browse_table_order_by(table)} LIMIT %s OFFSET %s", (limit, offset))
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
APP_API_TOKENS_TABLE = "app_api_tokens"
ADMIN_NOTES_TABLE = "admin_notes"


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def _hash_api_token(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def init_admin_notes_table():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {ADMIN_NOTES_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL DEFAULT '',
                    sort_order BIGINT,
                    is_done BOOLEAN NOT NULL DEFAULT FALSE,
                    created_by TEXT,
                    updated_by TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute(f"ALTER TABLE {ADMIN_NOTES_TABLE} ADD COLUMN IF NOT EXISTS sort_order BIGINT")
            cur.execute(f"ALTER TABLE {ADMIN_NOTES_TABLE} ADD COLUMN IF NOT EXISTS is_done BOOLEAN NOT NULL DEFAULT FALSE")
            cur.execute(f"""
                WITH ranked AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY updated_at DESC, id DESC) AS rn
                    FROM {ADMIN_NOTES_TABLE}
                )
                UPDATE {ADMIN_NOTES_TABLE} n
                SET sort_order = ranked.rn
                FROM ranked
                WHERE n.id = ranked.id
                  AND n.sort_order IS NULL
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{ADMIN_NOTES_TABLE}_sort_order
                ON {ADMIN_NOTES_TABLE} (sort_order ASC, id ASC)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{ADMIN_NOTES_TABLE}_updated_at
                ON {ADMIN_NOTES_TABLE} (updated_at DESC)
            """)
        conn.commit()
    finally:
        conn.close()


def list_admin_notes(limit: int = 100) -> list[dict]:
    init_admin_notes_table()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT *
                FROM {ADMIN_NOTES_TABLE}
                ORDER BY sort_order ASC NULLS LAST, updated_at DESC, id DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def create_admin_note(title: str, body: str, actor_email: str | None = None) -> dict:
    init_admin_notes_table()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT COALESCE(MAX(sort_order), 0) + 1 AS next_sort_order FROM {ADMIN_NOTES_TABLE}")
            next_sort_order = int(cur.fetchone()["next_sort_order"])
            cur.execute(
                f"""
                INSERT INTO {ADMIN_NOTES_TABLE}
                    (title, body, sort_order, is_done, created_by, updated_by, created_at, updated_at)
                VALUES (%s, %s, %s, FALSE, %s, %s, NOW(), NOW())
                RETURNING *
                """,
                (title or "", body or "", next_sort_order, actor_email, actor_email),
            )
            note = dict(cur.fetchone())
        conn.commit()
        return note
    finally:
        conn.close()


def update_admin_note(note_id: int, title: str, body: str, actor_email: str | None = None) -> dict | None:
    init_admin_notes_table()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                UPDATE {ADMIN_NOTES_TABLE}
                SET title = %s,
                    body = %s,
                    updated_by = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """,
                (title or "", body or "", actor_email, int(note_id)),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_admin_note(note_id: int) -> bool:
    init_admin_notes_table()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {ADMIN_NOTES_TABLE} WHERE id = %s", (int(note_id),))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def reorder_admin_notes(note_ids: list[int]) -> bool:
    init_admin_notes_table()
    normalized_ids = [int(nid) for nid in note_ids]
    if not normalized_ids:
        return False
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT id FROM {ADMIN_NOTES_TABLE} WHERE id = ANY(%s)",
                (normalized_ids,),
            )
            found_ids = {int(row["id"]) for row in cur.fetchall()}
            if len(found_ids) != len(set(normalized_ids)):
                return False
            for idx, nid in enumerate(normalized_ids, start=1):
                cur.execute(
                    f"UPDATE {ADMIN_NOTES_TABLE} SET sort_order = %s WHERE id = %s",
                    (idx, nid),
                )
        conn.commit()
        return True
    finally:
        conn.close()


def set_admin_note_done(note_id: int, is_done: bool, actor_email: str | None = None) -> dict | None:
    init_admin_notes_table()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                UPDATE {ADMIN_NOTES_TABLE}
                SET is_done = %s,
                    updated_by = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """,
                (bool(is_done), actor_email, int(note_id)),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        conn.close()


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
                CREATE TABLE IF NOT EXISTS {APP_API_TOKENS_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    email TEXT NOT NULL REFERENCES {APP_USERS_TABLE}(email) ON DELETE CASCADE,
                    label TEXT NOT NULL DEFAULT 'Signal follower',
                    token_hash TEXT NOT NULL UNIQUE,
                    token_prefix TEXT NOT NULL,
                    token_suffix TEXT NOT NULL DEFAULT '',
                    scopes TEXT NOT NULL DEFAULT 'signals:read,client:write',
                    last_used_at TIMESTAMPTZ,
                    revoked_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute(f"ALTER TABLE {APP_API_TOKENS_TABLE} ADD COLUMN IF NOT EXISTS token_suffix TEXT NOT NULL DEFAULT ''")
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
                    kalshi_ticker TEXT,
                    live_price DOUBLE PRECISION,
                    live_edge DOUBLE PRECISION,
                    current_price DOUBLE PRECISION,
                    current_value DOUBLE PRECISION,
                    unrealized_pnl DOUBLE PRECISION,
                    position_count DOUBLE PRECISION,
                    market_status TEXT,
                    last_checked_at TIMESTAMPTZ,
                    last_check_error TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    dry_run BOOLEAN NOT NULL DEFAULT TRUE,
                    result BOOLEAN,
                    profit_loss DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (email, game_pk)
                )
            """)
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS kalshi_ticker TEXT")
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS current_price DOUBLE PRECISION")
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS current_value DOUBLE PRECISION")
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS unrealized_pnl DOUBLE PRECISION")
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS position_count DOUBLE PRECISION")
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS market_status TEXT")
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE user_orders ADD COLUMN IF NOT EXISTS last_check_error TEXT")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS user_order_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    email TEXT NOT NULL REFERENCES {APP_USERS_TABLE}(email) ON DELETE CASCADE,
                    game_pk BIGINT NOT NULL,
                    kalshi_ticker TEXT,
                    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    current_price DOUBLE PRECISION,
                    current_value DOUBLE PRECISION,
                    unrealized_pnl DOUBLE PRECISION,
                    position_count DOUBLE PRECISION,
                    market_status TEXT,
                    source TEXT NOT NULL DEFAULT 'kalshi'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kalshi_market_snapshots (
                    ticker TEXT PRIMARY KEY,
                    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    current_price DOUBLE PRECISION,
                    market_status TEXT,
                    last_error TEXT
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS paper_orders (
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
                    kalshi_ticker TEXT,
                    live_price DOUBLE PRECISION,
                    live_edge DOUBLE PRECISION,
                    current_price DOUBLE PRECISION,
                    current_value DOUBLE PRECISION,
                    unrealized_pnl DOUBLE PRECISION,
                    position_count DOUBLE PRECISION,
                    market_status TEXT,
                    last_checked_at TIMESTAMPTZ,
                    last_check_error TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result BOOLEAN,
                    profit_loss DOUBLE PRECISION,
                    paper_bankroll_before DOUBLE PRECISION,
                    paper_bankroll_after DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (email, game_pk)
                )
            """)
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS kalshi_ticker TEXT")
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS current_price DOUBLE PRECISION")
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS current_value DOUBLE PRECISION")
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS unrealized_pnl DOUBLE PRECISION")
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS position_count DOUBLE PRECISION")
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS market_status TEXT")
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS last_check_error TEXT")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{APP_USERS_TABLE}_approval_status ON {APP_USERS_TABLE} (approval_status)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{APP_USERS_TABLE}_is_admin ON {APP_USERS_TABLE} (is_admin)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{APP_SESSIONS_TABLE}_email ON {APP_SESSIONS_TABLE} (email)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{APP_API_TOKENS_TABLE}_email ON {APP_API_TOKENS_TABLE} (email)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_balance_email_recorded ON user_balance (email, recorded_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_orders_email_status ON user_orders (email, status, game_date)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_orders_ticker ON user_orders (kalshi_ticker)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_orders_live_refresh ON user_orders (status, result, last_checked_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_order_snapshots_order_time ON user_order_snapshots (email, game_pk, checked_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_orders_email_status ON paper_orders (email, status, game_date)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_orders_ticker ON paper_orders (kalshi_ticker)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_orders_live_refresh ON paper_orders (status, result, last_checked_at)")

            # Universal paper bankroll pseudo-user (required for FK on paper_orders.email).
            cur.execute(
                f"""
                INSERT INTO {APP_USERS_TABLE} (
                    email, full_name, password_hash, is_admin, approval_status,
                    approved_at, created_at, updated_at
                )
                VALUES (%s, %s, %s, FALSE, 'approved', NOW(), NOW(), NOW())
                ON CONFLICT (email) DO NOTHING
                """,
                (PAPER_UNIVERSAL_EMAIL, "Universal paper bankroll", "!"),
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
                    'false',
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
    init_model_metrics_table()
    init_model_artifacts_table()
    init_admin_notes_table()


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


def create_api_token(email: str, label: str = "Signal API", scopes: str = "signals:read,client:write") -> dict:
    email = _norm_email(email)
    raw = f"mlbi_{secrets.token_urlsafe(32)}"
    token_hash = _hash_api_token(raw)
    token_prefix = raw[:12]
    token_suffix = raw[-6:]
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                UPDATE {APP_API_TOKENS_TABLE}
                SET revoked_at = COALESCE(revoked_at, NOW())
                WHERE email = %s
                  AND revoked_at IS NULL
                """,
                (email,),
            )
            cur.execute(
                f"""
                INSERT INTO {APP_API_TOKENS_TABLE} (
                    email, label, token_hash, token_prefix, token_suffix, scopes
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, email, label, token_prefix, token_suffix, scopes,
                          last_used_at, revoked_at, created_at
                """,
                (
                    email,
                    (label or "Signal API").strip() or "Signal API",
                    token_hash,
                    token_prefix,
                    token_suffix,
                    scopes,
                ),
            )
            row = dict(cur.fetchone())
        conn.commit()
        row["token"] = raw
        return row
    finally:
        conn.close()


def list_api_tokens(email: str) -> list[dict]:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, email, label, token_prefix, token_suffix, scopes,
                       last_used_at, revoked_at, created_at
                FROM {APP_API_TOKENS_TABLE}
                WHERE email = %s
                  AND revoked_at IS NULL
                ORDER BY created_at DESC, id DESC
                """,
                (email,),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def revoke_api_token(email: str, token_id: int) -> bool:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {APP_API_TOKENS_TABLE}
                SET revoked_at = COALESCE(revoked_at, NOW())
                WHERE email = %s AND id = %s
                """,
                (email, int(token_id)),
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        conn.close()


def get_user_for_api_token(token: str, required_scope: str = "signals:read") -> dict | None:
    token_hash = _hash_api_token(token)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT u.*
                FROM {APP_API_TOKENS_TABLE} t
                JOIN {APP_USERS_TABLE} u ON u.email = t.email
                WHERE t.token_hash = %s
                  AND t.revoked_at IS NULL
                """,
                (token_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                f"""
                SELECT scopes
                FROM {APP_API_TOKENS_TABLE}
                WHERE token_hash = %s
                  AND revoked_at IS NULL
                """,
                (token_hash,),
            )
            scopes_row = cur.fetchone()
            scopes = {
                s.strip()
                for s in str((scopes_row or {}).get("scopes") or "").split(",")
                if s.strip()
            }
            if required_scope and required_scope not in scopes:
                return None
            cur.execute(
                f"""
                UPDATE {APP_API_TOKENS_TABLE}
                SET last_used_at = NOW()
                WHERE token_hash = %s
                """,
                (token_hash,),
            )
            return dict(row)
    finally:
        conn.commit()
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
    return get_setting("global_live_betting", "false").lower() == "true"


def is_user_live_betting(email: str) -> bool:
    return get_user_setting(email, "live_betting", "false").lower() == "true"


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
                    encrypt_field(key_id.strip()),
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
            if not row:
                return None
            d = dict(row)
            d["key_id"] = decrypt_field(d["key_id"])
            return d
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
            rows = [dict(row) for row in cur.fetchall()]
            for r in rows:
                r["key_id"] = decrypt_field(r["key_id"])
            return rows
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


def get_user_balance_daily_history(email: str) -> pd.DataFrame:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT DISTINCT ON ((recorded_at AT TIME ZONE 'UTC')::date)
                       recorded_at, balance_cents, balance_dollars, source
                FROM user_balance
                WHERE email = %s
                ORDER BY (recorded_at AT TIME ZONE 'UTC')::date, recorded_at, id
                """,
                (email,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def get_paper_bankroll_dollars(email: str) -> float:
    # Paper mode is universal (same for every user).
    _ = email
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(profit_loss), 0)
                FROM paper_orders
                WHERE email = %s
                  AND profit_loss IS NOT NULL
                """,
                (PAPER_UNIVERSAL_EMAIL,),
            )
            row = cur.fetchone()
            paper_profit = float(row[0] or 0.0) if row else 0.0
            return round(PAPER_STARTING_BANKROLL_DOLLARS + paper_profit, 2)
    finally:
        conn.close()


def get_paper_bankroll_history(email: str) -> list[dict]:
    # Paper mode is universal (same for every user).
    _ = email
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT game_date, game_pk, paper_bankroll_after
                FROM paper_orders
                WHERE email = %s
                  AND paper_bankroll_after IS NOT NULL
                ORDER BY game_date ASC NULLS LAST, game_pk ASC
                """,
                (PAPER_UNIVERSAL_EMAIL,),
            )
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    history = [
        {
            "recorded_at": None,
            "balance_dollars": PAPER_STARTING_BANKROLL_DOLLARS,
            "source": "paper",
        }
    ]
    for row in rows:
        history.append(
            {
                "recorded_at": row.get("game_date"),
                "balance_dollars": float(row["paper_bankroll_after"]),
                "game_pk": row.get("game_pk"),
                "source": "paper",
            }
        )
    return history


def backfill_paper_orders_from_bets(email: str | None = None) -> int:
    """Create missing paper orders from historical model-qualified bet signals.

    Paper mode is universal: the same simulated bets for every user. We store the
    universal stream in `paper_orders` under `PAPER_UNIVERSAL_EMAIL`.
    """
    _ = email
    user_email = PAPER_UNIVERSAL_EMAIL
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT b.game_pk, b.game_date, b.home_team, b.away_team,
                       b.predicted_prob, b.market_implied_prob, b.edge,
                       b.bet_side, b.bet_frac, g.home_win
                FROM bets b
                LEFT JOIN games g ON g.game_pk = b.game_pk
                WHERE b.bet_side IN ('home', 'away')
                  AND COALESCE(b.bet_frac, 0) > 0
                  AND b.predicted_prob IS NOT NULL
                ORDER BY b.game_date ASC NULLS LAST, b.game_pk ASC
                """
            )
            signals = [dict(row) for row in cur.fetchall()]
            if not signals:
                return 0

            cur.execute(
                """
                SELECT game_pk, profit_loss
                FROM paper_orders
                WHERE email = %s
                """,
                (user_email,),
            )
            existing = {int(row["game_pk"]): dict(row) for row in cur.fetchall()}
            bankroll = PAPER_STARTING_BANKROLL_DOLLARS
            rows_to_insert = []
            inserted = 0

            for signal in signals:
                game_pk = int(signal["game_pk"])
                existing_row = existing.get(game_pk)
                if existing_row:
                    if existing_row.get("profit_loss") is not None:
                        bankroll += float(existing_row["profit_loss"])
                    continue

                bet_frac = float(signal.get("bet_frac") or 0.0)
                bet_dollars = round(bankroll * bet_frac, 2)
                bet_side = str(signal.get("bet_side") or "none")
                market_prob = signal.get("market_implied_prob")
                live_price = None
                live_edge = None
                if market_prob is not None:
                    market_prob = float(market_prob)
                    live_price = market_prob if bet_side == "home" else 1.0 - market_prob
                    model_prob = float(signal["predicted_prob"])
                    if bet_side == "away":
                        model_prob = 1.0 - model_prob
                    live_edge = model_prob - live_price

                result = signal.get("home_win")
                profit_loss = None
                bankroll_after = bankroll
                if result is not None and bet_dollars > 0:
                    won = (bool(result) and bet_side == "home") or (
                        not bool(result) and bet_side == "away"
                    )
                    if won and live_price and 0 < live_price < 1:
                        profit_loss = round(bet_dollars * (1.0 / live_price - 1.0), 2)
                    elif not won:
                        profit_loss = -bet_dollars
                    if profit_loss is not None:
                        bankroll_after = round(bankroll + profit_loss, 2)

                rows_to_insert.append(
                    (
                        user_email,
                        game_pk,
                        str(signal.get("game_date") or "")[:10] or None,
                        signal.get("home_team") or "",
                        signal.get("away_team") or "",
                        float(signal["predicted_prob"]),
                        float(market_prob) if market_prob is not None else None,
                        float(signal["edge"]) if signal.get("edge") is not None else None,
                        bet_side,
                        bet_frac,
                        bet_dollars,
                        None,
                        live_price,
                        live_edge,
                        "dry_run" if bet_dollars > 0 else "skipped_too_small",
                        bool(result) if result is not None else None,
                        profit_loss,
                        bankroll,
                        bankroll_after,
                    )
                )
                inserted += 1
                if profit_loss is not None:
                    bankroll = bankroll_after

            if rows_to_insert:
                execute_values(
                    cur,
                    """
                    INSERT INTO paper_orders (
                        email, game_pk, game_date, home_team, away_team,
                        predicted_prob, market_implied_prob, edge,
                        bet_side, bet_frac, bet_dollars, n_contracts,
                        live_price, live_edge, status, result, profit_loss,
                        paper_bankroll_before, paper_bankroll_after, updated_at
                    )
                    VALUES %s
                    ON CONFLICT (email, game_pk) DO NOTHING
                    """,
                    rows_to_insert,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                )
        conn.commit()
        return inserted
    finally:
        conn.close()


def _paper_side_model_prob(predicted_prob: float | None, bet_side: str) -> float | None:
    if predicted_prob is None:
        return None
    try:
        p = float(predicted_prob)
    except Exception:
        return None
    if bet_side == "away":
        return 1.0 - p
    if bet_side == "home":
        return p
    return None


def _paper_side_price(
    *,
    bet_side: str,
    predicted_prob: float | None,
    edge: float | None,
    live_price: float | None,
    market_implied_prob: float | None,
) -> float | None:
    """
    Return the best available contract price (side probability) in [0,1].
    Priority:
      1) live_price if present
      2) market_implied_prob converted to side-price
      3) derived from (model_prob_side - edge) if both present
    """
    for v in (live_price,):
        if v is None:
            continue
        try:
            p = float(v)
        except Exception:
            continue
        if 0 < p < 1:
            return p

    if market_implied_prob is not None:
        try:
            mp = float(market_implied_prob)
        except Exception:
            mp = None
        if mp is not None and 0 < mp < 1:
            side_price = mp if bet_side == "home" else (1.0 - mp if bet_side == "away" else None)
            if side_price is not None and 0 < side_price < 1:
                return side_price

    if edge is not None:
        try:
            e = float(edge)
        except Exception:
            e = None
        if e is not None:
            model_side = _paper_side_model_prob(predicted_prob, bet_side)
            if model_side is not None:
                side_price = model_side - e
                if 0 < side_price < 1:
                    return side_price
    return None


def _paper_profit_loss(
    *,
    won: bool,
    stake: float,
    side_price: float | None,
    n_contracts: int | None = None,
) -> float | None:
    if stake <= 0:
        return None
    if not won:
        return round(-stake, 2)
    if n_contracts is not None:
        return round(float(n_contracts) - stake, 2)
    if side_price is None or not (0 < side_price < 1):
        return None
    return round(stake * (1.0 / side_price - 1.0), 2)


def recompute_paper_order_financials(email: str | None = None) -> int:
    """
    Fill missing paper bet financials:
      - paper_bankroll_before / paper_bankroll_after
      - bet_dollars (stake) when missing
      - profit_loss when missing and result is known

    Uses universal chronological ordering and PAPER_STARTING_BANKROLL_DOLLARS.
    Profit is computed from filled n_contracts OR side-price (live_price / market / model-edge derived).
    """
    _ = email
    with pooled_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            updated = 0
            cur.execute(
                """
                SELECT id, game_pk, game_date, bet_side, bet_frac, bet_dollars,
                       predicted_prob, edge, market_implied_prob, live_price,
                       n_contracts, result, profit_loss,
                       paper_bankroll_before, paper_bankroll_after
                FROM paper_orders
                WHERE email = %s
                ORDER BY game_date ASC NULLS LAST, game_pk ASC, id ASC
                """,
                (PAPER_UNIVERSAL_EMAIL,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            bankroll = PAPER_STARTING_BANKROLL_DOLLARS

            for r in rows:
                bet_side = str(r.get("bet_side") or "none")
                bet_frac = r.get("bet_frac")
                try:
                    frac = float(bet_frac) if bet_frac is not None else 0.0
                except Exception:
                    frac = 0.0
                frac = max(0.0, frac)

                stake_existing = r.get("bet_dollars")
                try:
                    stake_existing_f = float(stake_existing) if stake_existing is not None else None
                except Exception:
                    stake_existing_f = None
                stake = stake_existing_f
                if stake is None and frac > 0:
                    stake = round(bankroll * frac, 2)
                if stake is not None and stake < 0:
                    stake = None

                pb_before = r.get("paper_bankroll_before")
                try:
                    pb_before_f = float(pb_before) if pb_before is not None else None
                except Exception:
                    pb_before_f = None
                pb_after = r.get("paper_bankroll_after")
                try:
                    pb_after_f = float(pb_after) if pb_after is not None else None
                except Exception:
                    pb_after_f = None

                result = r.get("result")
                pnl_existing = r.get("profit_loss")
                try:
                    pnl_existing_f = float(pnl_existing) if pnl_existing is not None else None
                except Exception:
                    pnl_existing_f = None

                n_contracts = r.get("n_contracts")
                try:
                    n_contracts_i = int(n_contracts) if n_contracts is not None else None
                except Exception:
                    n_contracts_i = None

                pnl = pnl_existing_f
                if pnl is None and result is not None and stake is not None and stake > 0 and bet_side in {"home", "away"}:
                    side_price = _paper_side_price(
                        bet_side=bet_side,
                        predicted_prob=r.get("predicted_prob"),
                        edge=r.get("edge"),
                        live_price=r.get("live_price"),
                        market_implied_prob=r.get("market_implied_prob"),
                    )
                    won = (bool(result) and bet_side == "home") or (not bool(result) and bet_side == "away")
                    pnl = _paper_profit_loss(won=won, stake=stake, side_price=side_price, n_contracts=n_contracts_i)

                desired_before = bankroll
                desired_after = bankroll
                if pnl is not None:
                    desired_after = round(bankroll + pnl, 2)

                needs_update = (
                    (stake_existing_f is None and stake is not None)
                    or (pb_before_f is None)
                    or (pb_after_f is None and result is not None)
                    or (pnl_existing_f is None and pnl is not None)
                )
                if needs_update:
                    cur.execute(
                        """
                        UPDATE paper_orders
                        SET bet_dollars = COALESCE(bet_dollars, %s),
                            profit_loss = COALESCE(profit_loss, %s),
                            paper_bankroll_before = COALESCE(paper_bankroll_before, %s),
                            paper_bankroll_after = COALESCE(paper_bankroll_after, %s),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            stake,
                            pnl,
                            desired_before,
                            desired_after,
                            int(r["id"]),
                        ),
                    )
                    updated += cur.rowcount

                bankroll = desired_after

        conn.commit()
        return updated


def upsert_paper_order(
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
    kalshi_ticker: str | None = None,
    live_price=None,
    live_edge=None,
    status: str = "pending",
    paper_bankroll_before=None,
    paper_bankroll_after=None,
):
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_orders (
                    email, game_pk, game_date, home_team, away_team,
                    predicted_prob, market_implied_prob, edge,
                    bet_side, bet_frac, bet_dollars, n_contracts,
                    kalshi_ticker, live_price, live_edge, status,
                    paper_bankroll_before, paper_bankroll_after, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, NOW()
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
                    bet_dollars = COALESCE(EXCLUDED.bet_dollars, paper_orders.bet_dollars),
                    n_contracts = COALESCE(EXCLUDED.n_contracts, paper_orders.n_contracts),
                    kalshi_ticker = COALESCE(EXCLUDED.kalshi_ticker, paper_orders.kalshi_ticker),
                    live_price = EXCLUDED.live_price,
                    live_edge = EXCLUDED.live_edge,
                    status = EXCLUDED.status,
                    paper_bankroll_before = COALESCE(EXCLUDED.paper_bankroll_before, paper_orders.paper_bankroll_before),
                    paper_bankroll_after = COALESCE(EXCLUDED.paper_bankroll_after, paper_orders.paper_bankroll_after),
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
                    kalshi_ticker,
                    float(live_price) if live_price is not None else None,
                    float(live_edge) if live_edge is not None else None,
                    status,
                    float(paper_bankroll_before) if paper_bankroll_before is not None else None,
                    float(paper_bankroll_after) if paper_bankroll_after is not None else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_paper_orders(email: str) -> pd.DataFrame:
    # Paper mode is universal (same for every user).
    _ = email
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM paper_orders
                WHERE email = %s
                ORDER BY game_date DESC NULLS LAST, game_pk DESC
                """,
                (PAPER_UNIVERSAL_EMAIL,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.drop(columns=["created_at", "updated_at"], errors="ignore")
    return df


def get_all_paper_orders() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM paper_orders
                ORDER BY game_date DESC NULLS LAST, game_pk DESC
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.drop(columns=["created_at", "updated_at"], errors="ignore")
    return df


def get_paper_order(email: str, game_pk: str | int) -> dict | None:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM paper_orders WHERE email = %s AND game_pk = %s",
                (email, int(game_pk)),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


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
    kalshi_ticker: str | None = None,
    live_price=None,
    live_edge=None,
    status: str = "pending",
    dry_run: bool = True,
    result=None,
    profit_loss=None,
    last_check_error: str | None = None,
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
                    kalshi_order_id, kalshi_ticker, live_price, live_edge, status, dry_run,
                    result, profit_loss, last_check_error, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, NOW()
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
                    kalshi_ticker = COALESCE(EXCLUDED.kalshi_ticker, user_orders.kalshi_ticker),
                    live_price = EXCLUDED.live_price,
                    live_edge = EXCLUDED.live_edge,
                    status = EXCLUDED.status,
                    dry_run = EXCLUDED.dry_run,
                    result = COALESCE(EXCLUDED.result, user_orders.result),
                    profit_loss = COALESCE(EXCLUDED.profit_loss, user_orders.profit_loss),
                    last_check_error = EXCLUDED.last_check_error,
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
                    kalshi_ticker,
                    float(live_price) if live_price is not None else None,
                    float(live_edge) if live_edge is not None else None,
                    status,
                    bool(dry_run),
                    bool(result) if result is not None else None,
                    float(profit_loss) if profit_loss is not None else None,
                    last_check_error,
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


def get_all_user_orders() -> pd.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM user_orders
                ORDER BY game_date DESC NULLS LAST, game_pk DESC
                """
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


def get_open_live_user_orders_for_refresh(
    stale_seconds: int = 300,
    limit: int = 200,
) -> list[dict]:
    """Return unsettled live orders that need Kalshi mark-to-market refresh."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT uo.*, ka.key_id, ka.key_path, ka.kalshi_env
                FROM user_orders uo
                JOIN kalshi_accounts ka ON ka.email = uo.email
                WHERE ka.is_active = TRUE
                  AND uo.status = 'filled'
                  AND uo.dry_run = FALSE
                  AND uo.result IS NULL
                  AND COALESCE(uo.bet_dollars, 0) > 0
                  AND (
                      uo.last_checked_at IS NULL
                      OR uo.last_checked_at < NOW() - (%s * INTERVAL '1 second')
                  )
                ORDER BY uo.last_checked_at NULLS FIRST, uo.game_date NULLS LAST, uo.id
                LIMIT %s
                """,
                (int(stale_seconds), int(limit)),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for row in rows:
                row["key_id"] = decrypt_field(row.get("key_id") or "")
            return rows
    finally:
        conn.close()


def get_open_orders_for_market_refresh(
    stale_seconds: int = 300,
    limit: int = 200,
) -> list[dict]:
    """Return unsettled live + paper orders needing global market-price marks."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM (
                    SELECT 'live' AS mode,
                           uo.email,
                           uo.game_pk,
                           uo.game_date,
                           uo.home_team,
                           uo.away_team,
                           uo.bet_side,
                           uo.bet_dollars,
                           uo.n_contracts,
                           uo.live_price,
                           uo.kalshi_ticker,
                           uo.last_checked_at,
                           ka.kalshi_env,
                           g.game_time_utc
                    FROM user_orders uo
                    JOIN kalshi_accounts ka ON ka.email = uo.email
                    LEFT JOIN games g ON g.game_pk = uo.game_pk
                    WHERE ka.is_active = TRUE
                      AND uo.status = 'filled'
                      AND uo.dry_run = FALSE
                      AND uo.result IS NULL
                      AND COALESCE(uo.bet_dollars, 0) > 0
                    UNION ALL
                    SELECT 'paper' AS mode,
                           po.email,
                           po.game_pk,
                           po.game_date,
                           po.home_team,
                           po.away_team,
                           po.bet_side,
                           po.bet_dollars,
                           po.n_contracts,
                           po.live_price,
                           po.kalshi_ticker,
                           po.last_checked_at,
                           NULL::TEXT AS kalshi_env,
                           g.game_time_utc
                    FROM paper_orders po
                    LEFT JOIN games g ON g.game_pk = po.game_pk
                    WHERE po.status = 'dry_run'
                      AND po.email = %s
                      AND po.result IS NULL
                      AND COALESCE(po.bet_dollars, 0) > 0
                ) q
                WHERE q.kalshi_ticker IS NULL
                   OR q.last_checked_at IS NULL
                   OR q.last_checked_at < NOW() - (%s * INTERVAL '1 second')
                ORDER BY q.last_checked_at NULLS FIRST, q.game_date NULLS LAST, q.mode, q.game_pk
                LIMIT %s
                """,
                (PAPER_UNIVERSAL_EMAIL, int(stale_seconds), int(limit)),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def update_order_kalshi_ticker(
    mode: str,
    email: str,
    game_pk: str | int,
    kalshi_ticker: str,
) -> None:
    table = "paper_orders" if mode == "paper" else "user_orders"
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {table}
                SET kalshi_ticker = %s,
                    updated_at = NOW()
                WHERE email = %s
                  AND game_pk = %s
                """,
                (kalshi_ticker, email, int(game_pk)),
            )
        conn.commit()
    finally:
        conn.close()


def update_user_order_kalshi_ticker(
    email: str,
    game_pk: str | int,
    kalshi_ticker: str,
) -> None:
    update_order_kalshi_ticker("live", email, game_pk, kalshi_ticker)


def upsert_kalshi_market_snapshot(
    ticker: str,
    *,
    current_price=None,
    market_status: str | None = None,
    last_error: str | None = None,
) -> None:
    ticker = (ticker or "").strip()
    if not ticker:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kalshi_market_snapshots (
                    ticker, checked_at, current_price, market_status, last_error
                )
                VALUES (%s, NOW(), %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    checked_at = EXCLUDED.checked_at,
                    current_price = EXCLUDED.current_price,
                    market_status = EXCLUDED.market_status,
                    last_error = EXCLUDED.last_error
                """,
                (
                    ticker,
                    float(current_price) if current_price is not None else None,
                    market_status,
                    last_error,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def apply_market_snapshot_to_open_orders(
    ticker: str,
    *,
    current_price=None,
    market_status: str | None = None,
    last_error: str | None = None,
) -> int:
    ticker = (ticker or "").strip()
    if not ticker:
        return 0
    price = float(current_price) if current_price is not None else None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            total = 0
            if last_error is not None:
                for table, status_clause in (
                    ("user_orders", "status = 'filled' AND dry_run = FALSE"),
                    ("paper_orders", "status = 'dry_run'"),
                ):
                    cur.execute(
                        f"""
                        UPDATE {table}
                        SET last_checked_at = NOW(),
                            last_check_error = %s,
                            updated_at = NOW()
                        WHERE kalshi_ticker = %s
                          AND result IS NULL
                          AND {status_clause}
                        """,
                        (last_error, ticker),
                    )
                    total += cur.rowcount
            else:
                for table, status_clause in (
                    ("user_orders", "status = 'filled' AND dry_run = FALSE"),
                    ("paper_orders", "status = 'dry_run'"),
                ):
                    cur.execute(
                        f"""
                        UPDATE {table}
                        SET current_price = %s,
                            position_count = COALESCE(
                                n_contracts::double precision,
                                CASE
                                    WHEN live_price IS NOT NULL AND live_price > 0
                                    THEN bet_dollars / live_price
                                    ELSE NULL
                                END
                            ),
                            current_value = ROUND((
                                COALESCE(
                                    n_contracts::double precision,
                                    CASE
                                        WHEN live_price IS NOT NULL AND live_price > 0
                                        THEN bet_dollars / live_price
                                        ELSE NULL
                                    END
                                ) * %s
                            )::numeric, 2),
                            unrealized_pnl = ROUND((
                                COALESCE(
                                    n_contracts::double precision,
                                    CASE
                                        WHEN live_price IS NOT NULL AND live_price > 0
                                        THEN bet_dollars / live_price
                                        ELSE NULL
                                    END
                                ) * %s - bet_dollars
                            )::numeric, 2),
                            market_status = %s,
                            last_checked_at = NOW(),
                            last_check_error = NULL,
                            updated_at = NOW()
                        WHERE kalshi_ticker = %s
                          AND result IS NULL
                          AND {status_clause}
                          AND COALESCE(bet_dollars, 0) > 0
                        """,
                        (price, price, price, market_status, ticker),
                    )
                    total += cur.rowcount
            cur.execute(
                """
                INSERT INTO user_order_snapshots (
                    email, game_pk, kalshi_ticker, current_price,
                    current_value, unrealized_pnl, position_count,
                    market_status, source
                )
                SELECT email, game_pk, kalshi_ticker, current_price,
                       current_value, unrealized_pnl, position_count,
                       market_status, 'kalshi_market'
                FROM user_orders
                WHERE kalshi_ticker = %s
                  AND result IS NULL
                  AND status = 'filled'
                  AND dry_run = FALSE
                  AND %s IS NULL
                """,
                (ticker, last_error),
            )
        conn.commit()
        return total
    finally:
        conn.close()


def update_user_order_live_snapshot(
    email: str,
    game_pk: str | int,
    *,
    kalshi_ticker: str | None = None,
    current_price=None,
    current_value=None,
    unrealized_pnl=None,
    position_count=None,
    market_status: str | None = None,
    last_check_error: str | None = None,
    source: str = "kalshi",
) -> None:
    email = _norm_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if last_check_error is not None:
                cur.execute(
                    """
                    UPDATE user_orders
                    SET kalshi_ticker = COALESCE(%s, kalshi_ticker),
                        last_checked_at = NOW(),
                        last_check_error = %s,
                        updated_at = NOW()
                    WHERE email = %s
                      AND game_pk = %s
                    """,
                    (kalshi_ticker, last_check_error, email, int(game_pk)),
                )
            else:
                cur.execute(
                    """
                    UPDATE user_orders
                    SET kalshi_ticker = COALESCE(%s, kalshi_ticker),
                        current_price = %s,
                        current_value = %s,
                        unrealized_pnl = %s,
                        position_count = %s,
                        market_status = %s,
                        last_checked_at = NOW(),
                        last_check_error = NULL,
                        updated_at = NOW()
                    WHERE email = %s
                      AND game_pk = %s
                    """,
                    (
                        kalshi_ticker,
                        float(current_price) if current_price is not None else None,
                        float(current_value) if current_value is not None else None,
                        float(unrealized_pnl) if unrealized_pnl is not None else None,
                        float(position_count) if position_count is not None else None,
                        market_status,
                        email,
                        int(game_pk),
                    ),
                )
            if last_check_error is None:
                cur.execute(
                    """
                    INSERT INTO user_order_snapshots (
                        email, game_pk, kalshi_ticker, current_price,
                        current_value, unrealized_pnl, position_count,
                        market_status, source
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        email,
                        int(game_pk),
                        kalshi_ticker,
                        float(current_price) if current_price is not None else None,
                        float(current_value) if current_value is not None else None,
                        float(unrealized_pnl) if unrealized_pnl is not None else None,
                        float(position_count) if position_count is not None else None,
                        market_status,
                        source[:64],
                    ),
                )
        conn.commit()
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


def backfill_paper_order_results():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE paper_orders po
                SET result = g.home_win,
                    profit_loss = CASE
                        WHEN po.bet_side IS NULL OR po.bet_side = 'none'
                             OR po.bet_dollars IS NULL THEN NULL
                        WHEN (g.home_win AND po.bet_side = 'home') OR (NOT g.home_win AND po.bet_side = 'away') THEN
                            ROUND((
                                CASE
                                    WHEN po.n_contracts IS NOT NULL THEN po.n_contracts::numeric - po.bet_dollars
                                    WHEN po.live_price IS NOT NULL AND po.live_price > 0 THEN po.bet_dollars * (1.0 / po.live_price - 1)
                                    WHEN po.market_implied_prob IS NOT NULL AND po.bet_side = 'home' THEN po.bet_dollars * (1.0 / NULLIF(po.market_implied_prob, 0) - 1)
                                    WHEN po.market_implied_prob IS NOT NULL AND po.bet_side = 'away' THEN po.bet_dollars * (1.0 / NULLIF(1.0 - po.market_implied_prob, 0) - 1)
                                    ELSE NULL
                                END
                            )::numeric, 2)
                        ELSE -po.bet_dollars
                    END,
                    paper_bankroll_after = CASE
                        WHEN po.paper_bankroll_before IS NULL THEN NULL
                        WHEN po.bet_side IS NULL OR po.bet_side = 'none'
                             OR po.bet_dollars IS NULL THEN po.paper_bankroll_before
                        WHEN (g.home_win AND po.bet_side = 'home') OR (NOT g.home_win AND po.bet_side = 'away') THEN
                            ROUND((
                                po.paper_bankroll_before + (
                                    CASE
                                        WHEN po.n_contracts IS NOT NULL THEN po.n_contracts::numeric - po.bet_dollars
                                        WHEN po.live_price IS NOT NULL AND po.live_price > 0 THEN po.bet_dollars * (1.0 / po.live_price - 1)
                                        WHEN po.market_implied_prob IS NOT NULL AND po.bet_side = 'home' THEN po.bet_dollars * (1.0 / NULLIF(po.market_implied_prob, 0) - 1)
                                        WHEN po.market_implied_prob IS NOT NULL AND po.bet_side = 'away' THEN po.bet_dollars * (1.0 / NULLIF(1.0 - po.market_implied_prob, 0) - 1)
                                        ELSE 0
                                    END
                                )
                            )::numeric, 2)
                        ELSE ROUND((po.paper_bankroll_before - po.bet_dollars)::numeric, 2)
                    END,
                    updated_at = NOW()
                FROM games g
                WHERE po.game_pk = g.game_pk
                  AND g.home_win IS NOT NULL
                  AND (po.result IS DISTINCT FROM g.home_win OR po.profit_loss IS NULL)
                """
            )
            count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model metric snapshots
# ---------------------------------------------------------------------------

MODEL_METRIC_SNAPSHOTS_TABLE = "model_metric_snapshots"


def init_model_metrics_table():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT to_regclass('public.model_training_runs'),
                       to_regclass('public.model_metric_snapshots')
                """
            )
            old_table, new_table = cur.fetchone()
            if old_table and not new_table:
                cur.execute("ALTER TABLE model_training_runs RENAME TO model_metric_snapshots")
                cur.execute("SELECT to_regclass('public.model_training_runs_id_seq')")
                if cur.fetchone()[0]:
                    cur.execute("ALTER SEQUENCE model_training_runs_id_seq RENAME TO model_metric_snapshots_id_seq")
                    cur.execute(
                        """
                        ALTER TABLE model_metric_snapshots
                        ALTER COLUMN id SET DEFAULT nextval('model_metric_snapshots_id_seq')
                        """
                    )
                cur.execute("DROP INDEX IF EXISTS idx_mtr_trained_at")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {MODEL_METRIC_SNAPSHOTS_TABLE} (
                    id SERIAL PRIMARY KEY,
                    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    model_type TEXT NOT NULL,
                    git_commit TEXT,
                    num_features INTEGER,
                    training_rows INTEGER,
                    val_rows INTEGER,
                    num_folds INTEGER,
                    mean_brier DOUBLE PRECISION,
                    mean_roi DOUBLE PRECISION,
                    total_bets INTEGER,
                    duration_seconds DOUBLE PRECISION,
                    feature_importances JSONB,
                    fold_results JSONB,
                    edge_distribution JSONB,
                    config JSONB,
                    feature_accuracy JSONB,
                    monthly_accuracy JSONB
                )
            """)
            cur.execute(f"""
                ALTER TABLE {MODEL_METRIC_SNAPSHOTS_TABLE}
                ADD COLUMN IF NOT EXISTS monthly_accuracy JSONB
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mms_trained_at
                ON {MODEL_METRIC_SNAPSHOTS_TABLE} (trained_at DESC)
            """)
        conn.commit()
    finally:
        conn.close()


def save_model_metric_snapshot(data: dict) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {MODEL_METRIC_SNAPSHOTS_TABLE}
                    (model_type, git_commit, num_features, training_rows, val_rows,
                     num_folds, mean_brier, mean_roi, total_bets, duration_seconds,
                     feature_importances, fold_results, edge_distribution, config,
                     feature_accuracy, monthly_accuracy)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                data.get("model_type"),
                data.get("git_commit"),
                data.get("num_features"),
                data.get("training_rows"),
                data.get("val_rows"),
                data.get("num_folds"),
                data.get("mean_brier"),
                data.get("mean_roi"),
                data.get("total_bets"),
                data.get("duration_seconds"),
                json.dumps(data.get("feature_importances", {})),
                json.dumps(data.get("fold_results", [])),
                json.dumps(data.get("edge_distribution", {})),
                json.dumps(data.get("config", {})),
                json.dumps(data.get("feature_accuracy", {})),
                json.dumps(data.get("monthly_accuracy", [])),
            ))
            run_id = cur.fetchone()[0]
        conn.commit()
        return run_id
    finally:
        conn.close()


def get_model_metric_snapshots(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM {MODEL_METRIC_SNAPSHOTS_TABLE}
                ORDER BY trained_at DESC LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_latest_model_metric_snapshot() -> dict | None:
    snapshots = get_model_metric_snapshots(limit=1)
    return snapshots[0] if snapshots else None


# ---------------------------------------------------------------------------
# Persisted prediction model artifacts
# ---------------------------------------------------------------------------

MODEL_ARTIFACTS_TABLE = "model_artifacts"


def init_model_artifacts_table():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {MODEL_ARTIFACTS_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    model_type TEXT NOT NULL,
                    training_fingerprint TEXT NOT NULL UNIQUE,
                    artifact_format TEXT NOT NULL DEFAULT 'pickle',
                    artifact_bytes BYTEA NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    settled_row_count INTEGER NOT NULL,
                    max_settled_game_date DATE,
                    active_season INTEGER,
                    target_train_cutoff DATE,
                    num_features INTEGER,
                    feature_columns JSONB NOT NULL,
                    early_feature_columns JSONB NOT NULL,
                    feature_config JSONB NOT NULL,
                    metrics JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    feature_importances JSONB NOT NULL DEFAULT '[]'::jsonb,
                    git_commit TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT FALSE
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_model_artifacts_created_at
                ON {MODEL_ARTIFACTS_TABLE} (created_at DESC)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_model_artifacts_active
                ON {MODEL_ARTIFACTS_TABLE} (is_active, created_at DESC)
            """)
            cur.execute(
                f"ALTER TABLE {MODEL_ARTIFACTS_TABLE} "
                "ADD COLUMN IF NOT EXISTS feature_importances JSONB NOT NULL DEFAULT '[]'::jsonb"
            )
            conn.commit()
            try:
                cur.execute("ALTER TABLE IF EXISTS bets ADD COLUMN IF NOT EXISTS model_artifact_id BIGINT")
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF to_regclass('public.bets') IS NOT NULL THEN
                            CREATE INDEX IF NOT EXISTS idx_bets_model_artifact_id
                            ON bets (model_artifact_id);
                        END IF;
                    END
                    $$;
                    """
                )
                conn.commit()
            except psycopg2.Error:
                conn.rollback()
                # Some production roles can read/write bets but do not own it.
                # Artifact caching still works; update_bet_prediction falls back
                # if model_artifact_id is unavailable.
    finally:
        conn.close()


def get_model_artifact_by_fingerprint(training_fingerprint: str) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT *
                FROM {MODEL_ARTIFACTS_TABLE}
                WHERE training_fingerprint = %s
                LIMIT 1
                """,
                (training_fingerprint,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def save_model_artifact(data: dict) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {MODEL_ARTIFACTS_TABLE} SET is_active = FALSE WHERE is_active")
            cur.execute(
                f"""
                INSERT INTO {MODEL_ARTIFACTS_TABLE} (
                    model_type, training_fingerprint, artifact_format,
                    artifact_bytes, artifact_sha256, settled_row_count,
                    max_settled_game_date, active_season, target_train_cutoff,
                    num_features, feature_columns, early_feature_columns,
                    feature_config, metrics, feature_importances, git_commit, is_active
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
                ON CONFLICT (training_fingerprint) DO UPDATE SET
                    is_active = TRUE
                RETURNING id
                """,
                (
                    data["model_type"],
                    data["training_fingerprint"],
                    data.get("artifact_format", "pickle"),
                    psycopg2.Binary(data["artifact_bytes"]),
                    data["artifact_sha256"],
                    data["settled_row_count"],
                    data.get("max_settled_game_date"),
                    data.get("active_season"),
                    data.get("target_train_cutoff"),
                    data.get("num_features"),
                    json.dumps(data.get("feature_columns", [])),
                    json.dumps(data.get("early_feature_columns", [])),
                    json.dumps(data.get("feature_config", {})),
                    json.dumps(data.get("metrics", {})),
                    json.dumps(data.get("feature_importances", [])),
                    data.get("git_commit"),
                ),
            )
            artifact_id = cur.fetchone()[0]
        conn.commit()
        return int(artifact_id)
    finally:
        conn.close()


def get_model_artifact_fi_history(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, created_at, model_type, training_fingerprint,
                       num_features, settled_row_count, max_settled_game_date,
                       feature_importances
                FROM {MODEL_ARTIFACTS_TABLE}
                ORDER BY created_at DESC NULLS LAST, id DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def migrate_encrypt_kalshi_keys() -> int:
    """One-time migration: encrypt any plaintext key_id values in kalshi_accounts."""
    if not _fernet:
        print("ENCRYPTION_KEY not set — skipping migration.")
        return 0
    conn = get_connection()
    count = 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email, key_id FROM kalshi_accounts")
            for email, key_id in cur.fetchall():
                if key_id and not key_id.startswith(_ENCRYPTED_PREFIX):
                    cur.execute(
                        "UPDATE kalshi_accounts SET key_id = %s, updated_at = NOW() WHERE email = %s",
                        (encrypt_field(key_id), email),
                    )
                    count += 1
        conn.commit()
    finally:
        conn.close()
    print(f"Encrypted {count} kalshi key_id(s).")
    return count


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
