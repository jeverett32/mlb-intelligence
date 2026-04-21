"""
dashboard/app.py — FastAPI backend for the MLB betting dashboard.
Serves the frontend at / and JSON data at /api/*.

Run from project root: uv run uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
"""

import math
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import db as DB

import bcrypt as _bcrypt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=[])
app = FastAPI(title="MLB Betting Dashboard")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests — slow down."},
        headers={"Retry-After": "60"},
    )

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


# Ensure auth tables exist on startup
DB.init_auth_tables()

COOKIE_NAME = "mlb_session"
COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30 days in seconds
# Set MLB_COOKIE_SECURE=1 when serving over HTTPS (behind Caddy/nginx).
COOKIE_SECURE = os.environ.get("MLB_COOKIE_SECURE", "0") == "1"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_session_email(request: Request):
    """Return the email for the current session, or None if not logged in."""
    session_id = request.cookies.get(COOKIE_NAME)
    return DB.get_session_email(session_id)


def require_auth(request: Request):
    """FastAPI dependency — raises 401 redirect if not authenticated."""
    email = _get_session_email(request)
    if not email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return email


# ---------------------------------------------------------------------------
# Login / logout routes (no auth required)
# ---------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = ""):
    # Already logged in → go to dashboard
    if _get_session_email(request):
        return RedirectResponse("/", status_code=302)
    error_html = f'<div class="error-banner">{error}</div>' if error else ""
    return (
        (TEMPLATES_DIR / "login.html")
        .read_text(encoding="utf-8")
        .replace("{{ERROR_BANNER}}", error_html)
    )


@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    stored_hash = DB.get_user_hash(email)
    if not stored_hash or not _verify_password(password, stored_hash):
        return RedirectResponse(
            "/login?error=Invalid+email+or+password", status_code=302
        )

    session_id = DB.create_session(email)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
    )
    return response


@app.post("/logout")
def logout(request: Request):
    session_id = request.cookies.get(COOKIE_NAME)
    if session_id:
        DB.delete_session(session_id)
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie(COOKIE_NAME)
    return response


# ---------------------------------------------------------------------------
# Frontend (auth required)
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not _get_session_email(request):
        return (TEMPLATES_DIR / "public.html").read_text(encoding="utf-8")
    return (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/public", response_class=HTMLResponse)
@limiter.limit("60/minute")
def public_page(request: Request):
    return (TEMPLATES_DIR / "public.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# API — Settings (live betting toggle)
# ---------------------------------------------------------------------------


class SettingPayload(BaseModel):
    value: str


@app.get("/api/settings")
def get_settings(email: str = Depends(require_auth)):
    return {"live_betting": DB.is_live_betting()}


@app.post("/api/settings/{key}")
def update_setting(
    key: str, payload: SettingPayload, email: str = Depends(require_auth)
):
    allowed = {"live_betting"}
    if key not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown setting: {key}")
    DB.set_setting(key, payload.value)
    return {"key": key, "value": payload.value}


# ---------------------------------------------------------------------------
# API — Bets
# ---------------------------------------------------------------------------


@app.get("/api/bets")
def get_bets(limit: int = 100, offset: int = 0, email: str = Depends(require_auth)):
    df = DB.get_all_bets()
    if df.empty:
        return {"bets": [], "total": 0}

    if "bet_dollars" in df.columns:
        df = df[df["bet_dollars"].notna()].copy()
        if not df.empty:
            df = df[df["bet_dollars"].astype(float) > 0].copy()

    if df.empty:
        return {"bets": [], "total": 0}

    if "result" in df.columns and "bet_dollars" in df.columns:

        def calc_pnl(row):
            if row.get("result") is None or row.get("bet_dollars") is None:
                return None
            won = (row["result"] is True and row["bet_side"] == "home") or (
                row["result"] is False and row["bet_side"] == "away"
            )
            bd = float(row["bet_dollars"] or 0)
            n_contracts = row.get("n_contracts")
            if won and n_contracts is not None:
                return round(float(n_contracts) - bd, 2)
            if won:
                mp = row.get("market_implied_prob")
                if mp and float(mp) > 0:
                    ratio = (
                        (1 / float(mp) - 1)
                        if row["bet_side"] == "home"
                        else (1 / (1 - float(mp)) - 1)
                    )
                    return round(bd * ratio, 2)
            if not won:
                return -bd
            return None

        df["profit_loss"] = df.apply(calc_pnl, axis=1)

    total = len(df)
    return {"bets": _safe_records(df.iloc[offset : offset + limit]), "total": total}


# ---------------------------------------------------------------------------
# API — Balance
# ---------------------------------------------------------------------------


@app.get("/api/balance")
def get_balance(email: str = Depends(require_auth)):
    df = DB.get_balance_history()
    if df.empty:
        return {"history": [], "current_dollars": 0.0}
    return {
        "history": _safe_records(df),
        "current_dollars": float(df.iloc[-1]["balance_dollars"]),
    }


# ---------------------------------------------------------------------------
# API — Upcoming games
# ---------------------------------------------------------------------------


@app.get("/api/upcoming")
def get_upcoming(email: str = Depends(require_auth)):
    try:
        games_df = DB.get_games_df(season=2026, upcoming_only=True)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if games_df.empty:
        return {"games": []}

    today = pd.Timestamp.now(tz="America/Denver").normalize().tz_localize(None)
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    upcoming = games_df[
        (games_df["game_date"] >= today)
        & (games_df["game_date"] <= today + pd.Timedelta(days=1))
    ].copy()

    bets_df = DB.get_all_bets()
    if not bets_df.empty:
        pred_cols = [
            "game_pk",
            "predicted_prob",
            "edge",
            "bet_side",
            "bet_frac",
            "market_implied_prob",
        ]
        available = [c for c in pred_cols if c in bets_df.columns]
        bets_df["game_pk"] = bets_df["game_pk"].astype(str)
        upcoming["game_pk"] = upcoming["game_pk"].astype(str)
        upcoming = upcoming.merge(bets_df[available], on="game_pk", how="left")

    if "game_time_utc" in upcoming.columns:
        upcoming = upcoming.sort_values("game_time_utc", na_position="last")

    cols = [
        "game_pk",
        "game_date",
        "game_time_utc",
        "home_team",
        "away_team",
        "home_implied_prob",
        "away_implied_prob",
        "close_home_ml",
        "close_away_ml",
        "predicted_prob",
        "edge",
        "bet_side",
        "bet_frac",
    ]
    return {
        "games": _safe_records(upcoming[[c for c in cols if c in upcoming.columns]])
    }


# ---------------------------------------------------------------------------
# API — Performance
# ---------------------------------------------------------------------------


def _compute_performance():
    bets_df = DB.get_all_bets()
    if bets_df.empty:
        return _empty_perf()

    settled = bets_df[
        bets_df["result"].notna() & bets_df["bet_dollars"].notna()
    ].copy()
    settled = settled[settled["bet_dollars"].astype(float) > 0]
    if settled.empty:
        return _empty_perf()

    result = settled["result"].astype(bool)
    side = settled["bet_side"]
    won_mask = ((result & (side == "home")) | (~result & (side == "away"))).to_numpy()

    bd = settled["bet_dollars"].astype(float).to_numpy()
    n_contracts = settled.get("n_contracts")
    n_contracts_arr = (
        pd.to_numeric(n_contracts, errors="coerce").to_numpy()
        if n_contracts is not None
        else np.full(len(settled), np.nan)
    )
    mp = pd.to_numeric(settled.get("market_implied_prob"), errors="coerce").to_numpy()
    is_home = (side == "home").to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(is_home, np.where(mp > 0, 1.0 / mp, 1.0),
                         np.where((mp > 0) & (mp < 1), 1.0 / (1.0 - mp), 1.0))
    returned_row = np.where(
        ~np.isnan(n_contracts_arr), n_contracts_arr, bd * ratio
    )

    total_bets = len(settled)
    wins = int(won_mask.sum())
    total_wagered = float(bd.sum())
    total_returned = float(returned_row[won_mask].sum())
    roi = (total_returned - total_wagered) / total_wagered if total_wagered else 0.0

    calibration = []
    if "predicted_prob" in settled.columns:
        settled = settled.assign(
            _won=won_mask,
            _pred_bin=(settled["predicted_prob"].astype(float) * 10).astype(int) / 10,
        )
        grouped = settled.groupby("_pred_bin")["_won"].agg(["mean", "size"])
        for bucket, row in grouped.iterrows():
            calibration.append({
                "predicted": float(bucket),
                "actual": float(row["mean"]),
                "n": int(row["size"]),
            })

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": total_bets - wins,
        "accuracy": round(wins / total_bets, 4) if total_bets else 0.0,
        "total_wagered": round(total_wagered, 2),
        "roi_pct": round(roi * 100, 2),
        "calibration": calibration,
    }


def _empty_perf():
    return {
        "total_bets": 0,
        "wins": 0,
        "losses": 0,
        "accuracy": 0.0,
        "total_wagered": 0.0,
        "roi_pct": 0.0,
        "calibration": [],
    }


@app.get("/api/performance")
def get_performance(email: str = Depends(require_auth)):
    return _compute_performance()


@app.get("/api/public/performance")
@limiter.limit("30/minute")
def get_public_performance(request: Request):
    return _compute_performance()


@app.get("/api/public/model-accuracy")
@limiter.limit("30/minute")
def get_public_model_accuracy(request: Request):
    return _compute_model_accuracy(include_recent=False)


@app.get("/api/model-accuracy")
def get_model_accuracy(email: str = Depends(require_auth)):
    return _compute_model_accuracy(include_recent=True)


def _compute_model_accuracy(include_recent: bool = True):
    df = DB.get_model_picks()
    if df.empty:
        return {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "market_total": 0,
            "market_correct": 0,
            "market_incorrect": 0,
            "market_accuracy": 0.0,
            "calibration": [],
            "recent": [],
        }

    df["predicted_home_win"] = df["predicted_prob"].astype(float) > 0.5
    df["home_win"] = df["home_win"].astype(bool)
    df["correct"] = df["predicted_home_win"] == df["home_win"]

    # Market prediction: market_implied_prob > 0.5 means market predicts home win
    df["market_predicted_home_win"] = df["market_implied_prob"].astype(float) > 0.5
    df["market_correct"] = df["market_predicted_home_win"] == df["home_win"]

    total = len(df)
    correct = int(df["correct"].sum())

    market_total = len(df[df["market_implied_prob"].notna()])
    market_correct = int(df[df["market_implied_prob"].notna()]["market_correct"].sum())

    # Calibration buckets
    calibration = []
    df["pred_bin"] = (df["predicted_prob"].astype(float) * 10).apply(int) / 10
    for bucket, grp in df.groupby("pred_bin"):
        calibration.append(
            {
                "predicted": float(bucket),
                "actual": float(grp["home_win"].astype(float).mean()),
                "n": len(grp),
            }
        )

    # Recent picks (last 20 settled games)
    recent = []
    rows_iter = df.head(20).iterrows() if include_recent else iter([])
    for _, r in rows_iter:
        market_prob = r.get("market_implied_prob")
        has_market = pd.notna(market_prob)
        market_correct_val = bool(r["market_correct"]) if has_market else None
        market_pred_val = bool(r["market_predicted_home_win"]) if has_market else None
        recent.append(
            {
                "game_date": str(r["game_date"])[:10],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "predicted_prob": float(r["predicted_prob"]),
                "market_prob": _safe_value(market_prob),
                "bet_side": r.get("bet_side"),
                "home_win": bool(r["home_win"]),
                "model_correct": bool(r["correct"]),
                "market_correct": market_correct_val,
                "market_pred_home": market_pred_val,
            }
        )

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "market_total": market_total,
        "market_correct": market_correct,
        "market_incorrect": market_total - market_correct,
        "market_accuracy": round(market_correct / market_total, 4)
        if market_total
        else 0.0,
        "calibration": calibration,
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# API — Database browser
# ---------------------------------------------------------------------------


@app.get("/api/db/{table}")
def browse_table(
    table: str, limit: int = 50, offset: int = 0, email: str = Depends(require_auth)
):
    try:
        return DB.browse_table(table, limit=limit, offset=offset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_value(v):
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return str(v)[:19]
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _safe_records(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        d = {}
        for k, v in row.items():
            d[k] = _safe_value(v)
        records.append(d)
    return records
