"""
dashboard/app.py — FastAPI backend for the MLB betting dashboard.
Serves the frontend at / and JSON data at /api/*.

Run from project root: uv run uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
"""

import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).parent.parent))
import db as DB
from fetch.fetch_balance import fetch_balance_for_account

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

class _CachedStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if resp.status_code == 200:
            resp.headers["Cache-Control"] = "public, max-age=604800, immutable"
        return resp


app.mount("/static", _CachedStatic(directory=str(STATIC_DIR)), name="static")


class CalibrationPoint(BaseModel):
    predicted: float
    actual: float
    n: int


class PublicPerformanceResponse(BaseModel):
    total_bets: int
    wins: int
    losses: int
    accuracy: float
    total_wagered: float
    roi_pct: float
    calibration: list[CalibrationPoint]


class PublicModelAccuracyResponse(BaseModel):
    total: int
    correct: int
    incorrect: int
    accuracy: float
    market_total: int
    market_correct: int
    market_incorrect: int
    market_accuracy: float
    calibration: list[CalibrationPoint]


class RecentModelPick(BaseModel):
    game_date: str
    away_team: str
    home_team: str
    predicted_prob: float
    market_prob: float | None
    bet_side: str | None
    home_win: bool
    model_correct: bool
    market_correct: bool | None
    market_pred_home: bool | None


class PrivateModelAccuracyResponse(PublicModelAccuracyResponse):
    recent: list[RecentModelPick]


class PublicSummaryResponse(BaseModel):
    performance: PublicPerformanceResponse
    model_accuracy: PublicModelAccuracyResponse


def _hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def _render_template(template_name: str) -> str:
    return (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")


def _secret_dir() -> Path:
    return Path(os.environ.get("KALSHI_SECRETS_DIR", str(Path(__file__).parent.parent / "secrets" / "kalshi")))


def _secret_path_for_email(email: str) -> Path:
    safe = "".join(c if c.isalnum() else "-" for c in email.lower())
    return _secret_dir() / f"{safe}.pem"


def _write_secret_file(path: Path, pem_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pem_text.strip() + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


# Ensure auth tables exist on startup, then purge expired sessions
DB.init_auth_tables()
try:
    _purged = DB.purge_expired_sessions()
    if _purged:
        print(f"Purged {_purged} expired session(s) on startup.")
except Exception as _e:
    print(f"Session purge on startup failed: {_e}")

COOKIE_NAME = "mlb_session"
COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30 days in seconds
# Set MLB_COOKIE_SECURE=1 when serving over HTTPS (behind Caddy/nginx).
COOKIE_SECURE = os.environ.get("MLB_COOKIE_SECURE", "0") == "1"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_current_user(request: Request):
    """Return the current session user, or None if not logged in."""
    session_id = request.cookies.get(COOKIE_NAME)
    return DB.get_session_user(session_id)


def require_session_user(request: Request):
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_approved_user(user: dict = Depends(require_session_user)):
    if user["approval_status"] != DB.USER_STATUS_APPROVED:
        raise HTTPException(status_code=403, detail="Account not approved")
    return user


def require_admin(user: dict = Depends(require_approved_user)):
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------------------------------------------------------------------------
# Login / logout routes (no auth required)
# ---------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = ""):
    # Already logged in → go to dashboard
    user = _get_current_user(request)
    if user and user["approval_status"] == DB.USER_STATUS_APPROVED:
        return RedirectResponse("/", status_code=302)
    error_html = f'<div class="error-banner">{error}</div>' if error else ""
    return _render_template("login.html").replace("{{ERROR_BANNER}}", error_html)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, error: str = ""):
    user = _get_current_user(request)
    if user and user["approval_status"] == DB.USER_STATUS_APPROVED:
        return RedirectResponse("/", status_code=302)
    error_html = f'<div class="error-banner">{error}</div>' if error else ""
    return _render_template("register.html").replace("{{ERROR_BANNER}}", error_html)


@app.get("/pending", response_class=HTMLResponse)
def pending_page():
    return _render_template("pending.html")


@app.get("/contact", response_class=HTMLResponse)
def contact_page():
    return _render_template("contact.html")


@app.post("/login")
@limiter.limit("10/minute")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    stored_hash = DB.get_user_hash(email)
    if not stored_hash or not _verify_password(password, stored_hash):
        return RedirectResponse(
            "/login?error=Invalid+email+or+password", status_code=302
        )
    user = DB.get_user(email)
    status = (user or {}).get("approval_status")
    if status == DB.USER_STATUS_PENDING:
        return RedirectResponse(
            "/login?error=Account+pending+admin+approval", status_code=302
        )
    if status == DB.USER_STATUS_REJECTED:
        return RedirectResponse(
            "/login?error=Registration+was+rejected.+Contact+an+admin", status_code=302
        )
    if status == DB.USER_STATUS_DISABLED:
        return RedirectResponse(
            "/login?error=Account+is+disabled", status_code=302
        )

    session_id = DB.create_session(
        email,
        created_by_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", ""),
    )
    DB.mark_user_login(email)
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


@app.post("/register")
@limiter.limit("5/minute")
async def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(""),
):
    email = email.strip().lower()
    if len(password) < 8:
        return RedirectResponse(
            "/register?error=Password+must+be+at+least+8+characters", status_code=302
        )
    created = DB.create_pending_user(email, _hash_password(password), full_name=full_name)
    if not created:
        existing = DB.get_user(email)
        if existing and existing["approval_status"] == DB.USER_STATUS_REJECTED:
            return RedirectResponse(
                "/register?error=Registration+blocked.+Contact+an+admin", status_code=302
            )
    return RedirectResponse("/pending", status_code=302)


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
    user = _get_current_user(request)
    if not user or user["approval_status"] != DB.USER_STATUS_APPROVED:
        return _render_template("landing.html")
    return _render_template("index.html")


@app.get("/public", response_class=HTMLResponse)
@limiter.limit("60/minute")
def public_page(request: Request):
    return _render_template("public.html")


@app.get("/health")
def health():
    return {"ok": True, "db": DB.ping()}


@app.exception_handler(404)
def _not_found(request: Request, exc):
    path = (TEMPLATES_DIR / "404.html")
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"), status_code=404)
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.exception_handler(500)
def _server_error(request: Request, exc):
    path = (TEMPLATES_DIR / "500.html")
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"), status_code=500)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# API — Settings (live betting toggle)
# ---------------------------------------------------------------------------


class SettingPayload(BaseModel):
    value: str


class KalshiAccountPayload(BaseModel):
    label: str = "Primary account"
    key_id: str
    private_key_pem: str
    kalshi_env: str = "prod"


DEFAULT_TIMEZONE = "America/Denver"


def _get_dashboard_timezone() -> str:
    tz = DB.get_setting("dashboard_timezone", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        ZoneInfo(tz)
        return tz
    except Exception:
        return DEFAULT_TIMEZONE


@app.get("/api/settings")
def get_settings(user: dict = Depends(require_approved_user)):
    return {
        "live_betting": DB.is_user_live_betting(user["email"]),
        "dashboard_timezone": DB.get_user_setting(
            user["email"], "dashboard_timezone", _get_dashboard_timezone()
        ),
        "global_live_betting": DB.is_global_live_betting() if user["is_admin"] else None,
        "effective_live_betting": DB.is_global_live_betting() and DB.is_user_live_betting(user["email"]),
    }


@app.post("/api/settings/{key}")
def update_setting(
    key: str, payload: SettingPayload, user: dict = Depends(require_approved_user)
):
    allowed = {"live_betting", "dashboard_timezone"}
    if key not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown setting: {key}")
    if key == "dashboard_timezone":
        try:
            ZoneInfo(payload.value)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid IANA timezone") from e
    DB.set_user_setting(user["email"], key, payload.value)
    return {"key": key, "value": payload.value}


@app.post("/api/admin/settings/{key}")
def update_admin_setting(
    key: str, payload: SettingPayload, user: dict = Depends(require_admin)
):
    allowed = {"global_live_betting"}
    if key not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown admin setting: {key}")
    DB.set_setting(key, payload.value)
    return {"key": key, "value": payload.value}


@app.get("/api/me")
def get_me(user: dict = Depends(require_approved_user)):
    return {
        "email": user["email"],
        "full_name": user.get("full_name") or "",
        "is_admin": bool(user["is_admin"]),
        "approval_status": user["approval_status"],
    }


@app.get("/api/account/kalshi")
def get_kalshi_account(user: dict = Depends(require_approved_user)):
    account = DB.get_kalshi_account(user["email"])
    if not account:
        return {"connected": False}
    return {
        "connected": True,
        "label": account["label"],
        "key_id": account["key_id"],
        "kalshi_env": account["kalshi_env"],
        "is_active": account["is_active"],
        "last_verified_at": account["last_verified_at"],
        "last_error": account["last_error"],
    }


@app.post("/api/account/kalshi")
def connect_kalshi_account(
    payload: KalshiAccountPayload,
    user: dict = Depends(require_approved_user),
):
    if payload.kalshi_env not in {"prod", "demo"}:
        raise HTTPException(status_code=400, detail="Invalid Kalshi environment")
    path = _secret_path_for_email(user["email"])
    _write_secret_file(path, payload.private_key_pem)
    try:
        balance_cents = fetch_balance_for_account(
            key_id=payload.key_id,
            key_path=str(path),
            kalshi_env=payload.kalshi_env,
            email=user["email"],
        )
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    DB.upsert_kalshi_account(
        user["email"],
        label=payload.label,
        key_id=payload.key_id,
        key_path=str(path),
        kalshi_env=payload.kalshi_env,
        last_verified=True,
        last_error="",
    )
    return {
        "connected": True,
        "label": payload.label,
        "key_id": payload.key_id,
        "kalshi_env": payload.kalshi_env,
        "balance_cents": balance_cents,
    }


@app.delete("/api/account/kalshi")
def disconnect_kalshi_account(user: dict = Depends(require_approved_user)):
    account = DB.get_kalshi_account(user["email"])
    if account and account.get("key_path"):
        try:
            Path(account["key_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    DB.delete_kalshi_account(user["email"])
    return {"connected": False}


# ---------------------------------------------------------------------------
# API — Bets
# ---------------------------------------------------------------------------


@app.get("/api/bets")
def get_bets(
    limit: int = 100,
    offset: int = 0,
    status: str = "all",
    user: dict = Depends(require_approved_user),
):
    df = DB.get_user_orders(user["email"])
    if df.empty:
        return {"bets": [], "total": 0}

    if "bet_dollars" in df.columns:
        df = df[df["bet_dollars"].notna()].copy()
        if not df.empty:
            df = df[df["bet_dollars"].astype(float) > 0].copy()

    if df.empty:
        return {"bets": [], "total": 0}

    if status == "open" and "result" in df.columns:
        df = df[df["result"].isna()].copy()
    elif status == "settled" and "result" in df.columns:
        df = df[df["result"].notna()].copy()
    elif status != "all":
        raise HTTPException(status_code=400, detail="Invalid bets status filter")

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


@app.get("/api/open-bets")
def get_open_bets(user: dict = Depends(require_approved_user)):
    df = DB.get_user_orders(user["email"])
    if df.empty:
        return {"bets": [], "total": 0}

    if "bet_dollars" in df.columns:
        df = df[df["bet_dollars"].notna()].copy()
        if not df.empty:
            df = df[df["bet_dollars"].astype(float) > 0].copy()

    if df.empty or "result" not in df.columns:
        return {"bets": [], "total": 0}

    open_df = df[df["result"].isna()].copy()
    if open_df.empty:
        return {"bets": [], "total": 0}

    if "game_pk" in open_df.columns:
        open_df["game_pk"] = open_df["game_pk"].astype(str)

    games_df = DB.get_games_df(season=2026, upcoming_only=True)
    if not games_df.empty:
        games_df = games_df.copy()
        if "game_pk" in games_df.columns:
            games_df["game_pk"] = games_df["game_pk"].astype(str)
        game_cols = [c for c in ("game_pk", "game_time_utc") if c in games_df.columns]
        if game_cols:
            open_df = open_df.merge(games_df[game_cols], on="game_pk", how="left")

    sort_cols = [c for c in ("game_date", "game_time_utc", "game_pk") if c in open_df.columns]
    if sort_cols:
        ascending = {"game_date": False, "game_time_utc": True, "game_pk": True}
        open_df = open_df.sort_values(
            by=sort_cols,
            ascending=[ascending[c] for c in sort_cols],
        )
    return {"bets": _safe_records(open_df), "total": len(open_df)}


# ---------------------------------------------------------------------------
# API — Balance
# ---------------------------------------------------------------------------


@app.get("/api/balance")
def get_balance(user: dict = Depends(require_approved_user)):
    df = DB.get_user_balance_history(user["email"])
    if df.empty:
        account = DB.get_kalshi_account(user["email"])
        if account and account.get("is_active"):
            try:
                fetch_balance_for_account(
                    key_id=account["key_id"],
                    key_path=account["key_path"],
                    kalshi_env=account["kalshi_env"],
                    email=user["email"],
                )
                df = DB.get_user_balance_history(user["email"])
            except Exception:
                pass
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
def get_upcoming(user: dict = Depends(require_approved_user)):
    try:
        games_df = DB.get_games_df(season=2026, upcoming_only=True)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if games_df.empty:
        return {"games": []}

    games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce")
    if "game_time_utc" in games_df.columns:
        game_times = pd.to_datetime(games_df["game_time_utc"], errors="coerce", utc=True)
        now_utc = pd.Timestamp.now(tz=timezone.utc)
        cutoff_utc = now_utc + pd.Timedelta(hours=48)
        user_tz = ZoneInfo(
            DB.get_user_setting(user["email"], "dashboard_timezone", _get_dashboard_timezone())
        )
        today_local = now_utc.astimezone(user_tz).date()
        game_local_dates = game_times.dt.tz_convert(user_tz).dt.date
        in_forward_window = (game_times >= now_utc) & (game_times <= cutoff_utc)
        same_local_day = game_local_dates == today_local
        no_start_time = game_times.isna()
        date_window_start = pd.Timestamp(today_local)
        date_window_end = date_window_start + pd.Timedelta(days=1)
        same_day_fallback = no_start_time & (
            (games_df["game_date"] >= date_window_start) & (games_df["game_date"] <= date_window_end)
        )
        upcoming = games_df[in_forward_window | same_local_day | same_day_fallback].copy()
    else:
        user_tz = ZoneInfo(
            DB.get_user_setting(user["email"], "dashboard_timezone", _get_dashboard_timezone())
        )
        today = pd.Timestamp.now(tz=user_tz).normalize().tz_localize(None)
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
            "bet_dollars",
            "result",
            "kalshi_order_id",
        ]
        available = [c for c in pred_cols if c in bets_df.columns]
        bets_df["game_pk"] = bets_df["game_pk"].astype(str)
        upcoming["game_pk"] = upcoming["game_pk"].astype(str)
        upcoming = upcoming.merge(bets_df[available], on="game_pk", how="left")

    user_orders = DB.get_user_orders(user["email"])
    if not user_orders.empty:
        user_orders = user_orders.copy()
        user_orders["game_pk"] = user_orders["game_pk"].astype(str)
        order_cols = [
            "game_pk",
            "bet_dollars",
            "n_contracts",
            "kalshi_order_id",
            "status",
            "dry_run",
            "live_price",
            "live_edge",
            "result",
        ]
        available = [c for c in order_cols if c in user_orders.columns]
        upcoming = upcoming.merge(
            user_orders[available],
            on="game_pk",
            how="left",
            suffixes=("", "_user"),
        )

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
        "bet_dollars",
        "n_contracts",
        "result",
        "kalshi_order_id",
        "status",
        "dry_run",
        "live_price",
        "live_edge",
    ]
    return {
        "games": _safe_records(upcoming[[c for c in cols if c in upcoming.columns]])
    }


# ---------------------------------------------------------------------------
# API — Performance
# ---------------------------------------------------------------------------


def _build_public_performance(email: str | None = None) -> PublicPerformanceResponse:
    bets_df = DB.get_user_orders(email) if email else DB.get_all_bets()
    if bets_df.empty:
        return _empty_public_performance()

    settled = bets_df[
        bets_df["result"].notna() & bets_df["bet_dollars"].notna()
    ].copy()
    settled = settled[settled["bet_dollars"].astype(float) > 0]
    if settled.empty:
        return _empty_public_performance()

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

    calibration: list[CalibrationPoint] = []
    if "predicted_prob" in settled.columns:
        settled = settled.assign(
            _won=won_mask,
            _pred_bin=(settled["predicted_prob"].astype(float) * 10).astype(int) / 10,
        )
        grouped = settled.groupby("_pred_bin")["_won"].agg(["mean", "size"])
        for bucket, row in grouped.iterrows():
            calibration.append(
                CalibrationPoint(
                    predicted=float(bucket),
                    actual=float(row["mean"]),
                    n=int(row["size"]),
                )
            )

    return PublicPerformanceResponse(
        total_bets=total_bets,
        wins=wins,
        losses=total_bets - wins,
        accuracy=round(wins / total_bets, 4) if total_bets else 0.0,
        total_wagered=round(total_wagered, 2),
        roi_pct=round(roi * 100, 2),
        calibration=calibration,
    )


def _empty_public_performance() -> PublicPerformanceResponse:
    return PublicPerformanceResponse(
        total_bets=0,
        wins=0,
        losses=0,
        accuracy=0.0,
        total_wagered=0.0,
        roi_pct=0.0,
        calibration=[],
    )


@app.get("/api/performance", response_model=PublicPerformanceResponse)
def get_performance(user: dict = Depends(require_approved_user)):
    return _build_public_performance(user["email"])


@app.get("/api/public/performance", response_model=PublicPerformanceResponse)
@limiter.limit("30/minute")
def get_public_performance(request: Request):
    return _build_public_performance()


@app.get("/api/public/model-accuracy", response_model=PublicModelAccuracyResponse)
@limiter.limit("30/minute")
def get_public_model_accuracy(request: Request):
    return _build_public_model_accuracy()


@app.get("/api/public/summary", response_model=PublicSummaryResponse)
@limiter.limit("30/minute")
def get_public_summary(request: Request):
    return _build_public_summary()


@app.get("/api/model-accuracy", response_model=PrivateModelAccuracyResponse)
def get_model_accuracy(user: dict = Depends(require_approved_user)):
    return _build_private_model_accuracy()


def _compute_model_accuracy_metrics() -> tuple[dict, pd.DataFrame]:
    df = DB.get_model_picks()
    if df.empty:
        return (_empty_model_accuracy_metrics(), df)

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
    calibration: list[CalibrationPoint] = []
    df["pred_bin"] = (df["predicted_prob"].astype(float) * 10).apply(int) / 10
    for bucket, grp in df.groupby("pred_bin"):
        calibration.append(
            CalibrationPoint(
                predicted=float(bucket),
                actual=float(grp["home_win"].astype(float).mean()),
                n=len(grp),
            )
        )

    return (
        {
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
        },
        df,
    )


def _empty_model_accuracy_metrics() -> dict:
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
    }


def _build_recent_model_picks(df: pd.DataFrame, limit: int = 20) -> list[RecentModelPick]:
    recent: list[RecentModelPick] = []
    for _, r in df.head(limit).iterrows():
        market_prob = r.get("market_implied_prob")
        has_market = pd.notna(market_prob)
        market_correct_val = bool(r["market_correct"]) if has_market else None
        market_pred_val = bool(r["market_predicted_home_win"]) if has_market else None
        recent.append(
            RecentModelPick(
                game_date=str(r["game_date"])[:10],
                away_team=r["away_team"],
                home_team=r["home_team"],
                predicted_prob=float(r["predicted_prob"]),
                market_prob=_safe_value(market_prob),
                bet_side=r.get("bet_side"),
                home_win=bool(r["home_win"]),
                model_correct=bool(r["correct"]),
                market_correct=market_correct_val,
                market_pred_home=market_pred_val,
            )
        )
    return recent


def _compute_public_model_accuracy() -> tuple[PublicModelAccuracyResponse, pd.DataFrame]:
    metrics, df = _compute_model_accuracy_metrics()
    return PublicModelAccuracyResponse(**metrics), df


def _build_public_model_accuracy() -> PublicModelAccuracyResponse:
    model_accuracy, _ = _compute_public_model_accuracy()
    return model_accuracy


def _build_private_model_accuracy() -> PrivateModelAccuracyResponse:
    model_accuracy, df = _compute_public_model_accuracy()
    return PrivateModelAccuracyResponse(
        **model_accuracy.model_dump(),
        recent=_build_recent_model_picks(df),
    )


def _build_public_summary() -> PublicSummaryResponse:
    performance = _build_public_performance()
    model_accuracy = _build_public_model_accuracy()
    return PublicSummaryResponse(
        performance=performance,
        model_accuracy=model_accuracy,
    )


# ---------------------------------------------------------------------------
# API — Database browser
# ---------------------------------------------------------------------------


@app.get("/api/db/{table}")
def browse_table(
    table: str, limit: int = 50, offset: int = 0, user: dict = Depends(require_admin)
):
    try:
        return DB.browse_table(table, limit=limit, offset=offset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/users")
def get_admin_users(
    status: str = "all",
    user: dict = Depends(require_admin),
):
    if status != "all" and status not in DB.USER_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid user status")
    return {"users": DB.list_users(status=status)}


class AdminUserActionPayload(BaseModel):
    status: str | None = None
    is_admin: bool | None = None
    rejection_reason: str = ""


@app.post("/api/admin/users/{email}/status")
def admin_set_user_status(
    email: str,
    payload: AdminUserActionPayload,
    user: dict = Depends(require_admin),
):
    if payload.status not in DB.USER_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid user status")
    email = email.strip().lower()
    DB.set_user_approval_status(
        email,
        payload.status,
        actor_email=user["email"],
        rejection_reason=payload.rejection_reason,
    )
    if payload.status != DB.USER_STATUS_APPROVED:
        DB.revoke_sessions_for_email(email)
    updated = DB.get_user(email)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated


@app.post("/api/admin/users/{email}/admin")
def admin_set_user_admin(
    email: str,
    payload: AdminUserActionPayload,
    user: dict = Depends(require_admin),
):
    if payload.is_admin is None:
        raise HTTPException(status_code=400, detail="Missing is_admin")
    email = email.strip().lower()
    DB.set_user_admin(email, payload.is_admin)
    updated = DB.get_user(email)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_value(v):
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return None
        if v.tzinfo is None:
            v = v.tz_localize(timezone.utc)
        return v.isoformat()
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
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
