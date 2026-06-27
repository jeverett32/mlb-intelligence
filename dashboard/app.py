"""
dashboard/app.py — FastAPI backend for the MLB betting dashboard.
Serves the frontend at / and JSON data at /api/*.

Run from project root: uv run uvicorn dashboard.app:app --host <bind_addr> --port <port>
"""

import html
import json
import math
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).parent.parent))
import db as DB
from config import ACTIVE_SEASON
from fetch.fetch_balance import fetch_balance_for_account
from fetch.fetch_live_positions import refresh_due_orders
from fetch.fetch_live_scores import refresh_scores_for_date

import bcrypt as _bcrypt
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=[])
app = FastAPI(title="MLB Intelligence", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_middleware(GZipMiddleware, minimum_size=500)
SIGNAL_MIN_EDGE = float(os.environ.get("KALSHI_EXECUTION_MIN_EDGE", "0.0"))
BASE_URL = os.environ.get("MLB_BASE_URL", "https://mlbintelligence.com")


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


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    if COOKIE_SECURE:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


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


class PublicRecentBet(BaseModel):
    game_date: str
    matchup: str
    side: str
    result: str
    model_prob: float | None
    market_prob: float | None
    edge: float | None
    return_pct: float


class PublicRoiPoint(BaseModel):
    game_date: str
    roi_pct: float


class PublicReceiptsResponse(BaseModel):
    last_updated_utc: str
    settled_bets: int
    first_bet_date: str | None
    last_bet_date: str | None
    recent_bets: list[PublicRecentBet]
    roi_series: list[PublicRoiPoint]


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


def _db_secret_ref_for_email(email: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in email.lower())
    return f"db://kalshi_accounts/{safe}/private_key_pem"


def _write_secret_file(path: Path, pem_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pem_text.strip() + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _safe_next_path(value: str = "") -> str:
    value = (value or "").strip()
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


# Ensure auth tables exist on startup, then purge expired sessions
DB.init_auth_tables()
try:
    DB.init_bets_explainability()
except Exception as _e:
    print(f"Explainability column init failed: {_e}")
try:
    DB.init_bets_execution_tracking()
except Exception as _e:
    print(f"Execution-tracking column init failed: {_e}")
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
def login_page(request: Request, error: str = "", next: str = ""):
    # Already logged in → go to dashboard
    next_path = _safe_next_path(next)
    user = _get_current_user(request)
    if user and user["approval_status"] == DB.USER_STATUS_APPROVED:
        return RedirectResponse(next_path, status_code=302)
    error_html = f'<div class="error-banner" role="alert">{html.escape(error)}</div>' if error else ""
    next_input = f'<input type="hidden" name="next" value="{html.escape(next_path)}">' if next_path != "/" else ""
    return (
        _render_template("login.html")
        .replace("{{ERROR_BANNER}}", error_html)
        .replace("{{NEXT_INPUT}}", next_input)
    )


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, error: str = ""):
    user = _get_current_user(request)
    if user and user["approval_status"] == DB.USER_STATUS_APPROVED:
        return RedirectResponse("/", status_code=302)
    error_html = f'<div class="error-banner" role="alert">{html.escape(error)}</div>' if error else ""
    return _render_template("register.html").replace("{{ERROR_BANNER}}", error_html)


@app.get("/pending", response_class=HTMLResponse)
def pending_page():
    return _render_template("pending.html")


@app.get("/contact", response_class=HTMLResponse)
def contact_page():
    return _render_template("contact.html")


@app.get("/research", response_class=HTMLResponse)
def research_page():
    return _render_template("research.html")


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    return _render_template("privacy.html")


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /admin\n"
        "Disallow: /settings\n"
        "Disallow: /api/\n"
        f"\nSitemap: {BASE_URL}/sitemap.xml\n"
    )


@app.get("/sitemap.xml")
def sitemap_xml():
    pages = ["/", "/home", "/public", "/research", "/privacy", "/contact", "/docs/api"]
    urls = "\n".join(
        f"  <url><loc>{BASE_URL}{p}</loc></url>" for p in pages
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>"
    )
    return Response(content=xml, media_type="application/xml")


@app.get("/api-docs")
def legacy_api_docs_page():
    return RedirectResponse("/docs/api", status_code=301)


@app.get("/docs")
def docs_index_page():
    return RedirectResponse("/docs/api", status_code=302)


@app.get("/docs/api", response_class=HTMLResponse)
def api_docs_page():
    return _render_template("api-docs.html")


@app.get("/docs/kalshi", response_class=HTMLResponse)
def kalshi_docs_page():
    return _render_template("kalshi-setup.html")


@app.post("/login")
@limiter.limit("10/minute")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form(""),
):
    email = email.strip().lower()
    next_path = _safe_next_path(next)
    next_query = f"&next={quote_plus(next_path)}" if next_path != "/" else ""
    stored_hash = DB.get_user_hash(email)
    if not stored_hash or not _verify_password(password, stored_hash):
        return RedirectResponse(
            f"/login?error=Invalid+email+or+password{next_query}", status_code=302
        )
    user = DB.get_user(email)
    status = (user or {}).get("approval_status")
    if status == DB.USER_STATUS_PENDING:
        return RedirectResponse(
            f"/login?error=Account+pending+admin+approval{next_query}", status_code=302
        )
    if status == DB.USER_STATUS_REJECTED:
        return RedirectResponse(
            f"/login?error=Registration+was+rejected.+Contact+an+admin{next_query}", status_code=302
        )
    if status == DB.USER_STATUS_DISABLED:
        return RedirectResponse(
            f"/login?error=Account+is+disabled{next_query}", status_code=302
        )

    session_id = DB.create_session(
        email,
        created_by_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", ""),
    )
    DB.mark_user_login(email)
    response = RedirectResponse(next_path, status_code=302)
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
    effective_live = DB.is_global_live_betting() and DB.is_user_live_betting(user["email"])
    html = _render_template("index.html")
    html = html.replace("{{EFFECTIVE_LIVE_BETTING}}", "true" if effective_live else "false")
    return html


@app.get("/home", response_class=HTMLResponse)
def public_home():
    return _render_template("landing.html")


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user: dict = Depends(require_approved_user)):
    return HTMLResponse(_render_template("settings.html"))


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, user: dict = Depends(require_admin)):
    return HTMLResponse(_render_template("admin.html"))


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
    private_key_pem: str = ""
    kalshi_env: str = "prod"


class ApiTokenPayload(BaseModel):
    label: str = "Signal follower"


class ClientBalancePayload(BaseModel):
    balance_cents: int
    source: str = "self_custody"


class ClientOrderPayload(BaseModel):
    signal_id: str = ""
    game_pk: str
    game_date: str = ""
    home_team: str = ""
    away_team: str = ""
    predicted_prob: float | None = None
    market_implied_prob: float | None = None
    edge: float | None = None
    bet_side: str = "none"
    bet_frac: float = 0.0
    bet_dollars: float | None = None
    n_contracts: int | None = None
    kalshi_order_id: str | None = None
    kalshi_ticker: str | None = None
    live_price: float | None = None
    live_edge: float | None = None
    status: str = "pending"
    dry_run: bool = False
    result: bool | None = None
    profit_loss: float | None = None
    error: str = ""


class ClientHeartbeatPayload(BaseModel):
    version: str = ""
    kalshi_env: str = "prod"


DEFAULT_TIMEZONE = "America/Denver"

TEAM_INFO = {
    "ARI": {"name": "Arizona Diamondbacks", "league": "NL", "division": "West"},
    "ATL": {"name": "Atlanta Braves", "league": "NL", "division": "East"},
    "BAL": {"name": "Baltimore Orioles", "league": "AL", "division": "East"},
    "BOS": {"name": "Boston Red Sox", "league": "AL", "division": "East"},
    "CHC": {"name": "Chicago Cubs", "league": "NL", "division": "Central"},
    "CHW": {"name": "Chicago White Sox", "league": "AL", "division": "Central"},
    "CIN": {"name": "Cincinnati Reds", "league": "NL", "division": "Central"},
    "CLE": {"name": "Cleveland Guardians", "league": "AL", "division": "Central"},
    "COL": {"name": "Colorado Rockies", "league": "NL", "division": "West"},
    "DET": {"name": "Detroit Tigers", "league": "AL", "division": "Central"},
    "HOU": {"name": "Houston Astros", "league": "AL", "division": "West"},
    "KCR": {"name": "Kansas City Royals", "league": "AL", "division": "Central"},
    "LAA": {"name": "Los Angeles Angels", "league": "AL", "division": "West"},
    "LAD": {"name": "Los Angeles Dodgers", "league": "NL", "division": "West"},
    "MIA": {"name": "Miami Marlins", "league": "NL", "division": "East"},
    "MIL": {"name": "Milwaukee Brewers", "league": "NL", "division": "Central"},
    "MIN": {"name": "Minnesota Twins", "league": "AL", "division": "Central"},
    "NYM": {"name": "New York Mets", "league": "NL", "division": "East"},
    "NYY": {"name": "New York Yankees", "league": "AL", "division": "East"},
    "ATH": {"name": "Athletics", "league": "AL", "division": "West"},
    "PHI": {"name": "Philadelphia Phillies", "league": "NL", "division": "East"},
    "PIT": {"name": "Pittsburgh Pirates", "league": "NL", "division": "Central"},
    "SDP": {"name": "San Diego Padres", "league": "NL", "division": "West"},
    "SEA": {"name": "Seattle Mariners", "league": "AL", "division": "West"},
    "SFG": {"name": "San Francisco Giants", "league": "NL", "division": "West"},
    "STL": {"name": "St. Louis Cardinals", "league": "NL", "division": "Central"},
    "TBR": {"name": "Tampa Bay Rays", "league": "AL", "division": "East"},
    "TEX": {"name": "Texas Rangers", "league": "AL", "division": "West"},
    "TOR": {"name": "Toronto Blue Jays", "league": "AL", "division": "East"},
    "WSN": {"name": "Washington Nationals", "league": "NL", "division": "East"},
}

LEAGUE_LABELS = {"AL": "American League", "NL": "National League"}

DIVISION_ORDER = [
    ("NL", "East"),
    ("AL", "East"),
    ("NL", "Central"),
    ("AL", "Central"),
    ("NL", "West"),
    ("AL", "West"),
]


def _get_dashboard_timezone() -> str:
    tz = DB.get_setting("dashboard_timezone", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        ZoneInfo(tz)
        return tz
    except Exception:
        return DEFAULT_TIMEZONE


def _get_bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        return ""
    return auth[7:].strip()


def require_signal_reader(request: Request):
    bearer = _get_bearer_token(request)
    if bearer:
        user = DB.get_user_for_api_token(bearer, required_scope="signals:read")
        if not user or user["approval_status"] != DB.USER_STATUS_APPROVED:
            raise HTTPException(status_code=403, detail="Invalid API token")
        return user
    user = _get_current_user(request)
    if not user or user["approval_status"] != DB.USER_STATUS_APPROVED:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_client_writer(request: Request):
    bearer = _get_bearer_token(request)
    if not bearer:
        raise HTTPException(status_code=401, detail="Missing API token")
    user = DB.get_user_for_api_token(bearer, required_scope="client:write")
    if not user or user["approval_status"] != DB.USER_STATUS_APPROVED:
        raise HTTPException(status_code=403, detail="Invalid API token")
    return user


@app.get("/api/settings")
def get_settings(user: dict = Depends(require_approved_user)):
    last_seen = DB.get_user_setting(user["email"], "self_custody_last_seen_at", "")
    last_error = DB.get_user_setting(user["email"], "self_custody_last_error", "")
    out = {
        "live_betting": DB.is_user_live_betting(user["email"]),
        "dashboard_timezone": DB.get_user_setting(
            user["email"], "dashboard_timezone", _get_dashboard_timezone()
        ),
        "global_live_betting": DB.is_global_live_betting(),
        "can_manage_global_live_betting": bool(user["is_admin"]),
        "effective_live_betting": DB.is_global_live_betting() and DB.is_user_live_betting(user["email"]),
        "execution_mode": DB.get_user_setting(user["email"], "execution_mode", "server_managed"),
        "self_custody": {
            "last_seen_at": last_seen or None,
            "kalshi_env": DB.get_user_setting(user["email"], "self_custody_kalshi_env", ""),
            "client_version": DB.get_user_setting(user["email"], "self_custody_client_version", ""),
            "last_error": last_error or None,
        },
    }
    if user["is_admin"]:
        out["active_live_model_version"] = DB.get_active_live_model_version()
        out["live_model_versions"] = DB.list_live_model_versions()
    return out


@app.post("/api/settings/{key}")
def update_setting(
    key: str, payload: SettingPayload, user: dict = Depends(require_approved_user)
):
    allowed = {"live_betting", "dashboard_timezone", "execution_mode"}
    if key not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown setting: {key}")
    if key == "dashboard_timezone":
        try:
            ZoneInfo(payload.value)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid IANA timezone") from e
    if key == "execution_mode" and payload.value not in {"server_managed", "self_custody", "paper_only"}:
        raise HTTPException(status_code=400, detail="Invalid execution mode")
    DB.set_user_setting(user["email"], key, payload.value)
    return {"key": key, "value": payload.value}


@app.post("/api/admin/settings/{key}")
def update_admin_setting(
    key: str, payload: SettingPayload, user: dict = Depends(require_admin)
):
    allowed = {"global_live_betting", "active_live_model_version"}
    if key not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown admin setting: {key}")
    if key == "active_live_model_version":
        try:
            DB.set_active_live_model_version(payload.value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"key": key, "value": DB.get_active_live_model_version()}
    DB.set_setting(key, payload.value)
    return {"key": key, "value": payload.value}


@app.post("/api/admin/pipeline-action")
@limiter.limit("5/minute")
def run_pipeline_action(
    request: Request, payload: dict, user: dict = Depends(require_admin)
):
    action = payload.get("action")
    allowed = {"settle_games", "refresh_schedule", "refresh_balance", "refresh_live_scores", "refresh_live_bets"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    project_root = str(Path(__file__).parent.parent)

    if action == "settle_games":
        try:
            from settle_games import settle_completed_games
            stats = settle_completed_games()
            return {
                "ok": True,
                "message": (
                    f"Settled {stats['games_settled']}/{stats['games_checked']} game(s); "
                    f"updated {stats['user_orders_updated']} user + "
                    f"{stats['paper_orders_updated']} paper bet row(s)."
                ),
            }
        except Exception as e:
            return {"ok": False, "message": str(e)}

    elif action == "refresh_schedule":
        try:
            result = subprocess.run(
                [sys.executable, "fetch/fetch_data.py"],
                cwd=project_root,
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                return {"ok": True, "message": "Schedule refreshed."}
            return {"ok": False, "message": result.stderr[:500] or "Fetch failed."}
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": "Timed out after 120s."}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    elif action == "refresh_balance":
        try:
            account = DB.get_kalshi_account(user["email"])
            if not account or not account.get("is_active"):
                return {"ok": False, "message": "No active Kalshi account linked."}
            balance = fetch_balance_for_account(
                key_id=account["key_id"],
                key_path=account["key_path"],
                kalshi_env=account["kalshi_env"],
                email=user["email"],
                private_key_pem=account.get("private_key_pem") or None,
            )
            return {"ok": True, "message": f"Balance: ${balance / 100:.2f}"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    elif action == "refresh_live_scores":
        try:
            today = pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d")
            updated = refresh_scores_for_date(today)
            return {"ok": True, "message": f"Refreshed live scores for {today} ({updated} game(s) updated)."}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    elif action == "refresh_live_bets":
        try:
            from fetch.fetch_live_positions import refresh_due_orders
            stats = refresh_due_orders(stale_seconds=0)
            return {
                "ok": True,
                "message": f"Checked {stats.checked} order(s), updated {stats.updated}, skipped {stats.skipped}, errors {stats.errors}.",
            }
        except Exception as e:
            return {"ok": False, "message": str(e)}


@app.get("/api/me")
def get_me(user: dict = Depends(require_approved_user)):
    return {
        "email": user["email"],
        "full_name": user.get("full_name") or "",
        "is_admin": bool(user["is_admin"]),
        "approval_status": user["approval_status"],
    }


@app.get("/api/account/api-tokens")
def get_api_tokens(user: dict = Depends(require_approved_user)):
    return {"tokens": _safe_records(pd.DataFrame(DB.list_api_tokens(user["email"])))}


@app.post("/api/account/api-tokens")
def create_api_token(
    payload: ApiTokenPayload,
    user: dict = Depends(require_approved_user),
):
    created = DB.create_api_token(user["email"], label=payload.label)
    return created


@app.delete("/api/account/api-tokens/{token_id}")
def delete_api_token(token_id: int, user: dict = Depends(require_approved_user)):
    deleted = DB.revoke_api_token(user["email"], token_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API token not found")
    return {"ok": True}


@app.get("/api/account/kalshi")
def get_kalshi_account(user: dict = Depends(require_approved_user)):
    account = DB.get_kalshi_account(user["email"])
    if not account:
        return {"connected": False}
    try:
        balance_cents = DB.get_last_user_balance_cents(user["email"])
    except Exception:
        balance_cents = None
    return {
        "connected": True,
        "label": account["label"],
        "key_id": account["key_id"],
        "kalshi_env": account["kalshi_env"],
        "is_active": account["is_active"],
        "last_verified_at": account["last_verified_at"],
        "last_error": account["last_error"],
        "balance_cents": balance_cents,
    }


@app.post("/api/account/kalshi")
def connect_kalshi_account(
    payload: KalshiAccountPayload,
    user: dict = Depends(require_approved_user),
):
    if payload.kalshi_env not in {"prod", "demo"}:
        raise HTTPException(status_code=400, detail="Invalid Kalshi environment")
    uploaded_private_key_pem = payload.private_key_pem.strip()
    private_key_pem = uploaded_private_key_pem
    if not private_key_pem:
        account = DB.get_kalshi_account(user["email"])
        private_key_pem = (account or {}).get("private_key_pem", "").strip()
    if not private_key_pem:
        raise HTTPException(
            status_code=400,
            detail="Private key PEM required. Upload one or connect with an existing stored PEM.",
        )
    key_ref = _db_secret_ref_for_email(user["email"])
    try:
        balance_cents = fetch_balance_for_account(
            key_id=payload.key_id,
            key_path=key_ref,
            kalshi_env=payload.kalshi_env,
            email=user["email"],
            private_key_pem=private_key_pem,
        )
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    DB.upsert_kalshi_account(
        user["email"],
        label=payload.label,
        key_id=payload.key_id,
        key_path=key_ref,
        private_key_pem=uploaded_private_key_pem or None,
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
    if account and account.get("key_path") and not str(account["key_path"]).startswith("db://"):
        try:
            Path(account["key_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    DB.delete_kalshi_account(user["email"])
    return {"connected": False}


def _signal_probability(row: dict, column: str) -> float | None:
    value = row.get(column)
    if value is None or value == "":
        return None
    prob = float(value)
    if row.get("bet_side") == "away":
        prob = 1.0 - prob
    return round(prob, 4)


def _signal_expires_at(row: dict) -> str | None:
    game_time_utc = row.get("game_time_utc")
    if game_time_utc:
        try:
            game_dt = pd.to_datetime(game_time_utc, utc=True)
            expires = (game_dt - pd.Timedelta(minutes=5)).to_pydatetime()
            return expires.replace(microsecond=0).isoformat()
        except Exception:
            pass
    game_date = str(row.get("game_date") or "")[:10]
    if not game_date:
        return None
    try:
        expires = datetime.fromisoformat(f"{game_date}T23:59:59+00:00")
        return expires.isoformat()
    except Exception:
        return None


def _build_signal_payload(row: dict) -> dict:
    model_prob = _signal_probability(row, "predicted_prob")
    market_prob = _signal_probability(row, "market_implied_prob")
    max_price = None
    if model_prob is not None:
        max_price = round(max(0.01, min(0.99, model_prob - SIGNAL_MIN_EDGE)), 4)
    updated_at = row.get("updated_at")
    generated_at = None
    if updated_at is not None:
        try:
            generated_at = pd.to_datetime(updated_at, utc=True).to_pydatetime().replace(microsecond=0).isoformat()
        except Exception:
            generated_at = None
    return {
        "signal_id": f"mlb-{str(row.get('game_date') or '')[:10]}-{row.get('game_pk')}-v1",
        "game_pk": str(row.get("game_pk")),
        "game_date": str(row.get("game_date") or "")[:10],
        "game_time_utc": _safe_value(row.get("game_time_utc")),
        "away_team": row.get("away_team"),
        "home_team": row.get("home_team"),
        "bet_side": row.get("bet_side"),
        "predicted_prob": _safe_value(row.get("predicted_prob")),
        "market_implied_prob": _safe_value(row.get("market_implied_prob")),
        "edge": _safe_value(row.get("edge")),
        "recommended_stake_frac": round(float(row.get("bet_frac") or 0.0), 6),
        "model_side_prob": model_prob,
        "market_side_prob": market_prob,
        "max_price": max_price,
        "min_edge": SIGNAL_MIN_EDGE,
        "generated_at_utc": generated_at,
        "expires_at_utc": _signal_expires_at(row),
        "version": 1,
    }


# ---------------------------------------------------------------------------
# API — Bets
# ---------------------------------------------------------------------------


@app.get("/api/signals/upcoming")
def get_upcoming_signals(
    request: Request,
    limit: int = 25,
    user: dict = Depends(require_signal_reader),
):
    limit = max(1, min(int(limit), 200))
    rows = DB.get_upcoming_bet_signals(limit=limit)
    return {
        "signals": [_build_signal_payload(row) for row in rows],
        "count": len(rows),
        "reader_email": user["email"],
    }


@app.get("/api/bet-explanation")
def get_bet_explanation(
    game_pk: str,
    user: dict = Depends(require_approved_user),
):
    row = DB.get_bet(game_pk)
    if not row:
        raise HTTPException(status_code=404, detail="Game not found")
    explanation = row.get("explanation")
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except Exception:
            explanation = None
    return {
        "game_pk": str(game_pk),
        "available": explanation is not None,
        "explanation": explanation,
    }


@app.post("/api/client/balance")
def sync_client_balance(
    payload: ClientBalancePayload,
    user: dict = Depends(require_client_writer),
):
    DB.insert_user_balance(
        user["email"],
        int(payload.balance_cents),
        source=(payload.source or "self_custody")[:64],
    )
    return {
        "ok": True,
        "balance_cents": int(payload.balance_cents),
        "balance_dollars": round(int(payload.balance_cents) / 100.0, 2),
    }


@app.post("/api/client/orders")
def sync_client_order(
    payload: ClientOrderPayload,
    user: dict = Depends(require_client_writer),
):
    if payload.bet_side not in {"home", "away", "none"}:
        raise HTTPException(status_code=400, detail="Invalid bet_side")
    DB.upsert_user_order(
        user["email"],
        payload.game_pk,
        game_date=payload.game_date,
        home_team=payload.home_team,
        away_team=payload.away_team,
        predicted_prob=payload.predicted_prob,
        market_implied_prob=payload.market_implied_prob,
        edge=payload.edge,
        bet_side=payload.bet_side,
        bet_frac=payload.bet_frac,
        bet_dollars=payload.bet_dollars,
        n_contracts=payload.n_contracts,
        kalshi_order_id=payload.kalshi_order_id,
        kalshi_ticker=payload.kalshi_ticker,
        live_price=payload.live_price,
        live_edge=payload.live_edge,
        status=payload.status,
        dry_run=payload.dry_run,
        result=payload.result,
        profit_loss=payload.profit_loss,
    )
    if payload.status == "error" and payload.error:
        DB.set_user_setting(user["email"], "self_custody_last_error", payload.error[:500])
    try:
        DB.backfill_user_order_results()
    except Exception as exc:
        print(f"Client order result backfill failed: {exc}")
    return {"ok": True, "game_pk": str(payload.game_pk), "status": payload.status}


@app.post("/api/client/heartbeat")
def sync_client_heartbeat(
    payload: ClientHeartbeatPayload,
    user: dict = Depends(require_client_writer),
):
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    DB.set_user_setting(user["email"], "self_custody_last_seen_at", now_utc)
    DB.set_user_setting(user["email"], "self_custody_kalshi_env", payload.kalshi_env[:32])
    DB.set_user_setting(user["email"], "self_custody_client_version", payload.version[:64])
    return {
        "ok": True,
        "email": user["email"],
        "server_time_utc": now_utc,
        "kalshi_env": payload.kalshi_env,
    }


ACTIVE_BET_STATUSES = {"filled", "dry_run"}
LIVE_BET_STATUSES = {"filled"}
PAPER_BET_STATUSES = {"dry_run"}
TERMINAL_EXIT_STATUSES = {DB.SOLD_STOP_LOSS_STATUS}
VOIDED_ORDER_STATUSES = {DB.VOIDED_ORDER_STATUS}
CLOSED_ORDER_STATUSES = TERMINAL_EXIT_STATUSES | VOIDED_ORDER_STATUSES


def _order_status_series(df: pd.DataFrame) -> pd.Series:
    return df.get("status", pd.Series("", index=df.index)).fillna("").astype(str)


def _is_settled_order_frame(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    status_col = _order_status_series(df)
    result_settled = df["result"].notna() if "result" in df.columns else pd.Series(False, index=df.index)
    return result_settled | status_col.isin(CLOSED_ORDER_STATUSES)


def _stop_loss_exit_pnl(row: pd.Series) -> float | None:
    pl = row.get("profit_loss")
    if pl is not None and not pd.isna(pl):
        return round(float(pl), 2)
    stake = row.get("bet_dollars")
    if stake is not None and not pd.isna(stake):
        return round(-float(stake), 2)
    return None


def _order_payout_dollars(row: pd.Series) -> float | None:
    n_contracts = row.get("n_contracts")
    if n_contracts is not None and not pd.isna(n_contracts):
        return round(float(n_contracts), 2)
    stake = row.get("bet_dollars")
    profit = row.get("profit_loss")
    if stake is not None and not pd.isna(stake) and profit is not None and not pd.isna(profit):
        return round(float(stake) + float(profit), 2)
    bet_cents = row.get("bet_cents")
    pl_cents = row.get("profit_loss_cents")
    if bet_cents is not None and not pd.isna(bet_cents) and pl_cents is not None and not pd.isna(pl_cents):
        return round((int(bet_cents) + int(pl_cents)) / 100.0, 2)
    return None


def _pending_payout_orders_df(email: str) -> pd.DataFrame:
    try:
        DB.reconcile_kalshi_balance_refresh_jobs()
    except Exception as exc:
        print(f"Pending payout reconciliation failed: {exc}")
    df = DB.get_pending_payout_user_orders(email)
    if df.empty:
        return df
    df = df.copy()
    df["payout_dollars"] = df.apply(_order_payout_dollars, axis=1)
    return df


def _pending_payout_game_pks(email: str) -> set[str]:
    df = _pending_payout_orders_df(email)
    if df.empty or "game_pk" not in df.columns:
        return set()
    return {str(pk) for pk in df["game_pk"].dropna().astype(str)}


def _validate_bet_mode(mode: str) -> str:
    mode = (mode or "paper").strip().lower()
    if mode not in {"paper", "live"}:
        raise HTTPException(status_code=400, detail="Invalid bet mode")
    return mode


def _ensure_paper_backfill(email: str) -> None:
    try:
        DB.backfill_paper_orders_from_bets(email)
    except Exception as exc:
        print(f"Paper order backfill failed for {email}: {exc}")
    try:
        DB.recompute_paper_order_financials(email)
    except Exception as exc:
        print(f"Paper financials recompute failed for {email}: {exc}")


def _orders_for_mode(email: str, mode: str, version: str = "v1") -> pd.DataFrame:
    if mode == "paper":
        _ensure_paper_backfill(email)
        if version == "v2":
            return DB.get_paper_orders(email, version="v2")
        return DB.get_paper_orders(email)
    return DB.get_user_orders(email)


def _all_orders_for_mode(mode: str, version: str = "v1") -> pd.DataFrame:
    if mode == "paper":
        if version == "v2":
            return DB.get_all_paper_orders(version="v2")
        return DB.get_all_paper_orders()
    return DB.get_all_user_orders()


def _is_open_position_frame(df: pd.DataFrame, active_statuses: set[str] = ACTIVE_BET_STATUSES) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    result_open = df["result"].isna() if "result" in df.columns else pd.Series(True, index=df.index)
    dollars_src = df["bet_dollars"] if "bet_dollars" in df.columns else pd.Series(0, index=df.index)
    dollars = pd.to_numeric(dollars_src, errors="coerce").fillna(0)
    statuses = _order_status_series(df)
    return (
        result_open
        & (dollars > 0)
        & statuses.isin(active_statuses)
        & ~statuses.isin(CLOSED_ORDER_STATUSES)
    )


@app.get("/api/bets")
def get_bets(
    limit: int = 100,
    offset: int = 0,
    status: str = "all",
    mode: str = "paper",
    model: str = "v1",
    user: dict = Depends(require_approved_user),
):
    if model not in ("v1", "v2"):
        model = "v1"
    mode = _validate_bet_mode(mode)
    if mode == "paper":
        if model == "v2":
            df = DB.get_paper_orders(user["email"], version="v2")
        else:
            df = DB.get_paper_orders(user["email"])
    else:
        # Live: show this user's Kalshi rows (same source as /api/open-bets + performance).
        # Using get_all_bets() was wrong — the `bets` table is model output and often has
        # no bet_dollars; /api/bets then filters everything out.
        df = DB.get_user_orders(user["email"])

    if df.empty:
        return {"bets": [], "total": 0}

    if "bet_dollars" in df.columns:
        df = df[df["bet_dollars"].notna()].copy()
        if not df.empty:
            df = df[df["bet_dollars"].astype(float) > 0].copy()

    if df.empty:
        return {"bets": [], "total": 0}

    if status == "open":
        statuses = PAPER_BET_STATUSES if mode == "paper" else LIVE_BET_STATUSES
        df = df[_is_open_position_frame(df, statuses)].copy()
    elif status == "settled":
        df = df[_is_settled_order_frame(df)].copy()
        if mode == "live" and not df.empty:
            pending_pks = _pending_payout_game_pks(user["email"])
            if pending_pks:
                df = df[~df["game_pk"].astype(str).isin(pending_pks)].copy()
    elif status != "all":
        raise HTTPException(status_code=400, detail="Invalid bets status filter")

    if df.empty:
        return {"bets": [], "total": 0}

    if "result" in df.columns and "bet_dollars" in df.columns:

        def calc_pnl(row):
            if str(row.get("status") or "") in CLOSED_ORDER_STATUSES:
                if str(row.get("status") or "") in VOIDED_ORDER_STATUSES:
                    return 0.0
                return _stop_loss_exit_pnl(row)
            if pd.isna(row.get("result")) or pd.isna(row.get("bet_dollars")):
                return None
            won = (bool(row["result"]) and row["bet_side"] == "home") or (
                not bool(row["result"]) and row["bet_side"] == "away"
            )
            bd = float(row["bet_dollars"] or 0)
            n_contracts = row.get("n_contracts")
            if won and n_contracts is not None and not pd.isna(n_contracts):
                return round(float(n_contracts) - bd, 2)
            if won:
                lp = row.get("live_price")
                if lp and float(lp) > 0:
                    return round(bd * (1.0 / float(lp) - 1.0), 2)
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

    if "bet_side" in df.columns:
        side = df["bet_side"].astype("string")
        if "market_implied_prob" in df.columns:
            mp = pd.to_numeric(df["market_implied_prob"], errors="coerce")
            df["market_side_prob"] = mp.where(side != "away", 1.0 - mp)
        if "predicted_prob" in df.columns:
            pred = pd.to_numeric(df["predicted_prob"], errors="coerce")
            df["model_side_prob"] = pred.where(side != "away", 1.0 - pred)
        nan_series = pd.Series(np.nan, index=df.index)
        stake = pd.to_numeric(df.get("bet_dollars", nan_series), errors="coerce")
        contracts = pd.to_numeric(df.get("n_contracts", nan_series), errors="coerce")
        live_price = pd.to_numeric(df.get("live_price", nan_series), errors="coerce")
        effective_entry = stake / contracts
        df["entry_price"] = effective_entry.where(
            (stake > 0) & (contracts > 0),
            live_price.where((live_price > 0) & (live_price < 1), df.get("market_side_prob")),
        )

    total = len(df)
    return {"bets": _safe_records(df.iloc[offset : offset + limit]), "total": total}


@app.get("/api/open-bets")
def get_open_bets(mode: str = "paper", model: str = "v1", user: dict = Depends(require_approved_user)):
    if model not in ("v1", "v2"):
        model = "v1"
    mode = _validate_bet_mode(mode)
    if mode == "paper":
        _ensure_paper_backfill(user["email"])
    try:
        DB.void_orders_for_inactive_games()
    except Exception as exc:
        print(f"Inactive-game order void skipped during open-bets read: {exc}")
    try:
        refresh_due_orders(stale_seconds=300, limit=25)
    except Exception as exc:
        print(f"Market mark refresh skipped during open-bets read: {exc}")
    df = _orders_for_mode(user["email"], mode, version=model)
    if df.empty:
        return {"bets": [], "total": 0}

    if "bet_dollars" in df.columns:
        df = df[df["bet_dollars"].notna()].copy()
        if not df.empty:
            df = df[df["bet_dollars"].astype(float) > 0].copy()

    if df.empty or "result" not in df.columns:
        return {"bets": [], "total": 0}

    statuses = PAPER_BET_STATUSES if mode == "paper" else LIVE_BET_STATUSES
    open_df = df[_is_open_position_frame(df, statuses)].copy()
    if open_df.empty:
        return {"bets": [], "total": 0}

    if "game_pk" in open_df.columns:
        open_df["game_pk"] = open_df["game_pk"].astype(str)

    games_df = DB.get_games_df(season=ACTIVE_SEASON, upcoming_only=True)
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


@app.get("/api/pending-payouts")
def get_pending_payouts(mode: str = "paper", user: dict = Depends(require_approved_user)):
    mode = _validate_bet_mode(mode)
    if mode != "live":
        return {"bets": [], "total": 0, "pending_payout_dollars": 0.0}

    df = _pending_payout_orders_df(user["email"])
    if df.empty:
        return {"bets": [], "total": 0, "pending_payout_dollars": 0.0}

    if "game_pk" in df.columns:
        df["game_pk"] = df["game_pk"].astype(str)

    sort_cols = [c for c in ("game_date", "balance_refresh_run_after", "game_pk") if c in df.columns]
    if sort_cols:
        ascending = {"game_date": False, "balance_refresh_run_after": True, "game_pk": True}
        df = df.sort_values(
            by=sort_cols,
            ascending=[ascending[c] for c in sort_cols],
        )

    pending_total = float(
        pd.to_numeric(df.get("payout_dollars"), errors="coerce").fillna(0).sum()
    )
    return {
        "bets": _safe_records(df),
        "total": len(df),
        "pending_payout_dollars": round(pending_total, 2),
    }


# ---------------------------------------------------------------------------
# API — Balance
# ---------------------------------------------------------------------------


def _order_profit_loss(row: pd.Series) -> float | None:
    if pd.isna(row.get("result")) or pd.isna(row.get("bet_dollars")):
        return None
    won = (bool(row["result"]) and row.get("bet_side") == "home") or (
        not bool(row["result"]) and row.get("bet_side") == "away"
    )
    stake = float(row.get("bet_dollars") or 0)
    n_contracts = row.get("n_contracts")
    if won and not pd.isna(n_contracts):
        return round(float(n_contracts) - stake, 2)
    if won:
        live_price = row.get("live_price")
        if live_price and float(live_price) > 0:
            return round(stake * (1.0 / float(live_price) - 1.0), 2)
    if not won:
        return -stake
    return None


def _live_order_bankroll_history(email: str, current_dollars: float) -> list[dict]:
    orders = DB.get_user_orders(email)
    if orders.empty:
        return []
    orders = orders.copy()
    orders["bet_dollars"] = pd.to_numeric(orders.get("bet_dollars"), errors="coerce")
    statuses = orders.get("status", pd.Series("", index=orders.index)).fillna("").astype(str)
    orders = orders[(orders["bet_dollars"] > 0) & statuses.isin(LIVE_BET_STATUSES)].copy()
    if orders.empty:
        return []

    sort_cols = [col for col in ["game_date", "game_pk"] if col in orders.columns]
    if sort_cols:
        orders = orders.sort_values(sort_cols, ascending=True)

    events: list[tuple[pd.Series, float]] = []
    for _, row in orders.iterrows():
        if not pd.isna(row.get("result")):
            effect = row.get("profit_loss")
            if pd.isna(effect):
                effect = _order_profit_loss(row)
        else:
            effect = -float(row.get("bet_dollars") or 0)
        if effect is None or pd.isna(effect):
            continue
        events.append((row, float(effect)))

    if not events:
        return []

    running = round(float(current_dollars) - sum(effect for _, effect in events), 2)
    history = [{"recorded_at": None, "balance_dollars": running, "source": "live_order"}]
    for row, effect in events:
        running = round(running + effect, 2)
        game_pk = row.get("game_pk")
        history.append(
            {
                "recorded_at": row.get("game_date"),
                "balance_dollars": running,
                "game_pk": None if pd.isna(game_pk) else str(game_pk),
                "source": "live_order",
            }
        )
    return history


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
                    private_key_pem=account.get("private_key_pem") or None,
                )
                df = DB.get_user_balance_history(user["email"])
            except Exception:
                pass
    if df.empty:
        return {
            "history": [],
            "current_dollars": 0.0,
            "pending_payout_dollars": 0.0,
            "effective_dollars": 0.0,
        }
    current_dollars = float(df.iloc[-1]["balance_dollars"])
    daily_df = DB.get_user_balance_daily_history(user["email"])
    pending_df = _pending_payout_orders_df(user["email"])
    pending_dollars = 0.0
    if not pending_df.empty and "payout_dollars" in pending_df.columns:
        pending_dollars = round(
            float(pd.to_numeric(pending_df["payout_dollars"], errors="coerce").fillna(0).sum()),
            2,
        )
    return {
        "history": _safe_records(daily_df),
        "current_dollars": current_dollars,
        "pending_payout_dollars": pending_dollars,
        "effective_dollars": round(current_dollars + pending_dollars, 2),
    }


@app.get("/api/paper-bankroll")
def get_paper_bankroll(model: str = "v1", user: dict = Depends(require_approved_user)):
    if model not in ("v1", "v2"):
        model = "v1"
        
    if model == "v1":
        _ensure_paper_backfill(user["email"])
    else:
        # V2 paper settlement is global
        try:
            DB.backfill_paper_order_v2_results()
        except Exception as exc:
            print(f"V2 paper backfill failed: {exc}")

    if model == "v2":
        current_dollars = DB.get_paper_bankroll_dollars(user["email"], version="v2")
        history = DB.get_paper_bankroll_history(user["email"], version="v2")
    else:
        current_dollars = DB.get_paper_bankroll_dollars(user["email"])
        history = DB.get_paper_bankroll_history(user["email"])

    return {
        "starting_dollars": DB.PAPER_STARTING_BANKROLL_DOLLARS,
        "current_dollars": current_dollars,
        "history": history,
    }


# ---------------------------------------------------------------------------
# API — Teams / standings
# ---------------------------------------------------------------------------


def _empty_team_row(abbr: str) -> dict:
    info = TEAM_INFO[abbr]
    return {
        "abbr": abbr,
        "name": info["name"],
        "league": info["league"],
        "league_name": LEAGUE_LABELS[info["league"]],
        "division": info["division"],
        "division_name": f"{info['league']} {info['division']}",
        "wins": 0,
        "losses": 0,
        "win_pct": 0.0,
        "runs_for": 0,
        "runs_against": 0,
        "run_diff": 0,
        "home_wins": 0,
        "home_losses": 0,
        "away_wins": 0,
        "away_losses": 0,
        "last_ten_wins": 0,
        "last_ten_losses": 0,
        "streak": "—",
        "division_rank": None,
        "games_back": None,
        "last_game": None,
        "next_game": None,
        "_results": [],
    }


_INACTIVE_GAME_STATUSES = {"postponed", "cancelled", "canceled", "suspended"}


def _is_known_game(row: pd.Series) -> bool:
    # `home_win` is the canonical "final result recorded" signal (matches
    # `db.get_games_df(upcoming_only=True)`). Scheduled / in-progress rows can
    # carry placeholder 0/0 scores, so we cannot use score presence alone.
    return (
        pd.notna(row.get("home_team"))
        and pd.notna(row.get("away_team"))
        and pd.notna(row.get("home_win"))
        and pd.notna(row.get("home_score"))
        and pd.notna(row.get("away_score"))
    )


def _is_inactive_game(row: pd.Series) -> bool:
    status = str(row.get("game_status") or "").strip().lower()
    return status in _INACTIVE_GAME_STATUSES


def _team_game_summary(row: pd.Series, team: str, completed: bool) -> dict:
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
    opponent = away if team == home else home
    is_home = team == home
    summary = {
        "game_pk": _safe_value(row.get("game_pk")),
        "game_date": _safe_value(row.get("game_date")),
        "game_time_utc": _safe_value(row.get("game_time_utc")),
        "opponent": opponent,
        "home_team": home,
        "away_team": away,
        "venue": "Home" if is_home else "Away",
        "matchup": f"{away} @ {home}",
    }
    if completed:
        home_score = int(float(row.get("home_score")))
        away_score = int(float(row.get("away_score")))
        won = home_score > away_score if is_home else away_score > home_score
        summary.update(
            {
                "result": "W" if won else "L",
                "team_score": home_score if is_home else away_score,
                "opponent_score": away_score if is_home else home_score,
            }
        )
    return summary


def _record_streak(results: list[bool]) -> str:
    if not results:
        return "—"
    latest = results[-1]
    count = 0
    for result in reversed(results):
        if result != latest:
            break
        count += 1
    return ("W" if latest else "L") + str(count)


def _games_back(team: dict, leader: dict) -> float:
    return round(((leader["wins"] - team["wins"]) + (team["losses"] - leader["losses"])) / 2, 1)


def _build_team_overview(games_df: pd.DataFrame, season: int = ACTIVE_SEASON) -> dict:
    teams = {abbr: _empty_team_row(abbr) for abbr in TEAM_INFO}
    if not games_df.empty:
        games_df = games_df.copy()
        if "game_time_utc" in games_df.columns:
            games_df["game_time_utc"] = pd.to_datetime(games_df["game_time_utc"], errors="coerce", utc=True)
        if "game_date" in games_df.columns:
            games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce")
        sort_cols = [c for c in ("game_date", "game_time_utc", "game_pk") if c in games_df.columns]
        if sort_cols:
            games_df = games_df.sort_values(sort_cols, na_position="last")

        for _, row in games_df.iterrows():
            home = str(row.get("home_team") or "")
            away = str(row.get("away_team") or "")
            if home not in teams or away not in teams:
                continue

            completed = _is_known_game(row)
            if completed:
                home_score = int(float(row.get("home_score")))
                away_score = int(float(row.get("away_score")))
                home_won = bool(row.get("home_win")) if pd.notna(row.get("home_win")) else home_score > away_score

                home_row = teams[home]
                away_row = teams[away]
                home_row["runs_for"] += home_score
                home_row["runs_against"] += away_score
                away_row["runs_for"] += away_score
                away_row["runs_against"] += home_score

                if home_won:
                    home_row["wins"] += 1
                    home_row["home_wins"] += 1
                    away_row["losses"] += 1
                    away_row["away_losses"] += 1
                else:
                    away_row["wins"] += 1
                    away_row["away_wins"] += 1
                    home_row["losses"] += 1
                    home_row["home_losses"] += 1

                home_row["_results"].append(home_won)
                away_row["_results"].append(not home_won)
                home_row["last_game"] = _team_game_summary(row, home, completed=True)
                away_row["last_game"] = _team_game_summary(row, away, completed=True)
            else:
                if _is_inactive_game(row):
                    continue
                if teams[home]["next_game"] is None:
                    teams[home]["next_game"] = _team_game_summary(row, home, completed=False)
                if teams[away]["next_game"] is None:
                    teams[away]["next_game"] = _team_game_summary(row, away, completed=False)

    for team in teams.values():
        games = team["wins"] + team["losses"]
        team["win_pct"] = round(team["wins"] / games, 3) if games else 0.0
        team["run_diff"] = team["runs_for"] - team["runs_against"]
        last_ten = team["_results"][-10:]
        team["last_ten_wins"] = sum(1 for result in last_ten if result)
        team["last_ten_losses"] = len(last_ten) - team["last_ten_wins"]
        team["streak"] = _record_streak(team["_results"])

    divisions = []
    for league, division in DIVISION_ORDER:
        rows = [
            team for team in teams.values()
            if team["league"] == league and team["division"] == division
        ]
        rows.sort(key=lambda t: (-t["win_pct"], -t["wins"], t["losses"], -t["run_diff"], t["name"]))
        leader = rows[0] if rows else None
        for rank, team in enumerate(rows, start=1):
            team["division_rank"] = rank
            team["games_back"] = 0.0 if rank == 1 or leader is None else _games_back(team, leader)
        divisions.append(
            {
                "key": f"{league}-{division}",
                "league": league,
                "league_name": LEAGUE_LABELS[league],
                "division": division,
                "name": f"{league} {division}",
                "teams": [{k: v for k, v in team.items() if k != "_results"} for team in rows],
            }
        )

    public_teams = [{k: v for k, v in team.items() if k != "_results"} for team in teams.values()]
    public_teams.sort(key=lambda t: (-t["win_pct"], -t["wins"], t["losses"], t["name"]))
    leader = public_teams[0] if public_teams else None
    hottest = sorted(
        public_teams,
        key=lambda t: (
            -int(t["streak"][1:]) if t["streak"].startswith("W") else 0,
            -t["win_pct"],
            t["name"],
        ),
    )[0] if public_teams else None
    return {
        "season": season,
        "last_updated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "summary": {
            "teams": len(public_teams),
            "completed_games": sum(t["wins"] for t in public_teams),
            "leader": leader,
            "hottest": hottest,
        },
        "divisions": divisions,
        "teams": public_teams,
    }


@app.get("/api/teams")
def get_teams(user: dict = Depends(require_approved_user)):
    try:
        games_df = DB.get_games_df(season=ACTIVE_SEASON)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return _build_team_overview(games_df, season=ACTIVE_SEASON)


# ---------------------------------------------------------------------------
# API — Upcoming games
# ---------------------------------------------------------------------------


@app.get("/api/games-by-date")
def get_games_by_date(
    date: str = None,
    mode: str = "paper",
    model: str = "v1",
    user: dict = Depends(require_approved_user),
):
    if model not in ("v1", "v2"):
        model = "v1"
    user_tz = ZoneInfo(
        DB.get_user_setting(
            user["email"], "dashboard_timezone", _get_dashboard_timezone()
        )
    )
    if not date:
        date = pd.Timestamp.now(tz=user_tz).strftime("%Y-%m-%d")

    try:
        target = pd.Timestamp(date)
    except Exception:
        return JSONResponse({"error": "Invalid date format"}, status_code=400)

    try:
        refresh_scores_for_date(date)
    except Exception as exc:
        print(f"Live score refresh skipped for {date}: {exc}")

    try:
        games_df = DB.get_games_df(season=ACTIVE_SEASON)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if games_df.empty:
        return {"date": date, "games": []}

    games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce")

    if "game_time_utc" in games_df.columns:
        game_times = pd.to_datetime(
            games_df["game_time_utc"], errors="coerce", utc=True
        )
        game_local_dates = game_times.dt.tz_convert(user_tz).dt.date
        has_time = game_times.notna()
        matches_time = has_time & (game_local_dates == target.date())
        matches_date = ~has_time & (games_df["game_date"].dt.date == target.date())
        day_games = games_df[matches_time | matches_date].copy()
    else:
        day_games = games_df[games_df["game_date"].dt.date == target.date()].copy()

    if model == "v2":
        bets_df = DB.get_all_bets(version="v2")
    else:
        bets_df = DB.get_all_bets()
    if not bets_df.empty:
        pred_cols = [
            "game_pk", "predicted_prob", "edge", "bet_side", "bet_frac",
            "market_implied_prob", "bet_dollars", "result", "kalshi_order_id",
            "profit_loss", "n_contracts",
        ]
        available = [c for c in pred_cols if c in bets_df.columns]
        bets_df["game_pk"] = bets_df["game_pk"].astype(str)
        day_games["game_pk"] = day_games["game_pk"].astype(str)
        day_games = day_games.merge(bets_df[available], on="game_pk", how="left")

    orders = (
        DB.get_paper_orders(user["email"], version=model) if mode == "paper"
        else DB.get_user_orders(user["email"])
    )
    if not orders.empty:
        orders = orders.copy()
        orders["game_pk"] = orders["game_pk"].astype(str)
        order_cols = [
            "game_pk", "bet_dollars", "n_contracts", "kalshi_order_id",
            "status", "dry_run", "live_price", "live_edge", "result",
            "profit_loss",
        ]
        available = [c for c in order_cols if c in orders.columns]
        day_games = day_games.merge(
            orders[available], on="game_pk", how="left",
            suffixes=("_bets", "_user"),
        )
        for col in order_cols:
            if col == "game_pk":
                continue
            user_col = col + "_user"
            bets_col = col + "_bets"
            if user_col in day_games.columns and bets_col in day_games.columns:
                day_games[col] = day_games[user_col].combine_first(day_games[bets_col])
                day_games.drop(columns=[user_col, bets_col], inplace=True)
            elif user_col in day_games.columns:
                day_games.rename(columns={user_col: col}, inplace=True)
            elif bets_col in day_games.columns:
                day_games.rename(columns={bets_col: col}, inplace=True)

    if "game_time_utc" in day_games.columns:
        day_games = day_games.sort_values("game_time_utc", na_position="last")

    cols = [
        "game_pk", "game_date", "game_time_utc", "home_team", "away_team",
        "home_score", "away_score", "home_win",
        "game_status", "game_state", "detailed_state", "inning",
        "inning_ordinal", "inning_state", "is_top_inning", "outs",
        "balls", "strikes", "live_status_text", "live_updated_at",
        "home_implied_prob", "away_implied_prob", "close_home_ml", "close_away_ml",
        "predicted_prob", "edge", "bet_side", "bet_frac", "bet_dollars",
        "n_contracts", "result", "kalshi_order_id", "status", "dry_run",
        "live_price", "live_edge", "profit_loss",
    ]
    return {
        "date": date,
        "games": _safe_records(day_games[[c for c in cols if c in day_games.columns]]),
    }


@app.get("/api/upcoming")
def get_upcoming(model: str = "v1", user: dict = Depends(require_approved_user)):
    if model not in ("v1", "v2"):
        model = "v1"
    try:
        refresh_scores_for_date()
    except Exception as exc:
        print(f"Live score refresh skipped for upcoming: {exc}")

    try:
        games_df = DB.get_games_df(season=ACTIVE_SEASON, upcoming_only=True)
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

    if model == "v2":
        bets_df = DB.get_all_bets(version="v2")
    else:
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
            suffixes=("_bets", "_user"),
        )
        for col in order_cols:
            if col == "game_pk":
                continue
            user_col = col + "_user"
            bets_col = col + "_bets"
            if user_col in upcoming.columns and bets_col in upcoming.columns:
                upcoming[col] = upcoming[user_col].combine_first(upcoming[bets_col])
                upcoming.drop(columns=[user_col, bets_col], inplace=True)
            elif user_col in upcoming.columns:
                upcoming.rename(columns={user_col: col}, inplace=True)
            elif bets_col in upcoming.columns:
                upcoming.rename(columns={bets_col: col}, inplace=True)

    if "game_time_utc" in upcoming.columns:
        upcoming = upcoming.sort_values("game_time_utc", na_position="last")

    cols = [
        "game_pk",
        "game_date",
        "game_time_utc",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "game_status",
        "game_state",
        "detailed_state",
        "inning",
        "inning_ordinal",
        "inning_state",
        "is_top_inning",
        "outs",
        "balls",
        "strikes",
        "live_status_text",
        "live_updated_at",
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


def _build_public_performance(
    email: str | None = None,
    *,
    mode: str = "live",
    model: str = "v1",
) -> PublicPerformanceResponse:
    mode = _validate_bet_mode(mode)
    if email is None and mode == "paper":
        settled = _settled_public_bets(model=model)
    else:
        bets_df = _orders_for_mode(email, mode, version=model) if email else _all_orders_for_mode(mode, version=model)
        if bets_df.empty:
            return _empty_public_performance()

        settled = bets_df[bets_df["bet_dollars"].notna()].copy()
        settled = settled[_is_settled_order_frame(settled)].copy()
        settled = settled[settled["bet_dollars"].astype(float) > 0]
    if settled.empty:
        return _empty_public_performance()

    side = settled["bet_side"]
    status_col = _order_status_series(settled)
    settled = settled[~status_col.isin(VOIDED_ORDER_STATUSES)].copy()
    if settled.empty:
        return _empty_public_performance()
    status_col = _order_status_series(settled)
    stop_loss_mask = status_col.isin(TERMINAL_EXIT_STATUSES).to_numpy()
    if "_won" in settled.columns:
        won_mask = settled["_won"].astype(bool).to_numpy()
    else:
        result = settled["result"].astype(bool)
        won_mask = ((result & (side == "home")) | (~result & (side == "away"))).to_numpy()

    bd = pd.to_numeric(
        settled["_stake"] if "_stake" in settled.columns else settled["bet_dollars"],
        errors="coerce",
    ).fillna(0.0).to_numpy()
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

    if "profit_loss" in settled.columns:
        exit_pnl = pd.to_numeric(settled["profit_loss"], errors="coerce").to_numpy()
    else:
        exit_pnl = np.full(len(settled), np.nan)
    if stop_loss_mask.any():
        won_mask = np.where(
            stop_loss_mask,
            np.where(~np.isnan(exit_pnl), exit_pnl > 0, False),
            won_mask,
        )

    total_bets = len(settled)
    wins = int(won_mask.sum())
    total_wagered = float(bd.sum())
    if "_profit_loss" in settled.columns:
        row_pnl = pd.to_numeric(settled["_profit_loss"], errors="coerce").to_numpy()
    elif "profit_loss" in settled.columns:
        row_pnl = pd.to_numeric(settled["profit_loss"], errors="coerce").to_numpy()
    else:
        row_pnl = np.full(len(settled), np.nan)
    fallback_pnl = np.where(won_mask, returned_row - bd, -bd)
    row_pnl = np.where(
        stop_loss_mask,
        np.where(~np.isnan(row_pnl), row_pnl, -bd),
        np.where(~np.isnan(row_pnl), row_pnl, fallback_pnl),
    )
    roi = float(np.nansum(row_pnl)) / total_wagered if total_wagered else 0.0

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
def get_performance(mode: str = "paper", model: str = "v1", user: dict = Depends(require_approved_user)):
    return _build_public_performance(user["email"], mode=mode, model=model)


@app.get("/api/public/performance", response_model=PublicPerformanceResponse)
@limiter.limit("30/minute")
def get_public_performance(request: Request, model: str = "v1"):
    return _build_public_performance(mode="paper", model=model)


@app.get("/api/public/model-accuracy", response_model=PublicModelAccuracyResponse)
@limiter.limit("30/minute")
def get_public_model_accuracy(request: Request, model: str = "v1"):
    return _build_public_model_accuracy(model=model)


@app.get("/api/public/summary", response_model=PublicSummaryResponse)
@limiter.limit("30/minute")
def get_public_summary(request: Request, model: str = "v1"):
    return _build_public_summary(model=model)


@app.get("/api/public/receipts", response_model=PublicReceiptsResponse)
@limiter.limit("30/minute")
def get_public_receipts(request: Request, model: str = "v1"):
    return _build_public_receipts(model=model)


@app.get("/api/model-accuracy", response_model=PrivateModelAccuracyResponse)
def get_model_accuracy(model: str = "v1", user: dict = Depends(require_approved_user)):
    return _build_private_model_accuracy(model=model)


def _compute_model_accuracy_metrics(model: str = "v1") -> tuple[dict, pd.DataFrame]:
    if model == "v2":
        df = DB.get_model_picks(version="v2")
    else:
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


def _compute_public_model_accuracy(model: str = "v1") -> tuple[PublicModelAccuracyResponse, pd.DataFrame]:
    metrics, df = _compute_model_accuracy_metrics(model=model)
    return PublicModelAccuracyResponse(**metrics), df


def _build_public_model_accuracy(model: str = "v1") -> PublicModelAccuracyResponse:
    model_accuracy, _ = _compute_public_model_accuracy(model=model)
    return model_accuracy


def _build_private_model_accuracy(model: str = "v1") -> PrivateModelAccuracyResponse:
    model_accuracy, df = _compute_public_model_accuracy(model=model)
    return PrivateModelAccuracyResponse(
        **model_accuracy.model_dump(),
        recent=_build_recent_model_picks(df),
    )


def _build_public_summary(model: str = "v1") -> PublicSummaryResponse:
    performance = _build_public_performance(mode="paper", model=model)
    model_accuracy = _build_public_model_accuracy(model=model)
    return PublicSummaryResponse(
        performance=performance,
        model_accuracy=model_accuracy,
    )


def _dedupe_public_bet_games(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-user paper copies into one public row per model wager."""
    if df.empty or "game_pk" not in df.columns:
        return df

    sort_cols = [c for c in ("game_date", "game_pk", "email") if c in df.columns]
    if sort_cols:
        ascending = [False if c in ("game_date", "game_pk") else True for c in sort_cols]
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    return df.drop_duplicates(subset=["game_pk"], keep="first").copy()


def _settled_public_bets(model: str = "v1") -> pd.DataFrame:
    if model == "v2":
        bets_df = DB.get_all_paper_orders(version="v2")
    else:
        bets_df = DB.get_all_paper_orders()
    if bets_df.empty:
        return pd.DataFrame()

    required = {"result", "bet_dollars", "bet_side"}
    if not required.issubset(set(bets_df.columns)):
        return pd.DataFrame()

    settled = bets_df[
        bets_df["result"].notna() & bets_df["bet_dollars"].notna()
    ].copy()
    settled = settled[pd.to_numeric(settled["bet_dollars"], errors="coerce") > 0]
    settled = settled[settled["bet_side"].isin(["home", "away"])]
    if settled.empty:
        return settled
    settled = _dedupe_public_bet_games(settled)

    result = settled["result"].astype(bool)
    side = settled["bet_side"]
    won_mask = ((result & (side == "home")) | (~result & (side == "away"))).to_numpy()
    settled["_won"] = won_mask
    settled["_stake"] = pd.to_numeric(settled["bet_dollars"], errors="coerce").fillna(0.0)

    if "profit_loss" in settled.columns:
        pnl = pd.to_numeric(settled["profit_loss"], errors="coerce")
    else:
        pnl = pd.Series(np.nan, index=settled.index)

    missing_pnl = pnl.isna()
    if missing_pnl.any():
        mp_source = settled["market_implied_prob"] if "market_implied_prob" in settled.columns else pd.Series(np.nan, index=settled.index)
        contracts_source = settled["n_contracts"] if "n_contracts" in settled.columns else pd.Series(np.nan, index=settled.index)
        mp = pd.to_numeric(mp_source, errors="coerce")
        n_contracts = pd.to_numeric(contracts_source, errors="coerce")
        stake = settled["_stake"]
        home_ratio = np.where(mp > 0, 1.0 / mp, 1.0)
        away_ratio = np.where((mp > 0) & (mp < 1), 1.0 / (1.0 - mp), 1.0)
        ratio = np.where(side == "home", home_ratio, away_ratio)
        returned = np.where(n_contracts.notna(), n_contracts, stake * ratio)
        fallback = np.where(won_mask, returned - stake, -stake)
        pnl = pnl.mask(missing_pnl, fallback)

    settled["_profit_loss"] = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    return settled


def _side_probability(row: pd.Series, column: str) -> float | None:
    if column not in row or pd.isna(row[column]):
        return None
    value = float(row[column])
    if row.get("bet_side") == "away":
        value = 1.0 - value
    return round(value, 4)


def _build_public_receipts(model: str = "v1") -> PublicReceiptsResponse:
    settled = _settled_public_bets(model=model)
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if settled.empty:
        return PublicReceiptsResponse(
            last_updated_utc=now_utc,
            settled_bets=0,
            first_bet_date=None,
            last_bet_date=None,
            recent_bets=[],
            roi_series=[],
        )

    settled["_game_date"] = pd.to_datetime(settled.get("game_date"), errors="coerce")
    first_date = settled["_game_date"].min()
    last_date = settled["_game_date"].max()

    recent_bets: list[PublicRecentBet] = []
    recent_source = settled.sort_values(["_game_date", "game_pk"], ascending=[False, False], na_position="last").head(20)
    for _, row in recent_source.iterrows():
        away = str(row.get("away_team") or "Away")
        home = str(row.get("home_team") or "Home")
        pnl = float(row["_profit_loss"])
        stake = float(row["_stake"])
        return_pct = (pnl / stake * 100.0) if stake else 0.0
        recent_bets.append(
            PublicRecentBet(
                game_date=str(_safe_value(row.get("game_date")) or "")[:10],
                matchup=f"{away} @ {home}",
                side=str(row.get("bet_side") or "").upper(),
                result="W" if bool(row["_won"]) else "L",
                model_prob=_side_probability(row, "predicted_prob"),
                market_prob=_side_probability(row, "market_implied_prob"),
                edge=_safe_value(row.get("edge")),
                return_pct=round(return_pct, 2),
            )
        )

    roi_series: list[PublicRoiPoint] = []
    running = settled.sort_values(["_game_date", "game_pk"], ascending=[True, True], na_position="last")
    cumulative_pl = 0.0
    cumulative_wagered = 0.0
    for _, row in running.iterrows():
        cumulative_pl += float(row["_profit_loss"])
        cumulative_wagered += float(row["_stake"])
        roi_pct = (cumulative_pl / cumulative_wagered * 100.0) if cumulative_wagered else 0.0
        roi_series.append(
            PublicRoiPoint(
                game_date=str(_safe_value(row.get("game_date")) or "")[:10],
                roi_pct=round(roi_pct, 2),
            )
        )

    return PublicReceiptsResponse(
        last_updated_utc=now_utc,
        settled_bets=len(settled),
        first_bet_date=str(_safe_value(first_date))[:10] if pd.notna(first_date) else None,
        last_bet_date=str(_safe_value(last_date))[:10] if pd.notna(last_date) else None,
        recent_bets=recent_bets,
        roi_series=roi_series,
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


@app.get("/api/admin/errors")
def get_admin_errors(user: dict = Depends(require_admin)):
    lines = []
    keywords = ["ERROR", "WARNING", "error", "warning", "Traceback", "Exception"]
    for service in ["mlb-pipeline.service", "mlb-dashboard.service"]:
        try:
            result = subprocess.run(
                ["journalctl", "-u", service, "-n", "500", "--no-pager", "--output=short"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if any(k in line for k in keywords):
                    lines.append(f"[{service}] {line}")
        except Exception:
            pass

    try:
        resp = requests.get(
            "https://api.github.com/repos/jeverett32/mlb-intelligence/actions/runs",
            params={"per_page": 15, "branch": "main"},
            timeout=10,
        )
        if resp.ok:
            for run in resp.json().get("workflow_runs", []):
                conclusion = run.get("conclusion") or run.get("status")
                ts = run.get("created_at", "")[:19].replace("T", " ")
                name = run.get("name", "workflow")
                sha = run.get("head_sha", "")[:7]
                msg = run.get("head_commit", {}).get("message", "").split("\n")[0][:80]
                if conclusion in ("failure", "cancelled", "timed_out"):
                    lines.insert(0, f"[github-actions] {ts} {name} {conclusion} ({sha} {msg})")
    except Exception:
        pass

    return {"lines": lines}


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


class AdminNotePayload(BaseModel):
    title: str = ""
    body: str = ""


class AdminNoteReorderPayload(BaseModel):
    note_ids: list[int]


class AdminNoteDonePayload(BaseModel):
    is_done: bool


def _clean_note_payload(payload: AdminNotePayload) -> tuple[str, str]:
    title = (payload.title or "").strip()
    body = (payload.body or "").strip()
    if not title and not body:
        raise HTTPException(status_code=400, detail="Note needs title or body")
    if len(title) > 160:
        raise HTTPException(status_code=400, detail="Title too long")
    if len(body) > 20000:
        raise HTTPException(status_code=400, detail="Note body too long")
    return title, body


def _serialize_admin_note(note: dict) -> dict:
    out = dict(note)
    for key in ("created_at", "updated_at"):
        if out.get(key):
            out[key] = out[key].isoformat()
    return out


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


@app.get("/api/admin/notes")
def get_admin_notes(user: dict = Depends(require_admin)):
    notes = DB.list_admin_notes(limit=100)
    return {"notes": [_serialize_admin_note(n) for n in notes]}


@app.post("/api/admin/notes")
def create_admin_note(payload: AdminNotePayload, user: dict = Depends(require_admin)):
    title, body = _clean_note_payload(payload)
    note = DB.create_admin_note(title, body, actor_email=user["email"])
    return _serialize_admin_note(note)


@app.put("/api/admin/notes/{note_id}")
def update_admin_note(
    note_id: int,
    payload: AdminNotePayload,
    user: dict = Depends(require_admin),
):
    title, body = _clean_note_payload(payload)
    note = DB.update_admin_note(note_id, title, body, actor_email=user["email"])
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return _serialize_admin_note(note)


@app.delete("/api/admin/notes/{note_id}")
def delete_admin_note(note_id: int, user: dict = Depends(require_admin)):
    if not DB.delete_admin_note(note_id):
        raise HTTPException(status_code=404, detail="Note not found")
    return {"ok": True}


@app.post("/api/admin/notes/reorder")
def reorder_admin_notes(payload: AdminNoteReorderPayload, user: dict = Depends(require_admin)):
    note_ids = [int(nid) for nid in payload.note_ids]
    if not note_ids:
        raise HTTPException(status_code=400, detail="Missing note_ids")
    if len(note_ids) != len(set(note_ids)):
        raise HTTPException(status_code=400, detail="Duplicate note_ids")
    if not DB.reorder_admin_notes(note_ids):
        raise HTTPException(status_code=400, detail="Invalid note_ids")
    return {"ok": True}


@app.post("/api/admin/notes/{note_id}/done")
def set_admin_note_done(
    note_id: int,
    payload: AdminNoteDonePayload,
    user: dict = Depends(require_admin),
):
    note = DB.set_admin_note_done(note_id, payload.is_done, actor_email=user["email"])
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return _serialize_admin_note(note)


@app.get("/api/admin/model-metrics")
def get_admin_model_metrics(model: str = "v1", user: dict = Depends(require_admin)):
    if model == "v2":
        runs = DB.get_model_metric_snapshots(limit=50, version="v2")
    else:
        runs = DB.get_model_metric_snapshots(limit=50)
    for r in runs:
        if r.get("trained_at"):
            r["trained_at"] = r["trained_at"].isoformat()
    return {"runs": runs}


@app.get("/api/admin/pipeline-runs")
def get_admin_pipeline_runs(user: dict = Depends(require_admin)):
    runs = DB.get_pipeline_runs(limit=200)
    for r in runs:
        if r.get("started_at"):
            r["started_at"] = r["started_at"].isoformat()
        if r.get("completed_at"):
            r["completed_at"] = r["completed_at"].isoformat()
    return {"runs": runs}


@app.get("/api/admin/model-artifacts/fi-history")
def get_admin_model_artifact_fi_history(model: str = "v1", user: dict = Depends(require_admin)):
    if model == "v2":
        rows = DB.get_model_artifact_fi_history(limit=100, version="v2")
    else:
        rows = DB.get_model_artifact_fi_history(limit=100)
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].isoformat()
        if r.get("max_settled_game_date"):
            r["max_settled_game_date"] = r["max_settled_game_date"].isoformat()
    return {"artifacts": rows}


@app.get("/api/admin/model-accuracy-by-month")
def get_admin_model_accuracy_by_month(
    year: int | None = None,
    model: str = "v1",
    user: dict = Depends(require_admin),
):
    """
    Per-month accuracy/Brier from latest training run's walk-forward
    validation predictions across all training years.
    """
    if model == "v2":
        run = DB.get_latest_model_metric_snapshot(version="v2")
    else:
        run = DB.get_latest_model_metric_snapshot()
    raw = (run or {}).get("monthly_accuracy") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = []
    months = list(raw) if isinstance(raw, list) else []
    available_years = sorted({m["year"] for m in months if m.get("year") is not None})
    if year is not None:
        months = [m for m in months if m.get("year") == int(year)]
    else:
        buckets: dict[int, list[dict]] = {}
        for m in months:
            mo = m.get("month")
            if mo is None:
                continue
            buckets.setdefault(int(mo), []).append(m)
        agg = []
        for mo in sorted(buckets):
            rows = buckets[mo]
            total_n = sum(int(r.get("count") or 0) for r in rows)
            def wavg(field: str) -> float | None:
                num = 0.0
                den = 0
                for r in rows:
                    v = r.get(field)
                    n = int(r.get("count") or 0)
                    if v is None or n <= 0:
                        continue
                    num += float(v) * n
                    den += n
                return (num / den) if den > 0 else None
            agg.append({
                "year": None,
                "month": mo,
                "year_month": f"{mo:02d}",
                "count": total_n,
                "accuracy": wavg("accuracy"),
                "brier": wavg("brier"),
                "log_loss": wavg("log_loss"),
                "market_accuracy": wavg("market_accuracy"),
                "market_brier": wavg("market_brier"),
            })
        months = agg
    trained_at = (run or {}).get("trained_at")
    if hasattr(trained_at, "isoformat"):
        trained_at = trained_at.isoformat()
    return {
        "months": months,
        "available_years": available_years,
        "trained_at": trained_at,
    }


@app.get("/api/admin/model-metrics/latest")
def get_admin_model_metrics_latest(model: str = "v1", user: dict = Depends(require_admin)):
    if model == "v2":
        run = DB.get_latest_model_metric_snapshot(version="v2")
    else:
        run = DB.get_latest_model_metric_snapshot()
    if not run:
        return {"run": None}
    if run.get("trained_at"):
        run["trained_at"] = run["trained_at"].isoformat()
    return {"run": run}


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


def _safe_dict_list(rows: list[dict]) -> list[dict]:
    return [{k: _safe_value(v) for k, v in r.items()} for r in rows]


# ---------------------------------------------------------------------------
# Data quality endpoints (admin)
# ---------------------------------------------------------------------------


@app.get("/api/admin/data-quality/latest")
def get_data_quality_latest(user: dict = Depends(require_admin)):
    """Most recent audit run with full per-column report."""
    try:
        run = db.get_latest_data_quality_run()
        if not run:
            return {"run": None}
        return {"run": _safe_dict_list([run])[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/data-quality/runs")
def list_data_quality_runs(limit: int = 50, user: dict = Depends(require_admin)):
    try:
        rows = db.get_data_quality_runs(limit=limit)
        return {"runs": _safe_dict_list(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/data-quality/repairs")
def list_data_repairs(limit: int = 100, user: dict = Depends(require_admin)):
    try:
        from data_quality.repair_log import list_repairs
        rows = list_repairs(limit=limit)
        return {"repairs": _safe_dict_list(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Local dev server wrapper
# ---------------------------------------------------------------------------


def _run_dev_server() -> None:
    """Run the FastAPI app via uvicorn.

    This is a convenience wrapper so you can run:
        uv run dashboard/app.py --reload --host 0.0.0.0 --port 8000

    (Equivalent to: uv run uvicorn dashboard.app:app ...)
    """

    import argparse

    try:
        import uvicorn
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"uvicorn is required to run the dashboard: {e}")

    parser = argparse.ArgumentParser(description="Run the MLB dashboard (uvicorn)")
    parser.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DASHBOARD_PORT", "8000")),
    )
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=os.environ.get("UVICORN_LOG_LEVEL", "info"))
    args = parser.parse_args()

    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    _run_dev_server()
