import pytest


def test_public_page_renders(client):
    r = client.get("/public")
    assert r.status_code == 200
    assert "MLB Model Performance" in r.text
    assert "Public Analytics" in r.text
    assert 'href="/contact"' in r.text


def test_public_page_renders_for_approved_session(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/public")
    assert r.status_code == 200
    assert "MLB Model Performance" in r.text
    assert "Public Analytics" in r.text
    assert "MLB Betting Dashboard" not in r.text


def test_public_performance_shape(client):
    r = client.get("/api/public/performance")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "total_bets",
        "wins",
        "losses",
        "accuracy",
        "total_wagered",
        "roi_pct",
        "calibration",
    }
    assert isinstance(body["calibration"], list)


def test_public_performance_empty_state(monkeypatch, app_module, client):
    monkeypatch.setattr(app_module.DB, "get_all_bets", lambda: app_module.pd.DataFrame())

    r = client.get("/api/public/performance")
    assert r.status_code == 200
    assert r.json() == {
        "total_bets": 0,
        "wins": 0,
        "losses": 0,
        "accuracy": 0.0,
        "total_wagered": 0.0,
        "roi_pct": 0.0,
        "calibration": [],
    }


def test_public_model_accuracy_shape(client):
    r = client.get("/api/public/model-accuracy")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "total",
        "correct",
        "incorrect",
        "accuracy",
        "market_total",
        "market_correct",
        "market_incorrect",
        "market_accuracy",
        "calibration",
    }
    assert isinstance(body["calibration"], list)
    assert "recent" not in body


def test_public_model_accuracy_empty_state(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_model_picks",
        lambda: app_module.pd.DataFrame(),
    )

    r = client.get("/api/public/model-accuracy")
    assert r.status_code == 200
    assert r.json() == {
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


def test_model_accuracy_public_private_field_separation(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_model_picks",
        lambda: app_module.pd.DataFrame(
            [
                {
                    "game_date": "2026-04-20",
                    "away_team": "NYM",
                    "home_team": "ATL",
                    "predicted_prob": 0.64,
                    "market_implied_prob": 0.58,
                    "bet_side": "home",
                    "home_win": True,
                }
            ]
        ),
    )

    public_response = client.get("/api/public/model-accuracy")
    assert public_response.status_code == 200
    assert "recent" not in public_response.json()

    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    private_response = client.get("/api/model-accuracy")
    assert private_response.status_code == 200
    body = private_response.json()
    assert set(body) == {
        "total",
        "correct",
        "incorrect",
        "accuracy",
        "market_total",
        "market_correct",
        "market_incorrect",
        "market_accuracy",
        "calibration",
        "recent",
    }
    assert body["recent"] == [
        {
            "game_date": "2026-04-20",
            "away_team": "NYM",
            "home_team": "ATL",
            "predicted_prob": 0.64,
            "market_prob": 0.58,
            "bet_side": "home",
            "home_win": True,
            "model_correct": True,
            "market_correct": True,
            "market_pred_home": True,
        }
    ]


def test_public_summary_shape(client):
    r = client.get("/api/public/summary")
    assert r.status_code == 200
    body = r.json()

    assert set(body) == {
        "performance",
        "model_accuracy",
    }
    assert set(body["performance"]) == {
        "total_bets",
        "wins",
        "losses",
        "accuracy",
        "total_wagered",
        "roi_pct",
        "calibration",
    }
    assert set(body["model_accuracy"]) == {
        "total",
        "correct",
        "incorrect",
        "accuracy",
        "market_total",
        "market_correct",
        "market_incorrect",
        "market_accuracy",
        "calibration",
    }
    assert isinstance(body["performance"]["calibration"], list)
    assert isinstance(body["model_accuracy"]["calibration"], list)


def test_public_summary_empty_state(monkeypatch, app_module, client):
    monkeypatch.setattr(app_module.DB, "get_all_bets", lambda: app_module.pd.DataFrame())
    monkeypatch.setattr(
        app_module.DB,
        "get_model_picks",
        lambda: app_module.pd.DataFrame(),
    )

    r = client.get("/api/public/summary")
    assert r.status_code == 200
    assert r.json() == {
        "performance": {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": 0.0,
            "total_wagered": 0.0,
            "roi_pct": 0.0,
            "calibration": [],
        },
        "model_accuracy": {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "market_total": 0,
            "market_correct": 0,
            "market_incorrect": 0,
            "market_accuracy": 0.0,
            "calibration": [],
        },
    }


def test_public_receipts_empty_state(monkeypatch, app_module, client):
    monkeypatch.setattr(app_module.DB, "get_all_bets", lambda: app_module.pd.DataFrame())

    r = client.get("/api/public/receipts")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "last_updated_utc",
        "settled_bets",
        "first_bet_date",
        "last_bet_date",
        "recent_bets",
        "roi_series",
    }
    assert body["settled_bets"] == 0
    assert body["recent_bets"] == []
    assert body["roi_series"] == []


def test_public_receipts_returns_recent_bets_and_roi(monkeypatch, app_module, client):
    monkeypatch.setattr(
        app_module.DB,
        "get_all_bets",
        lambda: app_module.pd.DataFrame(
            [
                {
                    "game_pk": 1,
                    "game_date": "2026-04-20",
                    "away_team": "NYM",
                    "home_team": "ATL",
                    "result": True,
                    "bet_side": "home",
                    "predicted_prob": 0.64,
                    "market_implied_prob": 0.58,
                    "edge": 0.06,
                    "bet_dollars": 10.0,
                    "n_contracts": 18,
                },
                {
                    "game_pk": 2,
                    "game_date": "2026-04-21",
                    "away_team": "BOS",
                    "home_team": "NYY",
                    "result": True,
                    "bet_side": "away",
                    "predicted_prob": 0.42,
                    "market_implied_prob": 0.5,
                    "edge": 0.08,
                    "bet_dollars": 12.0,
                    "profit_loss": -12.0,
                },
            ]
        ),
    )

    r = client.get("/api/public/receipts")
    assert r.status_code == 200
    body = r.json()
    assert body["settled_bets"] == 2
    assert body["first_bet_date"] == "2026-04-20"
    assert body["last_bet_date"] == "2026-04-21"
    assert body["recent_bets"][0]["matchup"] == "BOS @ NYY"
    assert body["recent_bets"][0]["result"] == "L"
    assert body["recent_bets"][0]["model_prob"] == 0.58
    assert body["roi_series"][-1]["cumulative_profit_loss"] == -4.0


def test_private_api_requires_auth(client):
    r = client.get("/api/balance")
    assert r.status_code in (401, 403)


# ── Phase 2: shared design system ────────────────────────────────────────────


def test_landing_includes_shared_stylesheet(client):
    r = client.get("/")
    assert r.status_code == 200
    assert '/static/css/public-site.css?v=' in r.text


def test_landing_includes_shared_script(client):
    r = client.get("/")
    assert r.status_code == 200
    assert '/static/js/public-site.js?v=' in r.text


def test_public_includes_shared_stylesheet(client):
    r = client.get("/public")
    assert r.status_code == 200
    assert '/static/css/public-site.css?v=' in r.text


def test_public_includes_shared_script(client):
    r = client.get("/public")
    assert r.status_code == 200
    assert '/static/js/public-site.js?v=' in r.text


def test_public_includes_chart_assets(client):
    r = client.get("/public")
    assert r.status_code == 200
    assert "chart.umd.min.js" in r.text


def test_login_includes_shared_stylesheet(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert '/static/css/public-site.css?v=' in r.text


def test_login_includes_shared_script(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert '/static/js/public-site.js?v=' in r.text


def test_login_hero_content_intact(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert "Members Area" in r.text
    assert "Private MLB betting workspace" in r.text


def test_contact_page_renders(client):
    r = client.get("/contact")
    assert r.status_code == 200
    assert "Reach out about the model" in r.text
    assert "john.everett32@gmail.com" in r.text
    assert "github.com/jeverett32" in r.text


def test_register_includes_shared_stylesheet(client):
    r = client.get("/register")
    assert r.status_code == 200
    assert '/static/css/public-site.css?v=' in r.text


def test_register_includes_shared_script(client):
    r = client.get("/register")
    assert r.status_code == 200
    assert '/static/js/public-site.js?v=' in r.text


def test_public_summary_delegates_correctly(client):
    """Verify summary endpoint returns same data as individual endpoints."""
    performance_r = client.get("/api/public/performance")
    model_accuracy_r = client.get("/api/public/model-accuracy")
    summary_r = client.get("/api/public/summary")

    assert performance_r.status_code == 200
    assert model_accuracy_r.status_code == 200
    assert summary_r.status_code == 200

    performance_data = performance_r.json()
    model_accuracy_data = model_accuracy_r.json()
    summary_data = summary_r.json()

    # Summary should contain exactly the same data as individual endpoints
    assert summary_data["performance"] == performance_data
    assert summary_data["model_accuracy"] == model_accuracy_data


# ── Phase 4: Enhanced landing page with live data ────────────────────────────


def test_landing_contains_trust_metrics_strip(client):
    """Landing page should contain a trust/metrics strip section."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'class="trust-metrics"' in r.text
    assert "data-live-metrics" in r.text


def test_landing_contains_how_it_works_section(client):
    """Landing page should contain a how-it-works section."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="how-it-works"' in r.text
    assert "How it works" in r.text or "How It Works" in r.text


def test_landing_contains_model_methodology_section(client):
    """Landing page should contain a model/methodology section."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="methodology"' in r.text
    assert ("methodology" in r.text.lower() or "model" in r.text.lower())


def test_landing_contains_performance_summary_section(client):
    """Landing page should contain a performance summary section with live data."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="performance-summary"' in r.text
    assert "data-endpoint" in r.text
    assert "/api/public/summary" in r.text


def test_landing_contains_about_section(client):
    """Landing page should contain an about section."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="about"' in r.text
    assert ("about" in r.text.lower() or "About" in r.text)


def test_landing_maintains_existing_structure(client):
    """Landing page should maintain existing nav, hero, and footer structure."""
    r = client.get("/")
    assert r.status_code == 200
    # Existing topbar with nav
    assert 'class="topbar"' in r.text
    assert 'class="nav-links"' in r.text
    # Existing hero
    assert 'class="ps-hero"' in r.text
    assert "Public MLB model results, updated daily." in r.text
    # Existing CTA links
    assert 'href="/public"' in r.text
    assert 'href="/login"' in r.text
    assert 'href="/register"' in r.text


# ── Phase 5: Public page refresh to align with landing page ──────────────────


def test_public_page_has_clear_link_to_landing(client):
    """Public page should have clear navigation back to landing page."""
    r = client.get("/public")
    assert r.status_code == 200
    assert 'class="brand" href="/"' in r.text


def test_public_page_does_not_mislabel_root_for_approved_users(monkeypatch, app_module, client):
    """Approved users still hit the private dashboard at /, so /public should not promise a public overview there."""
    monkeypatch.setattr(
        app_module.DB,
        "get_session_user",
        lambda session_id: {
            "email": "user@example.com",
            "approval_status": app_module.DB.USER_STATUS_APPROVED,
            "is_admin": False,
        },
    )
    client.cookies.set(app_module.COOKIE_NAME, "fake-session")

    r = client.get("/public")
    assert r.status_code == 200
    assert "Back to Overview" not in r.text


def test_public_page_maintains_analytics_identity(client):
    """Public page should feel analytics-focused, not like a second landing page."""
    r = client.get("/public")
    assert r.status_code == 200
    # Should contain analytics-specific headers and content
    assert "Model Accuracy" in r.text
    assert "Market Accuracy" in r.text
    assert "Bet Performance" in r.text
    # Should NOT be a duplicate landing page
    assert "Track the MLB model in public" not in r.text
    assert "how-it-works" not in r.text
    assert "methodology" not in r.text.lower()


def test_public_page_header_aligns_with_landing_design(client):
    """Public page header should align with landing page design patterns."""
    r = client.get("/public")
    assert r.status_code == 200
    # Should use similar header structure to landing page
    assert 'class="topbar"' in r.text
    assert 'class="brand"' in r.text
    assert 'class="nav-links"' in r.text


def test_public_page_preserves_analytics_functionality(client):
    """Public page should preserve all existing analytics functionality."""
    r = client.get("/public")
    assert r.status_code == 200
    # Should contain chart containers
    assert 'id="chart-cal-model"' in r.text
    assert 'id="chart-cal-perf"' in r.text
    # Should contain stat elements
    assert 'id="m-total"' in r.text
    assert 'id="p-roi"' in r.text
    # Should load data from public endpoints
    assert "/api/public/performance" in r.text
    assert "/api/public/model-accuracy" in r.text


def test_public_page_feels_like_deeper_proof_surface(client):
    """Public page should feel like the intentional deeper proof surface for landing page."""
    r = client.get("/public")
    assert r.status_code == 200
    # Should have detailed analytics sections that landing page points to
    assert "Model Calibration" in r.text
    assert "Bet Calibration" in r.text
    # Should have more granular stats than landing page
    assert "Correct" in r.text and "Incorrect" in r.text
    # Should have chart visualizations
    assert "chart.umd.min.js" in r.text
