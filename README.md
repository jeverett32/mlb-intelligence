# MLB Betting Pipeline

An automated MLB moneyline betting system that fetches game data, trains a gradient-boosted model, identifies positive-expected-value bets, and places orders on Kalshi — with a self-hosted web dashboard for monitoring.

---

## How It Works

```
fetch/fetch_data.py      — pulls schedule, odds, pitcher stats, weather, FanGraphs
fetch/fetch_balance.py   — checks Kalshi account balance
model/predict.py         — runs the trained model, calculates edge, sizes the bet
bet/place_bet.py         — places the Kalshi order (or dry-runs if live toggle is off)
run_pipeline.py          — orchestrates all four steps; loops forever, fires 5 min
                           before each scheduled game
```

The pipeline stores everything in PostgreSQL (games, bets, balance history, settings). A FastAPI dashboard at port 8080 visualises predictions, bet history, model performance, and lets you toggle live betting on/off without restarting anything.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Your machine (Windows/Mac/Linux)            │
│  - Clone repo, edit .env, push to GitHub    │
└──────────────────┬──────────────────────────┘
                   │  SSH / git pull
┌──────────────────▼──────────────────────────┐
│  Proxmox VE Homelab                          │
│                                              │
│  LXC 106 — PostgreSQL 16                    │
│    IP: 10.1.23.160                           │
│    DB: mlb  User: mlb                        │
│                                              │
│  LXC 107 — App Server (Python 3.12 + uv)    │
│    IP: 10.1.23.161                           │
│    mlb-pipeline.service   (pipeline loop)   │
│    mlb-dashboard.service  (FastAPI :8080)   │
└─────────────────────────────────────────────┘
```

---

## Prerequisites

- **Python 3.10+** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)** on your local machine
- A **Proxmox VE** homelab (or any two Linux servers/VMs for Postgres + app)
- A **[Kalshi](https://kalshi.com)** account with API credentials
- A free **[The Odds API](https://the-odds-api.com)** key (500 requests/month free)

---

## Setup

### 1. Clone and install locally

```bash
git clone https://github.com/jeverett32/mlb-pipeline.git
cd mlb-pipeline
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `KALSHI_KEY_ID` | Your Kalshi API key ID (UUID format) |
| `KALSHI_KEY_PATH` | Path to your Kalshi PEM private key file (default: `kalshi-key.pem`) |
| `KALSHI_ENV` | `prod` for real money, `demo` for paper trading |
| `ODDS_API_KEY` | Free key from [the-odds-api.com](https://the-odds-api.com) |
| `DB_HOST` | PostgreSQL host (e.g. `10.1.23.160`) |
| `DB_PORT` | PostgreSQL port (default `5432`) |
| `DB_NAME` | Database name (`mlb`) |
| `DB_USER` | Database user (`mlb`) |
| `DB_PASSWORD` | Database password |

Save your Kalshi private key as `kalshi-key.pem` in the project root (the full `-----BEGIN PRIVATE KEY-----` block).

---

### 3. Set up PostgreSQL (LXC 106)

Create a Debian 12 LXC in Proxmox, then SSH into it:

```bash
# Install Postgres 16
apt update && apt install -y postgresql-16

# Create database and user
sudo -u postgres psql <<'SQL'
CREATE DATABASE mlb;
CREATE USER mlb WITH PASSWORD 'your-password-here';
GRANT ALL PRIVILEGES ON DATABASE mlb TO mlb;
\c mlb
GRANT ALL ON SCHEMA public TO mlb;
SQL

# Allow connections from your LAN (edit pg_hba.conf)
echo "host  mlb  mlb  10.1.23.0/24  scram-sha-256" >> /etc/postgresql/16/main/pg_hba.conf

# Allow Postgres to listen on all interfaces (edit postgresql.conf)
sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" \
    /etc/postgresql/16/main/postgresql.conf

systemctl restart postgresql
```

The pipeline creates all tables automatically on first run via `db.py`.

---

### 4. Migrate historical data

The repo includes `data/master_mlb.csv` with seasons 2010–2025. Run the migration once to load it into Postgres:

```bash
uv run migrate_to_postgres.py
```

This upserts ~36,000 historical game rows that the model trains on.

---

### 5. Set up the app server (LXC 107)

Create a second Debian 12 LXC, then:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the repo
git clone https://github.com/jeverett32/mlb-pipeline.git /opt/mlb/pipeline
cd /opt/mlb/pipeline

# Copy your .env and kalshi-key.pem
# (scp or paste contents manually)

uv sync
```

#### Create systemd services

**`/etc/systemd/system/mlb-pipeline.service`**
```ini
[Unit]
Description=MLB Betting Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
NotifyAccess=all
WatchdogSec=120
User=mlb
Group=mlb
WorkingDirectory=/opt/mlb/pipeline
ExecStart=/opt/mlb/pipeline/.venv/bin/python3 run_pipeline.py
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal
Environment=HOME=/opt/mlb
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/mlb-dashboard.service`**
```ini
[Unit]
Description=MLB Dashboard API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=mlb
Group=mlb
WorkingDirectory=/opt/mlb/pipeline
ExecStart=/opt/mlb/pipeline/.venv/bin/uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=HOME=/opt/mlb

[Install]
WantedBy=multi-user.target
```

```bash
useradd --system --home-dir /opt/mlb --shell /usr/sbin/nologin mlb
chown -R mlb:mlb /opt/mlb/pipeline
chmod 700 /opt/mlb/pipeline
chmod 600 /opt/mlb/pipeline/.env /opt/mlb/pipeline/kalshi-key.pem
systemctl daemon-reload
systemctl enable --now mlb-pipeline mlb-dashboard
```

---

### 6. Fix IPv6/DNS (LXC containers only)

If your LXC container resolves hostnames to IPv6 addresses but has no IPv6 routing, add this to force IPv4:

```bash
echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf
```

---

## Dashboard

Navigate to `http://<app-server-ip>:8080` in your browser.

**Default login:**
- Email: *(set during setup — see below)*
- Password: *(set during setup)*

To create or change the login credentials, run this on the app server:

```bash
cd /opt/mlb/pipeline
uv run python - <<'EOF'
import bcrypt, db as DB
DB.init_auth_tables()
h = bcrypt.hashpw(b'your-password', bcrypt.gensalt()).decode()
DB.upsert_user('your@email.com', h)
print("User created.")
EOF
```

### Dashboard tabs

| Tab | Description |
|---|---|
| **Overview** | Balance chart, win rate, ROI, total wagered |
| **Upcoming Games** | Today & tomorrow's games with market odds and model predictions |
| **Bet Log** | Full history of every bet placed with result and P&L |
| **Performance** | Model calibration chart and aggregate stats |
| **Database** | Browse the raw `games`, `bets`, `balance`, and `settings` tables |

### Live betting toggle

The **Live Betting** switch in the top-right controls whether `place_bet.py` places real Kalshi orders or just logs what it would have done. It defaults to **OFF** — flip it on when you're ready to go live. The state persists in the `settings` table, so restarting the service doesn't reset it.

---

## Running in production

Once the systemd services are enabled, everything runs automatically:

- **Pipeline** wakes up 5 minutes before each scheduled game, fetches fresh data, runs the model, and places (or dry-runs) the bet. Between games it sleeps.
- **Dashboard** runs continuously on port 8080.
- Both services restart automatically on failure and on server reboots.

Check the pipeline logs in real time:
```bash
journalctl -u mlb-pipeline -f
```

---

## Project structure

```
mlb-pipeline/
├── bet/
│   └── place_bet.py          # Kalshi order placement
├── dashboard/
│   ├── app.py                # FastAPI backend
│   └── templates/
│       ├── index.html        # Main dashboard UI
│       └── login.html        # Login page
├── data/
│   └── master_mlb.csv        # Historical game data (2010–2025)
├── docs/                     # Per-module documentation
├── fetch/
│   ├── fetch_data.py         # Schedule, odds, pitchers, weather, FanGraphs
│   ├── fetch_balance.py      # Kalshi balance sync
│   ├── odds_api.py           # The Odds API client
│   ├── scraper.py            # SBR odds scraper
│   └── pools.py              # Async helpers
├── model/
│   ├── train.py              # Model training (LightGBM/XGBoost)
│   └── predict.py            # Prediction + Kelly criterion sizing
├── db.py                     # All PostgreSQL helpers
├── kalshi_client.py          # Kalshi RSA-PSS auth + API helpers
├── migrate_to_postgres.py    # One-time CSV → Postgres migration
├── run_pipeline.py           # Main orchestrator loop
├── pyproject.toml
└── .env.example
```

---

## Data sources

| Source | What it provides |
|---|---|
| [MLB Stats API](https://statsapi.mlb.com) | Schedule, scores, pitcher IDs, boxscores |
| [SBR (SportsbookReview)](https://www.sportsbookreview.com) | Historical and current moneyline odds |
| [The Odds API](https://the-odds-api.com) | Fallback live odds when SBR is unavailable |
| [Open-Meteo](https://open-meteo.com) | Weather forecasts and historical weather |
| [FanGraphs](https://www.fangraphs.com) | Team batting/pitching stats (wRC+, FIP, wOBA, etc.) |
| [Kalshi](https://kalshi.com) | Prediction market order placement and balance |
