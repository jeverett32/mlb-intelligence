# Homelab Access Helper (SSH)

This repo includes a one-shot SSH helper script for running commands on the homelab **app** and **db** LXC hosts.

- Script: `homelab.py`
- Transport: Paramiko SSH
- Model: **one command per invocation** (no persistent shell)
- Execution: commands run via `bash -lc '<cmd>'` so common shell features (PATH, `cd`, etc.) work as expected.

> Note
> 
> `homelab.py` connects **directly** to each target over SSH. It does **not** use Proxmox, `pct exec`, CTIDs, or `lxc-attach`.

## Install dependencies

Dependencies are managed via `pyproject.toml`.

```bash
uv sync
```

## Configure environment

Create `.env` from `.env.example` if needed, then set SSH parameters.

### App target

`homelab.py app ...` uses:

```env
HOMELAB_HOST=<REDACTED_IP>
HOMELAB_PORT=22
HOMELAB_USER=root
HOMELAB_PASSWORD=...
# Optional key auth (preferred over password when available):
HOMELAB_SSH_KEY_PATH=~/.ssh/id_ed25519
HOMELAB_SSH_KEY_PASSPHRASE=...
```

### DB target

`homelab.py db ...` uses:

```env
# SSH host for DB machine/LXC. Falls back to DB_HOST if not set.
HOMELAB_DB_SSH_HOST=<REDACTED_IP>

# DB SSH creds can be distinct; they fall back to HOMELAB_USER/HOMELAB_PASSWORD.
HOMELAB_DB_SSH_PORT=22
HOMELAB_DB_SSH_USER=root
HOMELAB_DB_SSH_PASSWORD=...

# Optional key auth (falls back to HOMELAB_SSH_KEY_PATH/PASSPHRASE):
HOMELAB_DB_SSH_KEY_PATH=~/.ssh/id_ed25519
HOMELAB_DB_SSH_KEY_PASSPHRASE=...
```

## Usage

Run from repo root.

You can use either `python3` directly or `uv run`:

```bash
python3 homelab.py app "hostname"
uv run homelab.py app "hostname"
```

### 1) Run a command on the app target

```bash
python3 homelab.py app "systemctl status mlb-dashboard --no-pager -l | tail -40"
python3 homelab.py app "journalctl -u mlb-dashboard -n 100 --no-pager"
python3 homelab.py app "curl -s http://localhost:<REDACTED_PORT>/health"
```

### 2) Run a command on the db target

```bash
python3 homelab.py db "pg_lsclusters"
```

If you want to run `psql`, DB credentials come from `.env` (e.g. `DB_USER`, `DB_PASSWORD`, `DB_NAME`). Example pattern:

```bash
# locally load env, then inject into the remote command
python3 - <<'PY'
from dotenv import load_dotenv
load_dotenv()
import os
pw=os.getenv('DB_PASSWORD')
user=os.getenv('DB_USER')
db=os.getenv('DB_NAME')
print(f"python3 homelab.py db \"PGPASSWORD={pw} psql -U {user} -d {db} -h <REDACTED_IP> -c \\\"\\\\dt\\\"\"")
PY
```

### 3) Push a local file to app or db

`push` uploads a local file directly to the target via SFTP.

```bash
python3 homelab.py push app --local ./tmp_script.py --remote /tmp/tmp_script.py
python3 homelab.py push db  --local ./tmp.sql       --remote /tmp/tmp.sql
```

### 4) Override SSH params via CLI flags (escape hatch)

Env vars are preferred, but you can override per-invocation:

```bash
python3 homelab.py app --host <REDACTED_IP> --user <REDACTED_USER> --port <REDACTED_PORT> "uptime"
python3 homelab.py db  --ssh-key ~/.ssh/id_ed25519 "pg_isready"
```

Supported override flags on each subcommand:

- `--host`
- `--port`
- `--user`
- `--password`
- `--ssh-key`
- `--ssh-key-passphrase`

## Behavior notes

- Each run creates a fresh SSH connection.
- Stdout/stderr are printed through to your terminal.
- The script exits with the same exit code as the remote command.
- Default command timeout is 120s (override with `--timeout`).
