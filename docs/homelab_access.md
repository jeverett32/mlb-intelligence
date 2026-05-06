# LXC SSH Helper

This repo includes a one-shot SSH helper script for running commands on the **MLB app LXC** and **MLB db LXC** over SSH.

- Script: `homelab.py`
- Transport: Paramiko SSH
- Model: **one command per invocation** (no persistent shell)
- Execution: commands run via `bash -lc '<cmd>'`

> Note
>
> `homelab.py` connects **directly** to each target over SSH. It does **not** use Proxmox, `pct exec`, CTIDs, or `lxc-attach`.
>
> There is currently **no** SSH entry point for the Proxmox host node in this repo.

## Install dependencies

```bash
uv sync
```

## Configure environment

Create `.env` from `.env.example` if needed.

### App target

`homelab.py app ...` uses:

```env
APP_SSH_HOST=<mlb-app-lxc-ssh-host>
APP_SSH_PORT=<ssh_port>
APP_SSH_USER=<ssh_user>
APP_SSH_PASSWORD=<optional-if-not-using-keys>

# Optional key auth:
APP_SSH_KEY_PATH=~/.ssh/id_ed25519
APP_SSH_KEY_PASSPHRASE=<optional>
```

### DB target

`homelab.py db ...` uses:

```env
DB_SSH_HOST=<mlb-db-lxc-ssh-host>
DB_SSH_PORT=<ssh_port>
DB_SSH_USER=<ssh_user>
DB_SSH_PASSWORD=<optional-if-not-using-keys>

# Optional key auth:
DB_SSH_KEY_PATH=~/.ssh/id_ed25519
DB_SSH_KEY_PASSPHRASE=<optional>
```

## Usage

Run from repo root.

### 1) Run a command on the app target

```bash
python3 homelab.py app "systemctl status mlb-dashboard --no-pager -l | tail -40"
python3 homelab.py app "journalctl -u mlb-dashboard -n 100 --no-pager"
python3 homelab.py app "curl -s http://localhost:<port>/health"
```

### 2) Run a command on the db target

```bash
python3 homelab.py db "pg_lsclusters"
```

If you want to run `psql`, DB credentials come from `.env` (e.g. `DB_USER`, `DB_PASSWORD`, `DB_NAME`).

Example pattern (generate a command locally; do not paste real secrets into docs):

```bash
python3 - <<'PY'
from dotenv import load_dotenv
load_dotenv()
import os

pw=os.getenv('DB_PASSWORD')
user=os.getenv('DB_USER')
db=os.getenv('DB_NAME')

print(
  "python3 homelab.py db "
  f"\"PGPASSWORD={pw} psql -U {user} -d {db} -h localhost -c \\\"\\\\dt\\\"\""
)
PY
```

### 3) Push a local file to app or db

```bash
python3 homelab.py push app --local ./tmp_script.py --remote /tmp/tmp_script.py
python3 homelab.py push db  --local ./tmp.sql       --remote /tmp/tmp.sql
```

### 4) Override SSH params via CLI flags (escape hatch)

Env vars are preferred, but you can override per-invocation:

```bash
python3 homelab.py app --host <app-ssh-host> --user <REDACTED_USER> --port <ssh_port> "uptime"
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
