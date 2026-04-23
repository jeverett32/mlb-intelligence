# Homelab Access Helper

This repo includes a one-shot SSH helper script for Proxmox + LXC workflows:

- `homelab.py` opens a fresh Paramiko SSH session per run.
- Commands are executed as one-shot operations (no persistent interactive shell).
- LXC commands are bridged through `pct exec` by default, with optional `lxc-attach`.
- Complex file payloads can be uploaded via SFTP, then moved into containers with `pct push`.

## Install dependency

`paramiko` is included in `pyproject.toml`. Install/update dependencies:

```bash
uv sync
```

## Configure environment

Copy `.env.example` to `.env` if you have not already, then set homelab values:

```env
HOMELAB_PROXMOX_HOST=10.1.23.162
HOMELAB_PROXMOX_PORT=22
HOMELAB_PROXMOX_USER=root
HOMELAB_PROXMOX_PASSWORD=your-password
HOMELAB_APP_CTID=107
HOMELAB_DB_CTID=106
```

You can use SSH keys instead of password by setting `HOMELAB_PROXMOX_SSH_KEY_PATH` (and optional `HOMELAB_PROXMOX_SSH_KEY_PASSPHRASE`). Legacy env vars `HOMELAB_SSH_KEY_PATH` / `HOMELAB_SSH_KEY_PASSPHRASE` still work.

## Usage

All examples run from repo root.

### 1) Run directly on Proxmox host

```bash
uv run homelab.py host "pct list"
```

### 2) Run in app container (CT 107 default, via Proxmox `pct exec`)

```bash
uv run homelab.py app "cd /opt/mlb/pipeline && git pull"
```

Use `--attach` when quoting is difficult and you want `lxc-attach`:

```bash
uv run homelab.py app --attach "python -V"
```

### 3) Run in app host directly (skip Proxmox)

If you have SSH access straight to the app container/VM, set:

```env
HOMELAB_APP_SSH_HOST=mlb-app
# optional:
HOMELAB_APP_SSH_PORT=22
HOMELAB_APP_SSH_USER=root
HOMELAB_APP_SSH_PASSWORD=your-password
```

Then run:

```bash
uv run homelab.py app --direct "cd /opt/mlb/pipeline && git pull"
```

### 4) Run in db container (CT 106 default)

```bash
uv run homelab.py db "psql --version"
```

### 5) Push a local file into a container

```bash
uv run homelab.py push --ctid 107 --local ./tmp_script.py --remote /tmp/tmp_script.py
```

### 6) Push and run (best for complex quoting)

```bash
uv run homelab.py push-run --ctid 107 --local ./tmp_script.py --remote /tmp/tmp_script.py --exec "python /tmp/tmp_script.py"
```

## Notes

- Each invocation is independent and creates a new SSH connection.
- `homelab.py` prints stdout/stderr from the remote command and exits with the same exit code.
- Avoid hard-coding credentials in scripts; use `.env` or CLI args.
