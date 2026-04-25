from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import paramiko
from dotenv import load_dotenv


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str


class HomelabClient:
    def __init__(
        self,
        host: str,
        username: str,
        port: int = 22,
        password: str | None = None,
        key_path: str | None = None,
        key_passphrase: str | None = None,
    ) -> None:
        self.host = host
        self.username = username
        self.port = port
        self.password = password
        self.key_path = key_path
        self.key_passphrase = key_passphrase
        self.client: paramiko.SSHClient | None = None

    def __enter__(self) -> "HomelabClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def connect(self) -> None:
        if self.client is not None:
            return
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            key_filename=self.key_path,
            passphrase=self.key_passphrase,
            look_for_keys=self.key_path is None and self.password is None,
            allow_agent=True,
            timeout=20,
        )
        self.client = client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def run(self, command: str, timeout: int = 120) -> CommandResult:
        if self.client is None:
            raise RuntimeError("SSH client is not connected")
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out_text = stdout.read().decode("utf-8", errors="replace")
        err_text = stderr.read().decode("utf-8", errors="replace")
        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=out_text,
            stderr=err_text,
        )

    def push_file(self, local_path: Path, remote_path: str) -> None:
        if self.client is None:
            raise RuntimeError("SSH client is not connected")
        sftp = self.client.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _print_result(result: CommandResult) -> int:
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.exit_code


def _build_client_from_args(args: argparse.Namespace, target: str) -> HomelabClient:
    """
    Resolve SSH connection params for a target ('app' or 'db').

    App target reads: HOMELAB_HOST, HOMELAB_USER, HOMELAB_PASSWORD, HOMELAB_PORT
    DB target reads:  HOMELAB_DB_SSH_HOST (falls back to DB_HOST),
                      HOMELAB_DB_SSH_USER / HOMELAB_DB_SSH_PASSWORD (fall back to HOMELAB_USER / HOMELAB_PASSWORD),
                      HOMELAB_DB_SSH_PORT (falls back to HOMELAB_PORT, then 22)
    """
    if target == "app":
        host = args.host or _first_env("HOMELAB_HOST")
        user = args.user or _first_env("HOMELAB_USER", default="root")
        password = args.password or _first_env("HOMELAB_PASSWORD")
        port = args.port or int(_first_env("HOMELAB_PORT", default="22") or "22")
        key_path = args.ssh_key or _first_env("HOMELAB_SSH_KEY_PATH")
        key_passphrase = args.ssh_key_passphrase or _first_env("HOMELAB_SSH_KEY_PASSPHRASE")
    else:  # db
        host = args.host or _first_env("HOMELAB_DB_SSH_HOST", "DB_HOST")
        user = args.user or _first_env("HOMELAB_DB_SSH_USER", "HOMELAB_USER", default="root")
        password = args.password or _first_env("HOMELAB_DB_SSH_PASSWORD", "HOMELAB_PASSWORD")
        port = args.port or int(
            _first_env("HOMELAB_DB_SSH_PORT", "HOMELAB_PORT", default="22") or "22"
        )
        key_path = args.ssh_key or _first_env("HOMELAB_DB_SSH_KEY_PATH", "HOMELAB_SSH_KEY_PATH")
        key_passphrase = args.ssh_key_passphrase or _first_env(
            "HOMELAB_DB_SSH_KEY_PASSPHRASE", "HOMELAB_SSH_KEY_PASSPHRASE"
        )

    if not host:
        raise ValueError(
            f"No SSH host for target '{target}'. "
            "Set HOMELAB_HOST (app) or HOMELAB_DB_SSH_HOST / DB_HOST (db)."
        )

    return HomelabClient(
        host=host,
        username=user or "root",
        port=port,
        password=password,
        key_path=key_path,
        key_passphrase=key_passphrase,
    )


def _add_ssh_override_args(parser: argparse.ArgumentParser) -> None:
    """Add optional SSH override flags (env vars are preferred; these are escape hatches)."""
    parser.add_argument("--host", default=None, help="Override SSH host")
    parser.add_argument("--port", type=int, default=None, help="Override SSH port")
    parser.add_argument("--user", default=None, help="Override SSH username")
    parser.add_argument("--password", default=None, help="Override SSH password")
    parser.add_argument("--ssh-key", default=None, help="Override SSH private key path")
    parser.add_argument("--ssh-key-passphrase", default=None, help="Override SSH key passphrase")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SSH helper for MLB homelab LXC containers.\n\n"
            "Connects directly to each LXC via SSH — no Proxmox/pct needed.\n\n"
            "Env vars (set in .env):\n"
            "  App:  HOMELAB_HOST, HOMELAB_USER, HOMELAB_PASSWORD, HOMELAB_PORT\n"
            "  DB:   HOMELAB_DB_SSH_HOST (or DB_HOST), HOMELAB_DB_SSH_USER,\n"
            "        HOMELAB_DB_SSH_PASSWORD, HOMELAB_DB_SSH_PORT\n"
            "        (DB SSH creds fall back to HOMELAB_USER / HOMELAB_PASSWORD)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timeout", type=int, default=120, help="Command timeout in seconds")

    subparsers = parser.add_subparsers(dest="target", required=True)

    # app subcommand
    app = subparsers.add_parser("app", help="Run command in app LXC (HOMELAB_HOST)")
    app.add_argument("cmd", help="Shell command to run")
    _add_ssh_override_args(app)

    # db subcommand
    db = subparsers.add_parser(
        "db", help="Run command in DB LXC (HOMELAB_DB_SSH_HOST or DB_HOST)"
    )
    db.add_argument("cmd", help="Shell command to run")
    _add_ssh_override_args(db)

    # push subcommand
    push = subparsers.add_parser("push", help="Upload a local file to app or db LXC")
    push.add_argument("target_lxc", choices=["app", "db"], help="Destination LXC")
    push.add_argument("--local", required=True, help="Local file path")
    push.add_argument("--remote", required=True, help="Destination path on LXC")
    _add_ssh_override_args(push)

    return parser


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    if args.target in ("app", "db"):
        with _build_client_from_args(args, args.target) as client:
            result = client.run(f"bash -lc {__import__('shlex').quote(args.cmd)}", timeout=args.timeout)
            return _print_result(result)

    if args.target == "push":
        local_path = Path(args.local).expanduser().resolve()
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        with _build_client_from_args(args, args.target_lxc) as client:
            client.push_file(local_path, args.remote)
            print(f"Pushed {local_path} → {args.remote}")
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
