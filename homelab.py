from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import paramiko
from dotenv import load_dotenv


DEFAULT_APP_CTID = 107
DEFAULT_DB_CTID = 106


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

    def run_in_lxc(
        self,
        ctid: int,
        command: str,
        timeout: int = 120,
        use_attach: bool = False,
    ) -> CommandResult:
        if use_attach:
            wrapped = f"lxc-attach -n {ctid} -- bash -lc {shlex.quote(command)}"
        else:
            wrapped = f"pct exec {ctid} -- bash -lc {shlex.quote(command)}"
        return self.run(wrapped, timeout=timeout)

    def push_file_to_lxc(
        self, ctid: int, local_path: Path, remote_path: str
    ) -> CommandResult:
        if self.client is None:
            raise RuntimeError("SSH client is not connected")
        tmp_name = f"/tmp/{local_path.name}"
        sftp = self.client.open_sftp()
        try:
            sftp.put(str(local_path), tmp_name)
        finally:
            sftp.close()
        push_cmd = f"pct push {ctid} {shlex.quote(tmp_name)} {shlex.quote(remote_path)}"
        return self.run(push_cmd)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot SSH helper for Proxmox + LXC homelab access."
    )
    parser.add_argument(
        "--host",
        default=_first_env("HOMELAB_PROXMOX_HOST", "HOMELAB_HOST", default="10.1.23.162"),
        help="Proxmox host IP/DNS (pct available). Env: HOMELAB_PROXMOX_HOST (preferred), HOMELAB_HOST (legacy).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(_first_env("HOMELAB_PROXMOX_PORT", "HOMELAB_PORT", default="22") or "22"),
        help="SSH port for Proxmox host. Env: HOMELAB_PROXMOX_PORT (preferred), HOMELAB_PORT (legacy).",
    )
    parser.add_argument(
        "--user",
        default=_first_env("HOMELAB_PROXMOX_USER", "HOMELAB_USER", default="root"),
        help="SSH username for Proxmox host. Env: HOMELAB_PROXMOX_USER (preferred), HOMELAB_USER (legacy).",
    )
    parser.add_argument(
        "--password",
        default=_first_env("HOMELAB_PROXMOX_PASSWORD", "HOMELAB_PASSWORD"),
        help="SSH password for Proxmox host. Env: HOMELAB_PROXMOX_PASSWORD (preferred), HOMELAB_PASSWORD (legacy).",
    )
    parser.add_argument(
        "--ssh-key",
        default=_first_env("HOMELAB_PROXMOX_SSH_KEY_PATH", "HOMELAB_SSH_KEY_PATH"),
        help="Path to SSH private key for Proxmox host. Env: HOMELAB_PROXMOX_SSH_KEY_PATH (preferred), HOMELAB_SSH_KEY_PATH (legacy).",
    )
    parser.add_argument(
        "--ssh-key-passphrase",
        default=_first_env("HOMELAB_PROXMOX_SSH_KEY_PASSPHRASE", "HOMELAB_SSH_KEY_PASSPHRASE"),
        help="Private key passphrase for Proxmox host. Env: HOMELAB_PROXMOX_SSH_KEY_PASSPHRASE (preferred), HOMELAB_SSH_KEY_PASSPHRASE (legacy).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for remote command execution.",
    )

    subparsers = parser.add_subparsers(dest="target", required=True)

    host = subparsers.add_parser("host", help="Run command directly on Proxmox host")
    host.add_argument("cmd", help="Command to run")

    app = subparsers.add_parser("app", help="Run command in app container")
    app.add_argument("cmd", help="Command to run")
    app.add_argument(
        "--direct",
        action="store_true",
        help="SSH directly to app host (no pct). Env: HOMELAB_APP_SSH_HOST (+ optional HOMELAB_APP_SSH_*).",
    )
    app.add_argument(
        "--ctid",
        type=int,
        default=int(os.getenv("HOMELAB_APP_CTID", str(DEFAULT_APP_CTID))),
        help="App LXC container ID.",
    )
    app.add_argument(
        "--attach",
        action="store_true",
        help="Use lxc-attach instead of pct exec.",
    )
    app.add_argument("--app-ssh-host", default=os.getenv("HOMELAB_APP_SSH_HOST"), help="Direct app SSH host (overrides HOMELAB_APP_SSH_HOST).")
    app.add_argument("--app-ssh-port", type=int, default=int(os.getenv("HOMELAB_APP_SSH_PORT", "22")), help="Direct app SSH port (default 22).")
    app.add_argument("--app-ssh-user", default=os.getenv("HOMELAB_APP_SSH_USER", "root"), help="Direct app SSH user (default root).")
    app.add_argument("--app-ssh-password", default=os.getenv("HOMELAB_APP_SSH_PASSWORD"), help="Direct app SSH password.")
    app.add_argument("--app-ssh-key", default=os.getenv("HOMELAB_APP_SSH_KEY_PATH"), help="Direct app SSH key path.")
    app.add_argument("--app-ssh-key-passphrase", default=os.getenv("HOMELAB_APP_SSH_KEY_PASSPHRASE"), help="Direct app SSH key passphrase.")

    db = subparsers.add_parser("db", help="Run command in db container")
    db.add_argument("cmd", help="Command to run")
    db.add_argument(
        "--direct",
        action="store_true",
        help="SSH directly to db host (no pct). Env: HOMELAB_DB_SSH_HOST (+ optional HOMELAB_DB_SSH_*).",
    )
    db.add_argument(
        "--ctid",
        type=int,
        default=int(os.getenv("HOMELAB_DB_CTID", str(DEFAULT_DB_CTID))),
        help="DB LXC container ID.",
    )
    db.add_argument(
        "--attach",
        action="store_true",
        help="Use lxc-attach instead of pct exec.",
    )
    db.add_argument("--db-ssh-host", default=os.getenv("HOMELAB_DB_SSH_HOST"), help="Direct db SSH host (overrides HOMELAB_DB_SSH_HOST).")
    db.add_argument("--db-ssh-port", type=int, default=int(os.getenv("HOMELAB_DB_SSH_PORT", "22")), help="Direct db SSH port (default 22).")
    db.add_argument("--db-ssh-user", default=os.getenv("HOMELAB_DB_SSH_USER", "root"), help="Direct db SSH user (default root).")
    db.add_argument("--db-ssh-password", default=os.getenv("HOMELAB_DB_SSH_PASSWORD"), help="Direct db SSH password.")
    db.add_argument("--db-ssh-key", default=os.getenv("HOMELAB_DB_SSH_KEY_PATH"), help="Direct db SSH key path.")
    db.add_argument("--db-ssh-key-passphrase", default=os.getenv("HOMELAB_DB_SSH_KEY_PASSPHRASE"), help="Direct db SSH key passphrase.")

    push = subparsers.add_parser(
        "push", help="Upload local file to LXC using SFTP + pct push"
    )
    push.add_argument("--ctid", type=int, required=True, help="Target LXC container ID")
    push.add_argument("--local", required=True, help="Local file path")
    push.add_argument("--remote", required=True, help="Destination path inside LXC")

    push_run = subparsers.add_parser(
        "push-run",
        help="Upload local file to LXC and run command there",
    )
    push_run.add_argument(
        "--ctid", type=int, required=True, help="Target LXC container ID"
    )
    push_run.add_argument("--local", required=True, help="Local file path")
    push_run.add_argument("--remote", required=True, help="Destination path inside LXC")
    push_run.add_argument(
        "--exec", dest="exec_cmd", required=True, help="Command to run in LXC"
    )
    push_run.add_argument(
        "--attach",
        action="store_true",
        help="Use lxc-attach instead of pct exec.",
    )

    return parser


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    if args.target == "app" and args.direct:
        direct_host = args.app_ssh_host or _required_env("HOMELAB_APP_SSH_HOST")
        direct_user = args.app_ssh_user or os.getenv("HOMELAB_APP_SSH_USER") or "root"
        with HomelabClient(
            host=direct_host,
            username=direct_user,
            port=args.app_ssh_port,
            password=args.app_ssh_password,
            key_path=args.app_ssh_key,
            key_passphrase=args.app_ssh_key_passphrase,
        ) as direct:
            wrapped = f"bash -lc {shlex.quote(args.cmd)}"
            return _print_result(direct.run(wrapped, timeout=args.timeout))

    if args.target == "db" and args.direct:
        direct_host = args.db_ssh_host or _required_env("HOMELAB_DB_SSH_HOST")
        direct_user = args.db_ssh_user or os.getenv("HOMELAB_DB_SSH_USER") or "root"
        with HomelabClient(
            host=direct_host,
            username=direct_user,
            port=args.db_ssh_port,
            password=args.db_ssh_password,
            key_path=args.db_ssh_key,
            key_passphrase=args.db_ssh_key_passphrase,
        ) as direct:
            wrapped = f"bash -lc {shlex.quote(args.cmd)}"
            return _print_result(direct.run(wrapped, timeout=args.timeout))

    host = args.host
    user = args.user
    if not host:
        host = _required_env("HOMELAB_PROXMOX_HOST")
    if not user:
        user = _required_env("HOMELAB_PROXMOX_USER")

    with HomelabClient(
        host=host,
        username=user,
        port=args.port,
        password=args.password,
        key_path=args.ssh_key,
        key_passphrase=args.ssh_key_passphrase,
    ) as homelab:
        if args.target == "host":
            return _print_result(homelab.run(args.cmd, timeout=args.timeout))

        if args.target == "app":
            result = homelab.run_in_lxc(
                ctid=args.ctid,
                command=args.cmd,
                timeout=args.timeout,
                use_attach=args.attach,
            )
            return _print_result(result)

        if args.target == "db":
            result = homelab.run_in_lxc(
                ctid=args.ctid,
                command=args.cmd,
                timeout=args.timeout,
                use_attach=args.attach,
            )
            return _print_result(result)

        if args.target == "push":
            local_path = Path(args.local).expanduser().resolve()
            if not local_path.exists() or not local_path.is_file():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            result = homelab.push_file_to_lxc(
                ctid=args.ctid,
                local_path=local_path,
                remote_path=args.remote,
            )
            return _print_result(result)

        if args.target == "push-run":
            local_path = Path(args.local).expanduser().resolve()
            if not local_path.exists() or not local_path.is_file():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            push_result = homelab.push_file_to_lxc(
                ctid=args.ctid,
                local_path=local_path,
                remote_path=args.remote,
            )
            push_code = _print_result(push_result)
            if push_code != 0:
                return push_code
            run_result = homelab.run_in_lxc(
                ctid=args.ctid,
                command=args.exec_cmd,
                timeout=args.timeout,
                use_attach=args.attach,
            )
            return _print_result(run_result)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
