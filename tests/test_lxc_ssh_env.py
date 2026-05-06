from __future__ import annotations

import argparse

import homelab


SSH_ENV_NAMES = (
    "APP_SSH_HOST",
    "APP_SSH_PORT",
    "APP_SSH_USER",
    "APP_SSH_PASSWORD",
    "APP_SSH_KEY_PATH",
    "APP_SSH_KEY_PASSPHRASE",
    "DB_SSH_HOST",
    "DB_SSH_PORT",
    "DB_SSH_USER",
    "DB_SSH_PASSWORD",
    "DB_SSH_KEY_PATH",
    "DB_SSH_KEY_PASSPHRASE",
    "DB_HOST",
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        host=None,
        port=None,
        user=None,
        password=None,
        ssh_key=None,
        ssh_key_passphrase=None,
    )


def _clear_ssh_env(monkeypatch) -> None:
    for name in SSH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_app_target_prefers_app_ssh_env(monkeypatch) -> None:
    _clear_ssh_env(monkeypatch)
    monkeypatch.setenv("APP_SSH_HOST", "app-lxc")
    monkeypatch.setenv("APP_SSH_PORT", "2222")
    monkeypatch.setenv("APP_SSH_USER", "app-user")
    monkeypatch.setenv("APP_SSH_PASSWORD", "app-pass")

    client = homelab._build_client_from_args(_args(), "app")

    assert client.host == "app-lxc"
    assert client.port == 2222
    assert client.username == "app-user"
    assert client.password == "app-pass"


def test_db_target_reads_explicit_db_ssh_env(monkeypatch) -> None:
    _clear_ssh_env(monkeypatch)
    monkeypatch.setenv("DB_SSH_HOST", "db-lxc")
    monkeypatch.setenv("DB_SSH_PORT", "2223")
    monkeypatch.setenv("DB_SSH_USER", "db-user")
    monkeypatch.setenv("DB_SSH_PASSWORD", "db-pass")

    client = homelab._build_client_from_args(_args(), "db")

    assert client.host == "db-lxc"
    assert client.port == 2223
    assert client.username == "db-user"
    assert client.password == "db-pass"


def test_db_target_does_not_fall_back_to_app_ssh_creds(monkeypatch) -> None:
    _clear_ssh_env(monkeypatch)
    monkeypatch.setenv("APP_SSH_USER", "app-user")
    monkeypatch.setenv("APP_SSH_PASSWORD", "app-pass")
    monkeypatch.setenv("APP_SSH_PORT", "2222")
    monkeypatch.setenv("DB_SSH_HOST", "db-lxc")

    db_client = homelab._build_client_from_args(_args(), "db")

    assert db_client.host == "db-lxc"
    assert db_client.port == 22
    assert db_client.username == "root"
    assert db_client.password is None
