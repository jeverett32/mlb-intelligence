#!/usr/bin/env python3
"""Read-only Postgres health snapshot for the MLB pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


DEFAULT_APP_BACKUP_DIR = Path("/opt/mlb/backups")
DEFAULT_WARN_BACKUP_HOURS = 30
DEFAULT_WARN_DEAD_RATIO = 0.2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _age_hours(ts: datetime | None) -> float | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_utc_now() - ts).total_seconds() / 3600.0


def _fmt_age(hours: float | None) -> str:
    if hours is None:
        return "n/a"
    return f"{hours:.1f}h"


def _query_all(sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    with db.pooled_connection() as conn:
        with conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return cur.fetchall()


def _query_one(sql: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
    rows = _query_all(sql, params)
    return rows[0] if rows else None


def _local_backup_status(path: Path, warn_hours: int) -> dict[str, Any]:
    files = sorted(path.glob("mlb-*.sql.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {
            "status": "WARN",
            "message": f"no backups found in {path}",
            "path": str(path),
        }
    latest = files[0]
    mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    age = _age_hours(mtime)
    status = "OK" if age is not None and age <= warn_hours else "WARN"
    return {
        "status": status,
        "latest": str(latest),
        "size_bytes": latest.stat().st_size,
        "age_hours": age,
        "count": len(files),
    }


def _gdrive_backup_status(warn_hours: int) -> dict[str, Any]:
    script = Path("/root/backups/backup_to_gdrive.py")
    if not script.exists():
        return {"status": "SKIP", "message": f"{script} not present on this host"}
    try:
        result = subprocess.run(
            [str(script), "--list"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return {"status": "WARN", "message": f"GDrive list failed: {exc}"}
    if result.returncode != 0:
        return {
            "status": "WARN",
            "message": "GDrive list command failed",
            "stderr": result.stderr.strip(),
        }
    latest_name = None
    latest_created = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("• ") and not line.startswith("- "):
            continue
        parts = line.split(" - ")
        if len(parts) < 2:
            continue
        latest_name = parts[0].split()[1] if parts[0].startswith("• ") else parts[0]
        try:
            latest_created = datetime.fromisoformat(parts[-1].replace("Z", "+00:00"))
        except ValueError:
            latest_created = None
        break
    age = _age_hours(latest_created)
    status = "OK" if latest_created and age is not None and age <= warn_hours else "WARN"
    return {
        "status": status,
        "latest": latest_name,
        "age_hours": age,
    }


def _connections() -> dict[str, Any]:
    row = _query_one(
        """
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE state = 'active') AS active,
               COUNT(*) FILTER (WHERE state = 'idle in transaction') AS idle_tx
        FROM pg_stat_activity
        WHERE datname = current_database()
        """
    )
    settings = dict(
        _query_all(
            """
            SELECT name, setting
            FROM pg_settings
            WHERE name IN ('max_connections', 'statement_timeout',
                           'idle_in_transaction_session_timeout', 'lock_timeout')
            """
        )
    )
    return {
        "total": row[0] if row else 0,
        "active": row[1] if row else 0,
        "idle_in_transaction": row[2] if row else 0,
        "max_connections": int(settings.get("max_connections", 0)),
        "timeouts": {
            "statement_timeout": settings.get("statement_timeout"),
            "idle_in_transaction_session_timeout": settings.get("idle_in_transaction_session_timeout"),
            "lock_timeout": settings.get("lock_timeout"),
        },
    }


def _dead_tuples(warn_ratio: float) -> list[dict[str, Any]]:
    rows = _query_all(
        """
        SELECT relname, n_live_tup, n_dead_tup, last_autovacuum, last_autoanalyze
        FROM pg_stat_user_tables
        WHERE relname IN (
            'games', 'bets', 'user_orders', 'paper_orders',
            'user_order_snapshots', 'kalshi_market_snapshots',
            'pipeline_runs', 'engineered_feature_cache'
        )
        ORDER BY relname
        """
    )
    out = []
    for relname, live, dead, last_vacuum, last_analyze in rows:
        denom = max(int(live or 0) + int(dead or 0), 1)
        ratio = int(dead or 0) / denom
        out.append(
            {
                "status": "WARN" if ratio > warn_ratio and int(dead or 0) > 1000 else "OK",
                "table": relname,
                "live": int(live or 0),
                "dead": int(dead or 0),
                "dead_ratio": ratio,
                "last_autovacuum": last_vacuum,
                "last_autoanalyze": last_analyze,
            }
        )
    return out


def _money_parity() -> dict[str, int]:
    checks = {
        "bets.bet_cents": "SELECT COUNT(*) FROM bets WHERE bet_dollars IS NOT NULL AND bet_cents IS DISTINCT FROM ROUND(bet_dollars * 100)::bigint",
        "bets.profit_loss_cents": "SELECT COUNT(*) FROM bets WHERE profit_loss IS NOT NULL AND profit_loss_cents IS DISTINCT FROM ROUND(profit_loss * 100)::bigint",
        "user_orders.bet_cents": "SELECT COUNT(*) FROM user_orders WHERE bet_dollars IS NOT NULL AND bet_cents IS DISTINCT FROM ROUND(bet_dollars * 100)::bigint",
        "user_orders.profit_loss_cents": "SELECT COUNT(*) FROM user_orders WHERE profit_loss IS NOT NULL AND profit_loss_cents IS DISTINCT FROM ROUND(profit_loss * 100)::bigint",
        "paper_orders.bet_cents": "SELECT COUNT(*) FROM paper_orders WHERE bet_dollars IS NOT NULL AND bet_cents IS DISTINCT FROM ROUND(bet_dollars * 100)::bigint",
        "paper_orders.profit_loss_cents": "SELECT COUNT(*) FROM paper_orders WHERE profit_loss IS NOT NULL AND profit_loss_cents IS DISTINCT FROM ROUND(profit_loss * 100)::bigint",
        "paper_orders.paper_bankroll_before_cents": "SELECT COUNT(*) FROM paper_orders WHERE paper_bankroll_before IS NOT NULL AND paper_bankroll_before_cents IS DISTINCT FROM ROUND(paper_bankroll_before * 100)::bigint",
        "paper_orders.paper_bankroll_after_cents": "SELECT COUNT(*) FROM paper_orders WHERE paper_bankroll_after IS NOT NULL AND paper_bankroll_after_cents IS DISTINCT FROM ROUND(paper_bankroll_after * 100)::bigint",
    }
    out: dict[str, int] = {}
    for name, sql in checks.items():
        row = _query_one(sql)
        out[name] = int(row[0] if row else 0)
    return out


def _schema_summary() -> dict[str, Any]:
    fk_missing = _query_all(
        """
        SELECT c.conrelid::regclass::text, c.conname
        FROM pg_constraint c
        WHERE c.contype = 'f'
          AND c.connamespace = 'public'::regnamespace
          AND NOT EXISTS (
              SELECT 1
              FROM pg_index i
              WHERE i.indrelid = c.conrelid
                AND i.indisvalid
                AND (string_to_array(i.indkey::text, ' ')::smallint[])[1:array_length(c.conkey, 1)] = c.conkey
          )
        ORDER BY 1, 2
        """
    )
    encrypted_bad = _query_one(
        """
        SELECT COUNT(*)
        FROM kalshi_accounts
        WHERE key_id NOT LIKE 'enc:%'
           OR key_path NOT LIKE 'enc:%'
        """
    )
    game_time_type = _query_one(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'games'
          AND column_name = 'game_time_utc'
        """
    )
    return {
        "fk_without_index": fk_missing,
        "unencrypted_kalshi_accounts": int(encrypted_bad[0] if encrypted_bad else 0),
        "game_time_utc_type": game_time_type[0] if game_time_type else None,
    }


def _pg_stat_statements(limit: int) -> list[dict[str, Any]]:
    available = _query_one("SELECT to_regclass('public.pg_stat_statements')")
    if not available or available[0] is None:
        return []
    rows = _query_all(
        """
        SELECT calls,
               round(mean_exec_time::numeric, 2) AS mean_ms,
               round(total_exec_time::numeric, 2) AS total_ms,
               rows,
               left(regexp_replace(query, '\\s+', ' ', 'g'), 180) AS query
        FROM pg_stat_statements
        WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
        ORDER BY total_exec_time DESC
        LIMIT %s
        """,
        (limit,),
    )
    return [
        {
            "calls": int(calls),
            "mean_ms": float(mean_ms),
            "total_ms": float(total_ms),
            "rows": int(row_count),
            "query": query,
        }
        for calls, mean_ms, total_ms, row_count, query in rows
    ]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_at": _utc_now().isoformat(),
        "host": os.uname().nodename,
        "backups": {
            "local": _local_backup_status(args.local_backup_dir, args.warn_backup_hours),
            "gdrive": _gdrive_backup_status(args.warn_backup_hours) if args.check_gdrive else {"status": "SKIP"},
        },
        "connections": _connections(),
        "dead_tuples": _dead_tuples(args.warn_dead_ratio),
        "money_parity": _money_parity(),
        "schema": _schema_summary(),
        "pg_stat_statements": _pg_stat_statements(args.pgss_limit),
    }


def _status_line(label: str, status: str, detail: str = "") -> str:
    suffix = f" - {detail}" if detail else ""
    return f"[{status}] {label}{suffix}"


def print_text(report: dict[str, Any]) -> None:
    print(f"DB health: {report['generated_at']} host={report['host']}")

    local = report["backups"]["local"]
    print(
        _status_line(
            "local backup",
            local["status"],
            local.get("message")
            or f"latest={local.get('latest')} age={_fmt_age(local.get('age_hours'))} count={local.get('count')}",
        )
    )
    gdrive = report["backups"]["gdrive"]
    print(
        _status_line(
            "gdrive backup",
            gdrive["status"],
            gdrive.get("message") or f"latest={gdrive.get('latest')} age={_fmt_age(gdrive.get('age_hours'))}",
        )
    )

    c = report["connections"]
    conn_status = "WARN" if c["idle_in_transaction"] else "OK"
    print(
        _status_line(
            "connections",
            conn_status,
            f"total={c['total']} active={c['active']} idle_tx={c['idle_in_transaction']} max={c['max_connections']}",
        )
    )

    for item in report["dead_tuples"]:
        print(
            _status_line(
                f"dead tuples {item['table']}",
                item["status"],
                f"live={item['live']} dead={item['dead']} ratio={item['dead_ratio']:.3f}",
            )
        )

    parity_bad = {k: v for k, v in report["money_parity"].items() if v}
    print(_status_line("money parity", "OK" if not parity_bad else "WARN", str(parity_bad or "clean")))

    schema = report["schema"]
    schema_bad = []
    if schema["fk_without_index"]:
        schema_bad.append(f"fk_without_index={schema['fk_without_index']}")
    if schema["unencrypted_kalshi_accounts"]:
        schema_bad.append(f"unencrypted_kalshi_accounts={schema['unencrypted_kalshi_accounts']}")
    if schema["game_time_utc_type"] != "timestamp with time zone":
        schema_bad.append(f"game_time_utc_type={schema['game_time_utc_type']}")
    print(_status_line("schema", "OK" if not schema_bad else "WARN", "; ".join(schema_bad) or "invariants clean"))

    print("pg_stat_statements top:")
    for row in report["pg_stat_statements"]:
        print(
            f"  calls={row['calls']} mean_ms={row['mean_ms']:.2f} "
            f"total_ms={row['total_ms']:.2f} rows={row['rows']} query={row['query']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--local-backup-dir", type=Path, default=DEFAULT_APP_BACKUP_DIR)
    parser.add_argument("--warn-backup-hours", type=int, default=DEFAULT_WARN_BACKUP_HOURS)
    parser.add_argument("--warn-dead-ratio", type=float, default=DEFAULT_WARN_DEAD_RATIO)
    parser.add_argument("--pgss-limit", type=int, default=10)
    parser.add_argument("--check-gdrive", action="store_true", help="Also run /root/backups/backup_to_gdrive.py --list if present")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect(args)
    if args.json:
        print(json.dumps(report, default=str, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
