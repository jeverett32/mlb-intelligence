#!/usr/bin/env bash
# Configure PostgreSQL log rotation and sane logging on the DB LXC.
# Run as root on the database host:
#   bash scripts/configure_db_log_rotation.sh
#
# Safe to re-run (idempotent).

set -euo pipefail

LOGROTATE_FILE="/etc/logrotate.d/postgresql-common"
PG_CLUSTER="${PG_CLUSTER:-16/main}"

echo "==> Disk before"
df -h /

echo "==> Pruning old PostgreSQL log archives"
shopt -s nullglob
for f in /var/log/postgresql/postgresql-*-main.log.[0-9]* /var/log/postgresql/postgresql-*-main.log.[0-9]*.gz; do
  rm -f "$f"
done

echo "==> Truncating active PostgreSQL log (stderr capture file)"
for f in /var/log/postgresql/postgresql-*-main.log; do
  if [[ -f "$f" ]]; then
    sudo -u postgres truncate -s 0 "$f"
    echo "  truncated $f"
  fi
done

echo "==> Installing logrotate config"
cat >"$LOGROTATE_FILE" <<'EOF'
/var/log/postgresql/*.log {
    daily
    rotate 7
    maxsize 100M
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su root adm
}
EOF
chmod 644 "$LOGROTATE_FILE"

echo "==> Tuning PostgreSQL logging (ALTER SYSTEM)"
sudo -u postgres psql -v ON_ERROR_STOP=1 <<'SQL'
ALTER SYSTEM SET log_min_duration_statement = '5s';
ALTER SYSTEM SET log_connections = 'off';
ALTER SYSTEM SET log_disconnections = 'off';
SQL

echo "==> Reloading PostgreSQL cluster ${PG_CLUSTER}"
pg_ctlcluster "${PG_CLUSTER%%/*}" "${PG_CLUSTER#*/}" reload

echo "==> Forcing logrotate dry-run"
logrotate -d "$LOGROTATE_FILE" 2>&1 | tail -5 || true

echo "==> Disk after"
df -h /

echo "Done."
