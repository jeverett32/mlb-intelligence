#!/usr/bin/env bash
# Daily Postgres dump with 7-day retention.
# Reads DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD from /opt/mlb/pipeline/.env
set -euo pipefail

ENV_FILE=/opt/mlb/pipeline/.env
BACKUP_DIR=/opt/mlb/backups
RETAIN_DAYS=7

if [[ ! -f "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

mkdir -p "$BACKUP_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$BACKUP_DIR/mlb-${STAMP}.sql.gz"

PGPASSWORD="$DB_PASSWORD" pg_dump \
  -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" \
  --no-owner --no-privileges --clean --if-exists \
  "$DB_NAME" | gzip -9 > "$OUT"

chmod 600 "$OUT"
find "$BACKUP_DIR" -name 'mlb-*.sql.gz' -mtime +"$RETAIN_DAYS" -delete
echo "backup: $OUT ($(du -h "$OUT" | cut -f1))"
