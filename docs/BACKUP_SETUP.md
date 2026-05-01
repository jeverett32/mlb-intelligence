# Database Backup Setup

Nightly Postgres dumps on the homelab app LXC (`10.1.23.161`).

- **Script**: `scripts/backup_db.sh` (deployed at `/opt/mlb/pipeline/scripts/backup_db.sh`)
- **Schedule**: systemd timer `mlb-backup.timer`, runs daily at **03:30 UTC**
- **Output**: `/opt/mlb/backups/mlb-YYYYMMDDTHHMMSSZ.sql.gz`
- **Retention**: 7 days (older `mlb-*.sql.gz` deleted by `find -mtime +7`)
- **Compression**: `gzip -9`
- **Env source**: `/opt/mlb/pipeline/.env` (reads `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`)
- **pg_dump flags**: `--no-owner --no-privileges --clean --if-exists`

No off-site / cloud backup. Dumps live only on the LXC's local disk.

---

## systemd Units

`/etc/systemd/system/mlb-backup.timer`:

```ini
[Unit]
Description=Run MLB DB backup nightly at 3:30 UTC

[Timer]
OnCalendar=*-*-* 03:30:00
Persistent=true
Unit=mlb-backup.service

[Install]
WantedBy=timers.target
```

`/etc/systemd/system/mlb-backup.service`:

```ini
[Unit]
Description=MLB Postgres nightly backup
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/mlb/pipeline
ExecStart=/opt/mlb/pipeline/scripts/backup_db.sh
StandardOutput=journal
StandardError=journal
```

Enable/start (one-time):

```bash
systemctl daemon-reload
systemctl enable --now mlb-backup.timer
```

---

## Operations

Run on the app LXC (e.g. `python homelab.py app "<cmd>"`).

| Action | Command |
|---|---|
| Trigger backup now | `systemctl start mlb-backup.service` |
| Check timer status | `systemctl list-timers mlb-backup.timer` |
| View last run logs | `journalctl -u mlb-backup.service -n 100` |
| List backups | `ls -lh /opt/mlb/backups` |
| Manual dump | `/opt/mlb/pipeline/scripts/backup_db.sh` |

---

## Restore

```bash
# Pick a backup
ls /opt/mlb/backups

# Restore (⚠️ overwrites target DB; --clean --if-exists drops existing objects)
gunzip -c /opt/mlb/backups/mlb-20260501T033001Z.sql.gz \
  | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME"
```

To copy a backup to your workstation:

```bash
scp root@10.1.23.161:/opt/mlb/backups/mlb-20260501T033001Z.sql.gz .
```

---

## Troubleshooting

| Symptom | Check |
|---|---|
| No new files in `/opt/mlb/backups` | `systemctl list-timers mlb-backup.timer`, `journalctl -u mlb-backup.service` |
| `pg_dump: error: connection ...` | `DB_*` vars in `/opt/mlb/pipeline/.env`; DB LXC reachable |
| `env file not found` | Confirm `/opt/mlb/pipeline/.env` exists and is readable by root |
| Disk filling up | Lower `RETAIN_DAYS` in `scripts/backup_db.sh` or prune manually |
