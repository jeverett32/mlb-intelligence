# Database Backup Setup

Two independent backup jobs run nightly. Both named `mlb-backup.{timer,service}` but live on different hosts.

| Layer | Host | When (UTC) | Destination | Retention |
|---|---|---|---|---|
| Off-site | DB LXC `10.1.23.160` | 03:00 (+0–5 min jitter) | Google Drive folder `mlb-db-backups` | 3 newest |
| On-box | App LXC `10.1.23.161` | 03:30 | `/opt/mlb/backups/` (local disk) | 7 days |

The DB LXC job is the off-site copy of record. The app LXC job is a local floor in case GDrive auth breaks.

---

## Off-site: DB LXC → Google Drive

- **Script**: `/root/backups/backup_to_gdrive.py` (not in repo; lives on `mlb-db` LXC)
- **Venv**: `/root/backups/.venv`
- **Auth**: OAuth (personal Google account), scope `drive.file`
  - `/root/backups/google_drive_credentials.json` (client secrets)
  - `/root/backups/google_drive_token.pickle` (refresh token)
- **Dump**: `sudo -u postgres pg_dump -d mlb` → `gzip` → `/tmp/mlb-db-YYYYMMDD_HHMMSS.sql.gz` → upload → delete temp
- **Retention**: keeps 3 newest in folder, deletes older after upload
- **Timeout**: 1800 s

### systemd units (DB LXC)

`/etc/systemd/system/mlb-backup.timer`:

```ini
[Unit]
Description=Daily MLB DB backup

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
```

`/etc/systemd/system/mlb-backup.service`:

```ini
[Unit]
Description=MLB DB backup to Google Drive
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/root/backups/backup_to_gdrive.py
User=root
StandardOutput=journal
StandardError=journal
```

### Operations (DB LXC)

Run on `10.1.23.160` (`python homelab.py db "<cmd>"`).

| Action | Command |
|---|---|
| Trigger backup now | `systemctl start mlb-backup.service` |
| Check timer | `systemctl list-timers mlb-backup.timer` |
| View last run | `journalctl -u mlb-backup.service -n 100` |
| List GDrive backups | `/root/backups/backup_to_gdrive.py --list` |
| Download a backup | `/root/backups/backup_to_gdrive.py --download mlb-db-YYYYMMDD_HHMMSS.sql.gz` |
| Re-auth (initial) | `/root/backups/backup_to_gdrive.py --init`, then `--save-token <CODE>` |
| Wipe all GDrive backups | `/root/backups/backup_to_gdrive.py --cleanup` |

### Restore from GDrive

```bash
# On DB LXC or workstation
/root/backups/backup_to_gdrive.py --download mlb-db-20260503_030000.sql.gz
gunzip -c mlb-db-20260503_030000.sql.gz \
  | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME"
```

---

## On-box: App LXC → local disk

Secondary safety net. Faster restore than GDrive download. Disappears if the LXC dies — never the only copy.

- **Script**: `scripts/backup_db.sh` (deployed at `/opt/mlb/pipeline/scripts/backup_db.sh`)
- **Schedule**: systemd timer `mlb-backup.timer` at 03:30 UTC
- **Output**: `/opt/mlb/backups/mlb-YYYYMMDDTHHMMSSZ.sql.gz`
- **Retention**: 7 days (`find -mtime +7 -delete`)
- **pg_dump flags**: `--no-owner --no-privileges --clean --if-exists`
- **Env**: `/opt/mlb/pipeline/.env` (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`)

### systemd units (App LXC)

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

### Operations (App LXC)

Run on `10.1.23.161` (`python homelab.py app "<cmd>"`).

| Action | Command |
|---|---|
| Trigger backup now | `systemctl start mlb-backup.service` |
| Check timer | `systemctl list-timers mlb-backup.timer` |
| View last run | `journalctl -u mlb-backup.service -n 100` |
| List backups | `ls -lh /opt/mlb/backups` |
| Manual dump | `/opt/mlb/pipeline/scripts/backup_db.sh` |

### Restore from local

```bash
ls /opt/mlb/backups
gunzip -c /opt/mlb/backups/mlb-20260501T033001Z.sql.gz \
  | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME"
```

To copy a backup to your workstation:

```bash
scp root@10.1.23.161:/opt/mlb/backups/mlb-20260501T033001Z.sql.gz .
```

---

## Gaps / future work

- No WAL archiving / PITR — recovery granularity is "yesterday's dump." Acceptable for ~245 MB DB; revisit if data volume or RPO tightens.
- No automated restore drill. Manually verify quarterly: download newest GDrive dump, restore to scratch DB, run smoke queries.
- GDrive OAuth token is single-user. If revoked or quota changes, off-site stops silently. Watch journald for `mlb-backup.service` failures.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| No new files in `/opt/mlb/backups` | `systemctl list-timers mlb-backup.timer` on app LXC; `journalctl -u mlb-backup.service` |
| No new GDrive uploads | `journalctl -u mlb-backup.service` on DB LXC; token expiry; GDrive quota |
| `pg_dump: error: connection ...` (app LXC) | `DB_*` vars in `/opt/mlb/pipeline/.env`; DB LXC reachable |
| `env file not found` | Confirm `/opt/mlb/pipeline/.env` exists and is readable by root |
| Disk filling up (app LXC) | Lower `RETAIN_DAYS` in `scripts/backup_db.sh` or prune manually |
| OAuth token broken | Re-run `/root/backups/backup_to_gdrive.py --init` on DB LXC |
