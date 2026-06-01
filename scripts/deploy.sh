#!/bin/bash

# MLB Pipeline Auto-Deploy Script
# This script safely deploys updates from the main branch to the homelab
# Includes health checks, rollback capability, and service management

set -euo pipefail

# Configuration
REPO_DIR="/opt/mlb/pipeline"
BACKUP_DIR="/opt/mlb/backup"
HEALTH_CHECK_URL="http://localhost:8080/health"
HEALTH_CHECK_TIMEOUT=30
MAX_HEALTH_CHECK_RETRIES=6
PIPELINE_FILES_CHANGED_PATTERNS="run_pipeline.py|run_pipeline_v2.py|fetch/|models/|bet/|db.py|kalshi_client.py"
LOCK_FILE="/tmp/mlb-deploy.lock"
BACKUP_PATH_FILE="/tmp/mlb_deploy_backup_path"
SYSTEMCTL_BIN="$(command -v systemctl)"
UV_CACHE_DIR_DEFAULT="${REPO_DIR}/.cache/uv"
DEFAULT_REPO_SLUG="jeverett32/mlb-intelligence"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp}: $message" >&1
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp}: $message" >&2
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp}: $message" >&2
            ;;
        DEBUG)
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp}: $message" >&1
            ;;
    esac

    # Also log to journal for systemd
    echo "deploy-script[$level]: $message" | systemd-cat
}

# Error handler
error_exit() {
    log ERROR "Deploy failed: $1"
    exit 1
}

cleanup() {
    rm -f "$BACKUP_PATH_FILE"
}

# Check if we're running as the correct user
check_user() {
    if [[ $EUID -eq 0 ]]; then
        error_exit "This script should not be run as root. Run as the mlb user."
    fi

    if [[ "$(whoami)" != "mlb" ]]; then
        error_exit "This script must be run as the mlb user."
    fi
}

check_repo_state() {
    if ! git -C "$REPO_DIR" diff --quiet || ! git -C "$REPO_DIR" diff --cached --quiet; then
        log WARN "Repository has local tracked changes."
        return 1
    fi

    if [[ -n "$(git -C "$REPO_DIR" ls-files --others --exclude-standard)" ]]; then
        log WARN "Repository has untracked files."
        return 1
    fi

    return 0
}

ensure_repo_aligned() {
    # Make on-disk tree and git index match origin/main exactly. This prevents
    # "archive deploy" drift where files update but HEAD stays at an old commit.
    if [[ ! -d "${REPO_DIR}/.git" ]]; then
        log ERROR "No .git directory found; cannot align repository metadata"
        return 1
    fi

    if ! repair_git_permissions; then
        log ERROR "Git metadata not writable; cannot align HEAD to origin/main"
        return 1
    fi

    if ! git -C "$REPO_DIR" fetch origin main; then
        log ERROR "git fetch failed; cannot align HEAD to origin/main"
        return 1
    fi

    git -C "$REPO_DIR" reset --hard origin/main || return 1

    # Keep secrets/state on box, but ensure the repo itself is clean.
    git -C "$REPO_DIR" clean -fd \
        -e '.env' \
        -e 'kalshi-key.pem' \
        -e '.venv/' \
        -e '.cache/' \
        -e 'data/' \
        -e '.pytest_cache/' \
        -e '__pycache__/' || return 1

    if ! check_repo_state; then
        return 1
    fi

    return 0
}

run_systemctl() {
    sudo "$SYSTEMCTL_BIN" "$@"
}

install_stop_loss_units() {
    local svc_src="${REPO_DIR}/deploy/mlb-stop-loss.service"
    local timer_src="${REPO_DIR}/deploy/mlb-stop-loss.timer"
    local svc_dst="/etc/systemd/system/mlb-stop-loss.service"
    local timer_dst="/etc/systemd/system/mlb-stop-loss.timer"

    if [[ ! -f "$svc_src" || ! -f "$timer_src" ]]; then
        log INFO "Stop-loss unit files not present in repo; skipping timer install"
        return 0
    fi

    log INFO "Installing stop-loss systemd units"
    # Important: GitHub Actions deploy runs non-interactively. Some targets may not have
    # passwordless sudo, so prefer `sudo -n` and gracefully skip installation when it
    # would otherwise prompt for a password.
    if ! sudo -n install -o root -g root -m 0644 "$svc_src" "$svc_dst"; then
        log WARN "Could not install ${svc_dst} (sudo requires password or lacks privileges); continuing"
    fi
    if ! sudo -n install -o root -g root -m 0644 "$timer_src" "$timer_dst"; then
        log WARN "Could not install ${timer_dst} (sudo requires password or lacks privileges); continuing"
    fi

    # If the unit files are already present from a previous deploy/manual install, it's
    # still worth trying to (re)enable them even when the update step couldn't run.
    if [[ ! -f "$svc_dst" || ! -f "$timer_dst" ]]; then
        log WARN "Stop-loss unit files not present on target (${svc_dst}, ${timer_dst}); skipping enable/restart"
        return 0
    fi

    run_systemctl daemon-reload || log WARN "Failed to daemon-reload systemd (continuing)"
    run_systemctl enable mlb-stop-loss.timer || log WARN "Failed to enable mlb-stop-loss.timer (continuing)"
    run_systemctl restart mlb-stop-loss.timer || log WARN "Failed to restart mlb-stop-loss.timer (continuing)"
    return 0
}

repair_git_permissions() {
    local git_dir="${REPO_DIR}/.git"
    local objects_dir="${git_dir}/objects"

    [[ -d "$git_dir" ]] || return 1
    [[ -d "$objects_dir" ]] || return 1

    if touch "${objects_dir}/.permtest" 2>/dev/null; then
        rm -f "${objects_dir}/.permtest"
        return 0
    fi

    # Try to self-heal ownership/perms via sudo so HEAD can be advanced
    # rather than silently leaving git index behind on archive deploys.
    if command -v sudo >/dev/null 2>&1; then
        log WARN "Repairing ${git_dir} ownership via sudo"
        sudo chown -R "$(whoami):$(id -gn)" "$git_dir" 2>/dev/null || true
        sudo chmod -R u+rwX "$git_dir" 2>/dev/null || true
    fi

    if touch "${objects_dir}/.permtest" 2>/dev/null; then
        rm -f "${objects_dir}/.permtest"
        return 0
    fi

    log WARN "Direct write to ${objects_dir} failed; falling back to archive deploy"
    return 1
}

get_repo_slug() {
    local remote_url
    remote_url="$(git -C "$REPO_DIR" config --get remote.origin.url 2>/dev/null || true)"

    case "$remote_url" in
        git@github.com:*)
            remote_url="${remote_url#git@github.com:}"
            remote_url="${remote_url%.git}"
            ;;
        https://github.com/*)
            remote_url="${remote_url#https://github.com/}"
            remote_url="${remote_url%.git}"
            ;;
        *)
            remote_url="$DEFAULT_REPO_SLUG"
            ;;
    esac

    printf '%s\n' "$remote_url"
}

sync_repo_from_archive() {
    local repo_slug archive_url tmp_dir extracted_dir

    repo_slug="$(get_repo_slug)"
    archive_url="https://codeload.github.com/${repo_slug}/tar.gz/refs/heads/main"
    tmp_dir="$(mktemp -d)"

    trap 'rm -rf "$tmp_dir"; cleanup' EXIT

    log INFO "Downloading archive from ${archive_url}"
    curl -fsSL "$archive_url" | tar -xzf - -C "$tmp_dir" || error_exit "Failed to download repository archive"

    extracted_dir="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [[ -n "$extracted_dir" ]] || error_exit "Failed to locate extracted archive contents"

    log INFO "Syncing archive contents into ${REPO_DIR}"
    rsync -a --delete \
        --exclude '.env' \
        --exclude 'kalshi-key.pem' \
        --exclude '.venv/' \
        --exclude '.cache/' \
        --exclude 'data/' \
        --exclude '.git/' \
        --exclude '.pytest_cache/' \
        --exclude '__pycache__/' \
        "${extracted_dir}/" "${REPO_DIR}/" || error_exit "Failed to sync archive contents"

    rm -rf "$tmp_dir"
    trap cleanup EXIT
}

deploy_from_archive() {
    local backup_ready="${1:-false}"

    log WARN "Git metadata not writable. Using archive-based deploy path."

    if [[ "$backup_ready" != "true" ]]; then
        create_backup
    fi

    sync_repo_from_archive

    if ! ensure_repo_aligned; then
        error_exit "Archive deploy left repository dirty or misaligned"
    fi

    log INFO "Syncing dependencies with uv"
    uv sync --quiet || error_exit "Dependency sync failed"

    if ! install_stop_loss_units; then
        error_exit "Failed to install stop-loss systemd units"
    fi

    log INFO "Stopping services for restart"
    run_systemctl stop mlb-dashboard || log WARN "Failed to stop mlb-dashboard"
    run_systemctl stop mlb-pipeline || log WARN "Failed to stop mlb-pipeline"
    run_systemctl stop mlb-pipeline-v2 || log WARN "Failed to stop mlb-pipeline-v2"

    log INFO "Starting mlb-dashboard service"
    run_systemctl start mlb-dashboard || error_exit "Failed to start mlb-dashboard"

    log INFO "Starting mlb-pipeline service"
    run_systemctl start mlb-pipeline || error_exit "Failed to start mlb-pipeline"

    log INFO "Starting mlb-pipeline-v2 service"
    run_systemctl start mlb-pipeline-v2 || log WARN "Failed to start mlb-pipeline-v2"

    sleep 5

    if ! health_check; then
        error_exit "Archive deployment failed health check"
    fi

    log INFO "Archive deployment completed successfully"
}

# Create backup of current state
create_backup() {
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="${BACKUP_DIR}/${backup_timestamp}"

    log INFO "Creating backup at ${backup_path}"

    mkdir -p "$backup_path"

    # Get current git commit
    local current_commit=$(git -C "$REPO_DIR" rev-parse HEAD)
    echo "$current_commit" > "${backup_path}/commit.txt"

    # Backup critical files that might change
    cp "${REPO_DIR}/.env" "${backup_path}/" 2>/dev/null || log WARN ".env file not found for backup"
    cp "${REPO_DIR}/kalshi-key.pem" "${backup_path}/" 2>/dev/null || log WARN "kalshi-key.pem not found for backup"

    # Keep only the last 5 backups
    find "$BACKUP_DIR" -maxdepth 1 -type d -name "*_*" | sort | head -n -5 | xargs rm -rf 2>/dev/null || true

    echo "$backup_path" > "$BACKUP_PATH_FILE"
    log INFO "Backup created successfully"
}

# Rollback to previous state
rollback() {
    local backup_path=$(cat "$BACKUP_PATH_FILE" 2>/dev/null || echo "")

    if [[ -z "$backup_path" || ! -d "$backup_path" ]]; then
        error_exit "No backup path found for rollback"
    fi

    log WARN "Rolling back to backup: $backup_path"

    # Get the commit from backup
    local rollback_commit=$(cat "${backup_path}/commit.txt" 2>/dev/null || echo "")

    if [[ -z "$rollback_commit" ]]; then
        error_exit "No commit found in backup for rollback"
    fi

    cd "$REPO_DIR"

    # Reset to backup commit
    git reset --hard "$rollback_commit" || error_exit "Failed to reset to backup commit"

    # Restore env files
    cp "${backup_path}/.env" . 2>/dev/null || log WARN "Could not restore .env from backup"
    cp "${backup_path}/kalshi-key.pem" . 2>/dev/null || log WARN "Could not restore kalshi-key.pem from backup"

    # Sync dependencies
    uv sync --quiet || log WARN "Failed to sync dependencies during rollback"

    log INFO "Rollback completed"
}

# Check if pipeline files changed (affects whether we restart mlb-pipeline.service)
check_pipeline_files_changed() {
    local from_commit=$1
    local to_commit=$2

    if git -C "$REPO_DIR" diff --name-only "$from_commit" "$to_commit" | grep -qE "$PIPELINE_FILES_CHANGED_PATTERNS"; then
        return 0 # Files changed
    else
        return 1 # No pipeline files changed
    fi
}

# Health check function
health_check() {
    local retries=0

    while [[ $retries -lt $MAX_HEALTH_CHECK_RETRIES ]]; do
        if curl -f -s --connect-timeout 5 --max-time "$HEALTH_CHECK_TIMEOUT" "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
            log INFO "Health check passed"
            return 0
        fi

        retries=$((retries + 1))
        log WARN "Health check failed (attempt $retries/$MAX_HEALTH_CHECK_RETRIES), retrying in 10 seconds..."
        sleep 10
    done

    log ERROR "Health check failed after $MAX_HEALTH_CHECK_RETRIES attempts"
    return 1
}

# Main deployment function
deploy() {
    log INFO "Starting deployment process"
    trap cleanup EXIT

    # Check prerequisites
    check_user
    mkdir -p "$(dirname "$LOCK_FILE")"
    exec 9>"$LOCK_FILE"
    flock -n 9 || error_exit "Another deployment is already running"

    cd "$REPO_DIR"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_CACHE_DIR_DEFAULT}"
    mkdir -p "$UV_CACHE_DIR"

    if ! repair_git_permissions; then
        deploy_from_archive
        return 0
    fi

    if ! check_repo_state; then
        log WARN "Repository dirty. Switching to archive deploy."
        deploy_from_archive
        return 0
    fi

    # Verify we're on the main branch
    local current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "main" ]]; then
        error_exit "Not on main branch. Current branch: $current_branch"
    fi

    # Get current commit before pulling
    local pre_deploy_commit=$(git rev-parse HEAD)
    log INFO "Current commit: $pre_deploy_commit"

    # Create backup
    create_backup

    # Fetch and fast-forward to the requested commit
    log INFO "Fetching latest changes from origin"
    if ! git fetch origin; then
        log WARN "Git fetch failed. Switching to archive deploy."
        deploy_from_archive true
        return 0
    fi

    # Check if there are any updates
    local latest_commit=$(git rev-parse origin/main)
    if [[ "$pre_deploy_commit" == "$latest_commit" ]]; then
        log INFO "Already up to date. No deployment needed."
        return 0
    fi

    log INFO "Updating from $pre_deploy_commit to $latest_commit"

    # Check if pipeline files changed
    local pipeline_restart_needed=false
    if check_pipeline_files_changed "$pre_deploy_commit" "$latest_commit"; then
        pipeline_restart_needed=true
        log INFO "Pipeline files changed - mlb-pipeline.service will be restarted"
    else
        log INFO "No pipeline files changed - mlb-pipeline.service will not be restarted"
    fi

    # Reset hard to origin/main so force-pushes (rewritten history) deploy cleanly.
    # Backup created above covers rollback if downstream steps fail.
    git reset --hard origin/main || {
        log ERROR "Failed to reset to origin/main"
        rollback
        error_exit "Reset to origin/main failed and rollback completed"
    }

    if ! ensure_repo_aligned; then
        rollback
        error_exit "Post-merge repository verification failed; rolled back"
    fi

    # Sync dependencies
    log INFO "Syncing dependencies with uv"
    uv sync --quiet || {
        log ERROR "Failed to sync dependencies"
        rollback
        error_exit "Dependency sync failed and rollback completed"
    }

    if ! install_stop_loss_units; then
        log ERROR "Failed to install stop-loss systemd units"
        rollback
        error_exit "Stop-loss unit install failed and rollback completed"
    fi

    # Stop services before restart
    log INFO "Stopping services for restart"
    run_systemctl stop mlb-dashboard || log WARN "Failed to stop mlb-dashboard"

    if [[ "$pipeline_restart_needed" == true ]]; then
        run_systemctl stop mlb-pipeline || log WARN "Failed to stop mlb-pipeline"
        run_systemctl stop mlb-pipeline-v2 || log WARN "Failed to stop mlb-pipeline-v2"
    fi

    # Start services
    log INFO "Starting mlb-dashboard service"
    run_systemctl start mlb-dashboard || {
        log ERROR "Failed to start mlb-dashboard"
        rollback
        run_systemctl start mlb-dashboard || log ERROR "Failed to start mlb-dashboard after rollback"
        error_exit "Service start failed"
    }

    if [[ "$pipeline_restart_needed" == true ]]; then
        log INFO "Starting mlb-pipeline service"
        run_systemctl start mlb-pipeline || {
            log ERROR "Failed to start mlb-pipeline"
            rollback
            run_systemctl start mlb-pipeline || log ERROR "Failed to start mlb-pipeline after rollback"
            run_systemctl start mlb-dashboard || log ERROR "Failed to start mlb-dashboard after rollback"
            error_exit "Pipeline service start failed"
        }

        log INFO "Starting mlb-pipeline-v2 service"
        run_systemctl start mlb-pipeline-v2 || log WARN "Failed to start mlb-pipeline-v2"
    fi

    # Wait a moment for services to initialize
    sleep 5

    # Health check
    if ! health_check; then
        log ERROR "Health check failed after deployment"
        rollback

        # Restart services after rollback
        run_systemctl start mlb-dashboard || log ERROR "Failed to start mlb-dashboard after rollback"
        if [[ "$pipeline_restart_needed" == true ]]; then
            run_systemctl start mlb-pipeline || log ERROR "Failed to start mlb-pipeline after rollback"
        fi

        error_exit "Deployment failed health check - rolled back"
    fi

    log INFO "Deployment completed successfully"
    log INFO "Deployed commit: $latest_commit"

    # Show service status
    log INFO "Service status:"
    run_systemctl status mlb-dashboard --no-pager -l || true
    if [[ "$pipeline_restart_needed" == true ]]; then
        run_systemctl status mlb-pipeline --no-pager -l || true
    fi
}

# Script entry point
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    health-check)
        health_check
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health-check}"
        exit 1
        ;;
esac
