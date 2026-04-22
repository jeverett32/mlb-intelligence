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
PIPELINE_FILES_CHANGED_PATTERNS="run_pipeline.py|fetch/|model/|bet/|db.py|kalshi_client.py"
LOCK_FILE="/tmp/mlb-deploy.lock"
BACKUP_PATH_FILE="/tmp/mlb_deploy_backup_path"
SYSTEMCTL_BIN="$(command -v systemctl)"

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
        error_exit "Repository has local tracked changes. Refusing to deploy over a dirty worktree."
    fi

    if [[ -n "$(git -C "$REPO_DIR" ls-files --others --exclude-standard)" ]]; then
        error_exit "Repository has untracked files. Refusing to deploy over a dirty worktree."
    fi
}

run_systemctl() {
    sudo "$SYSTEMCTL_BIN" "$@"
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
    check_repo_state
    mkdir -p "$(dirname "$LOCK_FILE")"
    exec 9>"$LOCK_FILE"
    flock -n 9 || error_exit "Another deployment is already running"

    cd "$REPO_DIR"

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
    git fetch origin || error_exit "Failed to fetch from origin"

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

    # Fast-forward only keeps deploy history linear and avoids accidental merges
    git merge --ff-only origin/main || {
        log ERROR "Failed to fast-forward to origin/main"
        rollback
        error_exit "Fast-forward failed and rollback completed"
    }

    # Sync dependencies
    log INFO "Syncing dependencies with uv"
    uv sync --quiet || {
        log ERROR "Failed to sync dependencies"
        rollback
        error_exit "Dependency sync failed and rollback completed"
    }

    # Stop services before restart
    log INFO "Stopping services for restart"
    run_systemctl stop mlb-dashboard || log WARN "Failed to stop mlb-dashboard"

    if [[ "$pipeline_restart_needed" == true ]]; then
        run_systemctl stop mlb-pipeline || log WARN "Failed to stop mlb-pipeline"
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
