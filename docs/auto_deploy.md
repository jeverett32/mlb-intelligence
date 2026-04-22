# Auto-Deploy with Self-Hosted GitHub Actions Runner

This document explains how to set up automated deployment from GitHub to your homelab using a self-hosted GitHub Actions runner. When you push to the `main` branch, the deployment will automatically run on your homelab.

## Overview

The auto-deploy system consists of:

1. **GitHub Actions Workflow** (`.github/workflows/deploy.yml`) - Triggers on pushes to main
2. **Self-Hosted Runner** - Runs on your homelab LXC container
3. **Deployment Script** (`scripts/deploy.sh`) - Safely updates the application
4. **Health Checks** - Verifies deployment success

## Architecture

```
┌─────────────────────────────┐
│   GitHub Repository         │
│   Push to main branch   ──┐ │
└─────────────────────────────┘ │
                                │ webhook
┌─────────────────────────────┐ │
│   Self-Hosted Runner        │ │
│   (LXC 107 - <REDACTED_IP>)   │◄┘
│   ┌─────────────────────┐   │
│   │  GitHub Actions     │   │
│   │  Agent Process      │   │
│   └─────────────────────┘   │
│             │               │
│             ▼               │
│   ┌─────────────────────┐   │
│   │  Deploy Script      │   │
│   │  /opt/mlb/pipeline/ │   │
│   │  scripts/deploy.sh  │   │
│   └─────────────────────┘   │
└─────────────────────────────┘
```

## Setting Up the Self-Hosted Runner

### 1. Install GitHub Actions Runner on Homelab

SSH into your app server (LXC 107):

```bash
ssh <REDACTED_USER>@<REDACTED_IP>
```

Switch to the mlb user and navigate to the home directory:

```bash
su - mlb
cd /opt/mlb
```

Download and set up the GitHub Actions runner:

```bash
# Create runner directory
mkdir -p /opt/mlb/actions-runner
cd /opt/mlb/actions-runner

# Download the latest runner
curl -o actions-runner-linux-x64-2.334.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.334.0/actions-runner-linux-x64-2.334.0.tar.gz

# Extract the installer
tar xzf ./actions-runner-linux-x64-2.334.0.tar.gz

# Clean up
rm actions-runner-linux-x64-2.334.0.tar.gz
```

### 2. Configure the Runner

Go to your GitHub repository settings to get the registration token:

1. Navigate to `https://github.com/jeverett32/mlb-pipeline/settings/actions/runners`
2. Click "New self-hosted runner"
3. Select "Linux" and "x64"
4. Copy the configuration command (it will include your unique token)

Run the configuration command (replace with your actual token):

```bash
# Example - use the actual command from GitHub
./config.sh --url https://github.com/jeverett32/mlb-pipeline --token YOUR_TOKEN_HERE --labels homelab --name homelab-runner
```

**Important Configuration Options:**
- **Name**: `homelab-runner` (or your preferred name)
- **Labels**: `homelab` (this must match the workflow file)
- **Work folder**: Accept the default (`_work`)
- **Run as service**: We'll set this up next

### 3. Install Runner as a System Service

Install the official runner service as root:

```bash
exit  # Exit from mlb user back to root

chown -R mlb:mlb /opt/mlb/actions-runner
cd /opt/mlb/actions-runner
./svc.sh install mlb
./svc.sh start

# Check status; the service name includes the repo and runner name
systemctl list-units --type=service | grep actions.runner
```

### 4. Configure Sudo Permissions

The deployment script needs to restart systemd services. Add sudo permissions for the mlb user:

```bash
cat > /etc/sudoers.d/mlb-deploy << 'EOF'
# Allow mlb user to manage specific systemd services for deployment
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl start mlb-dashboard
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop mlb-dashboard
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart mlb-dashboard
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl status mlb-dashboard *
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl is-active mlb-dashboard
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl start mlb-pipeline
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop mlb-pipeline
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart mlb-pipeline
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl status mlb-pipeline *
mlb ALL=(ALL) NOPASSWD: /usr/bin/systemctl is-active mlb-pipeline
mlb ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u mlb-dashboard *
mlb ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u mlb-pipeline *
EOF

chmod 440 /etc/sudoers.d/mlb-deploy

# Test sudo permissions
su - mlb -c "sudo systemctl status mlb-dashboard"
```

## How the Deployment Process Works

### Deployment Trigger

1. Developer pushes commits to the `main` branch
2. GitHub webhook triggers the deploy workflow
3. Workflow runs on the self-hosted runner (LXC 107)

### Deployment Steps

1. **Pre-deployment health check** - Verifies dashboard is responding
2. **Git operations** - Fetches latest changes and fast-forwards to `origin/main`
3. **Dependency sync** - Runs `uv sync` to update Python dependencies
4. **Service management**:
   - Always restarts `mlb-dashboard.service`
   - Only restarts `mlb-pipeline.service` if pipeline-related files changed
5. **Post-deployment verification** - Tests health endpoints and service status

### Smart Service Restart Logic

The deployment script analyzes which files changed to minimize service disruptions:

- **Dashboard always restarts**: Ensures web interface reflects latest changes
- **Pipeline conditional restart**: Only restarts if files like `run_pipeline.py`, `fetch/`, `model/`, `bet/`, `db.py`, or `kalshi_client.py` changed

### Rollback Capability

If deployment fails:

1. Automatically rolls back to the previous Git commit
2. Restores service configuration
3. Attempts to restart services
4. Logs failure details for debugging

The deploy script refuses to run if `/opt/mlb/pipeline` has local tracked or untracked changes. That is intentional: this box should behave as an immutable deploy target.

## Security Considerations

### Runner Security

- Runner runs as dedicated `mlb` user with minimal privileges
- Uses systemd security features (`NoNewPrivileges`, `ProtectSystem`, etc.)
- Only has sudo access to specific systemd commands
- Network access is limited to necessary endpoints

### Repository Security

- Runner only responds to events from the configured repository
- Uses GitHub's runner authentication and validation
- Deployment script validates it's running on the correct branch

### Sensitive Data

- Environment variables (`.env`) and keys (`kalshi-key.pem`) are preserved during deployment
- Backup system protects against accidental data loss
- File permissions are maintained

## Monitoring and Troubleshooting

### Check Runner Status

```bash
# Check if runner service is running
systemctl status github-actions-runner

# Check runner logs
journalctl -u github-actions-runner -f

# Check if runner is connected to GitHub
# (Look for "Connected to GitHub" in logs)
```

### Check Deployment History

```bash
# View deployment script logs
journalctl -t deploy-script -f

# View recent deployments
journalctl -u mlb-dashboard --since "1 hour ago"
```

### Common Issues

#### Runner Not Connecting
- Verify network connectivity: `curl -I https://api.github.com`
- Check if token expired (re-run config.sh with new token)
- Ensure firewall allows outbound HTTPS

#### Deployment Failures
- Check disk space: `df -h /opt`
- Verify file permissions: `ls -la /opt/mlb/pipeline`
- Test manual deployment: `su - mlb -c "cd /opt/mlb/pipeline && scripts/deploy.sh"`

#### Service Issues
- Check service status: `systemctl status mlb-dashboard mlb-pipeline`
- Review service logs: `journalctl -u mlb-dashboard -n 50`
- Test health endpoint: `curl http://localhost:<REDACTED_PORT>/health`

### Manual Operations

#### Manual Deployment
```bash
su - mlb
cd /opt/mlb/pipeline
scripts/deploy.sh deploy
```

#### Manual Rollback
```bash
su - mlb
cd /opt/mlb/pipeline
scripts/deploy.sh rollback
```

#### Health Check Only
```bash
scripts/deploy.sh health-check
```

## Updating the Runner

To update the GitHub Actions runner to a newer version:

```bash
# Stop the service
systemctl stop github-actions-runner

# Download new version (check GitHub releases for latest)
su - mlb
cd /opt/mlb/actions-runner
curl -o actions-runner-linux-x64-NEW_VERSION.tar.gz -L https://github.com/actions/runner/releases/download/vNEW_VERSION/actions-runner-linux-x64-NEW_VERSION.tar.gz

# Extract (this will overwrite existing files)
tar xzf ./actions-runner-linux-x64-NEW_VERSION.tar.gz

# Clean up
rm actions-runner-linux-x64-NEW_VERSION.tar.gz

# Start the service
exit  # Back to root
systemctl start github-actions-runner
```

## Maintenance

### Regular Tasks

- **Monitor disk space** in `/opt/mlb` (backups and logs can accumulate)
- **Check runner logs** periodically for connection issues
- **Test deployment process** after system updates
- **Rotate backup files** (deployment script keeps last 5 automatically)

### Backup Retention

The deployment script automatically:
- Creates backups before each deployment
- Keeps the last 5 backup snapshots
- Stores backups in `/opt/mlb/backup/YYYYMMDD_HHMMSS/`

To manually clean up old backups:
```bash
find /opt/mlb/backup -maxdepth 1 -type d -name "*_*" | sort | head -n -5 | xargs rm -rf
```
