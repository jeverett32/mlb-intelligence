# Auto-Deploy with a Self-Hosted GitHub Actions Runner

This document describes how to set up automated deployment from GitHub to a homelab machine using a self-hosted GitHub Actions runner.

## Secrets / hygiene

- Do not put passwords, API keys, hostnames, or IP addresses in docs.
- Keep runtime secrets in `.env` on the deployed machine (not in the repo).

## Overview

The auto-deploy system consists of:

1. **GitHub Actions workflow**: `.github/workflows/deploy.yml`
2. **Self-hosted runner**: installed on the target machine
3. **Deployment script**: `scripts/deploy.sh`
4. **Health checks**: verifies the dashboard after deploy

## Architecture (conceptual)

- Push to `main` triggers the deploy workflow.
- The self-hosted runner executes `scripts/deploy.sh` on the target machine.
- Services are restarted as needed and checked via the dashboard health endpoint.

## Setting up the self-hosted runner

Follow GitHub’s official instructions for installing a self-hosted runner on Linux.

Repository settings path (replace placeholders with your repo):

- `https://github.com/<owner>/<repo>/settings/actions/runners`

When configuring the runner, ensure the labels match what the workflow expects (see `deploy.yml`).

## Sudo permissions

The deploy workflow may need permission to restart services.

Create a tightly-scoped sudoers file that only allows the exact `systemctl` and `journalctl` commands required for your deployment.

## Troubleshooting

- Check runner service status and logs using your system’s service manager.
- Check the dashboard health endpoint (locally on the deployed machine) to confirm the web service is up.

## Files to inspect

- `.github/workflows/deploy.yml`
- `scripts/deploy.sh`
