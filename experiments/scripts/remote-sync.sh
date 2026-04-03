#!/usr/bin/env bash
# Sync local code to remote server via git push + SSH pull.
#
# Usage: scripts/remote-sync.sh [key=value overrides for remote_config.py]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load remote config (pass through any override args like host=orca03)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "$@")"

# Use the current local branch, not a config value
BRANCH="$(cd "$PROJECT_DIR" && git rev-parse --abbrev-ref HEAD)"

echo "==> Pushing local changes to origin/$BRANCH..."
cd "$PROJECT_DIR"
git push origin "$BRANCH"

echo "==> Pulling on remote ($host)..."
if ! ssh -o ConnectTimeout=10 "$host" "cd $remote_dir && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH"; then
    echo "Error: SSH connection to $host failed." >&2
    echo "Check that the host is reachable and remote_dir exists." >&2
    exit 1
fi

echo "==> Sync complete."
